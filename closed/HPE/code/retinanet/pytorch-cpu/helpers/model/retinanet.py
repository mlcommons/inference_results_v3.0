import math
from collections import OrderedDict, namedtuple
from multiprocessing import dummy
import warnings

import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import Dict, List, Tuple, Optional

from .anchor_utils import AnchorGenerator
from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from .feature_pyramid_network import LastLevelP6P7
from .focal_loss import sigmoid_focal_loss
from .boxes import box_iou, clip_boxes_to_image, batched_nms
from .utils import Matcher, overwrite_eps, BoxCoder, DefaultBoxes

from intel_extension_for_pytorch import nn as ipex_nn

__all__ = [
    "retinanet_resnet50_fpn",
    "retinanet_resnext50_32x4d_fpn",
]

# Type for returning fpn outputs (ar --> aspect_ratio)
fpn_output = namedtuple('FpnOutput', ['ar1', 'ar2', 'ar3', 'ar4', 'ar5'])

def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def retinanet_resnext_anchors():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    image_size = (800,800)
    feature_sz = [[100,100], [50,50], [25,25], [13,13], [7,7]]
    num_anchors_per_level = [a*b for a,b in feature_sz]

    dboxes = DefaultBoxes(anchor_sizes, aspect_ratios)
    anchors = dboxes.get_feature_anchors(feature_sz, image_size)
    num_anchors_per_location = dboxes.num_anchors_per_location()[0]
    
    return anchors, num_anchors_per_location, num_anchors_per_level


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for f,features in enumerate(x):
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape

            cls_logits = cls_logits.view(N, -1, self.num_classes, H*W)

            cls_logits = cls_logits.permute(0, 3, 1, 2)

            #cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)
            cls_logits = cls_logits.reshape(N,-1) # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return fpn_output(*all_cls_logits) 

class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))


    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for f,features in enumerate(x):
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape

            bbox_regression = bbox_regression.view(N, -1, 4, H*W)
            bbox_regression = bbox_regression.permute(0, 3, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return fpn_output(*all_bbox_regression)


class RetinaNet(nn.Module):
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
    }

    def __init__(self, backbone, num_classes,
                 # transform parameters
                 image_size=None,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        self.anchor_generator = anchor_generator

        self.anchors, self.num_anchors_per_location, self.num_anchors_per_level = retinanet_resnext_anchors()

        HW = sum(self.num_anchors_per_level)
        HWA = HW * self.num_anchors_per_location # Unprocessed number of detections
        A = HWA // HW # A is same as self.num_anchors_per_location (per feature)

        self.split_num_anchors_per_level = [hw * A for hw in self.num_anchors_per_level]
        
        self.split_anchors = [list(a.split(self.split_num_anchors_per_level)) for a in self.anchors]
        self.logit_idxs = [torch.arange(num_classes * num) for num in self.split_num_anchors_per_level]

        self.HeadOutput = namedtuple('HeadOutput', ['cls_logits', 'bbox_regression'])
        self.output = namedtuple('output', ['keep', 'drop'])
        if head is None:
            head = RetinaNetHead(backbone.out_channels, self.num_anchors_per_location, num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.num_classes = num_classes

        # used only on torchscript mode
        self._has_warned = False

    def postprocess_detections(self, head_outputs): 
        # type: (NamedTuple[str[Tensor], str[Tensor]],', Tensor], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]

        class_logits = head_outputs[0]
        box_regression = head_outputs[1]

        image_shape=(800,800)
        num_images = box_regression[0].shape[0]

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [box_regression[ar][index] for ar in range(len(box_regression))]

            logits_per_image = [class_logits[ar][index] for ar in range(len(class_logits))]

            anchors_per_image = self.split_anchors[0]

            image_boxes = []
            image_scores = []
            image_labels = []
            i=0
            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image):

                # remove low scoring boxes
                #scores_per_level = torch.sigmoid(logits_per_level)#.flatten()
                #keep_idxs = scores_per_level > self.score_thresh
                #scores_per_level = scores_per_level[keep_idxs]

                keep_idxs = logits_per_level > -2.944 # Transform sigmoid(x) > 0.05
                scores_per_level = logits_per_level[keep_idxs]

                topk_idxs = self.logit_idxs[i][keep_idxs]

                #topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                if (topk_idxs.size(0) > self.topk_candidates):
                    num_topk = min(self.topk_candidates, topk_idxs.size(0))
                    scores_per_level, idxs = scores_per_level.topk(self.topk_candidates)#, sorted=False) #num_topk)
                    topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
                labels_per_level = topk_idxs % self.num_classes

                i += 1

                if box_regression_per_level[anchor_idxs].shape[0] < 1:
                    continue

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = clip_boxes_to_image(boxes_per_level, image_shape)

                #scores_per_level = torch.sigmoid(scores_per_level)
                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            if len(image_boxes)==0:
                continue

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep].sigmoid(),
                'labels': image_labels[keep],
            })

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        # get the features from the backbone
        features = self.backbone(images) #images.tensors) #.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features, and return for post-processing
        head_outputs = self.head(features)
        x = self.HeadOutput(**head_outputs)

        # Insert '_make_per_tensor_quantized_tensor to manipulate input dequantization later on
        dummy = torch._make_per_tensor_quantized_tensor(x.cls_logits.ar1[0].to(torch.int8), 0.02065122127532959, int(0)) # Histogram observer for activations-symmetric-qint8: mAP=53.899 for 128samples, mAP=37.195 for 24781 samples; fps=265fps;

        return self.output(keep=x, drop=dummy)


def retinanet_resnext50_32x4d_fpn(pretrained=False, progress=True,
                                  num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone('resnext50_32x4d', pretrained_backbone, norm_layer=nn.BatchNorm2d, returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256), trainable_layers=trainable_backbone_layers)
    model = RetinaNet(backbone, num_classes, **kwargs)
    if pretrained:
        raise ValueError("Torchvision doesn't have a pretrained retinanet_resnext50_32x4d_fpn model")

    return model
