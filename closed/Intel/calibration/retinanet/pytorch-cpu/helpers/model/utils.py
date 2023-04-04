import math
import torch

from collections import OrderedDict
from torch import Tensor, nn
from typing import List, Tuple, Dict

from torchvision.ops.misc import FrozenBatchNorm2d


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Args:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] # - boxes[:, 0]
        heights = boxes[:, 3]# - boxes[:, 1]
        ctr_x = boxes[:, 0] #+ 0.5 * widths
        ctr_y = boxes[:, 1] #+ 0.5 * heights

        #print("======boxes shape: {}".format(boxes.shape))
        wx, wy, ww, wh = self.weights
        # rel_codes[:, 2] = rel_codes[:, 2] / ww # Dont need these weight scaling unlike in SSDR34
        # rel_codes[:, 3] = rel_codes[:, 3] / wh
        # rel_codes[:, 0] = rel_codes[:, 0] / wx
        # rel_codes[:, 1] = rel_codes[:, 1] / wy

        #dx = rel_codes[:, 0::4] / wx
        #dy = rel_codes[:, 1::4] / wy
        #dw = rel_codes[:, 2::4] / ww
        #dh = rel_codes[:, 3::4] / wh

        #---------------------------------------------------------------------------
        # Prevent sending too large values into torch.exp()
        rel_codes[:, 2] = torch.clamp(rel_codes[:, 2], max=self.bbox_xform_clip)
        rel_codes[:, 3] = torch.clamp(rel_codes[:, 3], max=self.bbox_xform_clip)

        #dw = torch.clamp(dw, max=self.bbox_xform_clip)
        #dh = torch.clamp(dh, max=self.bbox_xform_clip)

        #---------------------------------------------------------------------------
        rel_codes[:, :2] = rel_codes[:, :2] * boxes[:, 2:] + boxes[:, :2]
        rel_codes[:, 2:] = rel_codes[:, 2:].exp() * boxes[:, 2:]

        #pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        #pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        #pred_w = torch.exp(dw) * widths[:, None]
        #pred_h = torch.exp(dh) * heights[:, None]

        #---------------------------------------------------------------------------
        # Distance from center to box's corner.
        l, t, r, b = rel_codes[:, 0] - 0.5*rel_codes[:, 2],\
                     rel_codes[:, 1] - 0.5*rel_codes[:, 3],\
                     rel_codes[:, 0] + 0.5*rel_codes[:, 2],\
                     rel_codes[:, 1] + 0.5*rel_codes[:, 3]

        rel_codes[:, 0] = l
        rel_codes[:, 1] = t
        rel_codes[:, 2] = r
        rel_codes[:, 3] = b

        #c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        #c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        #pred_boxes1 = pred_ctr_x - c_to_c_w
        #pred_boxes2 = pred_ctr_y - c_to_c_h
        #pred_boxes3 = pred_ctr_x + c_to_c_w
        #pred_boxes4 = pred_ctr_y + c_to_c_h
        #pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return rel_codes #pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class SSDMatcher(Matcher):

    def __init__(self, threshold):
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix):
        matches = super().__call__(match_quality_matrix)

        # For each gt, find the prediction with which it has the highest quality
        _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(highest_quality_pred_foreach_gt.size(0),
                                                                dtype=torch.int64,
                                                                device=highest_quality_pred_foreach_gt.device)

        return matches


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def retrieve_out_channels(model, size):
    """
    This method retrieves the number of output channels of a specific model.

    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.

    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels



class DefaultBoxes(object):

    def __init__(self, sizes, aspect_ratios):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        self.cell_anchors = [self.generate_anchors(size, aspect_ratio) 
                for size, aspect_ratio in zip(sizes, aspect_ratios)]

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype: torch.dtype = torch.float32,
                             device: torch.device = torch.device("cpu")):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width)*stride_width #, dtype=torch.float32, device="cpu"
            shifts_y = torch.arange(0, grid_height)*stride_height #, dtype=torch.float32, device=device

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.

            dboxes = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)

            #widths = dboxes[:, 2] - dboxes[:, 0]
            #heights = dboxes[:, 3] - dboxes[:, 1]
            #ctr_x = dboxes[:, 0] + 0.5 * widths
            #ctr_y = dboxes[:, 1] + 0.5 * heights

            # Simplifies above four formulas
            dboxes[:, 2:] = dboxes[:, 2:] - dboxes[:, :2]
            dboxes[:, :2] = dboxes[:, :2] + 0.5*dboxes[:, 2:]
            anchors.append(dboxes)

            #anchors.append(
            #    (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            #)

        return anchors

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=torch.int64, device="cpu")
                             for cell_anchor in self.cell_anchors]

    def get_feature_anchors(self, feature_grid_sizes: List[List[Tensor]], image_size=(800,800)) -> List[Tensor]:
        grid_sizes = feature_grid_sizes #[feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = torch.int64, torch.device("cpu") #feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device="cpu"),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device="cpu")] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        return [torch.cat(anchors_in_image)]

