# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from code.common.fix_sys_path import ScopedRestrictedImport

import argparse
import os
import pprint
import sys
import torch
import torch.onnx
import torchvision
from importlib import import_module


retinanet_training_path = "build/training/single_stage_detector/ssd"
with ScopedRestrictedImport([retinanet_training_path] + sys.path):
    retinanet_from_backbone = import_module("model.retinanet").retinanet_from_backbone


MODEL_DIR = os.path.join(os.environ.get("MLPERF_SCRATCH_PATH", "/home/mlperf_inference_data"),
                         "models", "retinanet-resnext50-32x4d")
MODEL_BACKBONE = "resnext50_32x4d"
N_CLASSES = 264
N_TRAINABLE_BACKBONE_LAYERS = 3
IMAGE_SIZE = (800, 800)
INPUT_FORMAT = "channels_first"  # Other option is 'channels_last' (NHWC), but NCHW is default


class RetinanetClassificationHeadWrapper(torch.nn.Module):
    def __init__(self, class_head):
        super().__init__()
        self.class_head = class_head

    def forward(self, x):
        # type: (List[Tensor]) -> (Dict[str, Tensor])
        all_cls_logits = dict()

        for features in x:
            cls_logits = self.class_head.conv(features)
            cls_logits = self.class_head.cls_logits(cls_logits)

            # Shape: (N, A * K, H, W)
            N, _, H, W = cls_logits.shape
            key = f"classification_head_{H}x{W}"
            print(f"{key}: {cls_logits.shape}")
            all_cls_logits[key] = cls_logits

        return all_cls_logits


class RetinanetRegressionHeadWrapper(torch.nn.Module):
    def __init__(self, reg_head):
        super().__init__()
        self.reg_head = reg_head

    def forward(self, x):
        # type: (List[Tensor]) -> (Dict[str, Tensor])
        all_bbox_regression = dict()

        for features in x:
            bbox_regression = self.reg_head.conv(features)
            bbox_regression = self.reg_head.bbox_reg(bbox_regression)

            # Shape: (N, 4 * A, H, W)
            N, _, H, W = bbox_regression.shape
            key = f"regression_head_{H}x{W}"
            print(f"{key}: {bbox_regression.shape}")
            all_bbox_regression[key] = bbox_regression

        return all_bbox_regression


class RetinanetHeadWrapper(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.classification_head = RetinanetClassificationHeadWrapper(head.classification_head)
        self.regression_head = RetinanetRegressionHeadWrapper(head.regression_head)

    def forward(self, x):
        logits_out = self.classification_head(x)
        bbox_out = self.regression_head(x)
        outputs = dict()
        # Note: we can't just use dict.update since Python dicts preserve ordering of inserted elements.
        # The expected output order is interleaved classification and regression outputs, in decreasing featuremap size.
        for size in [100, 50, 25, 13, 7]:
            logits_key = f"classification_head_{size}x{size}"
            outputs[logits_key] = logits_out[logits_key]

            bbox_key = f"regression_head_{size}x{size}"
            outputs[bbox_key] = bbox_out[bbox_key]
        return outputs


class RetinanetFPNWrapper(torch.nn.Module):
    def __init__(self, rtnet_module):
        super().__init__()
        self.rtnet_module = rtnet_module
        self.rtnet_module.head = RetinanetHeadWrapper(self.rtnet_module.head)

    def forward(self, x):
        features = self.rtnet_module.backbone(x)
        if isinstance(features, torch.Tensor):
            features = [features]
        elif isinstance(features, dict):
            features = list(features.values())
        else:
            raise ValueError(f"Unexpected type for RetinaNet features")

        return self.rtnet_module.head(features)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        default=os.path.join(MODEL_DIR, "retinanet.pth"),
                        help='.pth checkpoint file')
    parser.add_argument('--output',
                        default=os.path.join(MODEL_DIR, "retinanet-fpn.onnx"),
                        help='Path to save ONNX file')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='Input batch size override. If not, uses dynamic batch size')
    parser.add_argument('--device', default='cuda', help='device')
    return parser.parse_args()


def main(args):
    dynamic_batch_size = (args.batch_size is None)
    effective_batch_size = args.batch_size or 1

    print(">>> Creating model")
    model = retinanet_from_backbone(backbone=MODEL_BACKBONE,
                                    num_classes=N_CLASSES,
                                    image_size=IMAGE_SIZE,
                                    data_layout=INPUT_FORMAT,
                                    pretrained=False,
                                    trainable_backbone_layers=N_TRAINABLE_BACKBONE_LAYERS)
    device = torch.device(args.device)
    model.to(device)

    print("... Loading checkpoint")
    ckpt = torch.load(args.input)

    # For some reason the batchnorms in older checkpoint files during development do not have the same sizes as the
    # module object. The checkpoint batchnorms have a size of [1, N, 1, 1], while the model batchnorms just have a size
    # of [N].  However, this is fine, since (assuming the README in the retinanet repo is correct), the batchnorms were
    # frozen and were not modified during training.
    target_state_dict = model.state_dict()
    for k, v in target_state_dict.items():
        ckpt_val = ckpt["model"][k]
        if v.size() == ckpt_val.size():
            continue
        target_size = torch.tensor(v.size())
        actual_size = torch.tensor(ckpt_val.size())
        flattened = torch.flatten(actual_size)
        if all(target_size != flattened):
            raise ValueError(f"Real size mismatch for {k}: {target_size} vs {actual_size}")
        ckpt["model"][k] = ckpt["model"][k].view(target_size)
        print(f"... Reshaped '{k}':{actual_size} in ckpt to {target_size}")
    # Remove unexpected keys
    for k in [k for k in ckpt["model"] if k not in target_state_dict]:
        del ckpt["model"][k]
        print(f"... Removing unused key '{k}'")
    model.load_state_dict(ckpt['model'])
    model = RetinanetFPNWrapper(model)
    print("... Done")

    print(">>> Creating ONNX")
    rand = torch.randn(effective_batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1],
                       device=device,
                       requires_grad=False,
                       dtype=torch.float)
    inputs = torch.autograd.Variable(rand)

    print("... Setting dynamic axes")
    dynamic_axes = dict()
    output_names = []
    if dynamic_batch_size:
        dynamic_axes["images"] = {0: "batch_size"}
        print("... Dynamic batch size enabled")

    # Just like above in RetinanetHeadWrapper, dict ordering matters - dynamic_axes and output_names must follow the
    # same order as the outputs in RetinanetHeadWrapper, which is the ordering expected by NMSOptPlugin.
    for size in [100, 50, 25, 13, 7]:
        class_head_axes = dict()
        if dynamic_batch_size:
            class_head_axes[0] = "batch_size"
        class_head_axes[1] = f"klass_x_anchors"
        class_head_axes[2] = f"H_features_{size}"
        class_head_axes[3] = f"W_features_{size}"

        class_head_name = f"classification_head_{size}x{size}"
        output_names.append(class_head_name)
        dynamic_axes[class_head_name] = class_head_axes

        reg_head_axes = dict()
        if dynamic_batch_size:
            reg_head_axes[0] = "batch_size"
        reg_head_axes[1] = "coords_x_anchors"
        reg_head_axes[2] = f"H_features_{size}"
        reg_head_axes[3] = f"W_features_{size}"

        reg_head_name = f"regression_head_{size}x{size}"
        output_names.append(reg_head_name)
        dynamic_axes[reg_head_name] = reg_head_axes

    print("... Dynamic axes:")
    pprint.pprint(dynamic_axes)

    print("... Exporting model")
    model.eval()
    torch.onnx.export(model,
                      inputs,
                      args.output,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


if __name__ == "__main__":
    main(parse_args())
