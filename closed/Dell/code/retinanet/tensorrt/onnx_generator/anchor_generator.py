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


___doc___ = """Generate the default anchors for the NMS layers usage

The code is modified based on the original version from mlperf training repo:
https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/anchor_utils.py
The anchors can be generated in xywh and ltrb format, and scaled down 800x to fit in [0,1].
"""


import argparse
import json
import os
import sys
import glob
import random
import time
import math
from typing import Dict, Tuple, List, Optional

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import torch  # Retinanet model source requires GPU installation of PyTorch 1.10
    from torch import nn, Tensor
    import onnx
    import onnx_graphsurgeon as onnxgs
    import tensorrt as trt
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

from code.common import logging
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.runner import EngineRunner, get_input_format

# This is migrated and modified from: https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/anchor_utils.py


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Args:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio)
                             for size, aspect_ratio in zip(sizes, aspect_ratios)]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
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

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device)
                             for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

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
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    # def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
    def forward(self, scale_retinanet=False, order="ltrb") -> List[Tensor]:
        # grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        # image_size = image_list.tensors.shape[-2:]
        # Hard coded image size
        grid_sizes = [[100, 100], [50, 50], [25, 25], [13, 13], [7, 7]]
        image_size = [800, 800]
        dtype, device = torch.float32, torch.device("cpu")
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(1):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        # Scale the anchor by the image size (800, 800) for nms
        # TODO: Need to change this.
        if scale_retinanet:
            for idx, anchor in enumerate(anchors):
                anchors[idx] = torch.div(anchor, image_size[0])

        # Calculate the xywh version of the anchor from the ltrb
        if order == "xywh":
            anchors_xywh = []
            for anchor in anchors:
                anchor_xywh = anchor.clone()
                anchor_xywh[:, 0] = anchor[:, 0] + 0.5 * (anchor[:, 2] - anchor[:, 0])
                anchor_xywh[:, 1] = anchor[:, 1] + 0.5 * (anchor[:, 3] - anchor[:, 1])
                anchor_xywh[:, 2] = anchor[:, 2] - anchor[:, 0]
                anchor_xywh[:, 3] = anchor[:, 3] - anchor[:, 1]
                anchors_xywh.append(anchor_xywh)

            return anchors_xywh

        return anchors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',
                        type=str,
                        default='/tmp',
                        help='The directory to store the output npy file for the anchors')
    parser.add_argument('--box-coding',
                        type=str,
                        default='xywh',
                        help='The coding type of the box (ltrb or xywh)',
                        choices={'xywh', 'ltrb'})
    parser.add_argument('--is_scale',
                        action='store_true',
                        help='whether to scale the boxes down by 800x into [0,1]')
    parser.add_argument('--concat_variance',
                        action='store_true',
                        help='whether to concat the variance into the anchor boxes')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.info("Parsed args -- {}: {}".format(key, value))

    return args


def main():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    print("anchor_sizes:", anchor_generator.sizes)
    print("anchor ratios:", anchor_generator.aspect_ratios)

    o_tensor = anchor_generator()[0]
    np_anchor = o_tensor.detach().cpu().numpy()
    # print("anchor_ltrb shape: ", np_anchor.shape, " top 4:", np_anchor[0, :4])
    np.save("/tmp/retinanet_anchor.npy", np_anchor)

    # Generate anchors in xywh format for retinanet. Range [800, 800]
    np_anchor_xywh = anchor_generator(order="xywh")[0].detach().cpu().numpy()
    print("anchor_xywh shape: ", np_anchor_xywh.shape, " top 4:", np_anchor_xywh[0, :4])
    np.save("/tmp/retinanet_anchor_xywh.npy", np_anchor_xywh)

    # Generate anchors sclaed down by 800x800 for retinanet. Range [1,1]
    np_anchor_xywh = anchor_generator(scale_retinanet=True, order="xywh")[0].detach().cpu().numpy()
    print("anchor_xywh_1x1 shape: ", np_anchor_xywh.shape, " top 4:", np_anchor_xywh[0, :4])
    np.save("/tmp/retinanet_anchor_xywh_1x1.npy", np_anchor_xywh)

    # LTRB for nmsopt, scaled to [1,1]
    np_anchor_ltrb = anchor_generator(scale_retinanet=True, order="ltrb")[0].detach().cpu().numpy()
    print("anchor_ltrb_1x1 shape: ", np_anchor_ltrb.shape, " top 4:", np_anchor_ltrb[0, :4])
    np.save("/tmp/retinanet_anchor_ltrb_1x1.npy", np_anchor_ltrb)

    # The tensor size for retinanet should be (120087, 4)
    # (100^2 + 50^2 + 25^2 + 13^2 + 7^2) * 9 anchors
    num_anchors = 120087
    print("np anchor shape:", np_anchor.shape)
    assert np_anchor.shape[0] == num_anchors, "Number of anchors are not matching."

    # The variances seem to be 1 according to
    # https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/retinanet.py#L373
    anchor_flatten = np_anchor.reshape((num_anchors * 4, 1)).astype("float32")

    variances = np.array([1, 1, 1, 1], dtype=np.float32)
    variances = np.tile(variances, num_anchors).reshape(num_anchors * 4, 1)
    concat_anchor_var = np.concatenate((anchor_flatten, variances), axis=0)

    reshaped_concat_anchor_var = concat_anchor_var.reshape((2, num_anchors * 4, 1))
    print(reshaped_concat_anchor_var.shape)
    np.save("/tmp/concatted_anchor_var.npy", reshaped_concat_anchor_var)


if __name__ == "__main__":
    main()
