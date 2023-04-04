#! /usr/bin/env python3
# coding=utf-8
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
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

import os
import numpy as np
import time
from scipy import signal
import torch

from global_vars import *
from pathlib import Path

__doc__ = """
Collection of utilities 3D UNet MLPerf-Inference reference model uses.
gaussian_kernel(n, std):
    returns gaussian kernel; std is standard deviation and n is number of points
apply_norm_map(image, norm_map):
    applies normal map norm_map to image and return the outcome
apply_argmax(image):
    returns indices of the maximum values along the channel axis
finalize(image, norm_map):
    finalizes results obtained from sliding window inference
prepare_arrays(image, roi_shape):
    returns empty arrays required for sliding window inference upon roi_shape
get_slice_for_sliding_window(image, roi_shape, overlap):
    returns indices for image stride, to fulfill sliding window inference
timeit(function):
    custom-tailored decorator for runtime measurement of each inference
"""


def gaussian_kernel(n, std):
    """
    Returns gaussian kernel; std is standard deviation and n is number of points
    """
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return gaussian3D


def apply_norm_map(image, norm_map):
    """
    Applies normal map norm_map to image and return the outcome
    """
    image /= norm_map
    return image


def apply_argmax(image):
    """
    Returns indices of the maximum values along the channel axis
    Input shape is (bs=1, channel=3, (ROI_SHAPE)), float -- sub-volume inference result
    Output shape is (bs=1, channel=1, (ROI_SHAPE)), integer -- segmentation result
    """
    channel_axis = 1
    image = np.argmax(image, axis=channel_axis).astype(np.uint8)
    image = np.expand_dims(image, axis=0)

    return image


def finalize(image, norm_map, normalize=False):
    """
    Finalizes results obtained from sliding window inference
    """
    # NOTE: layout is assumed to be linear (NCDHW) always
    # apply norm_map
    if normalize:
        image = apply_norm_map(image, norm_map)

    # argmax
    image = apply_argmax(image)

    return image


def prepare_arrays(image, roi_shape=ROI_SHAPE):
    """
    Returns empty arrays required for sliding window inference such as:
    - result array where sub-volume inference results are gathered
    - norm_map where normal map is constructed upon
    - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"

    image_shape = list(image.shape[2:])

    result = torch.zeros((1, 3, *image_shape), dtype=image.dtype).to(memory_format=torch.channels_last_3d)
    norm_map = torch.zeros_like(result)
    norm_patch = gaussian_kernel(
        roi_shape[0], 0.125*roi_shape[0]).astype(np.float32)

    return result, norm_map, norm_patch


def get_slice_for_sliding_window(image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = list(image.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) //
            strides[i] + 1 for i in range(dim)]

    i_range = range(0, strides[0] * size[0], strides[0])
    j_range = range(0, strides[1] * size[1], strides[1])
    k_range = range(0, strides[2] * size[2], strides[2])
    total_itr_left = len(i_range) * len(j_range) * len(k_range)
    for i in i_range:
        for j in j_range:
            for k in k_range:
                total_itr_left -= 1
                yield i, j, k, total_itr_left


def runtime_measure(function):
    """
    A decorator for runtime measurement
    Custom-tailored for measuring inference latency
    Also prints str: mystr that summarizes work in SUT
    """

    def get_latency(*args, **kw):
        ts = time.time()
        result, mystr = function(*args, **kw)
        te = time.time()
        print('{:86} took {:>10.5f} sec'.format(mystr, te - ts))
        return result, ""

    return get_latency


def prepare_arrays_preconditioned(image, roi_shape=ROI_SHAPE):
    """
    Returns empty arrays required for sliding window inference such as:
    - result array where sub-volume inference results are gathered
    - norm_map where normal map is constructed upon
    - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"

    image_shape = list(image.shape[2:])
    result = torch.zeros((1, 3, *image_shape), dtype=image.dtype).to(memory_format=torch.channels_last_3d)
    norm_patches = list()

    preproc_dir_env = os.getenv("GAUSSIAN_DATA_DIR", default="build/gaussian_patches")
    patch_npy_path = Path(preproc_dir_env, "gaussian_patches.pt")

    if patch_npy_path.is_file():
        norm_patches = torch.load(patch_npy_path)
    else:
        norm_map = torch.zeros_like(result)
        norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0]).astype(np.float32)
        norm_patch = torch.from_numpy(norm_patch).float()

        norm_patches = [None, ] * 27

        for i, j, k, _ in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            norm_map_slice = norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_map_slice += norm_patch

        # each dim is: 0 -- start corner, 1 -- middle, 2 -- end corner
        for i, j, k, _ in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            my_id = [0 if i == 0 else 2 if i + ROI_SHAPE[0] == image_shape[0] else 1,
                     0 if j == 0 else 2 if j + ROI_SHAPE[1] == image_shape[1] else 1,
                     0 if k == 0 else 2 if k + ROI_SHAPE[2] == image_shape[2] else 1]
            patch_id = my_id[0] * 9 + my_id[1] * 3 + my_id[2]
            my_slice = norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_patches[patch_id] = norm_patch / my_slice

    return result, norm_patches


def get_slice_for_sliding_window_preconditioned(image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = list(image.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
    i_range = range(0, strides[0] * size[0], strides[0])
    j_range = range(0, strides[1] * size[1], strides[1])
    k_range = range(0, strides[2] * size[2], strides[2])
    total_itr_left = len(i_range) * len(j_range) * len(k_range)
    for i in i_range:
        for j in j_range:
            for k in k_range:
                total_itr_left -= 1
                my_id = [0 if i == 0 else 2 if i + ROI_SHAPE[0] == image_shape[0] else 1,
                         0 if j == 0 else 2 if j + ROI_SHAPE[1] == image_shape[1] else 1,
                         0 if k == 0 else 2 if k + ROI_SHAPE[2] == image_shape[2] else 1]
                patch_id = my_id[0] * 9 + my_id[1] * 3 + my_id[2]
                yield i, j, k, patch_id, total_itr_left
