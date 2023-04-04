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


import torch
import torch.nn as nn


# For modular building of the 3D-UNet
convolutions = {"transpose": nn.ConvTranspose3d, "regular": nn.Conv3d}

activations = {
    "relu": nn.ReLU(),
    "none": nn.Identity(),
}

normalizations = {
    "instancenorm": lambda n, _: nn.InstanceNorm3d(n, affine=True),
    "none": lambda _, __: nn.Identity(),
}

def _normalization(norm_type, num_features, num_groups=16):
    """
    A helper redirecting normalization function used in 3D-UNet
    """
    if norm_type in normalizations:
        return normalizations[norm_type](num_features, num_groups)
    raise ValueError(f"Unknown normalization {norm_type}")

def _activation(activation):
    """
    A helper redirecting activation function used in 3D-UNet
    """
    if activation in activations:
        return activations[activation]
    raise ValueError(f"Unknown activation {activation}")

def conv_block_factory(in_channels, out_channels,
                       kernel_size=3, stride=1, padding=1,
                       conv_type="regular",
                       normalization="instancenorm", activation="relu"):
    """
    A method used for building basic 3D Convolution block of 3D-UNet
    """
    conv = convolutions[conv_type]
    conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=normalization=="none")

    normalization = _normalization(normalization, out_channels)
    activation = _activation(activation)

    return nn.Sequential(conv, normalization, activation)    


class DownsampleBlock(nn.Module):
    """
    A class building encoder block of 3D-UNet
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, stride=2,
                                        normalization="instancenorm", activation="relu")
        self.conv2 = conv_block_factory(out_channels, out_channels, 
                                        normalization="instancenorm", activation="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(nn.Module):
    """
    A class building decoder block of 3D-UNet
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_conv = conv_block_factory(in_channels, out_channels,
                                                kernel_size=2, stride=2, padding=0,
                                                conv_type="transpose", 
                                                normalization="none", activation="none")

        self.conv1 = conv_block_factory(2 * out_channels, out_channels,
                                        normalization="instancenorm", activation="relu")
        self.conv2 = conv_block_factory(out_channels, out_channels,
                                        normalization="instancenorm", activation="relu")

    def forward(self, x, skip):
        x = self.upsample_conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InputBlock(nn.Module):
    """
    A class building the very first input block of 3D-UNet
    """
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, 
                                        normalization="instancenorm", activation="relu")
        self.conv2 = conv_block_factory(out_channels, out_channels,
                                        normalization="instancenorm", activation="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutputLayer(nn.Module):
    """
    A class building final output block of 3D-UNet
    """
    def __init__(self, in_channels, n_class):
        super(OutputLayer, self).__init__()
        self.conv = conv_block_factory(in_channels, n_class, kernel_size=1, padding=0,
                                       activation="none", normalization="none")

    def forward(self, x):
        return self.conv(x)


class Unet3D(nn.Module):
    """
    A class to build the 3D-UNet model used in MLPerf-Training:
    https://github.com/mlcommons/training/blob/master/image_segmentation/pytorch

    Attributes
    ----------
    in_channels: int
        number of channels of the input tensor
    n_class: int
        number of classes the segmentation ends up for
    """
    def __init__(self, in_channels=1, n_class=3):
        """
        Constructs 3D-UNet as in MLPerf-Training 3D-UNet:
        https://github.com/mlcommons/training/blob/master/image_segmentation/pytorch

        """        
        super(Unet3D, self).__init__()

        filters = [32, 64, 128, 256, 320]
        self.filters = filters

        self.inp = filters[:-1]
        self.out = filters[1:]
        input_dim = filters[0]

        self.input_block = InputBlock(in_channels, input_dim)

        self.downsample = nn.ModuleList(
            [DownsampleBlock(i, o)
             for (i, o) in zip(self.inp, self.out)]
        )
        self.bottleneck = DownsampleBlock(filters[-1], filters[-1])
        upsample = [UpsampleBlock(filters[-1], filters[-1])]
        upsample.extend([UpsampleBlock(i, o)
                         for (i, o) in zip(reversed(self.out), reversed(self.inp))])
        self.upsample = nn.ModuleList(upsample)
        self.output = OutputLayer(input_dim, n_class)

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= 1.0

    def forward(self, x):
        x = self.input_block(x)
        outputs = [x]

        for downsample in self.downsample:
            x = downsample(x)
            outputs.append(x)

        x = self.bottleneck(x)

        for upsample, skip in zip(self.upsample, reversed(outputs)):
            x = upsample(x, skip)

        x = self.output(x)

        return x

