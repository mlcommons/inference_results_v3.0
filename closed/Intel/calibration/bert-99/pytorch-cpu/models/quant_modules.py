# coding=utf-8
# Copyright 2021 The I-BERT Authors (Sehoon Kim, Amir Gholami, Zhewei Yao,
# Michael Mahoney, Kurt Keutzer - UC Berkeley) and The HuggingFace Inc. team.
# Copyright (c) 20121, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F

import _C as P

from torch.autograd import Function

from transformers.utils import logging


logger = logging.get_logger(__name__)

def round_and_clamp(input, _min : float, _max : float):
    return torch.clamp(input.round(), _min, _max)

def clamp_and_round(input, _min : float, _max : float):
    return torch.round(torch.clamp(input, _min, _max) * scaling_factor)

# TODO: if amax is too small
class TensorQuantizer(nn.Module):
    """
    Tensor Quantizer contain same buffers as pytorch-quantization
    """
    def __init__(self, has_amax=True, **kwargs):
        super(TensorQuantizer, self).__init__()
        self.register_buffer('_amax', torch.tensor(1.0))

    def _get_amax(self):
        return self._amax

    @property
    def scale(self):
        return torch.tensor(127., dtype=torch.float32) / self._get_amax()

    def forward(self, input):
        bound = torch.tensor(127., dtype=torch.float32)
        output = round_and_clamp(input * self.scale, -bound, bound)
        output = output.type(torch.int8)
        return output


class LinearWeightQuantizer(nn.Module):
    """
    Tensor Quantizer contain same buffers as pytorch-quantization
    """
    def __init__(self, has_amax=True, **kwargs):
        super(LinearWeightQuantizer, self).__init__()
        self.register_buffer('_amax', torch.tensor(1.0))

    def _get_amax(self, input):
        amax = input.abs().max()
        return amax

    def scale(self, input):
        bound = torch.tensor(127., dtype=torch.float32)
        return bound / self._get_amax(input)

    def forward(self, input):
        bound = torch.tensor(127., dtype=torch.float32)
        output = round_and_clamp(input * self.scale(input), -bound, bound)
        output = output.type(torch.int8)
        return P.prepack_linear_weight(output)


class QuantLinear(nn.Linear):
    """
    Quantized version of :obj:`torch.nn.Linear`. Add quantization parameters same
    as pytorch-quantization. A placeholder for quant param initialization

    Args:
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self._input_quantizer = TensorQuantizer(True)
        self._weight_quantizer = LinearWeightQuantizer(False)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        i_scale = self._input_quantizer.scale
        w_scale = self._weight_quantizer.scale(self.weight)

        scale = i_scale * w_scale
        bias = self.bias * scale

        output = P.linear(quant_input, quant_weight, bias, None, None)
        output = output / scale

        return output


def hasanyattr(module, name_list):
    for name in name_list:
        if hasattr(module, name):
            return name
    return None

def pushdown_quantizer(module, attr = '_output_quantizer'):
    curr_level = [(name, child) for (name, child) in module.named_children() if 'Bert' in str(child)]
    if len(curr_level) > 0:
        _, last = curr_level[-1]
        last._output_quantizer = module._output_quantizer
        delattr(module, attr)
        pushdown_quantizer(last, attr)

    return


def propagate_quantizer(
    module,
    input_quantizer_list = ['_input_quantizer']):

    """
    Modify nn.Modules and make output quantizer ref to input quantizer of next
    Layer. Presume no alias scenario

    Args:
    """
    curr_level = [(name, child) for (name, child) in module.named_children()]

    if len(curr_level) is 0:
        return module

    _, child = curr_level[0]
    child = propagate_quantizer(child)

    quantizer_name = hasanyattr(child, input_quantizer_list)
    if quantizer_name:
        module._input_quantizer = getattr(child, quantizer_name)
        if type(child) is nn.ModuleList:
            delattr(child, quantizer_name)

    for i in range(1, len(curr_level)):
        _, curr = curr_level[i]
        _, last = curr_level[i-1]

        curr = propagate_quantizer(curr)

        quantizer_name = hasanyattr(curr, input_quantizer_list)
        if quantizer_name:
            if not hasattr(last, '_output_quantizer'):
                last._output_quantizer = getattr(curr, quantizer_name)
                if last is nn.ModuleList:
                    delattr(curr, quantizer_name)
                pushdown_quantizer(last, '_output_quantizer')

        if hasattr(curr, 'final_input_quantizer'):
            curr._output_quantizer = curr.final_input_quantizer
            pushdown_quantizer(curr, '_output_quantizer')

    return module
