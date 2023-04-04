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

from . import _C as P

from torch.autograd import Function

from transformers.utils import logging


logger = logging.get_logger(__name__)


class QuantEmbedding(nn.Module):
    """
    Quantized version of :obj:`torch.nn.Embedding`. Adds quantization-specific arguments on top of
    :obj:`torch.nn.Embedding`.

    Args:
        weight_bit (:obj:`int`, `optiona`l, defaults to :obj:`8`):
            Bitwidth for the quantized weight.
        momentum (:obj:`float`, `optional, defaults to :obj:`0.95`):
            Momentum for updating the activation quantization range.
        quant_mode (:obj:`bool`, `optional, defaults to :obj:`False`):
            Whether or not the layer is quantized.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        weight_bit=8,
        momentum=0.95,
        quant_mode=False,
    ):
        super().__init__()
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
        self.register_buffer("weight_scaling_factor", torch.zeros(1))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))

        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.percentile_mode = False
        self.weight_function = SymmetricQuantFunction.apply

    def forward(self, x, positions=None, incremental_state=None):
        if not self.quant_mode:
            return (
                F.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ),
                None,
            )

        if self.training:
            w = self.weight
            w_transform = w.data.detach()
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

            self.weight_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, False)
            self.weight_integer = self.weight_function(
                self.weight, self.weight_bit, self.percentile_mode, self.weight_scaling_factor
            )

        emb_int = F.embedding(
            x,
            self.weight_integer.type(torch.int8),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return emb_int, self.weight_scaling_factor


class QuantAct(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (:obj:`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (:obj:`float`, `optional`, defaults to :obj:`0.95`):
            Momentum for updating the activation quantization range.
        per_channel (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to or not use channel-wise quantization.
        channel_len (:obj:`int`, `optional`, defaults to :obj:`None`):
            Specify the channel length when set the `per_channel` True.
        quant_mode (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit, act_range_momentum=0.95, per_channel=False, channel_len=None, quant_mode=False):
        super().__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.percentile = False
        self.act_function = SymmetricQuantAct.apply

        if not self.per_channel:
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
            self.x_min -= 1e-5
            self.x_max += 1e-5
        else:
            raise NotImplementedError("per-channel mode is not currently supported for activation.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
            f"quant_mode: {self.activation_bit}, Act_min: {self.x_min.item():.2f}, "
            f"Act_max: {self.x_max.item():.2f})"
        )

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        specified_min=None,
        specified_max=None,
    ):

        x_act = x if identity is None else identity + x
        # collect running stats if training
        if self.training:
            assert not self.percentile, "percentile mode is not currently supported for activation."
            assert not self.per_channel, "per-channel mode is not currently supported for activation."
            x_min = x_act.data.min()
            x_max = x_act.data.max()

            assert (
                x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0
            ), "NaN detected when computing min/max of the activation"

            # Initialization
            if self.x_min.min() > -1.1e-5 and self.x_max.max() < 1.1e-5:
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        if not self.quant_mode:
            return x_act, None

        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        # XXX: confirm it's constant expression when inference
        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            quant_act_int = self.act_function(
                    x, self.activation_bit,
                    self.percentile,
                    self.act_scaling_factor)
        else:
            quant_act_int = FixedPointMul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        return quant_act_int, self.act_scaling_factor


# What's the difference between these two?
def round_and_clamp(input, _min : float, _max : float):
    return torch.clamp(input.round(), _min, _max)

def clamp_and_round(input, _min : float, _max : float):
    return torch.round(torch.clamp(input, _min, _max) * scaling_factor)

def _tensor_quant(inputs, amax, num_bits = 8):
    unsigned = False
    max_bound = torch.tensor((2.0**(num_bits - 1 + int(unsigned))) - 1.0, device=amax.device)
    min_bound = -max_bound
    scale = max_bound / amax
    epsilon = 1. / (1<<24)
    if amax <= epsilon:
        scale[True] = 0.
    outputs = torch.clamp((inputs * scale).round_(), min_bound, max_bound)
    if amax <= epsilon:
        scale[True] = 1.
    return outputs, scale

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

    def forward(self, input):
        outputs, self.scale = _tensor_quant(input, self._get_amax())
        return outputs / self.scale.to(input.dtype)


class LinearWeightQuantizer(nn.Module):
    """
    Tensor Quantizer contain same buffers as pytorch-quantization
    """
    def __init__(self, has_amax=True, **kwargs):
        super(LinearWeightQuantizer, self).__init__()

    def _get_amax(self, input):
        if hasattr(self, '_amax'):
            return self._amax
        else:
            amax = input.abs().max()
            return amax

    @property
    def amax(self):
        if self.hasattr('_amax'):
            return self._amax
        else:
            return None

    @amax.setter
    def amax(self, amax):
        if not hasattr(self, "_amax"):
            self.register_buffer('_amax', torch.tensor(amax))
        else:
            torch.tensor(value, device=self._amax.device)
            self._amax.data.copy_(value.data)

    def forward(self, input):
        output, self.scale = _tensor_quant(input, self._get_amax(input))
        return output / self.scale.to(input.dtype)


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

        output = F.linear(quant_input, quant_weight, bias=self.bias)

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


class IntGELU(nn.Module):
    """
    Quantized version of :obj:`torch.nn.GELU`. Adds quantization-specific arguments on top of :obj:`torch.nn.GELU`.

    Args:
        quant_mode (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the layer is quantized.
        force_dequant (:obj:`str`, `optional`, defaults to :obj:`"none"`):
            Force dequantize the layer if either "gelu" or "nonlinear" is given.
    """

    def __init__(self, quant_mode=True, force_dequant="none"):
        super().__init__()
        self.quant_mode = quant_mode

        if force_dequant in ["nonlinear", "gelu"]:
            logger.info("Force dequantize gelu")
            self.quant_mode = False

        if not self.quant_mode:
            self.activation_fn = nn.GELU()

        self.k = 1.4142
        self.const = 14  # dummy integer constant
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor ** 2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor ** 2 * self.coeff[0]

        # avoid overflow
        y_int = floor_ste.apply(y_int / 2 ** self.const)
        scaling_factor = scaling_factor * 2 ** self.const

        return y_int, scaling_factor

    def erf(self, x):
        a = self.coeff[0]
        b = self.coeff[1]
        c = self.coeff[2]

        # sign extract and install using vpand
        s = torch.sign(x)
        x_abs = torch.abs(x)
        x_curv = torch.clip(x_abs, 0, -b)
        y = ((x_curv + b) ** 2 + c)

        return y

    def gelu(self,x):
        input_x = x * torch.rsqrt(torch.tensor([2]))
        y = self.erf(input_x) + 1
        ret = x * 0.5 * y
        return ret

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            return self.activation_fn(x), None

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = 1.0 // sigmoid_scaling_factor

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor


class IntSoftmax(nn.Module):
    """
    Quantized version of :obj:`torch.nn.Softmax`. Adds quantization-specific arguments on top of
    :obj:`torch.nn.Softmax`.

    Args:
        output_bit (:obj:`int`):
            Bitwidth for the layer output activation.
        quant_mode (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the layer is quantized.
        force_dequant (:obj:`str`, `optional`, defaults to :obj:`"none"`):
            Force dequantize the layer if either "softmax" or "nonlinear" is given.
    """

    def __init__(self, output_bit, quant_mode=False, force_dequant="none"):
        super().__init__()
        self.output_bit = output_bit
        self.max_bit = 32
        self.quant_mode = quant_mode

        if force_dequant in ["nonlinear", "softmax"]:
            logger.info("Force dequantize softmax")
            self.quant_mode = False

        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931  # -ln2
        self.const = 30  # dummy integer constant
        self.coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)
        z = (x_int + b_int) * x_int + c_int
        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.const * x0_int)

        z = floor_ste.apply(x_int / x0_int)
        m = x_int - x0_int * z
        exp_int, exp_scaling_factor = self.int_polynomial(m, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.const - z)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** self.const
        return exp_int, scaling_factor

    #
    # decomposite a quantized number into ln2 representation
    #   -ln2 * z + p
    #
    def decomp(self, x):
        with torch.no_grad():
            x0 = 0 - torch.log(torch.tensor([2]))
            r_x0 = torch.reciprocal(x0)

        z = floor_ste.apply(x * r_x0)
        p = x - x0 * z
        return p, z

    # a((x + b) * x + c)
    def polynomial(self, x):
        with torch.no_grad():
            a = self.coef[0]
            b = self.coef[1]
            c = self.coef[2]

        y = a*((x + b) * x + c)
        return y

    def approximate_exp(self, x):
        p, z = self.decomp(x)
        y = self.polynomial(p) * (2**(-z))
        return y

    def forward(self, x, scaling_factor):
        if not self.quant_mode:
            return nn.Softmax(dim=-1)(x), None

        x_int = x
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)

        # Avoid overflow
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor

        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = floor_ste.apply(2 ** self.max_bit / exp_int_sum)
        exp_int = floor_ste.apply(
            exp_int * factor / 2 ** (self.max_bit - self.output_bit))
        scaling_factor = 1 / 2 ** self.output_bit
        return exp_int * scaling_factor, scaling_factor


class IntLayerNorm(nn.Module):
    """
    Quantized version of :obj:`torch.nn.LayerNorm`. Adds quantization-specific arguments on top of
    :obj:`torch.nn.LayerNorm`.

    Args:
        output_bit (:obj:`int`, `optional`, defaults to :obj:`8`):
            Bitwidth for the layer output activation.
        quant_mode (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the layer is quantized.
        force_dequant (:obj:`str`, `optional`, defaults to :obj:`"none"`):
            Force dequantize the layer if either "layernorm" or "nonlinear" is given.
    """

    def __init__(self, normalized_shape, eps, output_bit=8, quant_mode=False, force_dequant="none"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            logger.info("Force dequantize layernorm")
            self.quant_mode = False

        self.register_buffer("shift", torch.zeros(1))
        self.output_bit = output_bit
        self.max_bit = 32
        self.dim_sqrt = None
        self.dim_log = None
        self.activation = QuantAct(self.output_bit, quant_mode=self.quant_mode)

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2 ** self.max_bit)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info(f"Dynamic shift adjustment: {int(shift_old)} -> {int(self.shift)}")

    def overflow_fallback(self, y_int):
        """
        This fallback function is called when overflow is detected during training time, and adjusts the `self.shift`
        to avoid overflow in the subsequent runs.
        """
        self.set_shift(y_int)  # adjusts `self.shift`
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor):
        if not self.quant_mode:
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        # compute sqrt of the feature dimension if it is the first run
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)
            # XXX: presume dim is 2's exponential
            self.dim_log = torch.log2(n).type(torch.int8).item()

        # Normalization: computes mean and variance(std)
        # x_int = x
        # mean_int = round_ste.apply(x.mean(axis=2, keepdim=True))

        adjust = self.shift.type(torch.int).item()
        x_lower = sr_round_na(x, adjust).type(torch.int16)

        # range: 16bit << 10
        Sx = x.sum(axis=-1, keepdim=True)

        # adjust if detected overflow
        Sx_lower = sr_round_na(Sx, adjust)

        # range: 16bit x 16bit possible overflow
        Sx_sqr_lower = (x_lower.type(torch.int32)**2).sum(axis=-1, keepdim=True)

        x_diviate = sr_round_na(
                (x.type(torch.int32) << self.dim_log) - Sx,
                self.dim_log // 2)

        # 1/sqrt(Sx_sqr - Sx * Sx / N + eps)
        var_lower = Sx_sqr_lower - Sx_lower * sr_round_na(Sx_lower, self.dim_log)

        # Second pass, should we undo the shift we did?
        gamma = self.weight * torch.rsqrt(var_lower.type(torch.float))

        x_float = x_diviate.type(torch.float) * gamma * 2**adjust + self.bias
        return x_float


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor

    Args:
        input (:obj:`torch.Tensor`):
            The target tensor to calculate percentile max and min.
        lower_percentile (:obj:`float`):
            If 0.1, means we return the value of the smallest 0.1% value in the tensor as percentile min.
        upper_percentile (:obj:`float`):
            If 99.9, means we return the value of the largest 0.1% value in the tensor as percentile max.
        output_tensor (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, this function returns tensors, otherwise it returns values.

    Returns:
        :obj:`Tuple(torch.Tensor, torch.Tensor)`: Percentile min and max value of `input`
    """
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
        # lower_index += 1
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.

    Args:
        input (:obj:`torch.Tensor`):
            Single-precision input tensor to be quantized.
        scale (:obj:`torch.Tensor`):
            Scaling factor for quantization.
        zero_pint (:obj:`torch.Tensor`):
            Shift for quantization.
        inplace (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to compute inplace or not.

    Returns:
        :obj:`torch.Tensor`: Linearly quantized value of `input` according to `scale` and `zero_point`.
    """
    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # quantized = float / scale + zero_point
    if inplace:
        input.mul_(1.0 / scale).add_(zero_point).round_()
        return input
    return torch.round(1.0 / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, saturation_min, saturation_max, per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.

    Args:
        saturation_min (:obj:`torch.Tensor`):
            Lower bound for quantization range.
        saturation_max (:obj:`torch.Tensor`):
            Upper bound for quantization range.
        per_channel (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to or not use channel-wise quantization.

    Returns:
        :obj:`torch.Tensor`: Scaling factor that linearly quantizes the given range between `saturation_min` and
        `saturation_max`.
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale


class SymmetricQuantAct(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):
        """
        Args:
            x (:obj:`torch.Tensor`):
                Floating point tensor to be quantized.
            k (:obj:`int`):
                Quantization bitwidth.
            percentile_mode (:obj:`bool`):
                Whether or not to use percentile calibration.
            scale (:obj:`torch.Tensor`):
                Pre-calculated scaling factor for `x`. Note that the current implementation of SymmetricQuantFunction
                requires pre-calculated scaling factor.

        Returns:
            :obj:`torch.Tensor`: Symmetric-quantized value of `input`.
        """
        zero_point = torch.tensor(0.0).to(scale.device)

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n-1, n)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):
        """
        Args:
            x (:obj:`torch.Tensor`):
                Floating point tensor to be quantized.
            k (:obj:`int`):
                Quantization bitwidth.
            percentile_mode (:obj:`bool`):
                Whether or not to use percentile calibration.
            scale (:obj:`torch.Tensor`):
                Pre-calculated scaling factor for `x`. Note that the current implementation of SymmetricQuantFunction
                requires pre-calculated scaling factor.

        Returns:
            :obj:`torch.Tensor`: Symmetric-quantized value of `input`.
        """
        zero_point = torch.tensor(0.0).to(scale.device)

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


#
# Round to nearest tie to away
#
def sr_round_na(tmp, shifter):
    if shifter == 0:
        return tmp

    bit = (tmp < 0).type(tmp.dtype)
    nudge = (1<<(shifter -1)) - bit
    tmp = tmp + nudge
    return tmp >> shifter

#
# Round to nearest, tie to even
#
def sr_round_ne(tmp, shifter):
    if shifter == 0:
        return tmp

    bit = ((tmp & (0x1<<shifter) == 0)).type(tmp.dtype)
    nudge = (1<<(shifter -1)) - bit
    tmp = tmp + nudge
    return tmp >> shifter

def sr_floor(tmp, shifter):
    if shifter == 0:
        return tmp

    nudge = (1<<(shifter -1))
    tmp = tmp + nudge
    return tmp >> shifter

# Accuracy > 16 for validation
def integer_multiply_shift_round(x, m, e, accuracy=23):
    #
    # turn m into integer and a slop, and m in [-1, 1]
    #
    if accuracy < 8:
        nm = torch.round(m * 2**accuracy).type(torch.int8)
    elif accuracy < 16:
        nm = torch.round(m * 2**accuracy).type(torch.int16)
    elif accuracy < 32:
        nm = torch.round(m * 2**accuracy).type(torch.int32)
    else:
        nm = torch.round(m * 2**accuracy).type(torch.int64)

    if accuracy < 16:
        tmp = x.type(torch.int32) * nm.type(torch.int32)
    else:
        tmp = x.type(torch.int64) * nm.type(torch.int64)

    if (e - accuracy) >= 0 :
        return tmp << (e - accuracy)
    else:
        return sr_round_na(tmp, (accuracy-e).item())

def integer_multiply_shift_floor(x, m, e, accuracy=23):
    #
    # turn m into integer and a slop, and m in [-1, 1]
    #
    if accuracy < 8:
        nm = torch.round(m * 2**accuracy).type(torch.int8)
    elif accuracy < 16:
        nm = torch.round(m * 2**accuracy).type(torch.int16)
    elif accuracy < 32:
        nm = torch.round(m * 2**accuracy).type(torch.int32)
    else:
        nm = torch.round(m * 2**accuracy).type(torch.int64)

    if accuracy < 16:
        tmp = x.type(torch.int32) * nm.type(torch.int32)
    else:
        tmp = x.type(torch.int64) * nm.type(torch.int64)

    if (e - accuracy) >= 0 :
        return tmp << (e - accuracy)
    else:
        return tmp >> (accuracy-e).item()

def integer_multiply_shift_acc(x, y, mx, sx, my, sy):
    pass

class FixedPointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on hardware.

    Args:
        pre_act (:obj:`torch.Tensor`):
            Input tensor.
        pre_act_scaling_factor (:obj:`torch.Tensor`):
            Scaling factor of the input tensor `pre_act`.
        bit_num (:obj:`int`):
            Quantization bitwidth.
        z_scaling_factor (:obj:`torch.Tensor`):
            Scaling factor of the output tensor.
        identity (:obj:`torch.Tensor`, `optional`, defaults to :obj:`None`):
            Identity tensor, if exists.
        identity_scaling_factor (:obj:`torch.Tensor`, `optional`, defaults to :obj:`None`):
            Scaling factor of the identity tensor `identity`, if exists.

    Returns:
        :obj:`torch.Tensor`: Output tensor(`pre_act` if `identity` is not given, otherwise the addition of `pre_act`
        and `identity`), whose scale is rescaled to `z_scaling_factor`.
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        z_scaling_factor,
        identity=None,
        identity_scaling_factor=None,
    ):

        ctx.identity = identity

        r = 2 ** (bit_num - 1) - 1

        with torch.no_grad():
            ctx.z_scaling_factor = z_scaling_factor

            z_int = pre_act

            # constant expression, could be precalculated
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            m, e = torch.frexp(new_scale)

            #
            # end constant expression
            # Fix representation needed, m in [-1, 1]
            #
            output = integer_multiply_shift_round(z_int, m, e)

            if identity is not None:
                wx_int = identity

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B

                m1, e1 = torch.frexp(new_scale)

                # Fix representation needed, m in [-1, 1]
                # output1 = wx_int.type(torch.double) * m1.type(torch.double)

                # shift operation
                # output1 = torch.round(output1 / (2.0 ** e1))
                output1 = integer_multiply_shift_round(wx_int, m1, e1)
                output = output1 + output

            o = torch.clamp(output, -r-1, r)

            #
            # Integer saturate truncate
            #
            if bit_num <= 8:
                return o.type(torch.int8)
            elif bit_num <= 16:
                return o.type(torch.int16)
            elif bit_num <= 32:
                return o.type(torch.int32)
            else:
                return o.type(torch.int64)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None
