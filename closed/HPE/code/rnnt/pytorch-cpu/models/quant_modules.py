import torch
import torch.nn as nn
import _C as P

from torch import Tensor


def round_and_clamp(inputs: Tensor, _min: float, _max: float):
    return torch.clamp(inputs.round(), _min, _max)


def clamp_and_round(inputs: Tensor, _min: float, _max: float):
    return torch.round(torch.clamp(inputs, _min, _max))


class QuantDescriptor:
    def __init__(self, axis=None, amax=None, mode="quant", **kwargs):
        self._axis = axis
        self._amax = amax
        self._mode = mode
        self._update_amax = kwargs.pop("update_amax", False)
        self._narrow_bound = kwargs.pop("narrow_bound", False)

    @property
    def axis(self):
        return self._axis

    @property
    def amax(self):
        return self._amax

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def update_amax(self):
        return self._update_amax

    @property
    def narrow_bound(self):
        return self._narrow_bound


QUANT_LINEAR_ACTIVA = QuantDescriptor(
    axis=None, amax=None, mode="fake_quant", update_amax=False
)
QUANT_LINEAR_WEIGHT = QuantDescriptor(axis=None, amax=None, mode="fake_quant")
QUANT_LSTM_ACTIVA = QuantDescriptor(
    axis=None, amax=None, mode="fake_quant", update_amax=False
)
QUANT_LSTM_WEIGHT = QuantDescriptor(axis=None, amax=None, mode="fake_quant")


class TensorQuantizer(nn.Module):
    """
    Tensor Quantizer contain same buffers as pytorch-quantization

    Args:
        mode: calib/quant/fake_quant

    """

    def __init__(self, quant_desc=QuantDescriptor(), **kwargs):
        super(TensorQuantizer, self).__init__()
        self._mode = quant_desc.mode
        self._axis = quant_desc._axis
        self._max_bound = torch.tensor(127.0, dtype=torch.float32)
        self._min_bound = (
            -self._max_bound if quant_desc.narrow_bound else -self._max_bound - 1
        )
        self.amax = (
            torch.tensor(quant_desc._amax)
            if quant_desc._amax != None
            else torch.tensor(0.0)
        )
        self._scale = torch.tensor(0.0)
        self._name = kwargs.pop("name", "TensorQuantizer")
        self._update_amax = quant_desc.update_amax  # dynamic
        self._track_amax = False
        self._amaxs = []

    @property
    def mode(self):
        return self._mode

    @property
    def amax(self):
        return self._amax if hasattr(self, "_amax") else None

    @amax.setter
    def amax(self, value):
        if not hasattr(self, "_amax"):
            self.register_buffer("_amax", value)
        else:
            self._amax = value

    @property
    def scale(self):
        return self._max_bound / self.amax

    @scale.setter
    def scale(self, value):
        self._scale = value

    @torch.jit.ignore
    def calib_amax(self, inputs):
        cur_amax = inputs.abs().max()
        self.amax = torch.max(self.amax, cur_amax)
        if self._track_amax:
            self._amaxs.append(cur_amax)

    @torch.jit.export
    def _quant_forward(self, inputs: Tensor) -> Tensor:
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs = outputs.type(torch.int8)
        return outputs

    @torch.jit.ignore
    def _fake_quant_forward(self, inputs):
        if self._update_amax:
            if self._axis != None:
                self.amax = torch.max(inputs.abs(), self._axis).values.unsqueeze(
                    self._axis
                )
            else:
                self.amax = torch.max(inputs.abs())

        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs /= self.scale
        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        if self._mode == "calib":
            self.calib_amax(inputs)
        elif self._mode == "quant":
            outputs = self._quant_forward(inputs)
        elif self._mode == "fake_quant":
            outputs = self._fake_quant_forward(inputs)
        return outputs

    def __str__(self):
        s = f" name=${self._name}\n"
        s += f" mode=${self._mode}\n"
        s += f" axis=${self._axis}\n"
        s += f" amax=${self._amax}\n"
        s += f" scale=${self._scale}\n"
        s += f" max_bound=${self._max_bound}\n"
        s += f" min_bound=${self._min_bound}"
        return s


def transpose_tile_weight_bf16(weight, padding: bool = False):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = (col + 31) // 32
    col_tile = (row + 31) // 32
    if padding:
        pad_size = (0, col_step * 32 - col, 0, col_tile * 32 - row)
        weight = torch.nn.functional.pad(weight, pad_size, "constant", 0.0)

    weight = weight.view(col_tile * 16, 2, col_step * 32)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col_step * 64)
    weight = weight.view(col_tile, 16, col_step * 64)
    weight = weight.view(col_tile, 16, col_step, 2, 32)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()

    return weight


def transpose_tile_weight(weight, padding: bool = False):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = (col + 63) // 64
    col_tile = (row + 63) // 64
    if padding:
        pad_size = (0, col_step * 64 - col, 0, col_tile * 64 - row)
        weight = torch.nn.functional.pad(weight, pad_size, "constant", 0.0)

    weight = weight.view(col_tile * 16, 4, col_step * 64)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col_step * 256)
    weight = weight.view(col_tile, 16, col_step * 256)
    weight = weight.view(col_tile, 16, col_step, 4, 64)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()

    return weight


class WeightQuantizer(TensorQuantizer):
    def __init__(self, quant_desc=QuantDescriptor(), **kwargs):
        super(WeightQuantizer, self).__init__(quant_desc)

    def _quant_forward(self, inputs: Tensor, first_layer: bool = False) -> Tensor:
        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs = outputs.type(torch.int8)
        if first_layer:
            # prepacked_outputs = P.prepack_linear_weight(outputs)
            prepacked_outputs = transpose_tile_weight(outputs.transpose(1, 0), True)
        else:
            prepacked_outputs = transpose_tile_weight(outputs.transpose(1, 0))
        return prepacked_outputs

    @torch.jit.ignore
    def _fake_quant_forward(self, inputs):
        if self._scale == None:
            if self._axis != None:
                self.amax = torch.max(inputs.abs(), self._axis).values.unsqueeze(
                    self._axis
                )
            else:
                self.amax = torch.max(inputs.abs())
            self.scale = self._max_bound / self.amax

        outputs = round_and_clamp(inputs * self.scale, self._min_bound, self._max_bound)
        outputs /= self.scale
        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        if self._mode == "quant":
            outputs = self._quant_forward(inputs)
        elif self._mode == "fake_quant":
            outputs = self._fake_quant_forward(inputs)
        return outputs
