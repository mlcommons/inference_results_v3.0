import torch
import _C as P

from quant_modules import *
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class QuantLinear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, **kwargs
    ) -> None:
        super(QuantLinear, self).__init__(in_features, out_features, bias, **kwargs)

    def _init_quantizers(self, run_mode=None) -> None:
        # set quant desc
        if run_mode == "calib":
            QUANT_LINEAR_ACTIVA.mode = "calib"
            QUANT_LINEAR_WEIGHT.mode = "quant"
        else:
            QUANT_LINEAR_ACTIVA.mode = run_mode
            QUANT_LINEAR_WEIGHT.mode = run_mode
        self.input_quantizer = TensorQuantizer(QUANT_LINEAR_ACTIVA)
        self.weight_quantizer = WeightQuantizer(QUANT_LINEAR_WEIGHT)

    def _init_parameters(self, weight: Tensor, bias: Tensor) -> None:
        self.weight = weight
        self.bias = bias

    def _quant_parameters(self, run_mode) -> None:
        if run_mode == "fake_quant":
            self.weight_quantizer.amax = torch.max(self.weight.abs())
            self.weight = Parameter(
                self.weight_quantizer(self.weight), requires_grad=False
            )

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "input_quantizer"):
            if self.input_quantizer.mode != None:
                x = self.input_quantizer(x)

        y = F.linear(x, self.weight, self.bias)
        return y


class iLinear(QuantLinear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, **kwargs
    ) -> None:
        super(iLinear, self).__init__(in_features, out_features, bias, **kwargs)

    def _quant_parameters(self, run_mode) -> None:
        if run_mode == "quant":
            self.weight_quantizer.amax = torch.max(self.weight.abs())
            self.weight = Parameter(
                self.weight_quantizer(self.weight), requires_grad=False
            )
            b_scale = self.input_quantizer.scale * self.weight_quantizer.scale
            self.bias = Parameter(self.bias * b_scale, requires_grad=False)
            self.o_scale = 1 / b_scale

    def forward(self, x: Tensor) -> Tensor:
        y = P.linear(x, self.weight, self.bias, self.o_scale.item(), None)
        return y
