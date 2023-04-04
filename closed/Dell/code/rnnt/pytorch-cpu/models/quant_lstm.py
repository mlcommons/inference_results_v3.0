import torch
import _C as P

from quant_modules import *
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple


class iLSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers, skip_quant_y, **kwargs):
        super(iLSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)
        self.skip_quant_y = skip_quant_y
        self.weights = []
        self.rb_scale = torch.zeros(num_layers)
        self.in_scale = torch.zeros(num_layers)
        self.out_scale = torch.zeros(num_layers)
        self.run_mode = None

    def _init_layers(self, run_mode=None):
        input_size = self.input_size
        for layer in range(self.num_layers):
            self.run_mode = run_mode
            # set quant desc
            if run_mode == "calib":
                QUANT_LSTM_ACTIVA.mode = "calib"
                QUANT_LSTM_WEIGHT.mode = "quant"
            else:
                QUANT_LSTM_ACTIVA.mode = run_mode
                QUANT_LSTM_WEIGHT.mode = run_mode
            # create lstm layer
            if run_mode == "quant":
                lstm_layer = iLSTMLayer(input_size, self.hidden_size)
            else:
                lstm_layer = QuantLSTMLayer(input_size, self.hidden_size)

            if run_mode != None and run_mode != "f32":
                lstm_layer._init_quantizers(
                    WeightQuantizer(QUANT_LSTM_WEIGHT),
                    TensorQuantizer(QUANT_LSTM_ACTIVA),
                    TensorQuantizer(QUANT_LSTM_ACTIVA),
                )

            lstm_layer._init_parameters(
                self.all_weights[layer][0],
                self.all_weights[layer][1],
                self.all_weights[layer][2],
                self.all_weights[layer][3],
            )

            setattr(self, f"lstm{layer}", lstm_layer)
            input_size = self.hidden_size

    def _process_parameters(self, run_mode):
        for layer in range(self.num_layers):
            if run_mode == "quant" or run_mode == "fake_quant":
                getattr(self, f"lstm{layer}")._quant_parameters(
                    self.weights, self.rb_scale
                )
            delattr(self, self._all_weights[layer][0])
            delattr(self, self._all_weights[layer][1])
            delattr(self, self._all_weights[layer][2])
            delattr(self, self._all_weights[layer][3])

    def _propagate_quantizers(self):
        # per-layer
        for layer in range(self.num_layers):
            if layer != self.num_layers - 1:
                cur_cell = getattr(self, f"lstm{layer}")
                next_cell = getattr(self, f"lstm{layer+1}")
                cur_cell.output_quantizer = next_cell.input_quantizer
            self.in_scale[layer] = getattr(
                self, f"lstm{layer}"
            ).input_quantizer.scale.item()
            self.out_scale[layer] = getattr(
                self, f"lstm{layer}"
            ).output_quantizer.scale.item()

    def forward(
        self, x: Tensor, hx: List[Tensor], cx: List[Tensor]
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        if hx is None and cx is None:
            hx = [
                torch.zeros((x.size(1), self.hidden_size), dtype=x.dtype)
                for layer in range(self.num_layers)
            ]
            cx = [
                torch.zeros((x.size(1), self.hidden_size), dtype=torch.float16)
                for layer in range(self.num_layers)
            ]
        x, hx, cx = P.lstm_amx_int8(
            x,
            hx,
            cx,
            self.weights,
            self.rb_scale,
            self.in_scale,
            self.out_scale,
            self.skip_quant_y,
        )
        return x, hx, cx


class QuantLSTM(iLSTM):
    def __init__(self, input_size, hidden_size, num_layers, skip_quant_y, **kwargs):
        super(iLSTM, self).__init__(input_size, hidden_size, num_layers, **kwargs)

    @torch.jit.ignore
    def forward(
        self, x: Tensor, hx: Optional[List[Tensor]], cx: Optional[List[Tensor]]
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        if hx is None and cx is None:
            hx = [
                torch.zeros((x.size(1), self.hidden_size), dtype=torch.float32)
                for layer in range(self.num_layers)
            ]
            cx = [
                torch.zeros((x.size(1), self.hidden_size), dtype=torch.float32)
                for layer in range(self.num_layers)
            ]
        for layer in range(self.num_layers):
            x, hx[layer], cx[layer] = getattr(self, f"lstm{layer}")(
                x, hx[layer], cx[layer]
            )
        return x, hx, cx


class QuantLSTMLayer(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(QuantLSTMLayer, self).__init__(input_size, hidden_size, **kwargs)

    def _init_quantizers(
        self,
        weight_quantizer: WeightQuantizer = None,
        input_quantizer: TensorQuantizer = None,
        output_quantizer: TensorQuantizer = None,
    ) -> None:
        self.weight_quantizer = weight_quantizer
        self.input_quantizer = input_quantizer
        self.output_quantizer = output_quantizer

    def _init_parameters(
        self, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor
    ) -> None:
        self.weight_ih = w_ih
        self.weight_hh = w_hh
        self.bias_ih = b_ih
        self.bias_hh = b_hh

    def _quant_parameters(self, weights, rb_scale_list) -> None:
        self.weight_quantizer.amax = torch.max(
            torch.cat([self.weight_ih, self.weight_hh], 1).abs()
        )
        self.weight_ih = Parameter(
            self.weight_quantizer(self.weight_ih), requires_grad=False
        )
        self.weight_hh = Parameter(
            self.weight_quantizer(self.weight_hh), requires_grad=False
        )

    def forward(
        self, x: Tensor, hx: Tensor, cx: Tensor, quant_y: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        xt_list = []
        for i in range(x.shape[0]):
            if hasattr(self, "input_quantizer"):
                if self.input_quantizer.mode != None:
                    x[i], hx = self.input_quantizer(torch.cat([x[i], hx], 1)).split(
                        [x[i].size(1), hx.size(1)], 1
                    )
            gates = F.linear(x[i], self.weight_ih, self.bias_ih)
            gates += F.linear(hx, self.weight_hh, self.bias_hh)
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            cx = (ft * cx) + (it * gt)
            hx = ot * torch.tanh(cx)
            xt_list.append(hx)
        x = torch.stack(xt_list, 0)
        return x, hx, cx


class iLSTMLayer(QuantLSTMLayer):
    def __init__(
        self, input_size: int, hidden_size: int, first_layer=False, **kwargs
    ) -> None:
        super(iLSTMLayer, self).__init__(input_size, hidden_size, **kwargs)
        self.first_layer = first_layer

    def _quant_parameters(self, weights, rb_scale) -> None:
        self.weight_quantizer.amax = torch.max(
            torch.cat([self.weight_ih, self.weight_hh], 1).abs()
        )
        self.weight_ih = Parameter(
            self.weight_quantizer._quant_forward(self.weight_ih, self.first_layer),
            requires_grad=False,
        )
        self.weight_hh = Parameter(
            self.weight_quantizer._quant_forward(self.weight_hh, self.first_layer),
            requires_grad=False,
        )
        b_scale = self.input_quantizer.scale * self.weight_quantizer.scale
        # self.bias_ih = Parameter(self.bias_ih * b_scale, requires_grad=False)
        self.bias_hh = Parameter(
            (self.bias_hh + self.bias_ih) * b_scale, requires_grad=False
        )
        self.rb_scale = 1 / b_scale.item()
        self.in_quant_scale = self.input_quantizer.scale.item()
        self.out_quant_scale = self.output_quantizer.scale.item()
        weights_layer = [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        rb_scale[len(weights)] = self.rb_scale
        weights.append(weights_layer)

    def forward(
        self, x: Tensor, hx: Tensor, cx: Tensor, skip_quant_y: bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.first_layer:
            x = F.pad(x, (0, 64 - x.shape[-1] % 64), "constant", 0.0)
        return P.lstm_layer_amx_int8(
            x,
            hx,
            cx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.rb_scale,
            self.in_quant_scale,
            self.out_quant_scale,
            skip_quant_y,
        )

        gates_list = []
        for i in range(x.shape[0]):
            gates = P.linear(x[i], self.weight_ih, self.bias_ih, self.rb_scale, None)
            gates_list.append(gates)

        xt_list = []
        for i in range(x.shape[0]):
            gates_list[i] += P.linear(
                hx, self.weight_hh, self.bias_hh, self.rb_scale, None
            )
            it, ft, gt, ot = gates_list[i].chunk(4, 1)
            (x_f32, x_int8, hx, cx) = P.lstm_postop(
                it,
                ft,
                gt,
                ot,
                cx,
                self.in_quant_scale,
                self.out_quant_scale,
                skip_quant_y,
            )

            if skip_quant_y:
                xt_list.append(x_f32)
            else:
                xt_list.append(x_int8)

        x = torch.stack(xt_list, 0)
        return x, hx, cx
