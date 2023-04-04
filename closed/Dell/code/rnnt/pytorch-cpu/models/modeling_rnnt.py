import torch
import torch.nn.functional as F
import _C as P

from config import RNNTParam
from quant_lstm import QuantLSTM, iLSTM
from quant_modules import transpose_tile_weight_bf16
from torch import Tensor
from torch.nn import Linear
from torch.nn.parameter import Parameter
from typing import List, Tuple
from utils import *


class RNNT(torch.nn.Module):
    def __init__(
        self, model_path=None, run_mode=None, enable_bf16=False, load_jit=False
    ):
        super().__init__()
        self.transcription = Transcription(run_mode)
        self.prediction = Prediction(enable_bf16)
        self.joint = Joint(enable_bf16)
        self.update = GreedyDecoderUpdate()
        if model_path is not None:
            saved_quantizers = (
                False
                if run_mode == None or run_mode == "f32" or run_mode == "calib"
                else True
            )
            self._load_model(
                model_path, run_mode, enable_bf16, load_jit, saved_quantizers
            )

    def _load_model(
        self,
        model_path,
        run_mode=None,
        enable_bf16=False,
        load_jit=False,
        saved_quantizers=False,
    ):
        if load_jit and os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location="cpu")
            self.transcription = model.transcription
            self.prediction = model.prediction
            self.joint = model.joint
            self.update = model.update
        else:
            model = torch.load(model_path, map_location="cpu")
            state_dict = migrate_state_dict(model)
            if saved_quantizers:
                self.transcription.pre_rnn._init_layers(run_mode)
                self.transcription.post_rnn._init_layers(run_mode)
                # self.joint.linear1_trans._init_quantizers(run_mode)
                self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False)
                self.transcription.pre_rnn._init_layers(run_mode)
                self.transcription.post_rnn._init_layers(run_mode)
                # self.joint.linear1_trans._init_quantizers(run_mode)

            self.transcription.pre_rnn.lstm0.first_layer = True
            self.transcription.pre_rnn._process_parameters(run_mode)
            self.transcription.post_rnn._process_parameters(run_mode)
            # self.joint.linear1_trans._quant_parameters(run_mode)
            if run_mode == "quant":
                self.transcription.pre_rnn.lstm1.output_quantizer = (
                    self.transcription.post_rnn.lstm0.input_quantizer
                )
                self.transcription.pre_rnn._propagate_quantizers()
                self.transcription.post_rnn._propagate_quantizers()
                self.transcription.pre_quantizer = (
                    self.transcription.pre_rnn.lstm0.input_quantizer
                )
                self.transcription.post_quantizer = (
                    self.transcription.post_rnn.lstm0.input_quantizer
                )
        if load_jit == False:
            self.prediction.prepack_weights()
        if load_jit == False and enable_bf16:
            self.joint.prepack_weights()


class Transcription(torch.nn.Module):
    def __init__(self, run_mode, **kwargs):
        super().__init__()
        if run_mode == "quant":
            self.pre_rnn = iLSTM(
                RNNTParam.trans_input_size,
                RNNTParam.trans_hidden_size,
                RNNTParam.pre_num_layers,
                skip_quant_y=False,
            )
            self.post_rnn = iLSTM(
                RNNTParam.trans_hidden_size * RNNTParam.stack_time_factor,
                RNNTParam.trans_hidden_size,
                RNNTParam.post_num_layers,
                skip_quant_y=True,
            )
        else:
            self.pre_rnn = QuantLSTM(
                RNNTParam.trans_input_size,
                RNNTParam.trans_hidden_size,
                RNNTParam.pre_num_layers,
                skip_quant_y=False,
            )
            self.post_rnn = QuantLSTM(
                RNNTParam.trans_hidden_size * RNNTParam.stack_time_factor,
                RNNTParam.trans_hidden_size,
                RNNTParam.post_num_layers,
                skip_quant_y=True,
            )
        self.stack_time = StackTime(RNNTParam.stack_time_factor, run_mode)
        self.run_mode = run_mode

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        pre_hx: List[Tensor],
        pre_cx: List[Tensor],
        post_hx: List[Tensor],
        post_cx: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Args:
          x: {T, N, IC}
          x_lens: {N}
          pre_hx: [{N, C} * PRE_L]
          pre_cx: [{N, C} * PRE_L]
          post_hx: [{N, C} * POST_L]
          post_hx: [{N, C} * POST_L]

        Returns:
          x: {T, N, OC}
          pre_hx: [{N, C} * PRE_L]
          pre_cx: [{N, C} * PRE_L]
          post_hx: [{N, C} * POST_L]
          post_hx: [{N, C} * POST_L]
        """
        x, pre_hx, pre_cx = self.pre_rnn(x, pre_hx, pre_cx)
        x = self.stack_time(x, x_lens)
        x, post_hx, post_cx = self.post_rnn(x, post_hx, post_cx)
        return x, pre_hx, pre_cx, post_hx, post_cx


class Prediction(torch.nn.Module):
    def __init__(self, enable_bf16, **kwargs):
        super().__init__()
        self.sos = RNNTParam.SOS
        self.embed = torch.nn.Embedding(
            RNNTParam.num_labels - 1, RNNTParam.pred_hidden_size
        )
        self.pred_rnn = torch.nn.LSTM(
            RNNTParam.pred_hidden_size,
            RNNTParam.pred_hidden_size,
            RNNTParam.pred_num_layers,
        )
        self.enable_bf16 = enable_bf16

    @torch.jit.ignore
    def prepack_weights(self):
        self.pred_rnn.weights = []
        for layer in range(self.pred_rnn.num_layers):
            w_ih, w_hh, b_ih, b_hh = self.pred_rnn.all_weights[layer][:]
            if self.enable_bf16:
                w_ih, w_hh = w_ih.to(torch.bfloat16), w_hh.to(torch.bfloat16)
                prepacked_w_ih = transpose_tile_weight_bf16(w_ih.transpose(1, 0))
                prepacked_w_hh = transpose_tile_weight_bf16(w_hh.transpose(1, 0))
                self.pred_rnn.weights.append(
                    [prepacked_w_ih, prepacked_w_hh, b_ih, b_hh + b_ih]
                )
            else:
                prepacked_w_ih, prepacked_w_hh = P.prepack_lstm_weights(w_ih, w_hh)
                self.pred_rnn.weights.append(
                    [prepacked_w_ih, prepacked_w_hh, b_ih, b_hh]
                )
        if self.enable_bf16:
            self.embed.weight = Parameter(
                self.embed.weight.to(torch.bfloat16), requires_grad=False
            )

    def forward(
        self, pre_g: Tensor, pre_hg: List[Tensor], pre_cg: List[Tensor]
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Args:
          pre_g: {U, N}
          pre_hg, pre_cg: [{N, C} * L]

        Returns:
          g: {U, N, C}
          hg, cg: [{N, C} * L]
        """
        # hack SOS, since there is no SOS in embedding table :(
        # TODO: fuse maskfill + embedding + maskfill_ + copy
        sos_mask = pre_g.eq(self.sos)
        g = pre_g.masked_fill(sos_mask, 0)
        g = self.embed(g)
        g = g.masked_fill_(sos_mask.unsqueeze(2), 0.0)
        if self.enable_bf16:
            g, hg, cg = P.lstm_amx_bf16(g, pre_hg, pre_cg, self.pred_rnn.weights)
        else:
            g, hg, cg = P.lstm(g, pre_hg, pre_cg, self.pred_rnn.weights)
        return g, hg, cg


class Joint(torch.nn.Module):
    def __init__(self, enable_bf16, **kwargs):
        super().__init__()
        self.linear1_trans = Linear(
            RNNTParam.trans_hidden_size, RNNTParam.joint_hidden_size
        )
        self.linear1_pred = torch.nn.Linear(
            RNNTParam.pred_hidden_size, RNNTParam.joint_hidden_size
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            RNNTParam.joint_hidden_size, RNNTParam.num_labels
        )
        self.enable_bf16 = enable_bf16

    @torch.jit.ignore
    def prepack_weights(self):
        if self.enable_bf16:
            self.linear1_bias = Parameter(
                self.linear1_trans.bias + self.linear1_pred.bias, requires_grad=False
            )
            self.linear1_trans.weight = Parameter(
                transpose_tile_weight_bf16(
                    self.linear1_trans.weight.to(torch.bfloat16).transpose(1, 0)
                ),
                requires_grad=False,
            )
            self.linear1_pred.weight = Parameter(
                transpose_tile_weight_bf16(
                    self.linear1_pred.weight.to(torch.bfloat16).transpose(1, 0)
                ),
                requires_grad=False,
            )
            self.linear2.weight = Parameter(
                transpose_tile_weight_bf16(
                    self.linear2.weight.to(torch.bfloat16).transpose(1, 0), True
                ),
                requires_grad=False,
            )
            b = self.linear2.bias
            self.linear2.bias = Parameter(
                F.pad(b, (0, 32 - b.shape[-1] % 32), "constant", 0), requires_grad=False
            )
        else:
            self.linear1_trans.weight = Parameter(
                P.prepack_linear_weight(self.linear1_trans.weight), requires_grad=False
            )
            self.linear1_pred.weight = Parameter(
                P.prepack_linear_weight(self.linear1_pred.weight), requires_grad=False
            )

    def forward(self, f: Tensor, g: Tensor, padded_batch_size: int = 1) -> Tensor:
        """
        Args:
          f: {T=1, N, TC}
          g: {U=1, N, PC}

        Returns:
          y: {N, T=1, U=1, K}
        """
        if self.enable_bf16:
            y = P.amx_linear_bf16_accum_relu(
                f,
                self.linear1_trans.weight,
                g,
                self.linear1_pred.weight,
                self.linear1_bias,
            )
            # pad N to ensure last linear compute
            remainder = y.shape[0] % padded_batch_size
            if remainder != 0:
                y = F.pad(y, (0, 0, 0, padded_batch_size - y.shape[0]), "constant", 0.0)
                y = P.amx_linear_i16o32(y, self.linear2.weight, self.linear2.bias)
                y = y[:remainder, ...]
            else:
                y = P.amx_linear_i16o32(y, self.linear2.weight, self.linear2.bias)
        else:
            y = self.linear1_trans(f)
            y += self.linear1_pred(g)
            y = self.relu(y)
            y = self.linear2(y)
        return y


class StackTime(torch.nn.Module):
    def __init__(self, factor, run_mode="f32"):
        super().__init__()
        self.factor = factor
        self.run_mode = run_mode

    def forward(self, x: Tensor, x_lens: Tensor) -> Tensor:
        """
        Args:
          x: {T, N, C}
          x_lens: {N}

        Returns:
          x: {⌈T/factor⌉, N, C*factor}
          x_lens: {N}
        """
        if self.run_mode == "quant":
            x = self.forward_quant(x, x_lens)
        else:
            x = self.forward_f32(x, x_lens)
        return x

    def forward_f32(self, x: Tensor, x_lens: Tensor) -> Tensor:
        x_lens = x_lens.to(torch.int64)
        for batch_idx in range(x.size(1)):
            x[x_lens[batch_idx] :, batch_idx, :] = 0
        T, N, C = x.shape
        x = torch.transpose(x, 0, 1)
        zeros = torch.zeros(N, T % self.factor, C, dtype=x.dtype, device=x.device)
        x = torch.cat([x, zeros], 1)
        x = torch.reshape(x, (N, x.shape[1] // self.factor, C * self.factor))
        x = torch.transpose(x, 0, 1).contiguous()
        return x

    def forward_quant(self, x: Tensor, x_lens: Tensor) -> Tensor:
        x = P.stack_time(x, x_lens, self.factor)
        return x


class GreedyDecoderUpdate(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        symbols: Tensor,
        symbols_added: Tensor,
        res: Tensor,
        res_idx: Tensor,
        f: Tensor,
        f_lens: Tensor,
        time_idx: Tensor,
        fi: Tensor,
        pre_g: Tensor,
        pre_hg: List[Tensor],
        pre_cg: List[Tensor],
        hg: List[Tensor],
        cg: List[Tensor],
    ) -> bool:
        return P.greedy_decode_update(
            symbols,
            symbols_added,
            res,
            res_idx,
            f,
            f_lens,
            time_idx,
            fi,
            pre_g,
            pre_hg,
            pre_cg,
            hg,
            cg,
        )
