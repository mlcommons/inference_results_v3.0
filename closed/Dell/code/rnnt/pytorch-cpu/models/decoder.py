import math
import os
import torch
import _C as P

from config import RNNTParam
from torch import Tensor
from utils import *


class GreedyDecoder(torch.nn.Module):
    def __init__(self, model, run_mode, enable_bf16=False, split_len=-1, batch_size=1):
        super().__init__()
        self.rnnt = model
        self.split_len = split_len
        self.enable_bf16 = enable_bf16
        self.run_mode = run_mode
        intra = torch.get_num_threads()
        self.padded_batch_size = math.ceil(batch_size / (intra * 16)) * (intra * 16)

    def forward(self, x: Tensor, x_lens: Tensor):
        """
        Args:
          x: {T, N, C}
          x_lens: {N}

        Returns:
          res: {N}
        """
        self.batch_size = x_lens.size(0)
        res_dtype = torch.int32 if self.run_mode == "quant" else torch.int64
        self.res = torch.full(
            (self.batch_size, RNNTParam.max_symbols_per_step * x_lens.max().item()),
            -1,
            dtype=res_dtype,
        )
        self.res_idx = torch.full((self.batch_size,), -1, dtype=res_dtype)
        self.step = torch.zeros((self.batch_size, 2))  # debug only
        # init transcription tensors
        trans_hx_dtype = torch.int8 if self.run_mode == "quant" else torch.float32
        trans_cx_dtype = torch.float16 if self.run_mode == "quant" else torch.float32
        self.pre_hx = [
            torch.zeros(
                (self.batch_size, RNNTParam.trans_hidden_size), dtype=trans_hx_dtype
            )
            for layer in range(RNNTParam.pre_num_layers)
        ]
        self.pre_cx = [
            torch.zeros(
                (self.batch_size, RNNTParam.trans_hidden_size), dtype=trans_cx_dtype
            )
            for layer in range(RNNTParam.pre_num_layers)
        ]
        self.post_hx = [
            torch.zeros(
                (self.batch_size, RNNTParam.trans_hidden_size), dtype=trans_hx_dtype
            )
            for layer in range(RNNTParam.post_num_layers)
        ]
        self.post_cx = [
            torch.zeros(
                (self.batch_size, RNNTParam.trans_hidden_size), dtype=trans_cx_dtype
            )
            for layer in range(RNNTParam.post_num_layers)
        ]
        # init prediction tensors
        self.pre_g = torch.tensor([[RNNTParam.SOS] * self.batch_size], dtype=res_dtype)
        pred_dtype = torch.bfloat16 if self.enable_bf16 else torch.float32
        self.pre_hg = [
            torch.zeros((self.batch_size, RNNTParam.pred_hidden_size), dtype=pred_dtype)
            for layer in range(RNNTParam.pred_num_layers)
        ]
        self.pre_cg = [
            torch.zeros(
                (self.batch_size, RNNTParam.pred_hidden_size), dtype=torch.float32
            )
            for layer in range(RNNTParam.pred_num_layers)
        ]

        if self.split_len != -1:
            max_len = x_lens.max().item()
            split_lens = torch.tensor(
                [self.split_len] * self.batch_size, dtype=res_dtype
            )
            for split_idx in range(0, max_len, self.split_len):
                # 0. split x, x_lens
                xi_lens = torch.min(
                    split_lens, torch.clamp((x_lens - split_idx), min=0)
                )
                xi = x[split_idx : split_idx + self.split_len]
                self.greedy_decode(xi, xi_lens)
        else:
            self.greedy_decode(x, x_lens)
        return self.res, self.res_idx + 1

    def greedy_decode(self, f: Tensor, f_lens: Tensor):
        if self.run_mode == "f32" or self.run_mode == "calib" or self.run_mode == None:
            self.greedy_decode_f32(f, f_lens)
        else:
            self.greedy_decode_quant(f, f_lens)

    def greedy_decode_f32(self, f: Tensor, f_lens: Tensor):
        # init flags
        self.symbols_added = torch.zeros(self.batch_size, dtype=torch.int64)
        self.time_idx = torch.zeros(self.batch_size, dtype=torch.int64)
        self.finish = f_lens.eq(0)
        self.batch_idx = torch.range(0, self.batch_size - 1, dtype=torch.int64)
        self.trace = [[0] * f_len for f_len in f_lens]  # debug only
        # 1. do transcription
        (
            f,
            self.pre_hx,
            self.pre_cx,
            self.post_hx,
            self.post_cx,
        ) = self.rnnt.transcription(
            f, f_lens, self.pre_hx, self.pre_cx, self.post_hx, self.post_cx
        )
        f_lens = torch.ceil(f_lens / RNNTParam.stack_time_factor).to(torch.int32)
        self.eos_idx = torch.clamp(f_lens - 1, min=0)
        if self.enable_bf16:
            f = f.to(torch.bfloat16)
        fi = f[0]

        while True:
            # 2. do prediction
            g, hg, cg = self.rnnt.prediction(self.pre_g, self.pre_hg, self.pre_cg)
            # 3. do joint
            y = self.rnnt.joint(fi, g[0], self.padded_batch_size)
            symbols = torch.argmax(y, dim=1)
            # 4. if (no BLANK and no MAX_SYMBOLS_PER_STEP) and no FINISH
            self.update_g = (
                symbols.ne(RNNTParam.BLANK)
                & self.symbols_added.ne(RNNTParam.max_symbols_per_step)
                & ~self.finish
            )
            if any(self.update_g):
                self.step[self.update_g, 1] += 1
                self.res_idx += self.update_g
                # 4.1. update res
                self.res[self.update_g, self.res_idx[self.update_g]] = symbols[
                    self.update_g
                ]
                # 4.2. update symbols_added
                self.symbols_added += self.update_g
                # 4.3. update g
                self.pre_g[0][self.update_g] = symbols[self.update_g]
                self.pre_hg[0][self.update_g, :] = hg[0][self.update_g, :]
                self.pre_hg[1][self.update_g, :] = hg[1][self.update_g, :]
                self.pre_cg[0][self.update_g, :] = cg[0][self.update_g, :]
                self.pre_cg[1][self.update_g, :] = cg[1][self.update_g, :]

            # 5. if (BLANK or MAX_SYMBOLS_PER_STEP) and no FINISH
            self.update_f = ~self.update_g & ~self.finish
            if any(self.update_f):
                self.step[self.update_f, 0] += 1
                # 5.1. update time_idx
                self.time_idx += self.update_f
                # 5.2. BCE
                self.finish |= self.time_idx.ge(f_lens)  # TODO: add early response
                self.time_idx = self.time_idx.min(self.eos_idx)
                if all(self.finish):
                    break
                # 5.3. update f
                fi = f[self.time_idx, self.batch_idx, :]
                # 5.4. reset symbols_added
                self.symbols_added *= ~self.update_f
            # self._dump_tensors()
        return self.res

    def greedy_decode_quant(self, f: Tensor, f_lens: Tensor):
        # init flags
        self.symbols_added = torch.zeros(self.batch_size, dtype=torch.int32)
        self.time_idx = torch.zeros(self.batch_size, dtype=torch.int32)
        # 1. do transcription
        (
            f,
            self.pre_hx,
            self.pre_cx,
            self.post_hx,
            self.post_cx,
        ) = self.rnnt.transcription(
            f, f_lens, self.pre_hx, self.pre_cx, self.post_hx, self.post_cx
        )
        f_lens = torch.ceil(f_lens / RNNTParam.stack_time_factor).to(torch.int32)
        fi = f[0]

        while True:
            # 2. do prediction
            g, hg, cg = self.rnnt.prediction(self.pre_g, self.pre_hg, self.pre_cg)
            # 3. do joint
            y = self.rnnt.joint(fi, g[0])
            symbols = torch.argmax(y, dim=1)
            # 4. update state
            finish = self.rnnt.update(
                symbols,
                self.symbols_added,
                self.res,
                self.res_idx,
                f,
                f_lens,
                self.time_idx,
                fi,
                self.pre_g,
                self.pre_hg,
                self.pre_cg,
                hg,
                cg,
            )
            if finish:
                break
        return self.res
