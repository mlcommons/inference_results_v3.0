import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import array
import math
import mlperf_loadgen as lg
import toml
import torch
import _C as P

from datasets.process_librispeech import AudioProcessing
from decoder import GreedyDecoder
from rnnt_qsl import RNNTQSL
from tqdm import tqdm
from utils import *


class PytorchSUT:
    def __init__(
        self, model_path, dataset_dir, batch_size=1, run_mode=None, args=None, **kwargs
    ):
        # create processor
        if args.enable_process and os.path.exists(args.toml_path):
            if args.load_jit and os.path.exists(args.processor_path):
                self.processor = torch.jit.load(args.processor_path)
            else:
                config = toml.load(args.toml_path)
                featurizer_config = config["input_eval"]
                self.processor = AudioProcessing(
                    run_mode, pad_batch_size=True, **featurizer_config
                ).eval()
        else:
            self.processor = None
        self.batch_size = batch_size
        # create model
        if run_mode == "quant":
            from modeling_rnnt import RNNT
        else:
            from modeling_rnnt import RNNT
        rnnt = RNNT(model_path, run_mode, args.enable_bf16, args.load_jit).eval()
        self.model = GreedyDecoder(
            rnnt, run_mode, args.enable_bf16, args.split_len, self.batch_size
        )
        self.enable_process = self.processor != None
        self.scenario = args.scenario if run_mode != "calib" else None
        self.batch_sort = True if self.scenario == "Offline" else False
        # jit processor & model
        if not args.load_jit and args.save_jit:
            if self.enable_process:
                self.processor = jit_module(self.processor)
            self.model.rnnt = jit_model(self.model.rnnt)
        # create qsl & sut
        self.qsl = RNNTQSL(dataset_dir)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, samples):
        if self.batch_sort:
            samples.sort(key=lambda s: self.qsl[s.index][1].item(), reverse=True)
        for i in tqdm(range(0, len(samples), self.batch_size)):
            batch_samples = samples[i : min(i + self.batch_size, len(samples))]
            batch_idx = [sample.index for sample in batch_samples]
            results, results_idx = self.inference(batch_idx)
            self.query_samples_complete(batch_samples, results, results_idx)

    def inference(self, batch_idx):
        self.actual_batch_size = len(batch_idx)
        with torch.no_grad():
            if self.enable_process:
                # pad T to max_len in batch
                wavs = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx], batch_first=True
                )
                wav_lens = torch.tensor([self.qsl[idx][1] for idx in batch_idx])
                feas, fea_lens = self.processor(
                    wavs,
                    wav_lens.to(torch.int32),
                    pad_batch_size=(self.scenario == "Offline"),
                )
                # {N, C, T} -> {T, N, C}
                feas = feas.permute(2, 0, 1).contiguous()
            else:
                # pad T to max_len in batch
                feas = torch.nn.utils.rnn.pad_sequence(
                    [self.qsl[idx][0] for idx in batch_idx]
                )
                fea_lens = torch.tensor([self.qsl[idx][1] for idx in batch_idx])
                # pad N to ensure last batch accuracy
                if self.actual_batch_size % 32 != 0:
                    padded_batch_size = math.ceil(self.actual_batch_size / 32) * 32
                    feas = torch.nn.functional.pad(
                        feas,
                        (0, 0, 0, padded_batch_size - self.actual_batch_size, 0, 0),
                        "constant",
                        0.0,
                    )
                    fea_lens = torch.nn.functional.pad(
                        fea_lens,
                        (0, padded_batch_size - self.actual_batch_size),
                        "constant",
                        0.0,
                    )
            results = self.model(feas, fea_lens)
        return results

    def query_samples_complete(self, samples, results, results_idx):
        batch_responses = []
        for i in range(self.actual_batch_size):
            res_arr = array.array("i", results[i])
            buf_inf = res_arr.buffer_info()
            response = lg.QuerySampleResponse(
                samples[i].id, buf_inf[0], results_idx[i] * res_arr.itemsize
            )
            lg.QuerySamplesComplete([response])
            # batch_responses.append(response)
            print(f"{samples[i].index}::{seq_to_sen(results[i], results_idx[i])}")
        # lg.QuerySamplesComplete(batch_responses)

    def flush_queries(self):
        pass

    def __del__(self):
        lg.DestroySUT(self.sut)
        print("Finished destroying SUT.")
