#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../"))

import multiprocessing
import functools
import torch
import torch.nn as nn

from datasets.parts.features import FeatureFactory
from tqdm import tqdm
from typing import Tuple


def process(
    data, input_dir, dest_dir, dest_list, target_sr=None, speed=None, overwrite=True
):
    import sox

    speed = speed or []
    speed.append(1)
    speed = list(set(speed))  # Make uniqe

    input_fname = os.path.join(input_dir, data["input_relpath"], data["input_fname"])
    input_sr = sox.file_info.sample_rate(input_fname)
    target_sr = target_sr or input_sr

    os.makedirs(os.path.join(dest_dir, data["input_relpath"]), exist_ok=True)

    output_dict = {}
    output_dict["transcript"] = data["transcript"].lower().strip()
    output_dict["files"] = []

    fname = os.path.splitext(data["input_fname"])[0]
    for s in speed:
        output_fname = fname + "{}.wav".format("" if s == 1 else "-{}".format(s))
        output_fpath = os.path.join(dest_dir, data["input_relpath"], output_fname)
        output_rel_fpath = os.path.join(
            "train-clean-100-wav", data["input_relpath"], output_fname + "\n"
        )

        if dest_list != None and not output_rel_fpath in dest_list:
            return None
        if not os.path.exists(output_fpath) or overwrite:
            cbn = sox.Transformer().speed(factor=s).convert(target_sr)
            cbn.build(input_fname, output_fpath)

        file_info = sox.file_info.info(output_fpath)
        file_info["fname"] = os.path.join(
            os.path.basename(dest_dir), data["input_relpath"], output_fname
        )
        file_info["speed"] = s
        output_dict["files"].append(file_info)

        if s == 1:
            file_info = sox.file_info.info(output_fpath)
            output_dict["original_duration"] = file_info["duration"]
            output_dict["original_num_samples"] = file_info["num_samples"]

    return output_dict


def parallel_process(
    dataset, input_dir, dest_dir, dest_list, target_sr, speed, overwrite, parallel
):
    with multiprocessing.Pool(parallel) as p:
        func = functools.partial(
            process,
            input_dir=input_dir,
            dest_dir=dest_dir,
            dest_list=dest_list,
            target_sr=target_sr,
            speed=speed,
            overwrite=overwrite,
        )
        dataset = list(tqdm(p.imap(func, dataset), total=len(dataset)))
        result = []
        for data in dataset:
            if data != None:
                result.append(data)
        return result


class AudioProcessing(nn.Module):
    def __init__(self, run_mode, **kwargs):
        nn.Module.__init__(self)  # For PyTorch API
        self.optim_level = kwargs.get("optimization_level", 0)
        kwargs["pad_out_feat"] = True if run_mode == "quant" else False
        self.featurizer = FeatureFactory.from_config(kwargs)

    def forward(
        self, wavs: torch.Tensor, wav_lens: torch.Tensor, pad_batch_size: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feas, fea_lens = self.featurizer(wavs, wav_lens, pad_batch_size)
        return feas, fea_lens
