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

sys.path.insert(0, os.getcwd())

import argparse
import json
import multiprocessing
import pandas as pd
import toml
import torch

from dataset import AudioToTextDataLayer
from glob import glob
from process_librispeech import AudioProcessing, parallel_process


def parse_args():
    parser = argparse.ArgumentParser(description="Process LibriSpeech.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Input downloaded dataset dir",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output pre-processed dataset dir",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        required=False,
        help="a file contains list of files needs to be converted.",
    )
    parser.add_argument(
        "--output_json", type=str, default="./", help="name of the output json file."
    )
    parser.add_argument(
        "-s", "--speed", type=float, nargs="*", help="Speed perturbation ratio"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=None,
        help="Target sample rate. " "defaults to the input sample rate",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite file if exists"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use when processing audio files",
    )
    parser.add_argument("--toml_path", type=str, default="configs/rnnt.toml")
    parser.add_argument("--max_duration", type=float, default=15.0)
    parser.add_argument(
        "--calibration_file_path", type=str, default="configs/calibration_files.txt"
    )
    args = parser.parse_args()
    args.input_dir = os.path.abspath(args.input_dir).rstrip("/")
    args.output_dir = os.path.abspath(args.output_dir).rstrip("/")
    return args


def build_input_arr(input_dir):
    txt_files = glob(os.path.join(input_dir, "**", "*.trans.txt"), recursive=True)
    input_data = []
    for txt_file in txt_files:
        rel_path = os.path.relpath(txt_file, input_dir)
        with open(txt_file) as fp:
            for line in fp:
                fname, _, transcript = line.partition(" ")
                input_data.append(
                    dict(
                        input_relpath=os.path.dirname(rel_path),
                        input_fname=fname + ".flac",
                        transcript=transcript,
                    )
                )
    return input_data


def flac_to_wav(args, name, dest_dir, output_json):
    print(f"==> Scaning source dir from {args.input_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    dataset = build_input_arr(args.input_dir)

    if args.output_list is not None:
        with open(args.output_list, "r") as dest_file:
            dest_list = dest_file.readlines()
    else:
        dest_list = None

    print(f"==> Converting audio files to {dest_dir}")
    dataset = parallel_process(
        dataset=dataset,
        input_dir=args.input_dir,
        dest_dir=dest_dir,
        dest_list=dest_list,
        target_sr=args.target_sr,
        speed=args.speed,
        overwrite=args.overwrite,
        parallel=args.parallel,
    )

    print(f"==> Generating json to {output_json}")
    df = pd.DataFrame(dataset, dtype=object)
    dataset = df.to_dict(orient="records")
    with open(output_json, "w") as fp:
        json.dump(dataset, fp, indent=2)


def process_dataset(args, name, data_layer, data_processor):
    x_npy_dir = os.path.join(args.output_dir, "npy", name, "fp32")
    x_len_npy_dir = os.path.join(args.output_dir, "npy", name, "int32")
    os.makedirs(x_npy_dir, exist_ok=True)
    os.makedirs(x_len_npy_dir, exist_ok=True)

    x_input_dir = os.path.join(args.output_dir, "input", name, "fp32")
    x_len_input_dir = os.path.join(args.output_dir, "input", name, "int32")
    os.makedirs(x_input_dir, exist_ok=True)
    os.makedirs(x_len_input_dir, exist_ok=True)

    wavs = []
    wav_lens = []
    feas = []
    fea_lens = []
    for idx, data in enumerate(data_layer.data_iterator):
        wavs.append(data[0].squeeze())
        wav_lens.append(data[1])

        fea, fea_len = data_processor(data[0], data[1], pad_batch_size=False)
        # {N, C, T} -> {T, N, C}
        fea = fea.permute(2, 0, 1).contiguous()
        feas.append(fea.squeeze().contiguous())
        fea_lens.append(fea_len)

    data_wav = {"x": wavs, "x_lens": wav_lens}
    torch.save(data_wav, os.path.join(args.output_dir, "..", f"{name}-npy.pt"))
    data_fea = {"x": feas, "x_lens": fea_lens}
    torch.save(data_fea, os.path.join(args.output_dir, "..", f"{name}-input.pt"))


def convert_dataset(args):
    name = os.path.split(args.input_dir)[-1]
    print(f"==> Converting flac -> wav for {name}")
    wav_dir = os.path.join(args.output_dir, "wav", name + "-wav")
    manifest_path = os.path.join(args.output_dir, "wav", name + "-wav.json")
    dataset = flac_to_wav(args, name, wav_dir, manifest_path)

    print(f"==> Converting wav -> npy for {name}")
    cfg = toml.load(args.toml_path)
    cfg["input_eval"]["max_duration"] = args.max_duration
    data_layer = AudioToTextDataLayer(
        dataset_dir=os.path.join(args.output_dir, "wav"),
        featurizer_config=cfg["input_eval"],
        manifest_filepath=manifest_path,
        labels=cfg["labels"]["labels"],
        batch_size=1,
        shuffle=False,
    )
    data_processor = AudioProcessing(run_mode="f32", **cfg["input_eval"]).eval()
    process_dataset(args, name, data_layer, data_processor)


if __name__ == "__main__":
    args = parse_args()
    print(f"==> args: {args}")
    convert_dataset(args)
