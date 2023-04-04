# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from code.common.constants import Benchmark
from configs.configuration import BenchmarkConfiguration


class GPUBaseConfig(BenchmarkConfiguration):
    benchmark = Benchmark.BERT

    tensor_path = "build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy"
    precision = "int8"
    input_dtype = "int32"
    input_format = "linear"
    use_graphs = False
    bert_opt_seqlen = 384
    coalesced_tensor = True


class CPUBaseConfig(BenchmarkConfiguration):
    benchmark = Benchmark.BERT

    tensor_path = "build/preprocessed_data/squad_tokenized/input_ids.npy,build/preprocessed_data/squad_tokenized/input_mask.npy,build/preprocessed_data/squad_tokenized/segment_ids.npy"
    input_dtype = "fp32"
    precision = "fp32"
    use_triton = True
    model_name = "bert_int8_openvino"
    bert_opt_seqlen = 384
    coalesced_tensor = True
