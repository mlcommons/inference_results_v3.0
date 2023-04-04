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
    benchmark = Benchmark.DLRM
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "build/preprocessed_data/criteo/full_recalib/sample_partition.npy"
    tensor_path = "build/preprocessed_data/criteo/full_recalib/numeric_int8_chw4.npy,build/preprocessed_data/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
