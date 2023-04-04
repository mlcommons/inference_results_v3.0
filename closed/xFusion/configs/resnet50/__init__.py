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
    benchmark = Benchmark.ResNet50
    input_dtype = "int8"
    input_format = "linear"
    precision = "int8"
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "build/preprocessed_data/imagenet/ResNet50/int8_linear"


class CPUBaseConfig(BenchmarkConfiguration):
    benchmark = Benchmark.ResNet50
    input_dtype = "fp32"
    precision = "int8"
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "build/preprocessed_data/imagenet/ResNet50/fp32_nomean"
    use_triton = True
    run_infer_on_copy_streams = False
    model_name = "resnet50_int8_openvino"
