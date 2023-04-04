# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import re

from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args


class DLRMHarness(BaseBenchmarkHarness):
    """DLRM benchmark harness."""

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_copy_streams",
            "complete_threads",
            "sample_partition_path",
            "warmup_duration",
            "gpu_inference_streams",
            "num_staging_threads",
            "num_staging_batches",
            "max_pairs_per_staging_thread",
            "gpu_num_bundles",
            "check_contiguity",
            "start_from_device",
            "use_jemalloc",
            "compress_categorical_inputs",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        return "./build/bin/harness_dlrm"

    def _build_custom_flags(self, flag_dict):
        # Handle use_jemalloc
        self.use_jemalloc = dict_get(flag_dict, "use_jemalloc", False)
        flag_dict['use_jemalloc'] = None
        # when compress_categorical_inputs is True, need to point to appropriate compressed preprocessed dataset
        if flag_dict.get('compress_categorical_inputs', False) and "categorical_int32_compressed" not in flag_dict['tensor_path']:
            flag_dict['tensor_path'] = flag_dict['tensor_path'].replace("categorical_int32", "categorical_int32_compressed")
            logging.info(f"tensor_path not set to compressed tensor with compress_categorical_inputs=True. Automatically using input tensor from {flag_dict['tensor_path']}")
        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        if self.system_id == 'L4x1':
            argstr += " --eviction_last=0.5"

        return argstr
