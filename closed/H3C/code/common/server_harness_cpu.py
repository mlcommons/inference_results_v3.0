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

import re
import os
import math
from functools import reduce

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np

import code.common.arguments as common_args
from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import G_BUILD_DIR, Benchmark
from code.common.harness import BaseBenchmarkHarness, benchmark_qsl_size_map
from code.common.submission import TRITON_VERSION


TRITON_OV_CONFIG = """name: "{config_name}"
backend: "openvino"
max_batch_size: {max_batch_size}
{parameters}
{io_info}

instance_group {{
    count: {instance_group_count}
    kind: KIND_CPU
}}
{dynamic_batching}
"""

TRITON_OV_PARAMETERS = """
parameters: {{
    key: "{key}"
    value: {{
        string_value : "{value}"
    }}
}}
"""
# OV doesn't use preferred batch size?
TRITON_OV_DYNAMIC_BATCHING_FORMAT = """

dynamic_batching {{
    max_queue_delay_microseconds: {max_queue_delay_usec}
    default_queue_policy {{
        timeout_action: DELAY
        default_timeout_microseconds: {request_timeout_usec}
    }}
}}
"""


class TritonHarnessCPU(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        self.enable_interleaved = False
        self.is_int8 = args['precision'] == 'int8'
        args["skip_file_checks"] = True
        super().__init__(args, benchmark)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS
        self.model_store_path = os.path.abspath("./build/model_repo")
        self.model_binaries = ["model.xml", "model.bin", "model.mapping"]
        self.abs_path = os.path.join(G_BUILD_DIR, "models", "Triton", args["model_name"])
        self.model_name = args["model_name"]
        self.model_version = "1"
        self.openvino_version = args["openvino_version"] if "openvino_version" in args else "f2f281e6"
        self.map_path = args["map_path"] if "map_path" in args else None
        self.test_mode = args["test_mode"] if "test_mode" in args else None
        self.coalesced = args["coalesced_tensor"] if "coalesced_tensor" in args else None
        self.tensor_path = args["tensor_path"]

    def _get_harness_executable(self):
        return "./build/bin/harness_triton_cpu"

    def get_system_name(self):
        return super().get_system_name(add_trt=False)

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        if perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self._get_submission_benchmark_name()] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def _build_custom_flags(self, flag_dict):
        # Triton does not use gpu_engines flag
        flag_dict["gpu_engines"] = None

        # Force performance sample count
        flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]

        flag_dict["model_store_path"] = self.model_store_path
        flag_dict["model_name"] = self.model_name
        flag_dict["model_version"] = self.model_version
        flag_dict["openvino_version"] = self.openvino_version
        flag_dict["buffer_manager_thread_count"] = self.args.get("buffer_manager_thread_count", 0)
        flag_dict["pinned_input"] = True

        # Inform the server to use different QSL
        flag_dict["use_dlrm_qsl"] = (self.name == Benchmark.DLRM)

        # Specify harness-specific flags here
        flag_dict["tensor_path"] = self.tensor_path
        if self.test_mode:
            flag_dict["test_mode"] = self.test_mode
        if self.map_path:
            flag_dict["map_path"] = self.map_path
        if self.coalesced:
            flag_dict["coalesced_tensor"] = self.coalesced

        self.setup_triton_model_repo()

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        # Assign proper callback function here
        # TODO: Retinanet is not supported on CPU yet.
        if self.name == Benchmark.ResNet50:
            argstr += " --response_postprocess ovrn50"

        return argstr

    def setup_triton_model_repo(self):
        # Create model dir where model is loaded from
        model_dir = os.path.join(self.model_store_path, self.model_name, self.model_version)
        os.makedirs(model_dir, exist_ok=True)

        # Create a sym link to model binaries
        for binary in self.model_binaries:
            dst = os.path.join(model_dir, binary)
            if os.path.exists(dst):
                os.remove(dst)
            bin_path = os.path.join(self.abs_path, self.model_version, binary)
            os.symlink(bin_path, dst)

        # Generate configs - common OV config args
        config = {}
        config["config_name"] = self.model_name
        config["max_batch_size"] = self.args["batch_size"]
        config["instance_group_count"] = self.args.get("num_instances", 1)

        # OV Parameters for Triton
        config["parameters"] = ""
        parameters = self.args.get("ov_parameters")
        for p in parameters:
            parameter = {}
            parameter["key"] = p
            parameter["value"] = parameters[p]
            config["parameters"] += TRITON_OV_PARAMETERS.format(**parameter)

        # OV dynamic batching parameters for Triton
        config["max_queue_delay_usec"] = self.args.get("max_queue_delay_usec", 1000000)
        config["request_timeout_usec"] = self.args.get("request_timeout_usec", 1000000000)
        config["dynamic_batching"] = TRITON_OV_DYNAMIC_BATCHING_FORMAT.format(**config)

        # Populate I/O information based on the model
        if self.model_name == "resnet50_int8_openvino":
            config["io_info"] = """input [
            {
                name: "input_tensor"
                data_type: TYPE_FP32
                format: FORMAT_NCHW
                dims: [ 3, 224, 224 ]
            }
            ]
            output [
            {
                name: "softmax_tensor"
                data_type: TYPE_FP32
                dims: [ 1001 ]
            }
            ]"""
        elif self.model_name == "3dunet_int8_openvino":
            config["io_info"] = """input [
            {
                name: "input"
                data_type: TYPE_FP32
                dims: [ 1, 4, 224, 224,160 ]
            #    reshape { shape: [ 1, 3, 160, 224, 224 ] }
            }
            ]
            output [
            {
                name: "output/add_"
                data_type: TYPE_FP32
                dims: [1,4,224,224,160]
            }
            ]"""
        elif self.model_name == "bert_int8_openvino":
            config["io_info"] = """input [
            {
                name: "input_ids"
                data_type: TYPE_INT32
                dims: [ 1, 384 ]
            },
            {
                name: "attention_mask"
                data_type: TYPE_INT32
                dims: [ 1, 384 ]
            },
            {
                name: "token_type_ids"
                data_type: TYPE_INT32
                dims: [ 1, 384 ]
            }
            ]
            output [
            {
                name: "6703"
                data_type: TYPE_FP32
                dims: [1,384,2]
            }
            ]"""

        # Write config.pbtxt
        config_file_path = os.path.join(self.model_store_path, self.model_name, "config.pbtxt")
        with open(config_file_path, 'w') as f:
            f.write(TRITON_OV_CONFIG.format(**config))
