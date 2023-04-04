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

import re
import os
import sys
import math
sys.path.insert(0, os.getcwd())

from functools import reduce
from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import Benchmark
from code.common.harness import BaseBenchmarkHarness, benchmark_qsl_size_map
from code.common.submission import TRITON_VERSION

import numpy as np
import code.common.arguments as common_args


class TritonHarnessInferentia(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        self.enable_interleaved = False
        self.is_int8 = False
        super().__init__(args, benchmark, skip_file_checks=True)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS
        self.model_store_path = os.path.abspath("./build/model_repo")
        self.model_binaries = ["model.py"]
        self.compiled_batch_size = args["inferentia_compiled_model_batch_size"]
        self.framework = self.args["inferentia_compiled_model_framework"]
        self.model_name = f"{benchmark.valstr()}_{self.framework}_{self.compiled_batch_size}"
        self.compiled_model_path = "/home/ubuntu/inferentia-compiled-models/"
        self.model_version = "1"
        self.map_path = args.get("map_path", None)
        self.test_mode = args["test_mode"] if "test_mode" in args else None
        self.coalesced = args["coalesced_tensor"] if "coalesced_tensor" in args else None
        self.tensor_path = args["tensor_path"]

    def _get_harness_executable(self):
        return "./build/bin/harness_triton_inferentia"

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
        elif benchmark_qsl_size_map[self.name] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]
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
        flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self.name]

        flag_dict["model_store_path"] = self.model_store_path
        flag_dict["model_name"] = self.model_name
        flag_dict["model_version"] = self.model_version
        flag_dict["buffer_manager_thread_count"] = self.args.get("buffer_manager_thread_count", 0)
        flag_dict["pinned_input"] = True
        flag_dict["python_backend_path"] = "/work/build/triton-inference-server/out/python/install/backends/"
        flag_dict["request_pool_count"] = 250000

        # Specify harness-specific flags here
        flag_dict["tensor_path"] = self.tensor_path
        if self.test_mode:
            flag_dict["test_mode"] = self.test_mode
        if self.map_path:
            flag_dict["map_path"] = self.map_path
        if self.coalesced:
            flag_dict["coalesced_tensor"] = self.coalesced

        flag_dict["batch_triton_requests"] = self.args.get("batch_triton_requests", False)
        flag_dict["num_batchers"] = self.args.get("num_concurrent_batchers")
        flag_dict["num_issuers"] = self.args.get("num_concurrent_issuers")
        self.setup_triton_model_repo()

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        # Assign proper callback function here
        if self.name == Benchmark.ResNet50:
            argstr += " --response_postprocess infrn50"

        return argstr

    def _handle_harness_result(self, result):
        if self.name == Benchmark.DLRM:
            partitions = np.load(os.path.expandvars(self.args["sample_partition_path"]))
            partition_mean_size = np.mean(partitions[1:] - partitions[:-1])

            # Attempt to calculate pairs per second metric
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", result)
            if len(nums) == 1:
                print("User-item pairs per second: {:.3f}".format(float(nums[0]) * partition_mean_size))

        return result

    def setup_triton_model_repo(self):
        # Create model dir where model is loaded from
        model_dir = os.path.join(self.model_store_path, self.model_name, self.model_version)
        os.makedirs(model_dir, exist_ok=True)

        # Generate configs - common OV config args
        config = {}
        config["config_name"] = self.model_name
        config["max_batch_size"] = self.args.get("inferentia_request_batch_size", self.compiled_batch_size * self.args["inferentia_neuron_core_count"] * self.args["inferentia_threads_per_core"])

        config["instance_group_count"] = self.args.get("instance_group_count", 1)

        benchmark_name = self.benchmark
        folder_name = benchmark_name.valstr() + "-" + self.framework + "-bs" + str(self.compiled_batch_size)
        compiled_model_base_path = "/home/ubuntu/inferentia-compiled-models"
        compiled_model_name = "bert-large-int32-bs{}-concat.pt".format(self.compiled_batch_size)
        if self.benchmark == Benchmark.ResNet50:
            compiled_model_name = "resnet50_neuron_bs{}_dynamic.pt".format(self.compiled_batch_size)

        config["compiled_model_path"] = os.path.join(compiled_model_base_path, benchmark_name.valstr(), folder_name, compiled_model_name)
        config["nc_start_idx"] = 0
        config["nc_end_idx"] = self.args["inferentia_neuron_core_count"] - 1
        config["threads_per_core"] = self.args["inferentia_threads_per_core"]

        if self.benchmark == Benchmark.ResNet50:
            inputs_str = "--triton_input INPUT__0,FP32,3x224x224"
            output_str = "--triton_output OUTPUT__0,FP32,1000"

        if self.benchmark == Benchmark.BERT:
            inputs_str = "--triton_input INPUT__0,INT32,384"
            inputs_str += " --triton_input INPUT__1,INT32,384"
            inputs_str += " --triton_input INPUT__2,INT32,384"
            output_str = " --triton_output OUTPUT__0,FP32,384x2"

        gen_triton_model_command = "python3 python_backend/inferentia/scripts/gen_triton_model.py --model_version=" + str(self.model_version) + " " + inputs_str + " " + output_str + " --compiled_model " + config["compiled_model_path"] + " --threads_per_core " + str(
             config["threads_per_core"]) + " --triton_model_dir " + str(self.model_store_path) + "/" + str(self.model_name) + " --neuron_core_range=0:" + str(self.args["inferentia_neuron_core_count"] - 1) + " --model_type " + self.framework + " --max_batch_size " + str(config["max_batch_size"]) + " --triton_model_instance_count " + str(config["instance_group_count"]) + " --disable_batch_requests_to_neuron"
        run_command(gen_triton_model_command, verbose=False)
