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

import code.common.arguments as common_args
from code.common import logging, args_to_string
from code.common.constants import Benchmark, Scenario
from code.common.harness import BaseBenchmarkHarness


response_postprocess_map = {
    Benchmark.Retinanet: "openimageeffnms",  # TODO: the output arrangement is not fully optimized yet.
    Benchmark.SSDResNet34: "coco",
    Benchmark.SSDMobileNet: "coco"
}


class LWISHarness(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)

        self.use_jemalloc = (Scenario.Server == self.scenario)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS

    def _get_harness_executable(self):
        return "./build/bin/harness_default"

    def _build_custom_flags(self, flag_dict):
        if self.has_dla:
            flag_dict["dla_engines"] = self.dla_engine

        if not self.has_gpu and not self.has_dla:
            raise ValueError("Cannot specify --no_gpu and --gpu_only at the same time")

        if self.has_gpu and not self.has_dla:
            flag_dict["max_dlas"] = 0

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        if self.name in response_postprocess_map:
            argstr += " --response_postprocess " + response_postprocess_map[self.name]

        return argstr


class LWISSUTHarness(LWISHarness):
    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_flags = [
            "sut_numa_config",
            "nic_mapping",
            "sut_nic_gpu_affinity",
            "lon_netid",
            "tcp_port",
            "enable_max_transactions_per_connection",
            "max_transactions_per_connection",
            "max_wait_before_sending_us",
            "num_ibqps_per_nic",
            "SUT_uses_host_mem_for_RDMA",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS + custom_flags

    def _get_harness_executable(self):
        return "./build/bin/harness_lwis_sut"

    def _build_custom_flags(self, flag_dict):
        if self.has_dla:
            raise ValueError("Does not support DLA yet")

        if not self.has_gpu and not self.has_dla:
            raise ValueError("Cannot specify --no_gpu and --gpu_only at the same time")

        if self.has_gpu and not self.has_dla:
            flag_dict["max_dlas"] = 0

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        if self.name in response_postprocess_map:
            raise ValueError("Does not support response postprocess yet")

        return argstr


class LWISLONHarness(LWISHarness):
    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_flags = [
            "lon_numa_config",
            "nic_mapping",
            "sut_netid",
            "tcp_port",
            "enable_max_transactions_per_connection",
            "max_transactions_per_connection",
            "max_wait_before_sending_us",
            "num_ibqps_per_nic",
            "lon_uses_one_issue_queue",
            "round_robin_samples_to_multi_issue_queue",
            "smart_balance_samples_to_multi_issue_queue",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.LWIS_ARGS + common_args.SHARED_ARGS + custom_flags

    def _get_harness_executable(self):
        return "./build/bin/harness_lwis_lon"

    def _build_custom_flags(self, flag_dict):
        if self.has_dla:
            raise ValueError("Does not support DLA yet")

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        if self.name in response_postprocess_map:
            raise ValueError("Does not support response postprocess yet")

        return argstr
