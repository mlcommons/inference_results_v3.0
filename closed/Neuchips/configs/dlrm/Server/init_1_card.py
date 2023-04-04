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
#
# Modified by Neuchips corp. on 2023

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.dlrm")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    complete_threads = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
#class N3000_CPU_2S_Neuchips_Custom(A100_PCIe_80GBx8):
class N3000_CPU_2S_Neuchips_Custom(ServerGPUBaseConfig):
    system = KnownSystem.N3000_CPU_2S_Neuchips
    scenario = Scenario.Server

    max_pairs_per_staging_thread = 192000
    gpu_batch_size = max_pairs_per_staging_thread

    #min_duration = 60000

    num_staging_batches = 2
    num_staging_threads = 1
    complete_threads = 1

    #1 card configuration
    numa_config = "0:0-7"

    server_num_issue_query_threads = len(numa_config.split('&'))
    server_target_qps = 107000 * len(numa_config.split('&'))

    check_contiguity = False
