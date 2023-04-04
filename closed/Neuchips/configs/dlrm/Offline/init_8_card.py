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

class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    check_contiguity = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class N3000_CPU_2S_Neuchips_Custom(ParentConfig.GPUBaseConfig):

    system = KnownSystem.N3000_CPU_2S_Neuchips
    scenario = Scenario.Offline

    #performance_sample_count_override = 330067

    max_pairs_per_staging_thread = 192000
    gpu_batch_size = max_pairs_per_staging_thread

    num_staging_batches = 2
    num_staging_threads = 1
    complete_threads = 1

    #8 cards configuration / 8 nodes
    numa_config = "0:0-7&1:8-15&2:16-23&3:24-31&4:32-39&5:40-47&6:48-55&7:56-63"

    offline_expected_qps = 107000 * len(numa_config.split('&'))

    check_contiguity = False
