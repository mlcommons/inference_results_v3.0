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
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.resnet50 import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server

    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 45000
    use_cuda_thread_per_device = True
    use_graphs = True




@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    use_deque_limit = True
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 5
    server_target_qps = 140000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_Triton(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4
    server_target_qps = 125000
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_inference_streams = 5
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
    gpu_copy_streams = 9
    use_cuda_thread_per_device = True


