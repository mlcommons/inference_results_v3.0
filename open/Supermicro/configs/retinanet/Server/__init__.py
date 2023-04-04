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

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *


ParentConfig = import_module("configs.retinanet")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100
    use_graphs = False
    use_cuda_thread_per_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(ServerGPUBaseConfig):
    system = KnownSystem.L4x1
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 155
    workspace_size = 20000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(ServerGPUBaseConfig):
    system = KnownSystem.L40x1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 250
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 1440
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    server_target_qps = 680
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 1440 * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8_Triton(H100_SXM_80GBx1_Triton):
    system = KnownSystem.H100_SXM_80GBx8
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 840 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 930
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    server_target_qps = 600
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 920 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    server_target_qps = 5600
    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx1_Triton):
    system = KnownSystem.H100_PCIe_80GBx8
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 700 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 700
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 700 * 4


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 580 * 8
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 3300
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 450 * 8
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_Triton):
    server_target_qps = 2100
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 700 * 8
    start_from_device = True
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    start_from_device = None
    server_target_qps = 3735
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 560 * 8
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_Triton):
    server_target_qps = 2400
    power_limit = 250


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 1
    gpu_inference_streams = 2
    server_target_qps = 45
    start_from_device = True

    # Early stopping. For some reason this breaks --fast...
    min_query_count = 1
    min_duration = 30 * 60 * 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 40
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 55


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True
    server_target_qps = 47
