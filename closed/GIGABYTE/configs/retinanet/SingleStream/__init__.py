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


GPUBaseConfig = import_module("configs.retinanet").GPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = False
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.L4x1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.L40x1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_02x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GB_02x1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    start_from_device = True
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    start_from_device = True
    single_stream_expected_latency_ns = 1900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A2x1
    single_stream_expected_latency_ns = 25000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_Triton(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(SingleStreamGPUBaseConfig):
    system = KnownSystem.Orin
    workspace_size = 30000000000
    use_graphs = True
    single_stream_expected_latency_ns = 19000000
    gpu_copy_streams = 2
    use_direct_host_access = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_Triton(Orin):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_MaxQ(Orin):
    soc_cpu_freq = 729600
    soc_gpu_freq = 624750000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    orin_num_cores = 4
    #single_stream_expected_latency_ns = 12197185.0
    single_stream_expected_latency_ns = 36669810


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_NX(SingleStreamGPUBaseConfig):
    system = KnownSystem.Orin_NX
    workspace_size = 30000000000
    use_graphs = True
    single_stream_expected_latency_ns = 45000000
    gpu_copy_streams = 2
    use_direct_host_access = True
