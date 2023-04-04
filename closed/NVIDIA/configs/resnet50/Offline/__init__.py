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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    run_infer_on_copy_streams = False
    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 2


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.L4x1
    gpu_batch_size = 32
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 13000
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(OfflineGPUBaseConfig):
    system = KnownSystem.L40x1
    gpu_batch_size = 2048
    offline_expected_qps = 40000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    gpu_batch_size = 2048
    offline_expected_qps = 90000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 90000 * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_02x1(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GB_02x1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    gpu_batch_size = 2048
    offline_expected_qps = 57000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    offline_expected_qps = 450000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    # numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"    $ for ipp1-1470


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    gpu_batch_size = 2048
    offline_expected_qps = 52000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    offline_expected_qps = 200000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    gpu_batch_size = 2048
    offline_expected_qps = 40000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = KnownSystem.A100_PCIe_80GBx8
    offline_expected_qps = 293000.0
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    power_limit = 195
    offline_expected_qps = 250000
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 256000.0


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 256000.0
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    gpu_batch_size = 2048
    offline_expected_qps = 36000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx2
    offline_expected_qps = 72000.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_TritonUnified(A100_PCIe_80GB_aarch64x2):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    offline_expected_qps = 140000.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 120000.0
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    gpu_batch_size = 2048
    offline_expected_qps = 36000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(A100_PCIe_aarch64x1):
    system = KnownSystem.A100_PCIe_40GB_ARMx2
    offline_expected_qps = 72000.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_TritonUnified(A100_PCIe_aarch64x2):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(A100_PCIe_aarch64x1):
    system = KnownSystem.A100_PCIe_40GB_ARMx4
    offline_expected_qps = 140000.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_TritonUnified(A100_PCIe_aarch64x4):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 120000.0
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_MIG_1x1g_5gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    offline_expected_qps = 5100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_TritonUnified(A100_PCIe_MIG_1x1g5gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    offline_expected_qps = 4800


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 5100


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 43000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(A100_SXM_80GBx1):
    system = KnownSystem.A100_SXM_80GBx8
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 340000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    use_triton = True
    offline_expected_qps = 340000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    use_triton = True
    offline_expected_qps = 340000
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    offline_expected_qps = 270000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    offline_expected_qps = 270000
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    gpu_batch_size = 2048
    offline_expected_qps = 39500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    offline_expected_qps = 300000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 250000
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 300000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 300000
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    offline_expected_qps = 5100


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 5100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 36800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    system = KnownSystem.A100_SXM4_40GBx8
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 294400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_TritonUnified(A100_SXM4_40GBx8):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    gpu_batch_size = 1024
    offline_expected_qps = 3100
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_Triton(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_TritonUnified(A2x1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(A2x1):
    system = KnownSystem.A2x2
    gpu_batch_size = 1024
    offline_expected_qps = 6200
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_TritonUnified(A2x2):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_batch_size = 384
    gpu_copy_streams = 2
    workspace_size = 1073741824
    offline_expected_qps = 4900


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 4575


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_batch_size = 1536
    run_infer_on_copy_streams = True
    offline_expected_qps = 18200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True
    offline_expected_qps = 18900


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_TritonUnified(A30x1):
    use_triton = True
    offline_expected_qps = 18900
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 138000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    use_triton = True
    offline_expected_qps = 151200.0


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_TritonUnified(A30x8):
    use_triton = True
    offline_expected_qps = 151200.0
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(OfflineGPUBaseConfig):
    system = KnownSystem.T4x1
    gpu_batch_size = 256
    gpu_copy_streams = 4
    offline_expected_qps = 6100
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_TritonUnified(T4x1):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(T4x1):
    system = KnownSystem.T4x20
    offline_expected_qps = 121500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    use_triton = True
    offline_expected_qps = 121500


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_TritonUnified(T4x20):
    use_triton = True
    offline_expected_qps = 121500
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(T4x1):
    system = KnownSystem.T4x8
    offline_expected_qps = 50000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_TritonUnified(T4x8):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(OfflineGPUBaseConfig):
    system = KnownSystem.Orin
    # GPU-only QPS
    _gpu_offline_expected_qps = 4500
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 600
    # GPU + 2 DLA QPS
    offline_expected_qps = 6100
    use_direct_host_access = True
    run_infer_on_copy_streams = None
    dla_batch_size = 2
    dla_loop_count = 4
    dla_copy_streams = 2
    dla_inference_streams = 1
    dla_core = 0
    gpu_batch_size = 256
    input_format = 'chw4'
    tensor_path = 'build/preprocessed_data/imagenet/ResNet50/int8_chw4/'


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_MaxQ(Orin):
    soc_cpu_freq = 576000
    soc_gpu_freq = 624750000
    soc_dla_freq = 883200000
    soc_emc_freq = 2133000000
    orin_num_cores = 4
    offline_expected_qps = 3500
    gpu_batch_size = 256
    dla_batch_size = 8
    dla_loop_count = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_NX(OfflineGPUBaseConfig):
    system = KnownSystem.Orin_NX
    # GPU-only QPS
    _gpu_offline_expected_qps = 1700
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 230
    # GPU + 2 DLA QPS
    offline_expected_qps = 2400
    use_direct_host_access = True
    run_infer_on_copy_streams = None
    dla_batch_size = 2
    dla_loop_count = 4
    dla_copy_streams = 2
    dla_inference_streams = 1
    dla_core = 0
    gpu_batch_size = 256
    input_format = 'chw4'
    tensor_path = 'build/preprocessed_data/imagenet/ResNet50/int8_chw4/'


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_8380x1_Triton(OfflineCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_8380
    batch_size = 2
    offline_expected_qps = 4000
    max_queue_delay_usec = 100
    num_instances = 80
    ov_parameters = {'CPU_THREADS_NUM': '80', 'CPU_THROUGHPUT_STREAMS': '80', 'ENABLE_BATCH_PADDING': 'YES', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_2XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_2XLARGE
    offline_expected_qps = 1200
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/imagenet/ResNet50/fp32_pytorch/"
    precision = "fp32"
    input_dtype = "fp32"
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 5
    instance_group_count = 4
    inferentia_request_batch_size = 100
    benchmark = Benchmark.ResNet50
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_6XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_6XLARGE
    offline_expected_qps = 4000
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/imagenet/ResNet50/fp32_pytorch/"
    precision = "fp32"
    input_dtype = "fp32"
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 5
    instance_group_count = 16
    inferentia_request_batch_size = 100
    benchmark = Benchmark.ResNet50
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 360000
    power_limit = 215
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
