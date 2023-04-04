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
from configs.bert import GPUBaseConfig, CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline

    max_queue_delay_usec = 100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 5700
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 5000
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_HighAccuracy):
    offline_expected_qps = 1800
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    offline_expected_qps = 46000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx1_HighAccuracy):
    system = KnownSystem.H100_PCIe_80GBx8
    offline_expected_qps = 5000 * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 4000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1_HighAccuracy(H100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1800
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    offline_expected_qps = 16000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x1_HighAccuracy):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    offline_expected_qps = 8000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_02x1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GB_02x1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 8900
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 7900


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy_Triton(H100_SXM_80GBx1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 9400 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx1_HighAccuracy):
    system = KnownSystem.H100_SXM_80GBx8
    offline_expected_qps = 7900 * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx8_HighAccuracy_Triton(H100_SXM_80GBx8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 8
    offline_expected_qps = 900
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    gpu_batch_size = 16
    offline_expected_qps = 440


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(OfflineGPUBaseConfig):
    system = KnownSystem.L40x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_TritonUnified(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = KnownSystem.A100_PCIe_80GBx8
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    offline_expected_qps = 12800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_TritonUnified(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    offline_expected_qps = 27200
    power_limit = 240


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    offline_expected_qps = 11000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx2
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_TritonUnified(A100_PCIe_80GB_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 10000
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy_Triton(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx2
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_TritonUnified(A100_PCIe_aarch64x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy_Triton(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_TritonUnified(A100_PCIe_aarch64x4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy_Triton(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 9000
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 4500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_MIG_1x1g_5gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy(A100_PCIe_MIG_1x1g5gb):
    precision = "fp16"
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_TritonUnified(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy_Triton(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    offline_expected_qps = 470


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_TritonUnified(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(A100_SXM_80GBx1):
    system = KnownSystem.A100_SXM_80GBx8
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_TritonUnified(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    power_limit = 275
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 11000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 2500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_HighAccuracy(A100_SXM_80GB_aarch64x1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_HighAccuracy_Triton(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    offline_expected_qps = 27500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 22000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 14000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_HighAccuracy):
    offline_expected_qps = 12000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 14000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 14000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 350
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    precision = "fp16"
    gpu_batch_size = 64
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_TritonUnified(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    system = KnownSystem.A100_SXM4_40GBx8
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy(A100_SXM4_40GBx8):
    precision = "fp16"
    gpu_batch_size = 1024
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_TritonUnified(A100_SXM4_40GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_Triton(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_TritonUnified(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x1_HighAccuracy(A2x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 120


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_Triton(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_TritonUnified(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x1_HighAccuracy_Triton(A2x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x1_HighAccuracy_TritonUnified(A2x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy(A2x2):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 240


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_TritonUnified(A2x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy_Triton(A2x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy_TritonUnified(A2x2_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 96
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 505
    workspace_size = 805306368


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    offline_expected_qps = 246.3


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 457.658


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    precision = "fp16"
    offline_expected_qps = 219.18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 240


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_TritonUnified(A30_MIG_1x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 240


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    precision = "fp16"
    offline_expected_qps = 1014.9999999999999


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True
    offline_expected_qps = 1739.9999999999998


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_TritonUnified(A30x1):
    use_triton = True
    offline_expected_qps = 1739.9999999999998


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_TritonUnified(A30x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    offline_expected_qps = 13000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    precision = "fp16"
    offline_expected_qps = 8119.999999999999


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_TritonUnified(A30x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_TritonUnified(A30x8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(OfflineGPUBaseConfig):
    system = KnownSystem.Orin
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 550


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_MaxQ(Orin):
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 384
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    soc_cpu_freq = 576000
    soc_gpu_freq = 652800000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    orin_num_cores = 4
    offline_expected_qps = 280


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_NX(OfflineGPUBaseConfig):
    system = KnownSystem.Orin_NX
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 154


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(OfflineGPUBaseConfig):
    system = KnownSystem.T4x1
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 430


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(T4x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_TritonUnified(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(T4x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 189


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_TritonUnified(T4x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 189


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(T4x1):
    system = KnownSystem.T4x20
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 8800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy(T4x20):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 4400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    use_triton = True
    offline_expected_qps = 7920


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_TritonUnified(T4x20):
    use_triton = True
    offline_expected_qps = 7920


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_Triton(T4x20_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 3960


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_TritonUnified(T4x20_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 3960


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(T4x1):
    system = KnownSystem.T4x8
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy(T4x8):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 1680


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    use_triton = True
    offline_expected_qps = 3150


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_TritonUnified(T4x8):
    use_triton = True
    offline_expected_qps = 3150


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_Triton(T4x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 1512


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_TritonUnified(T4x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 1512


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_8380x1(OfflineCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_8380
    batch_size = 0
    offline_expected_qps = 70
    num_instances = 16
    ov_parameters = {
        'CPU_THREADS_NUM': '80',
        'CPU_THROUGHPUT_STREAMS': '16',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_2XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_2XLARGE
    offline_expected_qps = 70
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 4


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Triton_Inferentia_HighAccuracy_INF1_2XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_2XLARGE
    offline_expected_qps = 70
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 4


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_6XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_6XLARGE
    offline_expected_qps = 275
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 4
    instance_group_count = 16


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Triton_Inferentia_HighAccuracy_INF1_6XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_6XLARGE
    offline_expected_qps = 275
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 4
    instance_group_count = 16


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 40100
    power_limit = 300


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_MaxQ):
    pass
