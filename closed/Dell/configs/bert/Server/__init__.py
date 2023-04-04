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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False   
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 4700
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    # soft_drop = 1.0
    soft_drop = 0.99

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    server_target_qps = 1500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True
    server_target_qps = 2300


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    server_target_qps = 1150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx8
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 64
    # server_target_qps = 4500 * 8
    server_target_qps = 4600 * 8
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    graphs_max_seqlen = 200
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    power_limit = 300
    server_target_qps = 20500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx8):
    precision = "fp16"
    server_target_qps = 12000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_HighAccuracy):
    power_limit = 300
    server_target_qps = 10000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True
    server_target_qps = 18500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    server_target_qps = 9250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = True
    use_graphs = False
    gpu_batch_size = 1280
    server_target_qps = 3000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1_HighAccuracy(H100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    server_target_qps = 1500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = True
    use_graphs = False
    gpu_batch_size = 1280
    server_target_qps = 12000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    server_target_qps = 6000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 128
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 8200
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 7500 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    use_graphs = False
    gpu_batch_size = 128
    server_target_qps = 6200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(ServerGPUBaseConfig):
    system = KnownSystem.L4x1
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 900
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 380
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(ServerGPUBaseConfig):
    system = KnownSystem.L40x1
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1500.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    server_target_qps = 1500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx8
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 23000.0
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    server_target_qps = 10800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 18000
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    server_target_qps = 18000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    server_target_qps = 9500
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_TritonUnified(A100_PCIe_80GBx8_HighAccuracy):
    server_target_qps = 9500
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 17300
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    server_target_qps = 7500
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 17000
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 17000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    server_target_qps = 9480
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 10400.0
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    server_target_qps = 10000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    server_target_qps = 10000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    server_target_qps = 4800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True
    server_target_qps = 4500


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True
    server_target_qps = 4500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 10000.0
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    server_target_qps = 4000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.99
    use_graphs = False


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 170


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 160


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 170
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 170
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.993
    deque_timeout_usec = 50000
    use_graphs = False


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 164


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 160


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 360
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 166
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 170
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 1.0
    gpu_copy_streams = 4
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 12820
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.A100_SXM_80GBx1
    server_target_qps = 1575


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 11205
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_TritonUnified(A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 11205
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 21500
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 10000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    server_target_qps = 22455
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    server_target_qps = 22455
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    gpu_batch_size = 48
    server_target_qps = 11205
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    gpu_batch_size = 48
    server_target_qps = 11205
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 21000
    power_limit = 300


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 12300


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_HighAccuracy):
    soft_drop = 0.995
    power_limit = 300
    server_target_qps = 8800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    server_target_qps = 22000
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    server_target_qps = 22000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 12000
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 12000
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 320
    soft_drop = 0.99
    use_graphs = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    server_target_qps = 220


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 220


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 320
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 320
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 220
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    precision = "fp16"
    server_target_qps = 220
    use_triton = True


class A2x1(ServerGPUBaseConfig):
    system = KnownSystem.A2x1
    enable_interleaved = False
    #active_sms = 10
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 900
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(A2x1):
    system = KnownSystem.A2x2
    server_target_qps = 325


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy(A2x2):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 150


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_TritonUnified(A2x2):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy_Triton(A2x2_HighAccuracy):
    use_triton = True
    gpu_batch_size = 4
    server_target_qps = 130


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy_TritonUnified(A2x2_HighAccuracy):
    use_triton = True
    gpu_batch_size = 4
    server_target_qps = 130


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_inference_streams = 1
    active_sms = 60
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 380
    soft_drop = 0.993
    deque_timeout_usec = 50000
    workspace_size = 805306368
    use_graphs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    gpu_batch_size = 12
    deque_timeout_usec = 70000
    server_target_qps = 160


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 360


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    server_target_qps = 145


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 380
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    server_target_qps = 380
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_HighAccuracy):
    gpu_batch_size = 8
    server_target_qps = 150
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_TritonUnified(A30_MIG_1x1g6gb_HighAccuracy):
    gpu_batch_size = 8
    server_target_qps = 150
    use_triton = True


class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1500
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    server_target_qps = 11500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    precision = "fp16"
    server_target_qps = 5250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 11000
    use_triton = True
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_TritonUnified(A30x8):
    server_target_qps = 11000
    use_triton = True
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_HighAccuracy):
    server_target_qps = 5200
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_TritonUnified(A30x8_HighAccuracy):
    server_target_qps = 5200
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_8380x1_Triton(ServerCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_8380
    batch_size = 0
    server_target_qps = 39
    num_instances = 4
    ov_parameters = {'CPU_THREADS_NUM': '80', 'CPU_THROUGHPUT_STREAMS': '4', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_2XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_2XLARGE
    server_target_qps = 37
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Server
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
    server_target_qps = 37
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Server
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
    server_target_qps = 130
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Server
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 16


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Triton_Inferentia_HighAccuracy_INF1_6XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_6XLARGE
    server_target_qps = 130
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Server
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 16
