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

ParentConfig = import_module("configs.dlrm")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    complete_threads = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(ServerGPUBaseConfig):
    system = KnownSystem.L4x1
    embedding_weights_on_gpu_part = 0.8
    num_staging_batches = 2
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 14000
    server_target_qps = 89000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4x1_HighAccuracy(L4x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(ServerGPUBaseConfig):
    system = KnownSystem.L40x1
    num_staging_batches = 2
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 20000
    server_target_qps = 132600
    use_jemalloc = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40x1_HighAccuracy(L40x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    start_from_device = True
    use_small_tile_gemm_plugin = False
    use_jemalloc = True
    gpu_batch_size = 350000
    gpu_num_bundles = 4
    num_staging_batches = 2
    num_staging_threads = 4
    gpu_inference_streams = 1
    max_pairs_per_staging_thread = 180000
    complete_threads = 4
    server_target_qps = 735000
    compress_categorical_inputs = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 720000 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    gpu_batch_size = 200000
    server_target_qps = 120000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 32
    max_queue_delay_usec = 1000
    use_triton = True
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_HighAccuracy_Triton(H100_SXM_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    gpu_batch_size = 200000
    server_target_qps = 120000 * 8
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 32
    max_queue_delay_usec = 1000
    use_triton = True
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx8_HighAccuracy_Triton(H100_SXM_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_02x1(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GB_02x1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_SXM_80GBx1_600W_HighAccuracy(H100_SXM_80GB_02x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    use_small_tile_gemm_plugin = False
    use_jemalloc = True
    deque_timeout_usec = 100
    gpu_batch_size = 350000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 400000
    compress_categorical_inputs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    gpu_batch_size = 200000
    server_target_qps = 150000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 32
    max_queue_delay_usec = 1000
    use_triton = True
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    system = KnownSystem.H100_PCIe_80GBx8
    server_target_qps = 2500000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 2
    complete_threads = 2
    check_contiguity = True
    #numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    server_num_issue_query_threads = 8
    numa_config = "0:0-13,112-125&3:14-27,126-139&1:28-41,140-153&2:42-55,154-167&4:56-69,168-181&7:70-83,182-195&5:84-97,196-209&6:98-111,210-223"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    server_target_qps = 2100000
    power_limit = 290
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    gpu_batch_size = 262100
    server_target_qps = 150000 * 8
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 32
    max_queue_delay_usec = 1000
    use_triton = True
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 350000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    system = KnownSystem.H100_PCIe_80GB_ARMx8
    num_staging_batches = 2
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 262100
    server_target_qps = 1400000
    use_jemalloc = False
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    compress_categorical_inputs = True
    tensor_path = "build/preprocessed_data/criteo/full_recalib/numeric_int8_chw4.npy,build/preprocessed_data/criteo/full_recalib/categorical_int32_compressed.npy"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = KnownSystem.A100_PCIe_80GBx8
    num_staging_batches = 2
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 262100
    server_target_qps = 1600000
    use_jemalloc = False
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    compress_categorical_inputs = True
    #tensor_path = "build/preprocessed_data/criteo/full_recalib/numeric_int8_chw4.npy,build/preprocessed_data/criteo/full_recalib/categorical_int32_compressed.npy"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    gpu_batch_size = 200000
    server_target_qps = 600000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 32
    max_queue_delay_usec = 1000
    use_triton = True
    numa_config = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 750000
    power_limit = 185
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 700000
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    power_limit = 225
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx2
    server_target_qps = 300000
    use_jemalloc = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x2):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    server_target_qps = 600000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True
    batch_triton_requests = True
    server_target_qps = 460000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 500000
    # TODO: Set numa
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.3
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 36000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 20000
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    deque_timeout_usec = 1
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 286000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    num_staging_batches = 6
    num_staging_threads = 4
    gpu_num_bundles = 3
    gpu_batch_size = 263000
    server_target_qps = 2388000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx8_HighAccuracy):
    system = KnownSystem.A100_SXM_80GBx1
    server_target_qps = 298500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 2000000
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    server_target_qps = 1950000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 1830000
    power_limit = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 600000
    use_jemalloc = False
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 31000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 30000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 25000.0
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 132000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 3
    gpu_batch_size = 272000
    server_target_qps = 1056000
    use_jemalloc = False
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 600000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gemm_plugin_fairshare_cache_size = 18
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 69000
    server_target_qps = 43500
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy(A2x2):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    server_target_qps = 35300
    buffer_manager_thread_count = 0
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x2_HighAccuracy_Triton(A2x2_Triton):
    pass
