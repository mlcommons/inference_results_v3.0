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
from configs.resnet50 import GPUBaseConfig, CPUBaseConfig, LONBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server


class ServerLONBaseConfig(LONBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(ServerGPUBaseConfig):
    system = KnownSystem.L40x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 16000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 236000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_inference_streams = 5
    server_target_qps = 230000
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8_Triton):
    batch_triton_requests = True
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_inference_streams = 5
    server_target_qps = 220000
    numa_config = None
    use_graphs = False
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    numa_config = None
    gpu_batch_size = 128
    gpu_inference_streams = 3
    server_target_qps = 203500
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 130000
    use_graphs = False
    power_limit = 175
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    server_target_qps = 130000
    use_graphs = False
    power_limit = 175
    use_triton = True


class A100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    system = KnownSystem.A100_PCIe_80GB_ARMx4
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 104000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    server_target_qps = 93200
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    server_target_qps = 93200
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 92500
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3530
    start_from_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3440
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 3440
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 59000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GB_02x1(H100_SXM_80GBx1):
    system = KnownSystem.H100_SXM_80GB_02x1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 300000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 200000
    gather_kernel_buffer_threshold = 0
    gpu_batch_size = 64
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_graphs = True
    start_from_device = False
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 200000
    gather_kernel_buffer_threshold = 0
    gpu_batch_size = 64
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_graphs = True
    start_from_device = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 229000
    power_limit = 225


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 200000
    use_graphs = False
    start_from_device = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3600
    use_graphs = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    server_target_qps = 3500
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 270000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    server_target_qps = 230000
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 186000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 186000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    # gpu_batch_size = 128
    gpu_batch_size = 16
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    # server_target_qps = 4691
    server_target_qps = 5400
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True
    gpu_batch_size = 16
    server_target_qps = 5150


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_TritonUnified(A2x2):
    use_triton = True
    gpu_batch_size = 16
    server_target_qps = 5150


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(ServerGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    use_deque_limit = True
    deque_timeout_usec = 200
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 3400
    use_cuda_thread_per_device = True
    use_graphs = True
    workspace_size = 1610612736


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 3080


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    server_target_qps = 1800
    gpu_inference_streams = 3
    use_graphs = False
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    server_target_qps = 1800
    gpu_inference_streams = 3
    use_graphs = False
    use_triton = True


class A30x1(ServerGPUBaseConfig):
    system = KnownSystem.A30x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 15079.999999999998
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = KnownSystem.A30x8
    server_target_qps = 116000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    server_target_qps = 110000
    use_graphs = False
    use_triton = True
    gpu_batch_size = 32
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    batch_triton_requests = False
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_TritonUnified(A30x8):
    server_target_qps = 110000
    use_graphs = False
    use_triton = True
    gpu_batch_size = 32
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    batch_triton_requests = False
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_8380x1_Triton(ServerCPUBaseConfig):
    system = KnownSystem.Triton_CPU_2S_8380
    use_deque_limit = True
    server_target_qps = 3140
    batch_size = 1
    num_instances = 20
    ov_parameters = {'CPU_THREADS_NUM': '80', 'CPU_THROUGHPUT_STREAMS': '20', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_2XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_2XLARGE
    server_target_qps = 485
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/imagenet/ResNet50/fp32_pytorch/"
    precision = "fp32"
    input_dtype = "fp32"
    use_triton = True
    scenario = Scenario.Server
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    instance_group_count = 4
    inferentia_request_batch_size = 1
    benchmark = Benchmark.ResNet50
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_Inferentia_INF1_6XLARGEx1(BenchmarkConfiguration):
    system = KnownSystem.Triton_Inferentia_INF1_6XLARGE
    server_target_qps = 2500
    map_path = "data_maps/imagenet/val_map.txt"
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/imagenet/ResNet50/fp32_pytorch/"
    precision = "fp32"
    input_dtype = "fp32"
    use_triton = True
    scenario = Scenario.Server
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    instance_group_count = 16
    inferentia_request_batch_size = 1
    benchmark = Benchmark.ResNet50
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.LON_Node, AccuracyTarget.k_99, PowerSetting.MaxP)
class LON_2S_AMD_7742_CX6x8(ServerLONBaseConfig):
    system = KnownSystem.LON_2S_AMD_7742_CX6x8
    server_target_qps = 300000

    lon_numa_config = "mlx5_3:0:0-15&mlx5_2:0:16-31&mlx5_1:0:32-47&mlx5_0:0:48-63&mlx5_9:0:64-79&mlx5_6:0:80-95&mlx5_5:0:96-111&mlx5_4:0:112-127"
    sut_numa_config = "mlx5_3:3:0-15&mlx5_2:2:16-31&mlx5_1:1:32-47&mlx5_0:0:48-63&mlx5_9:7:64-79&mlx5_8:6:80-95&mlx5_7:5:96-111&mlx5_6:4:112-127"
    nic_mapping = "mlx5_3:mlx5_3&mlx5_2:mlx5_2&mlx5_1:mlx5_1&mlx5_0:mlx5_0&mlx5_9:mlx5_9&mlx5_6:mlx5_8&mlx5_5:mlx5_7&mlx5_4:mlx5_6"
    sut_nic_gpu_affinity = "mlx5_0:0&mlx5_1:1&mlx5_2:2&mlx5_3:3&mlx5_6:4&mlx5_7:5&mlx5_8:6&mlx5_9:7"

    lon_netid = "ipp2-2031"
    sut_netid = "luna-prod-72-80gb"
    tcp_port = "7000"

    lon_uses_one_issue_queue = False
    round_robin_samples_to_multi_issue_queue = False
    smart_balance_samples_to_multi_issue_queue = True
    enable_max_transactions_per_connection = True
    max_transactions_per_connection = 8192
    max_wait_before_sending_us = 10
    num_ibqps_per_nic = 1


@ConfigRegistry.register(HarnessType.SUT_Node, AccuracyTarget.k_99, PowerSetting.MaxP)
class SUT_2S_A100_SXM_80GBx8_CX6x8(ServerGPUBaseConfig):
    system = KnownSystem.SUT_2S_A100_SXM_80GBx8_CX6x8
    server_target_qps = 37500 * 8

    use_graphs = True

    gpu_copy_streams = 4
    gpu_inference_streams = 2

    gpu_batch_size = 128
    use_deque_limit = True
    deque_timeout_usec = 2000

    lon_numa_config = "mlx5_3:0:0-15&mlx5_2:0:16-31&mlx5_1:0:32-47&mlx5_0:0:48-63&mlx5_9:0:64-79&mlx5_6:0:80-95&mlx5_5:0:96-111&mlx5_4:0:112-127"
    sut_numa_config = "mlx5_3:3:0-15&mlx5_2:2:16-31&mlx5_1:1:32-47&mlx5_0:0:48-63&mlx5_9:7:64-79&mlx5_8:6:80-95&mlx5_7:5:96-111&mlx5_6:4:112-127"
    nic_mapping = "mlx5_3:mlx5_3&mlx5_2:mlx5_2&mlx5_1:mlx5_1&mlx5_0:mlx5_0&mlx5_9:mlx5_9&mlx5_6:mlx5_8&mlx5_5:mlx5_7&mlx5_4:mlx5_6"
    sut_nic_gpu_affinity = "mlx5_0:0&mlx5_1:1&mlx5_2:2&mlx5_3:3&mlx5_6:4&mlx5_7:5&mlx5_8:6&mlx5_9:7"

    lon_netid = "ipp2-2031"
    sut_netid = "luna-prod-72-80gb"
    tcp_port = "7000"

    SUT_uses_host_mem_for_RDMA = False
    enable_max_transactions_per_connection = True
    max_transactions_per_connection = 8192
    max_wait_before_sending_us = 10
    num_ibqps_per_nic = 1
