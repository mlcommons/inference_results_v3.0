# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 5746
    gpu_batch_size = 205
    gpu_copy_streams = 11
    gpu_inference_streams = 9
    server_target_qps = 189000
    use_cuda_thread_per_device = True
    start_from_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 4650
    use_cuda_thread_per_device = False
    use_graphs = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 3965
    gpu_batch_size = 109
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 152700
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 305000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 192
    gpu_copy_streams = 4
    gpu_inference_streams = 6
    server_target_qps = 583500
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(ServerGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    use_deque_limit = True
    deque_timeout_usec = 2440
    gpu_batch_size = 12
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 12070
    use_cuda_thread_per_device = True
    use_graphs = True
