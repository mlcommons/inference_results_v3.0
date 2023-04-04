# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 141500
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 53700
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 142000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 52800
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 32
    gpu_copy_streams = 12
    gpu_inference_streams = 3
    server_target_qps = 94300
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 31400 * 3
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 32
    gpu_copy_streams = 12
    gpu_inference_streams = 3
    server_target_qps = 245000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 32020 * 8
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    use_cuda_thread_per_device = True
    use_graphs = True
    server_target_qps = 53900
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    use_cuda_thread_per_device = True
    use_graphs = True
    server_target_qps = 71000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    use_cuda_thread_per_device = True
    use_graphs = True
    server_target_qps = 142500
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    use_cuda_thread_per_device = True
    use_graphs = True
    server_target_qps = 143000
    start_from_device = True
