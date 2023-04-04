# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 2460
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 911
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 2435
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 896
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_copy_streams = 6
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 3
    server_target_qps = 453 * 3
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_copy_streams = 6
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 3
    server_target_qps = 1355
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_copy_streams = 6
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 3
    server_target_qps = 460 * 8
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_copy_streams = 6
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 3
    server_target_qps = 460 * 8
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 955
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 1280
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 2600
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 2600
    start_from_device = True

