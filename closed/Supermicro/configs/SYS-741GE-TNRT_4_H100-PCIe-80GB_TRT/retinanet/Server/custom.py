# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(ServerGPUBaseConfig):
    system = KnownSystem.H100X4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    workspace_size = 60000000000
    server_target_qps = 3750
