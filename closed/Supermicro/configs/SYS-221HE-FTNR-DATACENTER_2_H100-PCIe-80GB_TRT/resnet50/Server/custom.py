# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(ServerGPUBaseConfig):
    system = KnownSystem.H100x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 90000
    use_cuda_thread_per_device = True
    use_graphs = True


