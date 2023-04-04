# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(ServerGPUBaseConfig):
    system = KnownSystem.H100X4

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    use_cuda_thread_per_device = True
    use_graphs = True

    gpu_batch_size = 256

    server_target_qps = 194000
    numa_config = "0,1:0-31,64-95&2,3:32-63,96-127"

