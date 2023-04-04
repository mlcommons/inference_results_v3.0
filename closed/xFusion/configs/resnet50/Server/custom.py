# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

      

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 72
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps= 146600
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 72
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps=181000
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 243000
    use_cuda_thread_per_device = True
    use_graphs = True
    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 301000
    use_cuda_thread_per_device = True
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(ServerGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 70
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 71400
    use_cuda_thread_per_device = True
    use_graphs = True