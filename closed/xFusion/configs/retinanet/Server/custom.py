# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps= 2600
    workspace_size = 40000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps=  3350
    workspace_size = 40000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8

    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2  #4
    server_target_qps = 3520
    workspace_size = 70000000000
    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2  #4
    server_target_qps = 4200
    workspace_size = 70000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(ServerGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4   #8
    gpu_inference_streams = 2   #4
    server_target_qps = 1050
    workspace_size = 20000000000