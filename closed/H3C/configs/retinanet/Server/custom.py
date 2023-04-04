# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4 
    gpu_inference_streams = 2
    server_target_qps = 3350
    workspace_size =70000000000 
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    server_target_qps =3200 
    instance_group_count = 4
    use_triton = True

	
	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 1330
    workspace_size = 20000000000
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    server_target_qps = 1280
    instance_group_count = 4
    use_triton = True
