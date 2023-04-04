# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    gpu_batch_size = 8    
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 3500
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 500
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    use_triton = True

	
	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1400
    run_infer_on_copy_streams = False
    workspace_size = 30000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True
    offline_expected_qps = 1300