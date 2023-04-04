# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    run_infer_on_copy_streams = True
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 190000
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    offline_expected_qps = 190000 

	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    gpu_batch_size = 2048  
    run_infer_on_copy_streams = True
    offline_expected_qps = 19500
	
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    use_triton = True
    offline_expected_qps = 25000  
	
	
	
	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_batch_size = 1792
    gpu_copy_streams = 4
    offline_expected_qps = 85050
    run_infer_on_copy_streams = True
    numa_config = "0-1:0-43&2-3:44-87"

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True
    offline_expected_qps = 87050

	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2(OfflineGPUBaseConfig):
    system = KnownSystem.R4900G6_L4x2
    gpu_batch_size = 32
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 26000
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2_Triton(R4900G6_L4x2):
    use_triton = True