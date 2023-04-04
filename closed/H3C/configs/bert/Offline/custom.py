# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 19000 
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy(R5350G6_A30X10):
    precision = "fp16"
    offline_expected_qps = 8800  


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy_Triton(R5350G6_A30X10_HighAccuracy):
    use_triton = True

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768
   


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    use_triton = True
    offline_expected_qps = 1900     

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 7100
    workspace_size = 7516192768
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy(R5300G6_A30X4):
    precision = "fp16"
    offline_expected_qps = 4100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy_Triton(R5300G6_A30X4_HighAccuracy):
    use_triton = True

	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 550


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True

    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_R4950G6(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1_R4950G6
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 1760
    workspace_size = 20016192768 