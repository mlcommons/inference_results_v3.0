# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    offline_expected_qps = 18
    numa_config = "0-9:0-95"
   


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy(R5350G6_A30X10):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    offline_expected_qps = 18
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy_Triton(R5350G6_A30X10_Triton):
    pass

	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    gpu_batch_size = 2
    offline_expected_qps = 1.70

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X1_HighAccuracy(R5350G6_A30X1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    use_triton = True



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X1_HighAccuracy_Triton(R5350G6_A30X1_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_batch_size = 2
    offline_expected_qps = 6.76
    numa_config = "0-1:0-43&2-3:44-87"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy(R5300G6_A30X4):
    pass
	
	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2(OfflineGPUBaseConfig):
    system = KnownSystem.R4900G6_L4x2
    gpu_batch_size = 1
    offline_expected_qps = 2.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R4900G6_L4x2_HighAccuracy(R4900G6_L4x2):
    pass