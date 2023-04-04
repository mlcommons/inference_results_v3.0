# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    gpu_batch_size = 1
    offline_expected_qps=13

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30x8_HighAccuracy_Triton(G5500V7_A30x8_Triton):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps=13

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X10_HighAccuracy(G5500V7_A30X10):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8
    gpu_batch_size = 1
    offline_expected_qps = 24
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X8_HighAccuracy(G5500V7_L40X8):
    pass
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    gpu_batch_size = 1  
    offline_expected_qps = 30
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X10_HighAccuracy(G5500V7_L40X10):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(OfflineGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    gpu_batch_size = 1    
    offline_expected_qps = 8
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4X6_2288H_V7_HighAccuracy(L4X6_2288H_V7):
    pass

