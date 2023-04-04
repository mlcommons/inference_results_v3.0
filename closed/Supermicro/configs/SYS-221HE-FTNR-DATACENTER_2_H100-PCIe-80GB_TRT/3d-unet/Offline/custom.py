# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(OfflineGPUBaseConfig):
    system = KnownSystem.H100x2
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 9.6


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X2_HighAccuracy(H100X2):
    pass



