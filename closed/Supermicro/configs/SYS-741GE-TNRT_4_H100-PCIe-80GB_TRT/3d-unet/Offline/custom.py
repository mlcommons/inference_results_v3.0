# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(OfflineGPUBaseConfig):
    system = KnownSystem.H100X4
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 18.3


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X4_HighAccuracy(H100X4):
    offline_expected_qps = 17.5


