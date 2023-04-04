# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_batch_size = 1
    offline_expected_qps = 24
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50_HighAccuracy(A40X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_batch_size = 1
    offline_expected_qps = 9
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50_HighAccuracy(A40X3_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 1
    offline_expected_qps = 24
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40_HighAccuracy(A40X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_batch_size = 1
    offline_expected_qps = 9
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40_HighAccuracy(A40X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_batch_size = 1
    offline_expected_qps = 2.95 * 3
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40_HighAccuracy(L40X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_batch_size = 1
    offline_expected_qps = 2.98 * 3
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50_HighAccuracy(L40X3_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_batch_size = 1
    offline_expected_qps = 5 * 8
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40_HighAccuracy(L40X8_CUSTOM_X640_G40):
    offline_expected_qps = 5 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_batch_size = 2
    offline_expected_qps = 3 * 8
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50_HighAccuracy(L40X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    gpu_batch_size = 2
    offline_expected_qps = 21
    numa_config = "0-3:0-27,56-83&4-7:28-55,84-111"

    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50_HighAccuracy(A30X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    gpu_batch_size = 2
    offline_expected_qps = 19
    numa_config = "0-3:0-9,20-29&4-7:10-19,30-39"

    start_from_device = True
    end_on_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40_HighAccuracy(A30X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    gpu_batch_size = 2
    offline_expected_qps = 8
    numa_config = "0-1:0-43,88-131&2-3:44-87,132-175"

    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50_HighAccuracy(A30X4_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    gpu_batch_size = 2
    offline_expected_qps = 6
    start_from_device = True
    end_on_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40_HighAccuracy(A30X3_CUSTOM_X620_G40):
    pass
