# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_batch_size = 2048
    offline_expected_qps = 320000
    gpu_inference_streams = 1
    gpu_copy_streams = 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_batch_size = 2048
    offline_expected_qps = 120000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 2048
    offline_expected_qps = 320000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_batch_size = 2048
    offline_expected_qps = 120000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_inference_streams = 4
    gpu_copy_streams = 5
    gpu_batch_size = 32
    offline_expected_qps = 42000 * 3


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_inference_streams = 4
    gpu_copy_streams = 5
    gpu_batch_size = 32
    offline_expected_qps = 42000 * 3


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    gpu_batch_size = 64
    offline_expected_qps = 42000 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_inference_streams = 3
    gpu_copy_streams = 4
    gpu_batch_size = 64
    offline_expected_qps = 40000 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 61000
    run_infer_on_copy_streams = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    offline_expected_qps = 81000
    run_infer_on_copy_streams = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 167000
    run_infer_on_copy_streams = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 167000
    run_infer_on_copy_streams = True
    start_from_device = True
