# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    offline_expected_qps = 4000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1600
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    offline_expected_qps = 4000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1600
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_batch_size = 4
    gpu_copy_streams = 5
    gpu_inference_streams = 4
    offline_expected_qps = 550 * 3
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_batch_size = 4
    gpu_copy_streams = 5
    gpu_inference_streams = 4
    offline_expected_qps = 520 * 3
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_batch_size = 4
    gpu_copy_streams = 5
    gpu_inference_streams = 4
    offline_expected_qps = 620 * 8
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_batch_size = 4
    gpu_copy_streams = 5
    gpu_inference_streams = 3
    offline_expected_qps = 550 * 8
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1100
    run_infer_on_copy_streams = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1350
    run_infer_on_copy_streams = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 3900
    run_infer_on_copy_streams = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 3000
    run_infer_on_copy_streams = False
    start_from_device = True

