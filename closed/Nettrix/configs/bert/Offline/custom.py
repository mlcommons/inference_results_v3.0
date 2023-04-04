# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 24000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50_HighAccuracy(A40X8_CUSTOM_X640_G50):
    precision = "fp16"
    offline_expected_qps = A40X8_CUSTOM_X640_G50.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 10000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50_HighAccuracy(A40X3_CUSTOM_X620_G50):
    precision = "fp16"
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 1024
    offline_expected_qps = 24000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40_HighAccuracy(A40X8_CUSTOM_X640_G40):
    precision = "fp16"
    offline_expected_qps = A40X8_CUSTOM_X640_G40.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 10000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40_HighAccuracy(A40X3_CUSTOM_X620_G40):
    precision = "fp16"
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 3
    gpu_inference_streams = 4
    gpu_batch_size = 128
    offline_expected_qps = 3500 * 3
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40_HighAccuracy(L40X3_CUSTOM_X620_G40):
    precision = "fp16"
    gpu_inference_streams = 3
    gpu_copy_streams = 3
    gpu_batch_size = 128
    offline_expected_qps = 3400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    offline_expected_qps = 4000 * 3
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50_HighAccuracy(L40X3_CUSTOM_X620_G50):
    precision = "fp16"
    offline_expected_qps = L40X3_CUSTOM_X620_G50.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    gpu_batch_size = 128
    offline_expected_qps = 3400 * 8
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40_HighAccuracy(L40X8_CUSTOM_X640_G40):
    precision = "fp16"
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    offline_expected_qps = 3400 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    gpu_batch_size = 128
    offline_expected_qps = 4000 * 8
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50_HighAccuracy(L40X8_CUSTOM_X640_G50):
    precision = "fp16"
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    offline_expected_qps = L40X8_CUSTOM_X640_G50.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768
    offline_expected_qps = 5100
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40_HighAccuracy(A30X3_CUSTOM_X620_G40):
    precision = "fp16"
    offline_expected_qps = 2550


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768
    offline_expected_qps = 6700
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50_HighAccuracy(A30X4_CUSTOM_X620_G50):
    precision = "fp16"
    offline_expected_qps = 3400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768
    offline_expected_qps = 14000

    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40_HighAccuracy(A30X8_CUSTOM_X640_G40):
    precision = "fp16"
    offline_expected_qps = A30X8_CUSTOM_X640_G40.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768
    offline_expected_qps = 13600

    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50_HighAccuracy(A30X8_CUSTOM_X640_G50):
    precision = "fp16"
    offline_expected_qps = A30X8_CUSTOM_X640_G50.offline_expected_qps / 2

