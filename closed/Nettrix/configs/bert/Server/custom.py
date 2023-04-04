# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12185
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50_HighAccuracy(A40X8_CUSTOM_X640_G50):
    precision = "fp16"
    server_target_qps = 5425


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4710
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50_HighAccuracy(A40X3_CUSTOM_X620_G50):
    precision = "fp16"
    server_target_qps = 2110


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12080
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40_HighAccuracy(A40X8_CUSTOM_X640_G40):
    precision = "fp16"
    server_target_qps = 5410


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4550
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40_HighAccuracy(A40X3_CUSTOM_X620_G40):
    precision = "fp16"
    server_target_qps = 2052


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_batch_size = 32
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 6350
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40_HighAccuracy(L40X3_CUSTOM_X620_G40):
    precision = "fp16"
    gpu_batch_size = 64
    server_target_qps = 2830


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_batch_size = 32
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 6300
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50_HighAccuracy(L40X3_CUSTOM_X620_G50):
    precision = "fp16"
    server_target_qps = 2850


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_batch_size = 32
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 17350
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40_HighAccuracy(L40X8_CUSTOM_X640_G40):
    precision = "fp16"
    server_target_qps = 7650


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 17900
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50_HighAccuracy(L40X8_CUSTOM_X640_G50):
    precision = "fp16"
    server_target_qps = 7500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993
    server_target_qps = 4500
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40_HighAccuracy(A30X3_CUSTOM_X620_G40):
    precision = "fp16"
    server_target_qps = 2000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993
    server_target_qps = 5700
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50_HighAccuracy(A30X4_CUSTOM_X620_G50):
    precision = "fp16"
    server_target_qps = 2600


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993
    server_target_qps = 11600
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40_HighAccuracy(A30X8_CUSTOM_X640_G40):
    precision = "fp16"
    server_target_qps = 5000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    soft_drop = 0.993
    server_target_qps = 11400
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50_HighAccuracy(A30X8_CUSTOM_X640_G50):
    precision = "fp16"
    server_target_qps = 10100
