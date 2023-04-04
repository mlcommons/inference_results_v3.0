# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    offline_expected_qps = 13000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X8_HighAccuracy(G5500V7_A30X8):
    precision = "fp16"
    offline_expected_qps = 8119.999999999999

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 18000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X10_HighAccuracy(G5500V7_A30X10):
    precision = "fp16"
    gpu_inference_streams = 2
    offline_expected_qps = 8119.999999999999

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8

    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 27200
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X8_HighAccuracy(G5500V7_L40X8):
    precision = "fp16"
    offline_expected_qps = 7840
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 19000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X10_HighAccuracy(G5500V7_L40X10):
    precision = "fp16"
    offline_expected_qps = 9800

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(OfflineGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    offline_expected_qps = 9600
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4X6_2288H_V7_HighAccuracy(L4X6_2288H_V7):
    precision = "fp16"
    offline_expected_qps = 3300
    use_fp8 = True
    use_graphs = False