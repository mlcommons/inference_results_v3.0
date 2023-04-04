# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(OfflineGPUBaseConfig):
    system = KnownSystem.H100X4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 22000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X4_HighAccuracy(H100X4):
    system = KnownSystem.H100X4
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 20000

