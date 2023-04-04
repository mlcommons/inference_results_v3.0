# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X1(OfflineGPUBaseConfig):
    system = KnownSystem.H100x1
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 6000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X1_HighAccuracy(H100X1):
    pass


