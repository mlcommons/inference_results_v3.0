# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4_MIG_1g_10gb
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_HighAccuracy(NC96ads_A100_v4_MIG):
    precision = "fp16"
    offline_expected_qps = NC96ads_A100_v4_MIG.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_Triton(NC96ads_A100_v4_MIG):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_HighAccuracy_Triton(NC96ads_A100_v4_MIG_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32

