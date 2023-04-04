# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG(OfflineGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4_MIG_1g_10gb
    gpu_batch_size = 256
    run_infer_on_copy_streams = True
    offline_expected_qps = 5500

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_Triton(NC96ads_A100_v4_MIG):
    use_triton = True

