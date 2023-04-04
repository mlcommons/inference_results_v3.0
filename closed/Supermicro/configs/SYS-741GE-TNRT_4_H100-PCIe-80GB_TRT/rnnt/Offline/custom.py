# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(OfflineGPUBaseConfig):
    system = KnownSystem.H100X4
    gpu_batch_size = 2048
    use_graphs = False  # MLPINF-1773
    offline_expected_qps = 65000
    disable_encoder_plugin = False

