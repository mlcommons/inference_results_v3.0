# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X1(OfflineGPUBaseConfig):
    system = KnownSystem.H100x1
    gpu_batch_size = 2048
    offline_expected_qps = 57000

