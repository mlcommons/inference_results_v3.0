# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100x1

    gpu_batch_size = 1
    single_stream_expected_latency_ns = 472434000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X1_HighAccuracy(H100X1):
    pass


