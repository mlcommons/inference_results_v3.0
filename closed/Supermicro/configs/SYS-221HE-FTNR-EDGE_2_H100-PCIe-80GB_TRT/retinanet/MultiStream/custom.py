# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(MultiStreamGPUBaseConfig):
    system = KnownSystem.H100x2
    multi_stream_expected_latency_ns = 6000000
