# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520c_A2x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_A2x1

    multi_stream_expected_latency_ns = 114000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    multi_stream_expected_latency_ns = 114000000
    # Use early stopping estimate due to long runtime.
    min_duration = 3600000
    min_query_count = 16384


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1

    use_graphs = True
    multi_stream_expected_latency_ns = 68400000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    multi_stream_expected_latency_ns = 12000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    multi_stream_expected_latency_ns = 12000000
