# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    single_stream_expected_latency_ns = 996648


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520c_A2x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_A2x1

    single_stream_expected_latency_ns = 720000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    single_stream_expected_latency_ns = 730000
    disable_beta1_smallk = True
    power_limit = 60


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(A30x1):
    system = KnownSystem.XR4520c_A30x1

    single_stream_expected_latency_ns = 498000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True
