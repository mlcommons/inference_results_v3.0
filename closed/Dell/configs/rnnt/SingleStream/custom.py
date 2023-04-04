# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 105000000
    nouse_copy_kernel = True
    power_limit = 60


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False
