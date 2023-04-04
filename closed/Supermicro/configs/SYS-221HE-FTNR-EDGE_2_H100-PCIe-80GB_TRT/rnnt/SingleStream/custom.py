# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100x2

    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 5000000
    nouse_copy_kernel = False
