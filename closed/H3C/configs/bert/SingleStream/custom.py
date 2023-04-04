# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    single_stream_expected_latency_ns = 3400000
    use_triton = True
	

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_R4950G6(SingleStreamGPUBaseConfig):
    system = KnownSystem.A30x1_R4950G6
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000
