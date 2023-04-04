# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    single_stream_expected_latency_ns = 4000000	


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1_Triton(R5350G6_A30X1):
    use_triton = True



