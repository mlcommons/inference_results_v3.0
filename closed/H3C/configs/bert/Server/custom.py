# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps =15100    
    soft_drop = 0.993    
    


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy(R5350G6_A30X10):
    precision = "fp16"
    server_target_qps = 6800  


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    server_target_qps = 14800
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy_Triton(R5350G6_A30X10_HighAccuracy):
    use_triton = True
    server_target_qps = 7550   


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5950
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy(R5300G6_A30X4):
    precision = "fp16"
    server_target_qps = 2700


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True
    server_target_qps = 6070
    max_queue_delay_usec = 1000
    

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy_Triton(R5300G6_A30X4_HighAccuracy):
    use_triton = True
    server_target_qps = 2960
	
	
	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    server_target_qps = 325
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    soft_drop = 0.993
    gpu_batch_size = 8
    enable_interleaved = False



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True
    server_target_qps = 350

