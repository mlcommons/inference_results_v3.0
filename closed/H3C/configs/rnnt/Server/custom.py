# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x10
    gpu_batch_size = 1792
    server_target_qps = 44000
	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_batch_size = 1792
    server_target_qps = 8500
    numa_config = "0-1:0-43&2-3:44-87"
	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    audio_buffer_num_lines = 512
    dali_pipeline_depth = 1
    gpu_copy_streams = 4
    num_warmups = 32
    gpu_batch_size = 256
    audio_batch_size = 32
    server_target_qps = 2000
    
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2(ServerGPUBaseConfig):
    system = KnownSystem.R4900G6_L4x2
    gpu_batch_size = 512
    server_target_qps = 7450
    audio_batch_size = 64
    audio_buffer_num_lines = 1024
