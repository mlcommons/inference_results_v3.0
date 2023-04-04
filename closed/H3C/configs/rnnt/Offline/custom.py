# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x10
    gpu_batch_size = 2048
    offline_expected_qps = 69599.999999999999 

	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x1
    gpu_batch_size = 2048
    offline_expected_qps = 7500

	
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    gpu_batch_size = 2048
    offline_expected_qps = 29000
    numa_config = "0-1:0-43&2-3:44-87"
	
	
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(OfflineGPUBaseConfig):
    system = KnownSystem.A2x2
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 2460

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2(OfflineGPUBaseConfig):
    system = KnownSystem.R4900G6_L4x2
    gpu_batch_size = 512
    offline_expected_qps = 8000
    audio_batch_size = 64
    audio_buffer_num_lines = 1024