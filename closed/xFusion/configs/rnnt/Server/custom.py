# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.rnnt import GPUBaseConfig

class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    use_graphs = True
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    num_warmups = 20480
    nobatch_sorting = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    gpu_batch_size = 1792  #2048
    server_target_qps= 32100

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    gpu_batch_size = 1792
    server_target_qps= 44000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8
    gpu_batch_size = 2048
    server_target_qps = 76500

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    gpu_batch_size = 2048
    server_target_qps = 95300

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(ServerGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7

    gpu_batch_size = 1024  #1792
    server_target_qps = 6000