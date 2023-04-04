# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.bert import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    active_sms = 60
    gpu_batch_size = 56
    gpu_copy_streams = 1
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps= 11850
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X8_HighAccuracy(G5500V7_A30X8):
    precision = "fp16"
    server_target_qps=5400

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    active_sms = 60
    gpu_batch_size = 64
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps=14900
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X10_HighAccuracy(G5500V7_A30X10):
    precision = "fp16"
    server_target_qps= 6260
    gpu_batch_size = 60

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 16000
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

   
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X8_HighAccuracy(G5500V7_L40X8):
    precision = "fp16"
    server_target_qps = 6950
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 20995
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X10_HighAccuracy(G5500V7_L40X10):
    precision = "fp16"
    server_target_qps = 8700

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(ServerGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    gpu_batch_size = 32 
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5500
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4X6_2288H_V7_HighAccuracy(L4X6_2288H_V7):
    precision = "fp16"
    gpu_batch_size = 16 
    server_target_qps = 3100
    use_fp8 = True
    use_graphs = False
    use_small_tile_gemm_plugin = False






