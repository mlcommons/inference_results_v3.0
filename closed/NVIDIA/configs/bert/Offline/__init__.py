import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.bert import GPUBaseConfig, CPUBaseConfig

class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NVIDIA_A100(OfflineGPUBaseConfig):
    system = KnownSystem.NVIDIA_A100

