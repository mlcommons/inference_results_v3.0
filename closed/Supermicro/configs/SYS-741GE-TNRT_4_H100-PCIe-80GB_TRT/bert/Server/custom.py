# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(ServerGPUBaseConfig):
    system = KnownSystem.H100X4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 64
    # server_target_qps = 4500 * 8
    server_target_qps = 17400  
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    graphs_max_seqlen = 200
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X4_HighAccuracy(H100X4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 3800 * 4

