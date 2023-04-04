# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(ServerGPUBaseConfig):
    system = KnownSystem.H100x2
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 8600  # default: 9400
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    # soft_drop = 1.0
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X2_HighAccuracy(H100X2):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 7800
    soft_drop = 1.0


