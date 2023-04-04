# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4_MIG_1g_10gb
    gpu_inference_streams = 1
    active_sms = 100
    gpu_batch_size = 32
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 330
    soft_drop = 1
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_HighAccuracy(NC96ads_A100_v4_MIG):
    precision = "fp16"
    server_target_qps = 160

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_Triton(NC96ads_A100_v4_MIG):
    use_triton = True
    server_target_qps = 400

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_HighAccuracy_Triton(NC96ads_A100_v4_MIG_HighAccuracy):
    use_triton = True
    server_target_qps = 30

