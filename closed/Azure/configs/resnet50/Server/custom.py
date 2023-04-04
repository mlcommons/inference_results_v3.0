# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4_MIG_1g_10gb
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 4250
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG_Triton(NC96ads_A100_v4_MIG):
    use_triton = True
    server_target_qps = 3700

