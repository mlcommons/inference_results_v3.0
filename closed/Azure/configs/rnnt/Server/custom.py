# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NC96ads_A100_v4_MIG(ServerGPUBaseConfig):
    system = KnownSystem.NC96ads_A100_v4_MIG_1g_10gb
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    gpu_batch_size = 1024
    num_warmups = 64
    server_target_qps = 1000
    max_seq_length = 64


