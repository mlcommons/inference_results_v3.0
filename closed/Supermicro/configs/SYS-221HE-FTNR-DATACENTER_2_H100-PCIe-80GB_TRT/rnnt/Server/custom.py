# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(ServerGPUBaseConfig):
    system = KnownSystem.H100x2
    gpu_batch_size = 2048
    server_target_qps = 30000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


