# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(OfflineGPUBaseConfig):
    system = KnownSystem.H100X4
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 12000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


