# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(ServerGPUBaseConfig):
    system = KnownSystem.H100x2
    use_small_tile_gemm_plugin = False
    use_jemalloc = True
    deque_timeout_usec = 1
    gpu_batch_size = 350000
    # gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 800000
    compress_categorical_inputs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X2_HighAccuracy(H100X2):
    pass



