# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X2(OfflineGPUBaseConfig):
    system = KnownSystem.H100x2
    complete_threads = 1
    deque_timeout_usec = 1
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 350000
    offline_expected_qps = 940000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X2_HighAccuracy(H100X2):
    gpu_batch_size = 352000



