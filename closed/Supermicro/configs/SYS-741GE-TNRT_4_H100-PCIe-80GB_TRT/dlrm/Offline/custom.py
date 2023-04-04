# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(OfflineGPUBaseConfig):
    system = KnownSystem.H100X4
    complete_threads = 1
    deque_timeout_usec = 1
    use_small_tile_gemm_plugin = False
    compress_categorical_inputs = False
    gpu_batch_size = 350000
    offline_expected_qps = 1880000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    numa_config = "0,1:0-31,64-95&2,3:32-63,96-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X4_HighAccuracy(H100X4):
    pass




