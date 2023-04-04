# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X4(ServerGPUBaseConfig):
    system = KnownSystem.H100X4
    use_small_tile_gemm_plugin = False
    compress_categorical_inputs = True
    check_contiguity = True
    use_jemalloc = True
    complete_threads = 1
    deque_timeout_usec = 1
    num_staging_batches = 8
    num_staging_threads = 8
    gpu_batch_size = 350000
    server_target_qps = 1225000
    numa_config = "0,1:0-31,64-95&2,3:32-63,96-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X4_HighAccuracy(H100X4):
    pass

