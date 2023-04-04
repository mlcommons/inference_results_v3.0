# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_num_bundles = 3
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 300000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps= 1167000
    use_jemalloc = False
    numa_config = "0-7:0-39"   ##

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X8_HighAccuracy(G5500V7_A30X8):
    server_target_qps= 1167000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_num_bundles = 3
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 300000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps= 1400000
    use_jemalloc = False
    numa_config = "0-9:0-39"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X10_HighAccuracy(G5500V7_A30X10):
    server_target_qps= 1410000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8
    num_staging_batches = 8
    num_staging_threads = 8
    gpu_num_bundles = 3
    gpu_batch_size = 600000
    server_target_qps = 1500000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-7:0-39"   ##

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X8_HighAccuracy(G5500V7_L40X8):
    pass
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(ServerGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    num_staging_batches = 8
    num_staging_threads = 8
    gpu_num_bundles = 3
    gpu_batch_size = 600000
    server_target_qps = 1510000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-9:0-39"   

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X10_HighAccuracy(G5500V7_L40X10):
    pass
    server_target_qps = 1510000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(ServerGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gemm_plugin_fairshare_cache_size = 18
    max_pairs_per_staging_thread = 20000
    num_staging_batches = 2
    num_staging_threads = 2
    gpu_num_bundles = 2
    gpu_batch_size = 100000
    server_target_qps = 320000
    use_jemalloc = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4X6_2288H_V7_HighAccuracy(L4X6_2288H_V7):
    pass
    server_target_qps = 320000



