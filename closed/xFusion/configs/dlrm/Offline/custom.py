# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1110000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X8_HighAccuracy(G5500V7_A30X8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1420000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_A30X10_HighAccuracy(G5500V7_A30X10):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8

    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 262100
    offline_expected_qps = 1440000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X8_HighAccuracy(G5500V7_L40X8):
    pass
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 262100
    offline_expected_qps = 1800000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class G5500V7_L40X10_HighAccuracy(G5500V7_L40X10):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(OfflineGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    enable_interleaved_top_mlp = True
    output_padding_granularity = 32
    use_small_tile_gemm_plugin = False
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    complete_threads = 8
    offline_expected_qps = 380000
    num_staging_batches = 8
    num_staging_threads = 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L4X6_2288H_V7_HighAccuracy(L4X6_2288H_V7):
    pass