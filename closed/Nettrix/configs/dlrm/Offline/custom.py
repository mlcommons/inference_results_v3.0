# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 133000
    offline_expected_qps = 270000 * 8
    max_pairs_per_staging_thread = 133000
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 2
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50_HighAccuracy(A40X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 480000
    offline_expected_qps = 800000
    max_pairs_per_staging_thread = 480000
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50_HighAccuracy(A40X3_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 480000
    offline_expected_qps = 270000 * 8
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40_HighAccuracy(A40X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 480000
    offline_expected_qps = 800000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40_HighAccuracy(A40X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_batch_size = 65525
    offline_expected_qps = 300000 * 3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40_HighAccuracy(L40X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 65525
    offline_expected_qps = 320000 * 3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50_HighAccuracy(L40X3_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 262100
    offline_expected_qps = 320000 * 8
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40_HighAccuracy(L40X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_batch_size = 65525
    offline_expected_qps = 300000 * 8
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50_HighAccuracy(L40X8_CUSTOM_X640_G50):
    gpu_copy_streams = 3
    gpu_inference_streams = 2
    gpu_batch_size = 32763
    offline_expected_qps = 300000 * 8
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1400000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50_HighAccuracy(A30X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1120000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40_HighAccuracy(A30X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(OfflineGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 570000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50_HighAccuracy(A30X4_CUSTOM_X620_G50):
    offline_expected_qps = 571000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 441000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40_HighAccuracy(A30X3_CUSTOM_X620_G40):
    offline_expected_qps = 440000

