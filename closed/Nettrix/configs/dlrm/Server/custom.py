# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    num_staging_batches = 10
    num_staging_threads = 10
    gpu_num_bundles = 2
    gpu_batch_size = 130000
    server_target_qps = 1116000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50_HighAccuracy(A40X8_CUSTOM_X640_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 2
    gpu_batch_size = 200000
    server_target_qps = 472000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50_HighAccuracy(A40X3_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    num_staging_batches = 12
    num_staging_threads = 12
    gpu_num_bundles = 2
    gpu_batch_size = 100000
    server_target_qps = 800000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40_HighAccuracy(A40X8_CUSTOM_X640_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 2
    gpu_batch_size = 200000
    server_target_qps = 436000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40_HighAccuracy(A40X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    num_staging_batches = 4
    num_staging_threads = 4
    gpu_num_bundles = 2
    gpu_batch_size = 200000
    server_target_qps = 546000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40_HighAccuracy(L40X3_CUSTOM_X620_G40):
    server_target_qps = 547700


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    num_staging_batches = 4
    num_staging_threads = 4
    gpu_num_bundles = 2
    gpu_batch_size = 200000
    server_target_qps = 547000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50_HighAccuracy(L40X3_CUSTOM_X620_G50):
    server_target_qps = 548100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    num_staging_batches = 10
    num_staging_threads = 10
    gpu_num_bundles = 2
    gpu_batch_size = 100000
    server_target_qps = 727000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40_HighAccuracy(L40X8_CUSTOM_X640_G40):
    server_target_qps = 727000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    num_staging_batches = 10
    num_staging_threads = 10
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    gpu_batch_size = 150000
    server_target_qps = 1100000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50_HighAccuracy(L40X8_CUSTOM_X640_G50):
    server_target_qps = 1110000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 3
    gpu_batch_size = 272000
    server_target_qps = 433000
    use_jemalloc = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40_HighAccuracy(A30X3_CUSTOM_X620_G40):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 3
    gpu_batch_size = 272000
    server_target_qps = 565000
    use_jemalloc = False
    numa_config = "0-1:0-43,88-131&2-3:44-87,132-175"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50_HighAccuracy(A30X4_CUSTOM_X620_G50):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    num_staging_batches = 4
    num_staging_threads = 4
    gpu_num_bundles = 3
    gpu_batch_size = 272000
    server_target_qps = 775000
    use_jemalloc = False
    numa_config = "0-3:0-17,36-53&4-7:18-35,54-71"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40_HighAccuracy(A30X8_CUSTOM_X640_G40):
    server_target_qps = 750000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 3
    gpu_batch_size = 272000
    server_target_qps = 873000
    use_jemalloc = False
    numa_config = "0-3:0-55&4-7:56-111"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50_HighAccuracy(A30X8_CUSTOM_X640_G50):
     server_target_qps = 861000
