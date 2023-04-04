# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    use_jemalloc = True
    deque_timeout_usec = 100
    gpu_batch_size = 319687
    gpu_num_bundles = 3
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 1702350
    start_from_device = True
    compress_categorical_inputs = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved_top_mlp = True
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy(XE2420_T4X1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    num_staging_batches = 6
    num_staging_threads = 4
    gpu_num_bundles = 3
    gpu_batch_size = 263000
    server_target_qps = 2580000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8_HighAccuracy(XE9680_A100_SXM4_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(ServerGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 150000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_A30X1_HighAccuracy(XR4520C_A30X1):
    pass
