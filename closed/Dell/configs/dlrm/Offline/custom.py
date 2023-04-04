# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    complete_threads = 1
    deque_timeout_usec = 1
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 350000
    offline_expected_qps = 470000*4
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved_top_mlp = True
    output_padding_granularity = 32
    use_small_tile_gemm_plugin = False
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 262100
    offline_expected_qps = 34000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_T4X1_HighAccuracy(XE2420_T4X1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    complete_threads = 1
    gpu_batch_size = 334000
    offline_expected_qps = 2640000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8_HighAccuracy(XE9680_A100_SXM4_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    offline_expected_qps = 170000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_A30X1_HighAccuracy(XR4520C_A30X1):
    pass



