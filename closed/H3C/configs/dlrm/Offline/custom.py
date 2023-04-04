# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1450000
    numa_config = "0-9:0-95"
    


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy(R5350G6_A30X10):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    batch_triton_requests = True
    num_concurrent_batchers = 1
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy_Triton(R5350G6_A30X10_Triton):
    pass

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    offline_expected_qps = 600000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0-1:0-43&2-3:44-87"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy(R5300G6_A30X4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True
    batch_triton_requests = True
    num_concurrent_batchers = 1
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy_Triton(R5300G6_A30X4_Triton):
    pass


