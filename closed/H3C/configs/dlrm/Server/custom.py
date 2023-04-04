# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    num_staging_batches =8  
    num_staging_threads =8  
    gpu_num_bundles =  2    
    gpu_batch_size = 131000        
    server_target_qps = 1065000  
    use_jemalloc = False
    numa_config = "0-9:0-95"
    


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy(R5350G6_A30X10):
    server_target_qps = 1065000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(ServerGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 272000            
    server_target_qps = 1249000 
    use_jemalloc = False
    num_staging_batches = 3
    num_staging_threads = 3
    gpu_num_bundles = 3
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_A30X10_HighAccuracy_Triton(R5350G6_A30X10_Triton):
    server_target_qps = 1250000 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 170000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 570000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy(R5300G6_A30X4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    use_triton = True
    server_target_qps =480000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
	
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_A30X4_HighAccuracy_Triton(R5300G6_A30X4_HighAccuracy):
    use_triton = True
