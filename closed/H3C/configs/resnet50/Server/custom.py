# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10(ServerGPUBaseConfig):
    system = KnownSystem.R5350G6_A30x10
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 183000   
    use_cuda_thread_per_device = True
    use_graphs = True
    numa_config ="0-9:0-95"
    



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_A30X10_Triton(R5350G6_A30X10):
    use_triton = True
    gpu_batch_size = 32
    server_target_qps = 160000    
    use_graphs = False
    batch_triton_requests = False
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
 
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_A30x4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 73000
    use_cuda_thread_per_device = True
    use_graphs = True
    numa_config = "0-1:0-43&2-3:44-87"
    


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_A30X4_Triton(R5300G6_A30X4):
    server_target_qps = 60000
    use_graphs = False
    use_triton = True
    gpu_batch_size = 32
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    batch_triton_requests = False
    max_queue_delay_usec = 1000

	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2(ServerGPUBaseConfig):
    system = KnownSystem.R4900G6_L4x2
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 16
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 22400
    use_cuda_thread_per_device = True
    use_graphs = True



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R4900G6_L4x2_Triton(R4900G6_L4x2):
    use_triton = True
    gpu_batch_size = 16
    server_target_qps = 12000