# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIE_80GBX8_SUPERMICRO(H100_PCIe_80GBx8):
    system = KnownSystem.H100_PCIe_80GBx8_Supermicro
    numa_config = "0,1,2,3:0-15,32-47&4,5,6,7:16-31,48-63" 
    server_target_qps = 345000
    gpu_batch_size = 128
    gpu_inference_streams = 4
    gpu_copy_streams = 2
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    # gpu_batch_size: int = 0
    # input_dtype: str = ''
    # input_format: str = ''
    # map_path: str = ''
    # precision: str = ''
    # tensor_path: str = ''

    # # Optional fields:
    # active_sms: int = 0
    # assume_contiguous: bool = False
    # buffer_manager_thread_count: int = 0
    # cache_file: str = ''
    # complete_threads: int = 0
    # deque_timeout_usec: int = 0
    # disable_beta1_smallk: bool = False
    # gpu_copy_streams: int = 0
    # gpu_inference_streams: int = 0
    # instance_group_count: int = 0
    # model_path: str = ''
    # numa_config: bool = False
    # performance_sample_count_override: int = 0
    # preferred_batch_size: str = ''
    # request_timeout_usec: int = 0
    # run_infer_on_copy_streams: bool = False
    # schedule_rng_seed: int = 0
    # server_num_issue_query_threads: int = 0
    # server_target_latency_ns: int = 0
    # server_target_latency_percentile: float = 0.0
    # server_target_qps: int = 0
    # server_target_qps_adj_factor: float = 0.0
    # use_batcher_thread_per_device: bool = False
    # use_cuda_thread_per_device: bool = False
    # use_deque_limit: bool = False
    # use_graphs: bool = False
    # use_jemalloc: bool = False
    # use_same_context: bool = False
    # use_spin_wait: bool = False
    # warmup_duration: float = 0.0
    # workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIE_80GBX8_SUPERMICRO_Triton(H100_PCIe_80GBx8_Triton):
    system = KnownSystem.H100_PCIe_80GBx8_Supermicro
    numa_config = "2:0-3,32-35&3:4-7,36-39&1:8-11,40-43&0:12-15,44-47&6:16-19,48-51&7:20-23,52-55&5:24-27,56-59&4:28-31,60-63" 
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    # gpu_batch_size: int = 0
    # input_dtype: str = ''
    # input_format: str = ''
    # map_path: str = ''
    # precision: str = ''
    # tensor_path: str = ''

    # # Optional fields:
    # active_sms: int = 0
    # assume_contiguous: bool = False
    # batch_triton_requests: bool = False
    # buffer_manager_thread_count: int = 0
    # cache_file: str = ''
    # complete_threads: int = 0
    # deque_timeout_usec: int = 0
    # disable_beta1_smallk: bool = False
    # gather_kernel_buffer_threshold: int = 0
    # gpu_copy_streams: int = 0
    # gpu_inference_streams: int = 0
    # instance_group_count: int = 0
    # max_queue_delay_usec: int = 0
    # model_path: str = ''
    # num_concurrent_batchers: int = 0
    # num_concurrent_issuers: int = 0
    # numa_config: bool = False
    # output_pinned_memory: bool = False
    # performance_sample_count_override: int = 0
    # preferred_batch_size: str = ''
    # request_timeout_usec: int = 0
    # run_infer_on_copy_streams: bool = False
    # schedule_rng_seed: int = 0
    # server_num_issue_query_threads: int = 0
    # server_target_latency_ns: int = 0
    # server_target_latency_percentile: float = 0.0
    # server_target_qps: int = 0
    # server_target_qps_adj_factor: float = 0.0
    # use_batcher_thread_per_device: bool = False
    # use_concurrent_harness: bool = False
    # use_cuda_thread_per_device: bool = False
    # use_deque_limit: bool = False
    # use_graphs: bool = False
    # use_jemalloc: bool = False
    # use_same_context: bool = False
    # use_spin_wait: bool = False
    # warmup_duration: float = 0.0
    # workspace_size: int = 0


