# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_1115CS_TNR(OfflineGPUBaseConfig):
    system = KnownSystem.AS_1115CS_TNR

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: int = 0
    input_dtype: str = ''
    input_format: str = ''
    map_path: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    complete_threads: int = 0
    deque_timeout_usec: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    instance_group_count: int = 0
    model_path: str = ''
    offline_expected_qps: int = 0
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    slice_overlap_patch_kernel_cg_impl: bool = False
    unet3d_sw_gaussian_patch_path: str = ''
    use_batcher_thread_per_device: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_same_context: bool = False
    use_spin_wait: bool = False
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_1115CS_TNR_HighAccuracy(AS_1115CS_TNR):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_1115CS_TNR_Triton(AS_1115CS_TNR):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: int = 0
    input_dtype: str = ''
    input_format: str = ''
    map_path: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    complete_threads: int = 0
    deque_timeout_usec: int = 0
    gather_kernel_buffer_threshold: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    offline_expected_qps: int = 0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    slice_overlap_patch_kernel_cg_impl: bool = False
    unet3d_sw_gaussian_patch_path: str = ''
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    use_cuda_thread_per_device: bool = False
    use_deque_limit: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_same_context: bool = False
    use_spin_wait: bool = False
    warmup_duration: float = 0.0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_1115CS_TNR_HighAccuracy_Triton(AS_1115CS_TNR_HighAccuracy):
    use_triton = True


