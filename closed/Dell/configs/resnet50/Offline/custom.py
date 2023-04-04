# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 57000*4
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    gpu_batch_size = 256
    gpu_copy_streams = 4
    offline_expected_qps = 6100
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_inference_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 3
    run_infer_on_copy_streams = False
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 180000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 374000
    start_from_device = True
    gpu_batch_size = 2048


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
   system = KnownSystem.XE9680_H100_SXM_80GBx8

    gpu_batch_size = 2048
    offline_expected_qps = 90000 * 8
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520c_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A2x1

    gpu_batch_size = 1024
    offline_expected_qps = 4200
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    gpu_batch_size = 1024
    offline_expected_qps = 3100
    run_infer_on_copy_streams = None
    power_limit = 45


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_batch_size = 796
    gpu_copy_streams =  12
    gpu_inference_streams = 8
    run_infer_on_copy_streams = True
    offline_expected_qps = 22000
    complete_threads = 16
    deque_timeout_usec = 7


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 32
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 13000
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 32
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 13000
    use_graphs = True
