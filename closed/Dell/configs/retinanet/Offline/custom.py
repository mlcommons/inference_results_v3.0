# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 16
    gpu_copy_streams: int = 2
    gpu_inference_streams: int = 2
    offline_expected_qps: int = 2500
    run_infer_on_copy_streams: bool = False
    workspace_size: int = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx2
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1000*2
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1000*4
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_batch_size = 16
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    offline_expected_qps = 15000
    run_infer_on_copy_streams = False
    workspace_size = 70000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 6424
    start_from_device = True
    run_infer_on_copy_streams = False
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 13000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520c_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A2x1

    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 90


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 50
    power_limit = 45


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_inference_streams = 1
    gpu_batch_size = 12
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    offline_expected_qps = 396
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 170
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 170
    run_infer_on_copy_streams = False
    workspace_size = 20000000000
