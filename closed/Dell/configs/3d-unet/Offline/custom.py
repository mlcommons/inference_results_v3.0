# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 1
    offline_expected_qps: int = 14


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx2
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 4.8*2
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2_HighAccuracy(R750xa_H100_PCIe_80GBx2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 4.8*4
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_batch_size = 2
    offline_expected_qps = 350
    start_from_device = True
    end_on_device = True
    workspace_size: 7000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4_HighAccuracy(XE8545_A100_SXM4_80GBX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    offline_expected_qps = 40
    use_cuda_thread_per_device = True
    gpu_batch_size = 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8_HighAccuracy(XE9680_A100_SXM4_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    gpu_batch_size = 8
    offline_expected_qps = 60
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_batch_size = 1
    offline_expected_qps = 2.5


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_A30X1_HighAccuracy(XR4520C_A30X1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ_HighAccuracy(XR5610_L4x1_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR7620_L4x1_HighAccuracy(XR7620_L4x1):
    pass

