# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 1024
    gemm_plugin_fairshare_cache_size: int = 120
    offline_expected_qps: int = 15000
    use_small_tile_gemm_plugin: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    precision = "fp16"
    offline_expected_qps = R750XA_A100_PCIE_80GBX4.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 5700*4
    workspace_size = 7516192768
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 430


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 18000
    start_from_device = True
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4_HighAccuracy(XE8545_A100_SXM4_80GBX4):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 9000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    offline_expected_qps = 33000
    workspace_size = 7516192768
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8_HighAccuracy(XE9680_A100_SXM4_80GBX8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 16500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400 * 8
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 8200 * 8


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_Triton(XE9680_H100_SXM_80GBX8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520c_A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A2x1

    gpu_batch_size = 256
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 250
    power_limit = 45


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 2084
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_A30X1_HighAccuracy(XR4520C_A30X1):
    precision = "fp16"
    offline_expected_qps = 1042


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 8
    offline_expected_qps = 900
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    offline_expected_qps = 1000
    workspace_size = 7516192768
