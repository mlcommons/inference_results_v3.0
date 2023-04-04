# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 64
    active_sms: int = 60
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    graphs_max_seqlen: int = 200
    server_num_issue_query_threads: int = 1
    server_target_qps: int = 11700
    soft_drop: float = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    precision = "fp16"
    server_target_qps = R750XA_A100_PCIE_80GBX4.server_target_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 15500
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 15400
    soft_drop = 1.0
    gpu_batch_size = 94


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_T4X1(ServerGPUBaseConfig):
    system = KnownSystem.XE2420_T4x1
    gpu_batch_size = 4
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 290
    soft_drop = 0.993
    use_small_tile_gemm_plugin = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 13600
    soft_drop = 1.0
    gpu_copy_streams = 4
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4_HighAccuracy(XE8545_A100_SXM4_80GBX4):
    precision = "fp16"
    server_target_qps = 7000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 27500
    soft_drop = 1.0
    gpu_copy_streams = 4
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8_HighAccuracy(XE9680_A100_SXM4_80GBX8):
    precision = "fp16"
    server_target_qps = 14000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 128
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 55550
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 128
    server_target_qps = 49500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_Triton(XE9680_H100_SXM_80GBX8):
    use_triton = True
    server_target_qps = 39800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(ServerGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1500
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_A30X1_HighAccuracy(XR4520C_A30X1):
    precision = "fp16"
    server_target_qps = 680
