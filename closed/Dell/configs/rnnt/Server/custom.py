# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 2048
    server_target_qps: int = 48500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx2
    gpu_batch_size = 2048
    server_target_qps = 32000
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size = 2048
    server_target_qps = 64100
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_inference_streams = 1
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_num_issue_query_threads = 0
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    server_target_qps = 54000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    server_num_issue_query_threads = 0
    server_target_qps = 110000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    gpu_batch_size = 1728
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773
    server_target_qps = 180000
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    dali_pipeline_depth = 4
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(ServerGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_batch_size = 1650
    server_target_qps = 5100
    gpu_copy_streams = 14
    gpu_inference_streams = 1
    dali_pipeline_depth = 2
