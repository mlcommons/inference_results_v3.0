# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 2048
    offline_expected_qps: int = 55000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx2
    gpu_batch_size = 2048
    use_graphs = True  # MLPINF-1773
    offline_expected_qps = 17000*2
    disable_encoder_plugin = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size = 2048
    use_graphs = True  # MLPINF-1773
    offline_expected_qps = 17000*4
    disable_encoder_plugin = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_inference_streams = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 60000
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT
    workspace_size: 7000000000
    start_from_device = True
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 114400
    num_warmups = 40480
    nobatch_sorting = True
    gpu_batch_size = 2048


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    gpu_batch_size = 2048
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    offline_expected_qps = 180000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR4520C_MAXQ_A2X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_MaxQ_A2x1

    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1150
    power_limit = 45


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_batch_size = 2048
    offline_expected_qps = 8000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024
