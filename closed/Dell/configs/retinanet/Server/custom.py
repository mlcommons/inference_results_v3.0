# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 10
    deque_timeout_usec: int = 37985
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 3
    server_target_qps: int = 2145
    use_deque_limit: bool = True
    workspace_size: int = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx2
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 49042
    gpu_batch_size = 10
    gpu_inference_streams = 2
    server_target_qps = 3708//2
    workspace_size = 60000000000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 49042
    gpu_batch_size = 10
    gpu_inference_streams = 2
    server_target_qps = 3708
    workspace_size = 60000000000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM4_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM4_80GBx4
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 30734
    gpu_batch_size = 16
    gpu_inference_streams = 5
    server_target_qps = 2850
    workspace_size = 70000000000
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_A100_SXM4_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_A100_SXM4_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 5720
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8

    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_inference_streams = 2
    workspace_size = 60000000000
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 1440 * 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_A30X1(ServerGPUBaseConfig):
    system = KnownSystem.XR4520c_A30x1
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 12
    gpu_inference_streams = 1
    server_target_qps = 280
    workspace_size = 70000000000
