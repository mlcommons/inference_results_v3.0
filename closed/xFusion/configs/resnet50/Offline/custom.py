# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x8
    run_infer_on_copy_streams = True
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    offline_expected_qps=150000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_A30X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_A30x10
    run_infer_on_copy_streams = True
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    offline_expected_qps=200000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x8
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    offline_expected_qps = 200000
    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class G5500V7_L40X10(OfflineGPUBaseConfig):
    system = KnownSystem.G5500V7_L40x10

    gpu_batch_size = 2048
    gpu_copy_streams = 2 
    offline_expected_qps = 250000
    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4X6_2288H_V7(OfflineGPUBaseConfig):
    system = KnownSystem.L4x6_2288H_V7
    gpu_batch_size = 128    
    gpu_inference_streams = 1
    gpu_copy_streams = 2    
    offline_expected_qps = 78000