# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G50
    gpu_batch_size = 2048
    server_target_qps = 28010


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G50
    gpu_batch_size = 2048
    server_target_qps = 6820


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X8_CUSTOM_X640_G40
    gpu_batch_size = 2048
    server_target_qps = 28500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A40X3_CUSTOM_X620_G40
    gpu_batch_size = 2048
    server_target_qps = 7510


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G40
    gpu_batch_size = 1024
    server_target_qps = 9300 * 3


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X3_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x3_Custom_X620_G50
    gpu_batch_size = 1024
    server_target_qps = 28100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G40
    gpu_inference_streams = 3
    gpu_copy_streams = 3
    gpu_batch_size = 512
    server_target_qps = 9120 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.L40x8_Custom_X640_G50
    gpu_inference_streams = 3
    gpu_copy_streams = 3
    gpu_batch_size = 512
    server_target_qps = 9310 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X3_CUSTOM_X620_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_Custom_X620_G40
    gpu_batch_size = 1000
    server_target_qps = 7600


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X4_CUSTOM_X620_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x4_Custom_X620_G50
    gpu_batch_size = 1000
    server_target_qps = 12000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G40(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G40
    gpu_batch_size = 1792
    server_target_qps = 34500
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30X8_CUSTOM_X640_G50(ServerGPUBaseConfig):
    system = KnownSystem.A30x8_Custom_X640_G50
    gpu_batch_size = 1792
    server_target_qps = 34300
    start_from_device = True
