# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Any, Callable, Dict, Final, Optional, List, Union

import os
import sys
import importlib.util

from code.common import logging
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


# Dynamically build Enum for known systems
_system_confs = dict()


def add_systems(name_format_string: str,
                id_format_string: str,
                cpu: KnownCPU,
                accelerator: KnownGPU,
                accelerator_counts: List[int],
                mem_requirement: Memory,
                target_dict: Dict[str, SystemConfiguration] = _system_confs):
    """Adds a SystemConfiguration to a dictionary.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
        target_dict (Dict[str, SystemConfiguration]): The dictionary to add the SystemConfiguration to.
                                                      (Default: _system_confs)
    """
    for count in accelerator_counts:
        target_dict[name_format_string.format(count)] = SystemConfiguration(
            CPUConfiguration({cpu: MATCH_ANY}),
            min_memory_requirement(mem_requirement),
            AcceleratorConfiguration({accelerator: count}),
            numa_conf=MATCH_ANY,
            system_id=id_format_string.format(count))


# ADA Systems
add_systems("L40x{}", "L40x{}", KnownCPU.AMD_EPYC_7313P.value,
            KnownGPU.L40.value, [1, 2, 4, 8], Memory(100, ByteSuffix.GiB))
add_systems("L4x{}", "L4x{}", KnownCPU.AMD_EPYC_7313P.value,
            KnownGPU.L4.value, [1, 2, 4, 8], Memory(100, ByteSuffix.GiB))

# Hopper systems
add_systems("H100_SXM_80GBx{}", "DGX-H100_H100-SXM-80GBx{}", KnownCPU.Intel_Xeon_Platinum_8480C.value,
            KnownGPU.H100_SXM_80GB.value, [1, 2, 4, 8], Memory(30, ByteSuffix.GiB))
add_systems("H100_SXM_80GB_02x{}", "H100-SXM-80GB-02x{}", KnownCPU.Intel_Xeon_Silver_4314.value,
            KnownGPU.H100_SXM_80GB.value, [1], Memory(100, ByteSuffix.GiB))
add_systems("H100_PCIe_80GBx{}", "H100-PCIe-80GBx{}", KnownCPU.AMD_EPYC_7742.value,
            KnownGPU.H100_PCIe_80GB.value, [1, 2, 4, 8], Memory(100, ByteSuffix.GiB))
add_systems("H100_PCIe_80GB_ARMx{}", "H100-PCIe-80GB_aarch64x{}", KnownCPU.Neoverse_N1_ARM.value,
            KnownGPU.H100_PCIe_80GB.value, [1, 2, 4, 8], Memory(100, ByteSuffix.GiB))

# A100_PCIe_40GB and 80GB based systems:
add_systems("A100_PCIe_40GB_ARMx{}", "A100-PCIe_aarch64x{}", KnownCPU.Neoverse_N1_ARM.value,
            KnownGPU.A100_PCIe_40GB.value, [1, 2, 4], Memory(30, ByteSuffix.GiB))
add_systems("A100_PCIe_80GBx{}", "A100-PCIe-80GBx{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_PCIe_80GB.value, [1, 8], Memory(30, ByteSuffix.GiB))
add_systems("A100_PCIe_80GB_ARMx{}", "A100-PCIe-80GB_aarch64x{}", KnownCPU.Neoverse_N1_ARM.value,
            KnownGPU.A100_PCIe_80GB.value, [1, 2, 4], Memory(30, ByteSuffix.GiB))

add_systems("A100_PCIe_40GB_MIG_{}x1g_5gb", "A100-PCIe-MIG_{}x1g.5gb", KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_PCIe_40GB_1GPC.value, [1], Memory(1, ByteSuffix.TiB))
add_systems("A100_PCIe_40GB_ARM_MIG_{}x1g_5gb", "A100-PCIe_aarch64-MIG_{}x1g.5gb", KnownCPU.Neoverse_N1_ARM.value,
            KnownMIG.A100_PCIe_40GB_1GPC.value, [1], Memory(1, ByteSuffix.TiB))
# FIXME: not sure what's causing ipp1-1468's host memory to be less than 1 TiB, so using 1TB for now
add_systems("A100_PCIe_80GB_MIG_{}x1g_10gb", "A100-PCIe-80GB-MIG_{}x1g.10gb", KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_PCIe_80GB_1GPC.value, [1], Memory(1, ByteSuffix.TB))
add_systems("A100_PCIe_80GB_ARM_MIG_{}x1g_10gb", "A100-PCIe-80GB_aarch64-MIG_{}x1g.10gb",
            KnownCPU.Neoverse_N1_ARM.value, KnownMIG.A100_PCIe_80GB_1GPC.value, [1], Memory(1, ByteSuffix.TiB))

# A100_SXM4_40GB and SXM_80GB based systems:
add_systems("A100_SXM4_40GBx{}", "DGX-A100_A100-SXM4-40GBx{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_SXM4_40GB.value, [1, 8], Memory(30, ByteSuffix.GiB))
add_systems("A100_SXM_80GBx{}", "DGX-A100_A100-SXM-80GBx{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_SXM_80GB.value, [1, 8], Memory(30, ByteSuffix.GiB))
add_systems("A100_SXM_80GB_ARMx{}", "A100-SXM-80GB_aarch64x{}", KnownCPU.Neoverse_N1_ARM.value,
            KnownGPU.A100_SXM_80GB.value, [1, 8], Memory(1, ByteSuffix.TB))

add_systems("A100_SXM4_40GB_MIG_{}x1g_5gb", "DGX-A100_A100-SXM4-40GB-MIG_{}x1g.5gb", KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_SXM4_40GB_1GPC.value, [1], Memory(1, ByteSuffix.TiB))
add_systems("A100_SXM_80GB_MIG_{}x1g_10gb", "DGX-A100_A100-SXM-80GB-MIG_{}x1g.10gb", KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_SXM_80GB_1GPC.value, [1], Memory(1, ByteSuffix.TiB))
add_systems("A100_SXM_80GB_ARM_MIG_{}x1g_10gb", "A100-SXM-80GB_aarch64_MIG_{}x1g.10gb",
            KnownCPU.Neoverse_N1_ARM.value, KnownMIG.A100_SXM_80GB_1GPC.value, [1],
            Memory(1, ByteSuffix.TB))

# Other Ampere based systems
add_systems("GeForceRTX_3080x{}", "GeForceRTX3080x{}", MATCH_ANY, KnownGPU.GeForceRTX_3080.value,
            [1], Memory(30, ByteSuffix.GiB))
add_systems("GeForceRTX_3090x{}", "GeForceRTX3090x{}", MATCH_ANY, KnownGPU.GeForceRTX_3090.value,
            [1], Memory(30, ByteSuffix.GiB))
add_systems("A10x{}", "A10x{}", KnownCPU.AMD_EPYC_7742.value, KnownGPU.A10.value,
            [1, 8], Memory(0.9, ByteSuffix.TiB))
add_systems("A30x{}", "A30x{}", KnownCPU.AMD_EPYC_7742.value, KnownGPU.A30.value,
            [1, 8], Memory(0.5, ByteSuffix.TiB))
add_systems("A2x{}", "A2x{}", MATCH_ANY, KnownGPU.A2.value, [1, 2], Memory(100, ByteSuffix.GiB))
add_systems("A30_MIG_{}x1g_6gb", "A30-MIG_{}x1g.6gb", KnownCPU.AMD_EPYC_7742.value, KnownMIG.A30_1GPC.value,
            [1], Memory(0.5, ByteSuffix.TiB))

# Turing based systems
add_systems("T4x{}", "T4x{}", CPU(MATCH_ANY, CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY), KnownGPU.T4.value,
            [1, 8, 20], Memory(32, ByteSuffix.GiB))

# Embedded systems
add_systems("Orin", "Orin", KnownCPU.ARM_V8_Generic.value, KnownGPU.Orin.value,
            [1], Memory(7, ByteSuffix.GiB))
add_systems("Orin_NX", "Orin_NX", KnownCPU.ARM_V8_Generic.value, KnownGPU.Orin_NX.value,
            [1], Memory(7, ByteSuffix.GiB))

# Intel CPU-based systems (no discrete accelerator)
_system_confs["Triton_CPU_2S_8380"] = SystemConfiguration(
    CPUConfiguration({KnownCPU.Intel_Xeon_Platinum_8380.value: 2}),
    min_memory_requirement(Memory(500, ByteSuffix.GB)),
    AcceleratorConfiguration(dict()),
    numa_conf=None,
    system_id="Triton_CPU_2S_8380x1")

# Inferentia-based system
_system_confs["Triton_Inferentia_INF1_XLARGE"] = SystemConfiguration(
    MATCH_ANY,
    MATCH_ANY,
    AcceleratorConfiguration({KnownInferentia.Inferentia_INF1_XLARGE.value: 1}),
    numa_conf=None,
    system_id="Triton_Inferentia_INF1_XLARGEx1"
)
_system_confs["Triton_Inferentia_INF1_2XLARGE"] = SystemConfiguration(
    MATCH_ANY,
    MATCH_ANY,
    AcceleratorConfiguration({KnownInferentia.Inferentia_INF1_2XLARGE.value: 1}),
    numa_conf=None,
    system_id="Triton_Inferentia_INF1_2XLARGEx1"
)
_system_confs["Triton_Inferentia_INF1_6XLARGE"] = SystemConfiguration(
    MATCH_ANY,
    MATCH_ANY,
    AcceleratorConfiguration({KnownInferentia.Inferentia_INF1_6XLARGE.value: 1}),
    numa_conf=None,
    system_id="Triton_Inferentia_INF1_6XLARGEx1"
)
_system_confs["Triton_Inferentia_INF1_24XLARGE"] = SystemConfiguration(
    MATCH_ANY,
    MATCH_ANY,
    AcceleratorConfiguration({KnownInferentia.Inferentia_INF1_24XLARGE.value: 1}),
    numa_conf=None,
    system_id="Triton_Inferentia_INF1_24XLARGEx1"
)

"""
Handle custom systems to better support partner drops. The custom_list by default is located at
code.common.systems.custom_list, but for testing and developer use, you can set the environment variable
MLPINF_CUSTOM_DEFINITION_PATH to look in a different directory.
This expects the directory at this path to contain:
    - custom_systems/custom_list.py
    - custom_configs/<benchmark>/<scenario>/custom.py
"""
_custom_definition_path = os.environ.get("MLPINF_CUSTOM_DEFINITION_PATH", None)
if _custom_definition_path is None:
    if importlib.util.find_spec("code.common.systems.custom_list") is not None:
        from code.common.systems.custom_list import custom_systems
        _system_confs.update(custom_systems)
elif not os.path.exists(_custom_definition_path):
    raise FileNotFoundError(f"MLPINF_CUSTOM_DEFINITION_PATH {_custom_definition_path} does not exist.")
else:
    from code.common.fix_sys_path import ScopedRestrictedImport
    with ScopedRestrictedImport(restricted_path=[_custom_definition_path] + sys.path) as sri:
        if importlib.util.find_spec("custom_systems.custom_list") is not None:
            from custom_systems.custom_list import custom_systems
            _system_confs.update(custom_systems)
KnownSystem = MatchableEnum("KnownSystem", _system_confs)


_deprecated_systems = dict()
add_systems("A100_SXM_80GB_ROx{}", "DGX-Station-A100_A100-SXM-80GBx{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_SXM_80GB_RO.value, [1, 4], Memory(30, ByteSuffix.GiB),
            target_dict=_deprecated_systems)
add_systems("A100_SXM_80GB_RO_MIG_{}x1g_10gb", "DGX-Station-A100_A100-SXM-80GB-MIG_{}x1g.10gb",
            KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_SXM_80GB_RO_1GPC.value, [7, 28], Memory(1, ByteSuffix.TiB),
            target_dict=_deprecated_systems)
add_systems("A100_SXM4_40GB_MIG_{}x1g_5gb", "DGX-A100_A100-SXM4-40GB-MIG_{}x1g.5gb",
            KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_SXM4_40GB_1GPC.value, [56], Memory(1, ByteSuffix.TiB),
            target_dict=_deprecated_systems)
add_systems("A100_SXM_80GB_MIG_{}x1g_10gb", "DGX-A100_A100-SXM-80GB-MIG_{}x1g.10gb",
            KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_SXM_80GB_1GPC.value, [7, 56], Memory(1, ByteSuffix.TiB),
            target_dict=_deprecated_systems)
add_systems("A100_SXM_80GB_ARM_MIG_{}x1g_10gb", "A100-SXM-80GB_aarch64_MIG_{}x1g.10gb",
            KnownCPU.Neoverse_N1_ARM.value,
            KnownMIG.A100_SXM_80GB_1GPC.value, [7, 56], Memory(1, ByteSuffix.TB),
            target_dict=_deprecated_systems)
add_systems("A100_PCIe_80GB_MIG_{}x1g_10gb", "A100-PCIe-80GB-MIG_{}x1g.10gb",
            KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A100_PCIe_80GB_1GPC.value, [7, 56], Memory(1, ByteSuffix.TB),
            target_dict=_deprecated_systems)
add_systems("A30_MIG_{}x1g_6gb", "A30-MIG_{}x1g.6gb",
            KnownCPU.AMD_EPYC_7742.value,
            KnownMIG.A30_1GPC.value, [32], Memory(0.5, ByteSuffix.TiB),
            target_dict=_deprecated_systems)
add_systems("AGX_Xavier", "AGX_Xavier",
            KnownCPU.NVIDIA_Carmel_ARM_V8.value,
            KnownGPU.AGX_Xavier.value, [1], Memory(30, ByteSuffix.GiB),
            target_dict=_deprecated_systems)
add_systems("Xavier_NX", "Xavier_NX",
            KnownCPU.NVIDIA_Carmel_ARM_V8.value,
            KnownGPU.Xavier_NX.value, [1], Memory(7, ByteSuffix.GiB),
            target_dict=_deprecated_systems)
add_systems("DRIVE_A100_PCIE", "Drive-A100-PCIex1",
            KnownCPU.Intel_Xeon_Silver_4314.value, KnownGPU.DRIVE_A100_PCIE.value,
            [1], Memory(quantity=30.0, byte_suffix=ByteSuffix.GB),
            target_dict=_deprecated_systems)
add_systems("A100_PCIe_40GBx{}", "A100-PCIex{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_PCIe_40GB.value, [1, 8], Memory(30, ByteSuffix.GiB),
            target_dict=_deprecated_systems)
_deprecated_systems["Triton_CPU_2S_6258R"] = SystemConfiguration(
    CPUConfiguration({KnownCPU.Intel_Xeon_Gold_6258R.value: 2}),
    min_memory_requirement(Memory(500, ByteSuffix.GB)),
    AcceleratorConfiguration(dict()),
    numa_conf=None,
    system_id="Triton_CPU_2S_6258Rx1")
_deprecated_systems["Triton_CPU_4S_8380H"] = SystemConfiguration(
    CPUConfiguration({KnownCPU.Intel_Xeon_Platinum_8380H.value: 4}),
    min_memory_requirement(Memory(500, ByteSuffix.GB)),
    AcceleratorConfiguration(dict()),
    numa_conf=None,
    system_id="Triton_CPU_4S_8380Hx1")
DeprecatedSystem = MatchableEnum("DeprecatedSystem", _deprecated_systems)


def match_known_system(sys_conf):
    """Matches a SystemConfiguration with KnownSystems and returns the enum member sys_conf matched with. Also sets the
    system_id field of sys_conf to the system_id of the enum member.

    Returns None if no match was found."""
    match = KnownSystem.get_first_match(sys_conf)
    if match is None:
        # Check if the system is a deprecated system. If so, log a warning. It is up to the caller to throw the error
        # (main.py will throw an error, since we still return None for MATCHED_SYSTEM).
        match = DeprecatedSystem.get_first_match(sys_conf)
        if match is not None:
            logging.warn(f"Detected system is a deprecated system, and is no longer supported: {match}")
        return None
    sys_conf.set_id(match.value.system_id)
    return match


DETECTED_SYSTEM = SystemConfiguration.detect()
"""SystemConfiguration: The detected SystemConfiguration of the current system at runtime"""

MATCHED_SYSTEM = match_known_system(DETECTED_SYSTEM)
"""KnownSystem: The KnownSystem Enum member that DETECTED_SYSTEM matched with. Used to select BenchmarkConfigurations."""


def _default_on_matched(f):
    """Decorator that takes in a classmethod Callable that takes in a System as a parameter, and returns an equivalent
    Callable where that single parameter now has a default value of MATCHED_SYSTEM.
    """
    def _inner(cls, v=MATCHED_SYSTEM):
        # Make sure that v is a KnownSystem
        if v is None:
            return False
        elif type(v) is SystemConfiguration:
            v = match_known_system(v)
        elif type(v) is not KnownSystem:
            raise TypeError(f"Cannot classify system of object type {type(v)}")
        return f(cls, v)
    return _inner


class SystemClassifications:
    """Defines classmethods with the signature Callable[KnownSystem, bool] that returns True or False, representing
    whether or not a KnownSystem satisfies a certain condition.

    This is the equivalent of code.common.constants.AdHocSystemClassification for the old System description from MLPerf
    Inference v1.1."""

    @classmethod
    @_default_on_matched
    def is_hopper(cls, sys):
        return sys.value.get_compute_sm() in [90]

    @classmethod
    @_default_on_matched
    def is_ampere(cls, sys):
        return sys.value.get_compute_sm() in [80, 86, 87, 89]

    @classmethod
    @_default_on_matched
    def is_turing(cls, sys):
        return sys.value.get_compute_sm() == 75

    @classmethod
    @_default_on_matched
    def is_orin(cls, sys):
        return sys in [KnownSystem.Orin, KnownSystem.Orin_NX]

    @classmethod
    @_default_on_matched
    def is_aarch64(cls, sys):
        return sys.value.host_cpu_conf.get_architecture() == CPUArchitecture.aarch64

    @classmethod
    @_default_on_matched
    def is_soc(cls, sys):
        return SystemClassifications.is_orin(sys)

    @classmethod
    @_default_on_matched
    def start_from_device_enabled(cls, sys):
        return sys in [
            KnownSystem.H100_SXM_80GBx1,
            KnownSystem.H100_SXM_80GBx8,
            KnownSystem.H100_SXM_80GB_02x1,
            KnownSystem.A100_SXM_80GBx1,
            KnownSystem.A100_SXM_80GBx8,
            KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb,
            KnownSystem.A100_SXM4_40GBx1,
            KnownSystem.A100_SXM4_40GBx8,
            KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb,
            KnownSystem.A100_SXM_80GB_ARMx1,
            KnownSystem.A100_SXM_80GB_ARMx8,
            KnownSystem.A30x3_Custom_X620_G40,
            KnownSystem.A30x4_Custom_X620_G50,
            KnownSystem.A30x8_Custom_X640_G40,
            KnownSystem.A30x8_Custom_X640_G50,
        ]

    @classmethod
    @_default_on_matched
    def end_on_device_enabled(cls, sys):
        return sys in [
            KnownSystem.A100_SXM_80GBx1,
            KnownSystem.A100_SXM_80GBx8,
            KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb,
            KnownSystem.A100_SXM4_40GBx1,
            KnownSystem.A100_SXM4_40GBx8,
            KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb,
            KnownSystem.A100_SXM_80GB_ARMx1,
            KnownSystem.A100_SXM_80GB_ARMx8,
        ]

    @classmethod
    @_default_on_matched
    def intel_openvino(cls, sys):
        return len(sys.value.accelerator_conf.get_accelerators()) == 0 and \
            sys.value.host_cpu_conf.get_architecture() == CPUArchitecture.x86_64

    @classmethod
    @_default_on_matched
    def inferentia_based(cls, sys):
        return sys.value.accelerator_conf.num_inferentia() > 0

    @classmethod
    @_default_on_matched
    def gpu_based(cls, sys):
        return sys in [
            known
            for known in KnownSystem
            if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 0
        ]

    @classmethod
    @_default_on_matched
    def multi_gpu(cls, sys):
        return sys in [
            known
            for known in KnownSystem
            if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 1
        ]
