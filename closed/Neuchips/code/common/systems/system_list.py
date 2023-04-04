# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#
# Modified by NEUCHIPS on 2023
#

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Final, Optional, List, Union

import os
import importlib.util

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


# Dynamically build Enum for known systems
system_confs = dict()


def add_systems(name_format_string: str, id_format_string: str, cpu: KnownCPU, accelerator: KnownGPU,
                accelerator_counts: List[int], mem_requirement: Memory):
    """Adds a SystemConfiguration to system_confs.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
    """
    for count in accelerator_counts:
        system_confs[name_format_string.format(count)] = SystemConfiguration(
            CPUConfiguration({cpu: MATCH_ANY}),
            min_memory_requirement(mem_requirement),
            AcceleratorConfiguration({accelerator: count}),
            numa_conf=MATCH_ANY,
            system_id=id_format_string.format(count))


system_confs["N3000_CPU_2S_Neuchips"] = SystemConfiguration(
    CPUConfiguration({KnownCPU.AMD_EPYC_7742.value: 2}),
    min_memory_requirement(Memory(1000, ByteSuffix.GB)),
    KnownGPU.A100_PCIe_80GB.value, [1, 8],
    #AcceleratorConfiguration(dict()),
    #numa_conf=None,
    system_id="N3000_CPU_2S_Neuchips")


# Handle custom systems to better support partner drops
if importlib.util.find_spec("code.common.systems.custom_list") is not None:
    from code.common.systems.custom_list import custom_systems
    system_confs.update(custom_systems)

KnownSystem = Enum("KnownSystem", system_confs)

def match_known_system(sys_conf):
    """Matches a SystemConfiguration with KnownSystems and returns the enum member sys_conf matched with. Also sets the
    system_id field of sys_conf to the system_id of the enum member.

    Returns None if no match was found."""

    for sys in KnownSystem:
        if sys_conf == sys.value:
            sys_conf.set_id(sys.value.system_id)
            return sys
    return None


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

    #@classmethod
    #@_default_on_matched
    #def is_xavier_nx(cls, sys):
    #    return sys == KnownSystem.Xavier_NX

    #@classmethod
    #@_default_on_matched
    #def is_xavier_agx(cls, sys):
    #    return sys == KnownSystem.AGX_Xavier

    #@classmethod
    #@_default_on_matched
    #def is_xavier(cls, sys):
    #    return SystemClassifications.is_xavier_nx(sys) or SystemClassifications.is_xavier_agx(sys)

    @classmethod
    @_default_on_matched
    def is_ampere(cls, sys):
        return sys.value.get_compute_sm() in [80, 86, 87]

    @classmethod
    @_default_on_matched
    def is_turing(cls, sys):
        return sys.value.get_compute_sm() == 75

    #@classmethod
    #@_default_on_matched
    #def is_orin(cls, sys):
    #    return sys in [KnownSystem.Orin]

    #@classmethod
    #@_default_on_matched
    #def is_soc(cls, sys):
    #    return SystemClassifications.is_xavier(sys) or SystemClassifications.is_orin(sys)

    @classmethod
    @_default_on_matched
    def start_from_device_enabled(cls, sys):
        return False

    @classmethod
    @_default_on_matched
    def end_on_device_enabled(cls, sys):
        return False

    @classmethod
    @_default_on_matched
    def gpu_based(cls, sys):
        return True
        #return sys in [
        #    known
        #    for known in KnownSystem
        #    if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 0
        #]

    @classmethod
    @_default_on_matched
    def multi_gpu(cls, sys):
        return True

        #return sys in [
        #    known
        #    for known in KnownSystem
        #    if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 1
        #]
