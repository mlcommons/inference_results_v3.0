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
from dataclasses import dataclass
from enum import unique
from typing import Any, Callable, Final, Optional, List, Union

import re
import os

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.inferentia import Inferentia
from code.common.systems.memory import MemoryConfiguration


__doc__ = """Contains all known Hardware objects, which are components used to easily create new SystemConfiguration
objects. These enums are stored in this separate file to avoid a cyclical dependency between system_list.py and
custom_list.py, which both require these Enums."""


def mem_to_bytes(m): return m.to_bytes()
def match_float_approximate(m): return MatchFloatApproximate(m, mem_to_bytes, rel_tol=0.05)

"""Helper functions to create 'MatchFloatApproximate' representations of Memory"""


def mem_conf_to_bytes(mc):
    """Note: You cannot use a lambda function variation of this method (i.e. lambda mc: mem_to_bytes(...)). Doing so
    will make min_memory_requirement un-pickle-able, which is required for Python multi-threading to work.

    See: https://stackoverflow.com/a/52283968"""
    return mem_to_bytes(mc.host_memory_capacity)


def mem_conf_to_min_bytes(mc):
    """memory bytes considering comparison tolerance, to be used as a lower bound threshold"""
    return mem_conf_to_bytes(mc) * (1 - mc.comparison_tolerance)


def min_memory_requirement(m): return MatchNumericThreshold(MemoryConfiguration(m),
                                                            mem_conf_to_min_bytes,
                                                            min_threshold=True)

"""Helper function to create 'MatchNumericThreshold' representations of Memory (i.e. min. memory requirement)"""


def get_full_pci_id(s): return f"0x{s}10DE"


def pci_id_match_list(L):
    """Returns an AliasedName representing all the PCI IDs in L, where L is a list of shortened PCI IDs."""
    assert len(L) > 0
    full_pci_ids = list(map(get_full_pci_id, L))
    if len(full_pci_ids) == 1:
        return AliasedName(full_pci_ids[0])
    else:
        return AliasedName(full_pci_ids[0], tuple(full_pci_ids[1:]))


@unique
class KnownGPU(MatchableEnum):
    L4 = GPU(name="NVIDIA Graphics Device",
             accelerator_type=AcceleratorType.Discrete,
             vram=match_float_approximate(Memory(23, ByteSuffix.GiB)),
             max_power_limit=75.0,
             pci_id=pci_id_match_list(("27B8",)),
             compute_sm=89)
    L40 = GPU(name="NVIDIA Graphics Device",
              accelerator_type=AcceleratorType.Discrete,
              vram=match_float_approximate(Memory(45, ByteSuffix.GiB)),
              max_power_limit=300.0,
              pci_id=pci_id_match_list(("26B5",)),
              compute_sm=89)
    H100_SXM_80GB = GPU(name="NVIDIA Graphics Device",
                        accelerator_type=AcceleratorType.Discrete,
                        vram=match_float_approximate(Memory(80, ByteSuffix.GiB)),
                        max_power_limit=700.0,
                        pci_id=pci_id_match_list(("2330", "233F")),
                        compute_sm=90)
    H100_PCIe_80GB = GPU(name="NVIDIA Graphics Device",
                         accelerator_type=AcceleratorType.Discrete,
                         vram=match_float_approximate(Memory(80, ByteSuffix.GiB)),
                         max_power_limit=350.0,
                         pci_id=pci_id_match_list(("2331",)),
                         compute_sm=90)
    A100_PCIe_40GB = GPU(AliasedName("A100-PCIe-40GB", ("A100-PCIe", "NVIDIA A100-PCIe", "NVIDIA A100-PCIe-40GB", "NVIDIA A100 40GB PCIe")),
                         AcceleratorType.Discrete,
                         match_float_approximate(Memory(40, ByteSuffix.GiB)),
                         250.0,
                         pci_id_match_list(("20F1", "20BF")),
                         80)
    A100_PCIe_80GB = GPU(AliasedName("A100-PCIe-80GB", ("NVIDIA A100-PCIe-80GB", "NVIDIA A100 80GB PCIe")),
                         AcceleratorType.Discrete,
                         match_float_approximate(Memory(80, ByteSuffix.GiB)),
                         300.0,
                         pci_id_match_list(("20B5",)),
                         80)
    A100_SXM4_40GB = GPU(AliasedName("A100-SXM4-40GB", ("NVIDIA A100-SXM4-40GB",)),
                         AcceleratorType.Discrete,
                         match_float_approximate(Memory(40, ByteSuffix.GiB)),
                         400.0,
                         pci_id_match_list(("20B0",)),
                         80)
    A100_SXM_80GB = GPU(AliasedName("A100-SXM-80GB", ("NVIDIA A100-SXM-80GB", "NVIDIA A100-SXM4-80GB")),
                        AcceleratorType.Discrete,
                        match_float_approximate(Memory(80, ByteSuffix.GiB)),
                        400.0,
                        pci_id_match_list(("20B2",)),
                        80)
    A100_SXM_80GB_RO = GPU(AliasedName("A100-SXM-80GB", ("NVIDIA A100-SXM-80GB", "NVIDIA A100-SXM4-80GB")),
                           AcceleratorType.Discrete,
                           match_float_approximate(Memory(80, ByteSuffix.GiB)),
                           275.0,
                           pci_id_match_list(("20B2",)),
                           80)
    GeForceRTX_3080 = GPU(AliasedName("GeForce RTX 3080", ("NVIDIA GeForce RTX 3080",)),
                          AcceleratorType.Discrete,
                          match_float_approximate(Memory(10, ByteSuffix.GiB)),
                          320.0,
                          pci_id_match_list(("2206",)),
                          86)
    GeForceRTX_3090 = GPU(AliasedName("GeForce RTX 3090", ("NVIDIA GeForce RTX 3090", "Quadro RTX A6000", "RTX A6000")),
                          AcceleratorType.Discrete,
                          match_float_approximate(Memory(24, ByteSuffix.GiB)),
                          350.0,
                          pci_id_match_list(("2204", "2230")),
                          86)
    A10 = GPU(AliasedName("A10", ("NVIDIA A10",)),
              AcceleratorType.Discrete,
              match_float_approximate(Memory(24, ByteSuffix.GiB)),
              150.0,
              pci_id_match_list(("2236",)),
              86)
    A30 = GPU(AliasedName("A30", ("NVIDIA A30",)),
              AcceleratorType.Discrete,
              match_float_approximate(Memory(24, ByteSuffix.GiB)),
              165.0,
              pci_id_match_list(("20B7",)),
              80)
    A2 = GPU(AliasedName("A2", ("NVIDIA A2",)),
             AcceleratorType.Discrete,
             match_float_approximate(Memory(15, ByteSuffix.GiB)),
             60.0,
             pci_id_match_list(("25B6",)),
             86)
    DRIVE_A100_PCIE = GPU(AliasedName("DRIVE A100 PCIE", ("NVIDIA DRIVE A100 PCIE", "DRIVE A100")),
                          AcceleratorType.Discrete,
                          match_float_approximate(Memory(32, ByteSuffix.GiB)),
                          None,
                          pci_id_match_list(("20bb",)),
                          80)
    T4 = GPU(AliasedName("T4", ("Tesla T4", "T4 32GB", "NVIDIA T4", "NVIDIA Tesla T4")),
             AcceleratorType.Discrete,
             MatchAllowList([
                 match_float_approximate(Memory(16, ByteSuffix.GiB)),
                 match_float_approximate(Memory(32, ByteSuffix.GiB))]),
             70.0,
             pci_id_match_list(("1EB8", "1EB9")),
             75)
    AGX_Xavier = GPU(AliasedName("Jetson-AGX"),
                     AcceleratorType.Integrated,
                     MATCH_ANY,
                     MATCH_ANY,
                     MATCH_ANY,
                     72)
    Xavier_NX = GPU(AliasedName("NVIDIA Jetson Xavier NX Developer Kit", patterns=(re.compile(r".*Xavier NX.*"),)),
                    AcceleratorType.Integrated,
                    MATCH_ANY,
                    MATCH_ANY,
                    MATCH_ANY,
                    72)
    Orin = GPU(AliasedName("Orin", patterns=(re.compile(r"(Jetson AGX Orin)|(NVIDIA Orin Jetson-Small Developer Kit)"),)),
               AcceleratorType.Integrated,
               MATCH_ANY,
               MATCH_ANY,
               MATCH_ANY,
               MATCH_ANY)
    Orin_NX = GPU(AliasedName("Orin_NX", patterns=(re.compile(r"(NVIDIA Orin NX Developer Kit)|(NGX007)"),)),
                  AcceleratorType.Integrated,
                  MATCH_ANY,
                  MATCH_ANY,
                  MATCH_ANY,
                  MATCH_ANY)


@unique
class KnownMIG(MatchableEnum):
    A30_1GPC = MIG(KnownGPU.A30.value.name + " MIG-1g.6gb",
                   AcceleratorType.Discrete,
                   match_float_approximate(Memory(6, ByteSuffix.GiB)),
                   165.0,
                   pci_id_match_list(("20B7",)),
                   80,
                   num_gpcs=1)
    A100_PCIe_40GB_1GPC = MIG(KnownGPU.A100_PCIe_40GB.value.name + " MIG-1g.5gb",
                              AcceleratorType.Discrete,
                              match_float_approximate(Memory(5, ByteSuffix.GiB)),
                              250.0,
                              pci_id_match_list(("20F1", "20BF")),
                              80,
                              num_gpcs=1)
    A100_PCIe_80GB_1GPC = MIG(KnownGPU.A100_PCIe_80GB.value.name + " MIG-1g.10gb",
                              AcceleratorType.Discrete,
                              match_float_approximate(Memory(10, ByteSuffix.GiB)),
                              300.0,
                              pci_id_match_list(("20B5",)),
                              80,
                              num_gpcs=1)
    A100_SXM4_40GB_1GPC = MIG(KnownGPU.A100_SXM4_40GB.value.name + " MIG-1g.5gb",
                              AcceleratorType.Discrete,
                              match_float_approximate(Memory(5, ByteSuffix.GiB)),
                              400.0,
                              pci_id_match_list(("20B0",)),
                              80,
                              num_gpcs=1)
    A100_SXM_80GB_1GPC = MIG(KnownGPU.A100_SXM_80GB.value.name + " MIG-1g.10gb",
                             AcceleratorType.Discrete,
                             match_float_approximate(Memory(10, ByteSuffix.GiB)),
                             400.0,
                             pci_id_match_list(("20B2",)),
                             80,
                             num_gpcs=1)
    A100_SXM_80GB_RO_1GPC = MIG(KnownGPU.A100_SXM_80GB_RO.value.name + " MIG-1g.10gb",
                                AcceleratorType.Discrete,
                                match_float_approximate(Memory(10, ByteSuffix.GiB)),
                                275.0,
                                pci_id_match_list(("20B2",)),
                                80,
                                num_gpcs=1)


@unique
class KnownCPU(MatchableEnum):
    Intel_Xeon_Platinum_8480C = CPU(AliasedName('Intel(R) Xeon(R) Platinum 8480C'),
                                    architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=MatchAllowList([1, 2]))
    AMD_EPYC_7742 = CPU(AliasedName("AMD EPYC 7742 64-Core Processor", ("AMD EPYC 7742",)),
                        CPUArchitecture.x86_64, MatchAllowList([64, 32]), MatchAllowList([1, 2]))
    Intel_Xeon_Platinum_8380 = CPU(AliasedName("Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz"),
                                   CPUArchitecture.x86_64, 40, MatchAllowList([1, 2]))
    Intel_Xeon_Platinum_8380H = CPU(AliasedName("Intel(R) Xeon(R) Platinum 8380H CPU @ 2.90GHz"),
                                    CPUArchitecture.x86_64, 28, MatchAllowList([1, 2]))
    Intel_Xeon_Gold_6258R = CPU(AliasedName("Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz"),
                                CPUArchitecture.x86_64, 28, MatchAllowList([1, 2]))
    x86_64_Generic = CPU(AliasedName("x86-64 Generic", patterns=(re.compile(r"(?:AMD|Intel).*"),)),
                         CPUArchitecture.x86_64, MATCH_ANY, MATCH_ANY)
    Neoverse_N1_ARM = CPU(AliasedName("Neoverse-N1", ("Ampere Altra Q80-30",)), CPUArchitecture.aarch64, 80, 1)
    NVIDIA_Carmel_ARM_V8 = CPU(AliasedName("NVIDIA Carmel (ARMv8.2)", ("ARMv8 Processor rev 0 (v8l)",)),
                               CPUArchitecture.aarch64, 2, 1)
    ARM_V8_Generic = CPU(AliasedName("ARM v8 Generic", patterns=(re.compile(r"ARMv8 Processor rev \d+ \(v8.*\)"),)),
                         CPUArchitecture.aarch64, MATCH_ANY, MATCH_ANY)
    Intel_Xeon_Silver_4314 = CPU(AliasedName("Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"),
                                 CPUArchitecture.x86_64, 16, 2)
    GH100_SXM_CPU = CPU(AliasedName("AMD EPYC 7252 8-Core Processor", ("AMD EPYC 7252",)),
                        architecture=CPUArchitecture.x86_64, core_count=8, threads_per_core=2)
    AMD_EPYC_7313P = CPU(AliasedName("AMD EPYC 7313P 16-Core Processor", ("AMD EPYC 7313P",)),
                         architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2)
    Intel_Xeon_Platinum_8480P = CPU(AliasedName('Intel(R) Xeon(R) Platinum 8480+'),
                                        architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=MatchAllowList([1, 2]))



@unique
class KnownInferentia(MatchableEnum):
    Inferentia_INF1_XLARGE = Inferentia("inf1.xlarge")
    Inferentia_INF1_2XLARGE = Inferentia("inf1.2xlarge")
    Inferentia_INF1_6XLARGE = Inferentia("inf1.6xlarge")
    Inferentia_INF1_24XLARGE = Inferentia("inf1.24xlarge")
