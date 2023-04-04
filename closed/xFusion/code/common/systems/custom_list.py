# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


custom_systems = dict()


# Do not manually edit any lines below this. All such lines are generated via scripts/add_custom_system.py

###############################
### START OF CUSTOM SYSTEMS ###
###############################

custom_systems['G5500V7_L40x8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0563916999999998, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA L40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=47.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x26B510DE", compute_sm=89): 8}), numa_conf=None, system_id="G5500V7_L40x8")

custom_systems['G5500V7_L40x10'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0563916880000002, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA L40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=47.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x26B510DE", compute_sm=89): 10}), numa_conf=None, system_id="G5500V7_L40x10")

custom_systems['L4x6_2288H_V7'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0562149040000002, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 6}), numa_conf=None, system_id="L4x6_2288H_V7")

custom_systems['G5500V7_A30x10'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056391472, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 10}), numa_conf=None, system_id="G5500V7_A30x10")

custom_systems['G5500V7_A30x8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056391472, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 8}), numa_conf=None, system_id="G5500V7_A30x8")


###############################
#### END OF CUSTOM SYSTEMS ####
###############################
