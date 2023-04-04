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

custom_systems['L4x8_Custom_X640_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.05640928, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 8}), numa_conf=None, system_id="L4x8_Custom_X640_G50")
custom_systems['L4x8_Custom_X640_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8368 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=38, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564647, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 8}), numa_conf=None, system_id="L4x8_Custom_X640_G40")
custom_systems['L4x5_Custom_X620_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6458Q", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564746440000001, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 5}), numa_conf=None, system_id="L4x5_Custom_X620_G50")
custom_systems['L4x4_Custom_X620_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=28, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.9917240000001, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 4}), numa_conf=None, system_id="L4x4_Custom_X620_G40")
custom_systems['A30x3_Custom_X620_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=28, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.992384, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 3}), numa_conf=None, system_id="A30x3_Custom_X620_G40")
custom_systems['A30x4_Custom_X620_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8458P", architecture=CPUArchitecture.x86_64, core_count=44, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.73407199999997, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 4}), numa_conf=None, system_id="A30x4_Custom_X620_G50")
custom_systems['A30x8_Custom_X640_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6354 CPU @ 3.00GHz", architecture=CPUArchitecture.x86_64, core_count=18, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564846319999999, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 8}), numa_conf=None, system_id="A30x8_Custom_X640_G40")
custom_systems['A30x8_Custom_X640_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564421720000001, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 8}), numa_conf=None, system_id="A30x8_Custom_X640_G50")
custom_systems['L40x8_Custom_X640_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz", architecture=CPUArchitecture.x86_64, core_count=40, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.113446324, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L40.value: 8}), numa_conf=None, system_id="L40x8_Custom_X640_G40")
custom_systems['L40x8_Custom_X640_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564796, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L40.value: 8}), numa_conf=None, system_id="L40x8_Custom_X640_G50")
custom_systems['L40x3_Custom_X620_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=28, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=528.0080720000001, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L40.value: 3}), numa_conf=None, system_id="L40x3_Custom_X620_G40")
custom_systems['L40x3_Custom_X620_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6442Y", architecture=CPUArchitecture.x86_64, core_count=24, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056498956, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L40.value: 3}), numa_conf=None, system_id="L40x3_Custom_X620_G50")
custom_systems['A40X8_CUSTOM_X640_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.05646532, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=44.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x223510DE", compute_sm=86): 8}), numa_conf=None, system_id="A40X8_CUSTOM_X640_G50")
custom_systems['A40X3_CUSTOM_X620_G50'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056479084, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=44.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x223510DE", compute_sm=86): 3}), numa_conf=None, system_id="A40X3_CUSTOM_X620_G50")
custom_systems['A40X8_CUSTOM_X640_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8368 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=38, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056483568, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=44.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x223510DE", compute_sm=86): 8}), numa_conf=None, system_id="A40X8_CUSTOM_X640_G40")
custom_systems['A40X3_CUSTOM_X620_G40'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=28, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=528.008672, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A40", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=44.98828125, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x223510DE", compute_sm=86): 3}), numa_conf=None, system_id="A40X3_CUSTOM_X620_G40")



###############################
#### END OF CUSTOM SYSTEMS ####
###############################
