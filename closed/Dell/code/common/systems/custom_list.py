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


custom_systems['R750xa_A100_PCIE_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.794688, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 4}), numa_conf=None, system_id="R750xa_A100_PCIE_80GBx4")
custom_systems['R750xa_H100_PCIe_80GBx2'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.803324, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA H100 PCIe", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=79.6474609375, byte_suffix=ByteSuffix.GiB), max_power_limit=310.0, pci_id="0x233110DE", compute_sm=90): 2}), numa_conf=None, system_id="R750xa_H100_PCIe_80GBx2")
custom_systems['R750xa_H100_PCIe_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.803324, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA H100 PCIe", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=79.6474609375, byte_suffix=ByteSuffix.GiB), max_power_limit=310.0, pci_id="0x233110DE", compute_sm=90): 4}), numa_conf=None, system_id="R750xa_H100_PCIe_80GBx4")
custom_systems['XE9680_H100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8470", architecture=CPUArchitecture.x86_64, core_count=52, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.113263964, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 8}), numa_conf=None, system_id="XE9680_H100_SXM_80GBx8")
custom_systems['XE2420_T4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6238 CPU @ 2.10GHz", architecture=CPUArchitecture.x86_64, core_count=22, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=262.708312, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.T4.value: 1}), numa_conf=None, system_id="XE2420_T4x1")
custom_systems['XR7620_L4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Genuine Intel(R) CPU 0000%@", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.87606, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_c    onf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 1}), numa_conf=None, system_id="XR7620_L4x1")
custom_systems['XR5610_L4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Genuine Intel(R) CPU 0000%@", architecture=CPUArchitecture.x86_64, core_count=20, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.64824, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 1}), numa_conf=None, system_id="XR5610_L4x1")
custom_systems['XR4520c_MaxQ_A2x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) D-2776NT CPU @ 2.10GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=65.45722, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A2.value: 1}), numa_conf=None, system_id="XR4520c_MaxQ_A2x1")
custom_systems['XR4520c_A30x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) D-2712T CPU @ 1.90GHz", architecture=CPUArchitecture.x86_64, core_count=4, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.545456, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 1}), numa_conf=None, system_id="XR4520c_A30x1")
custom_systems['XR4520c_A2x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) D-2776NT CPU @ 2.10GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.53962, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A2.value: 1}), numa_conf=None, system_id="XR4520c_A2x1")
custom_systems['XE8545_A100_SXM4_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7763 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=64, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056384888, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM4-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=500.0, pci_id="0x20B210DE", compute_sm=80): 4}), numa_conf=None, system_id="XE8545_A100_SXM4_80GBx4")
custom_systems['XE9680_A100_SXM4_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8470", architecture=CPUArchitecture.x86_64, core_count=52, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.113263964, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM4-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=500.0, pci_id="0x20B210DE", compute_sm=80): 8}), numa_conf=None, system_id="XE9680_A100_SXM4_80GBx8")


###############################
#### END OF CUSTOM SYSTEMS ####
###############################
