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

custom_systems['NC96ads_A100_v4_MIG_1g_10gb'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7V13 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=48, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=908.198056, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={MIG(name="NVIDIA A100 80GB PCIe MIG-1g.10gb", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=10, byte_suffix=ByteSuffix.GiB), max_power_limit=300.0, pci_id="0x20B510DE", compute_sm=80, num_gpcs=1): 28}), numa_conf=None, system_id="NC96ads_A100_v4_MIG_1g_10gb")


###############################
#### END OF CUSTOM SYSTEMS ####
###############################
