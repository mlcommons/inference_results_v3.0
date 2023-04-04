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

from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, Final, Optional, List, Union

import re
import shutil
import textwrap

from code.common import run_command, logging
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.info_source import InfoSource


NV_GPU_AFFINITY_COMMAND: Final[Tuple[str, ...]] = ("nvidia-smi", "topo", "-m")
"""str: The command for NVIDIA-based systems to query for GPU NUMA affinity"""

CPU_AFFINITY_COMMAND: Final[Tuple[str, ...]] = ("taskset", "-c", "-p", "$$")
"""str: The executable to check CPU affinities"""

NUMA_CONTROL_COMMAND: Final[Tuple[str, ...]] = ("numactl", "--hardware")
"""str: Command to execute to get system NUMA information"""


@dataclass(eq=True, frozen=True)
class Interval:
    """Defines an inclusive range of integers [a, b]. i.e. Interval[1, 3] is the same as {1, 2, 3}.
    """
    start: int
    end: Optional[int] = None

    def __post_init__(self):
        if self.end is None:
            object.__setattr__(self, 'end', self.start)

        if self.start > self.end:
            tmp = self.start
            object.__setattr__(self, 'start', self.end)
            object.__setattr__(self, 'end', tmp)

    def intersects(self, o):
        """Returns whether or not another interval overlaps with self. This includes trivial overlaps where the start of
        one interval equals the end of the other, since Intervals are inclusive.

        Args:
            o (Any) - An arbitrary object.

        Returns:
            bool - If o is not an Interval, returns NotImplemented. Otherwise, returns whether or not the intervals
                   overlap.
        """
        if o.__class__ != Interval:
            return NotImplemented
        return (min(self.end, o.end) - max(self.start, o.start)) >= 0

    def __str__(self):
        if self.start == self.end:
            return str(self.start)
        else:
            return f"{self.start}-{self.end}"

    def __iter__(self):
        return range(self.start, self.end + 1)

    def to_list(self):
        return [i for i in range(self.start, self.end + 1)]

    def to_set(self):
        return set(self.to_list())

    @classmethod
    def from_str(cls, s: str):
        toks = s.split("-")
        if len(toks) > 2 or len(toks) == 0:
            raise ValueError(f"Cannot convert string '{s}' to Interval")
        elif len(toks) == 1:
            return Interval(int(toks[0]))
        else:
            return Interval(int(toks[0]), int(toks[1]))

    @classmethod
    def build_interval_list(cls, nums: List[int]) -> List[Interval]:
        """Returns a list of Intervals representing the numbers in `nums`.

        Args:
            nums (List[int]): A list of integers to turn into a list of Intervals

        Returns:
            List[Interval]: An list of Intervals that is equivalent to the input list.
        """
        if len(nums) == 0:
            return []

        nums = list(sorted(set(nums)))
        intervals = [(nums[0], nums[0])]
        for x in nums[1:]:
            if x == intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], x)
            else:
                intervals.append((x, x))
        return [Interval(*t) for t in intervals]


def interval_list_from_str(s):
    L = []
    for tok in s.split(","):
        L.append(Interval.from_str(tok))
    return L


def get_numa_info():
    """Reads information from numactl, nvidia-smi topology, and CPU taskset to check the NUMA configuration.

    Returns:
        List[Dict[str, Any]]: A list of length 1 containing a dict:
            {
                "num_numa_nodes": int,
                "active_cpus": List[Interval],
                "gpu_numa_map": {
                    gpu_index (int): {
                        "cpu_affinity": List[Interval],
                        "numa_affinity": numa_node_index (int),
                    },
                    ...
                }
            }
    """
    if None in list(map(shutil.which, [NV_GPU_AFFINITY_COMMAND[0], CPU_AFFINITY_COMMAND[0], NUMA_CONTROL_COMMAND[0]])):
        return []

    gpu_affinity_out = run_command(" ".join(NV_GPU_AFFINITY_COMMAND), get_output=True, tee=False, verbose=False)
    # Format:
    #         GPU0    GPU1 ... GPUN    CPU Affinity    NUMA Affinity
    # GPU0     X      NV4      X       16-31,80-95     1
    # GPU1    ...
    # ...
    found_mellanox_nic = False
    gpu_numa_map = dict()
    for line in gpu_affinity_out[1:]:  # Skip header
        line = line.strip()
        if len(line) == 0:
            continue

        if line.lower().startswith("legend:"):  # If we reach the legend, no more GPUs to parse
            break

        toks = line.split()
        if not toks[0].startswith("GPU"):
            if toks[0].startswith("mlx"):
                logging.warning(f"Found Mellanox core {toks[0]}. Skipping.")
                found_mellanox_nic = True
            else:
                logging.info(f"Found unknown device in GPU connection topology: {toks[0]}. Skipping.")
            continue
        if toks[-1] == "N/A":
            toks[-1] = None
        else:
            toks[-1] = Interval.from_str(toks[-1])
        gpu_numa_map[toks[0]] = (toks[-2], toks[-1])

    cpu_affinity = run_command(" ".join(CPU_AFFINITY_COMMAND), get_output=True, tee=False, verbose=False)
    active_cpus = cpu_affinity[0].split()[-1]

    system_numa = run_command(" ".join(NUMA_CONTROL_COMMAND), get_output=True, tee=False, verbose=False)
    # First line format:
    # available: 4 nodes (0-3)
    num_numa_nodes = int(system_numa[0].split()[1])
    # TODO: Handle the case where not all NUMA nodes are used.
    # Convert the remaining lines and figure out which NUMA Node ids are actually active.

    return [{
        "num_numa_nodes": num_numa_nodes,
        "active_cpus": active_cpus,
        "gpu_numa_map": {
            k: {
                "cpu_affinity": gpu_numa_map[k][0],
                "numa_affinity": gpu_numa_map[k][1],
            }
            for k in gpu_numa_map
        },
        "has_mellanox_nic": found_mellanox_nic,
    }]


NUMA_INFO_SOURCE: Final[InfoSource] = InfoSource(get_numa_info)


@dataclass(eq=True, frozen=True)
class NUMANode:
    """Represents a NUMA Node, containing the index of the node, and cpus / gpus connected to this node."""
    index: int
    cpus: List[Interval]
    gpus: List[int]


@dataclass
class NUMAConfiguration(Hardware):
    """Represents a configuration of NUMA nodes, mapping each numa node index to its associated cpu cores and gpus.
    """
    numa_nodes: Dict[int, NUMANode]
    num_numa_nodes: int

    @classmethod
    def detect(cls) -> NUMAConfiguration:
        """Detects the NUMAConfiguration from NUMA_INFO_SOURCE. Also validates that CPU cores in the NUMA topology are
        cores that are allowed by the session's cgroup, which is important when running on shared datacenters/clusters.

        Returns None if NUMA is not being used, or cannot be detected correctly."""
        NUMA_INFO_SOURCE.reset()
        if not NUMA_INFO_SOURCE.has_next():
            return None

        numa_info = next(NUMA_INFO_SOURCE)

        # WAR: Currently NUMA detection is broken on Luna machines, or machines that have NICs that mess up the
        # nvidia-smi topo -m command. Skip autodetect for these for now, as these are low priority: We generally use
        # start_from_device on these machines, so numa isn't used (unless it is the multi-MIG harness).
        if numa_info["has_mellanox_nic"]:
            logging.warning("System has Mellanox NICs. Skipping NUMA detection.")
            return None

        active_cpus = set()
        for interval in interval_list_from_str(numa_info["active_cpus"]):
            active_cpus = active_cpus.union(interval.to_set())

        numa_nodes = dict()
        for gpu_id, info in numa_info["gpu_numa_map"].items():
            indices = info["numa_affinity"]
            if indices is None:
                continue

            for index in indices.to_list():
                cpus = interval_list_from_str(info["cpu_affinity"])
                # Check if the CPUs in affinities are active from taskset
                cpu_ids = set()
                for cpu_interval in cpus:
                    cpu_ids = cpu_ids.union(cpu_interval.to_set())

                invalid_cpus = cpu_ids - active_cpus
                valid_cpus = cpu_ids.intersection(active_cpus)
                if len(invalid_cpus) > 0:
                    logging.warning(f"CPUs {invalid_cpus} are in reported NUMA affinity, but not in taskset. Removing.")
                    cpus = Interval.build_interval_list(valid_cpus)
                if len(cpus) == 0:
                    raise ValueError(f"No CPUs in taskset for NUMA Node {index}")

                gpu = int(gpu_id[len("GPU"):])

                if index not in numa_nodes:
                    numa_nodes[index] = NUMANode(index, cpus, [gpu])
                else:
                    if numa_nodes[index].cpus != cpus:
                        raise ValueError(f"Inconsistent CPU affinity for NUMA Node {index}")
                    numa_nodes[index].gpus.append(gpu)
        return NUMAConfiguration(numa_nodes, numa_info["num_numa_nodes"])

    @classmethod
    def from_str(self, s):
        numa_node_strs = s.split("&")
        numa_nodes = dict()
        for i in range(len(numa_node_strs)):
            gpu_str, cpu_str = numa_node_strs[i].split(":")
            numa_nodes[i] = NUMANode(i, interval_list_from_str(cpu_str), list(map(int, gpu_str.split(","))))
        return NUMAConfiguration(numa_nodes, len(numa_node_strs))

    def __str__(self):
        numa_config_strs = []
        for i in range(self.num_numa_nodes):
            if i not in self.numa_nodes:
                numa_config_strs.append("")
            else:
                gpu_str = ",".join(map(str, self.numa_nodes[i].gpus))
                cpu_str = ",".join(map(str, self.numa_nodes[i].cpus))
                numa_config_strs.append(f"{gpu_str}:{cpu_str}")
        return "&".join(numa_config_strs)


@dataclass(eq=True, frozen=True)
class SystemConfiguration(Hardware):
    host_cpu_conf: CPUConfiguration
    """CPUConfiguration: CPU configuration of the host machine"""

    host_mem_conf: MemoryConfiguration
    """MemoryConfiguration: Memory configuration of the host machine"""

    accelerator_conf: AcceleratorConfiguration
    """AcceleratorConfiguration: Configuration of the accelerators in the system"""

    numa_conf: Optional[NUMAConfiguration] = None
    """NUMAConfiguration: The NUMA configuration detected on the system. If this field is None, NUMA will not be used."""

    system_id: Optional[str] = None
    """str: The system ID string to identify the system as. The field is optional, and is only used to nickname a system
    for convenience"""

    @classmethod
    def detect(cls, no_auto_numa=False) -> SystemConfiguration:
        """Consolidates the detected CPU, Memory, Accelerator, and NUMA configurations into a single object.

        Args:
            no_auto_numa (bool): If True, disables NUMA configuration detection. (Default: False)

        Returns:
            SystemConfiguration: A SystemConfiguration object that contains all the detected components from runtime.
        """
        host_cpu_conf = CPUConfiguration.detect()
        host_mem_conf = MemoryConfiguration.detect()
        accelerator_conf = AcceleratorConfiguration.detect()
        numa_conf = None

        # Get NUMA Configuration
        if not no_auto_numa and accelerator_conf.num_gpus() > 0:
            numa_conf = NUMAConfiguration.detect()
        return SystemConfiguration(host_cpu_conf, host_mem_conf, accelerator_conf, numa_conf=numa_conf)

    def __hash__(self) -> int:
        """Generate a hash from the components using prime numbers - See
        https://stackoverflow.com/questions/1145217/why-should-hash-functions-use-a-prime-number-modulus/1147232#1147232"""
        return hash(self.host_cpu_conf) + \
            13 * hash(self.host_mem_conf) + \
            29 * hash(self.accelerator_conf) + \
            37 * hash(self.numa_conf)

    def get_id(self) -> str:
        return self.system_id

    def set_id(self, s):
        object.__setattr__(self, 'system_id', s)

    def get_compute_sm(self):
        """Returns the compute SM capability of the system, defaulting to the first GPU or MIG accelerator. If no GPUs
        or MIGs exist, returns None.
        """
        compute_sm = None
        for accelerator in self.accelerator_conf.layout:
            if accelerator.__class__ in [MIG, GPU]:
                if compute_sm is None:
                    compute_sm = accelerator.compute_sm
                elif accelerator.compute_sm != compute_sm:
                    logging.warning("Multiple accelerators detected, using architecture of first accelerator.")
        return compute_sm

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            return self.host_cpu_conf == other.host_cpu_conf and \
                self.host_mem_conf == other.host_mem_conf and \
                self.accelerator_conf == other.accelerator_conf
        return NotImplemented

    def __eq__(self, o) -> bool:
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the SystemConfiguration.

        Returns:
            str: 'Pretty-print' string representation of the SystemConfiguration
        """
        lines = ["SystemConfiguration:",
                 f"System ID (Optional Alias): {self.system_id}",
                 self.host_cpu_conf.pretty_string(),
                 self.host_mem_conf.pretty_string(),
                 self.accelerator_conf.pretty_string(),
                 f"NUMA Config String: {self.numa_conf}"]
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s
