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
from abc import ABC, abstractmethod, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, unique
from subprocess import CalledProcessError
from typing import Any, Callable, Final, Optional, List, Union

import os
import re
import math
import shutil
import textwrap
import json

from code.common import run_command, logging
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.inferentia import Inferentia
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.info_source import InfoSource
from code.common.systems.known_hardware import KnownCPU, KnownGPU, KnownMIG


NV_GPU_AFFINITY_COMMAND: Final[Tuple[str, ...]] = ("nvidia-smi", "topo", "-m")
"""str: The command for NVIDIA-based systems to query for GPU NUMA affinity"""

CPU_AFFINITY_COMMAND: Final[Tuple[str, ...]] = ("taskset", "-c", "-p", "$$")
"""str: The executable to check CPU affinities"""

NUMA_CONTROL_COMMAND: Final[Tuple[str, ...]] = ("numactl", "--hardware")
"""str: Command to execute to get system NUMA information"""

SYSTEM_JSON_MIG_MARKETING_NAME_FORMAT = re.compile(r"(.+) \((\d+)x(\d+)g\.(\d+)gb MIG\)")
"""re.Pattern: Regex to parse the marketing name for MIG systems, which are not consistent with nvidia-smi output or our
               internal system ID.

               Example matching strings:
                   "NVIDIA A100 (1x1g.5gb MIG)"
                   "A30 (12x1g.6gb MIG)"
                   "A100-SXM4-80GB (3x2g.20gb MIG)"

               match(1) - parent GPU name
               match(2) - number of MIG slices
               match(3) - GPCs per slice
               match(4) - mem per slice (in GB)
"""


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

    try:
        gpu_affinity_out = run_command(" ".join(NV_GPU_AFFINITY_COMMAND), get_output=True, tee=False, verbose=False)
    except CalledProcessError:
        logging.warning("nvidia-smi command exists but failed to execute - Ignoring NUMA detection.")
        return []
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
        if not no_auto_numa and accelerator_conf.num_gpus() > 1:  # MLPINF-1846: Do not detect NUMA on single accelerator nodes
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

    def get_full_name(self, workload_setting: WorkloadSetting, trt_version: str = "") -> Optional[str]:
        """Intended to replace code.common.harness.get_system_name.

        Args:
            workload_setting (WorkloadSetting): The WorkloadSetting to get the name for, as the power setting and
                                                harness changes the submission system name
            trt_version (str): String indicating the TensorRT version (i.e. "84" for TensorRT 8.4). (Default: "")

        Returns:
            str: The full system name used for submission, which includes the SW stack and power setting. If
                 `self.system_id` is not set, returns None instead.
        """
        name = self.system_id
        if name is None:
            return None

        # Add software stack
        if type(self.accelerator_conf.get_primary_accelerator()) in (GPU, MIG):
            name += f"_TRT{trt_version}"

        if workload_setting.harness_type == HarnessType.Triton:
            name += "_Triton"
        elif workload_setting.harness_type == HarnessType.HeteroMIG:
            name += "_HeteroMultiUse"

        # Add power setting
        if workload_setting.power_setting == PowerSetting.MaxQ:
            name += "_MaxQ"

        return name

    def find_system_json_paths(self) -> List[str]:
        """Finds the JSON files in systems/ that match this system's system_id. If system_id is not set, returns an
        empty list."""
        if self.system_id is None:
            return []

        harness_list = [HarnessType.Custom, HarnessType.Triton]
        if self.accelerator_conf.num_migs() == 1:
            harness_list.append(HarnessType.HeteroMIG)
        elif self.accelerator_conf.num_migs() > 1:
            harness_list = [HarnessType.Triton]
        power_settings = [PowerSetting.MaxP, PowerSetting.MaxQ]

        json_paths = []
        for harness_type in harness_list:
            for power_setting in power_settings:
                workload_setting = WorkloadSetting(harness_type, AccuracyTarget.k_99, power_setting)
                full_name = self.get_full_name(workload_setting)
                json_path = os.path.join("systems", f"{full_name}.json")
                if os.path.exists(json_path):
                    json_paths.append(json_path)
        return json_paths

    def _verify_cpu_fields(self, json_dict: Dict[str, object]) -> List[str]:
        """Verifies the fields in a JSON dict that pertain to the host CPU.

        MLPINF-970: There is a mismatch between the hardware names in the system.json and the detected settings from
        hwinfo, mostly because some names are truncated or renamed for marketing or clarity purposes. Therefore we
        cannot actually compare these directly. Instead we can use a "transitive property" of Matchables. While not
        *strictly* true, in this case it is sufficient if both the detected CPU object and CPU object constructed from
        the system.json fields match with the same KnownCPU.

        The fields in the system JSON used to construct the CPU object are:
            - "host_processor_model_name"
            - "host_processor_core_count"

        Args:
            json_dict: A dictionary representing the system JSON file

        Returns:
            List[str]: A list of keys in the JSON where there were value mismatches from this system
        """
        mismatched_keys = []

        constructed_cpu = CPU(json_dict["host_processor_model_name"],
                              MATCH_ANY,  # CPU architecture is not detailed in system.json
                              json_dict["host_processor_core_count"],
                              MATCH_ANY)  # Threads/core is not detailed in system.json
        constructed_match = KnownCPU.get_first_match(constructed_cpu)
        logging.debug(f"CPU from JSON matched: {constructed_match}")

        primary_cpu = self.host_cpu_conf.get_primary_cpu()
        detected_match = KnownCPU.get_first_match(primary_cpu)
        logging.debug(f"Detected CPU matched: {detected_match}")
        assert detected_match != None, "Detected CPU has no match. This means it was manually set in KnownSystems without being added to KnownCPU. This is disallowed."

        if detected_match != constructed_match:
            logging.warn(f"Mismatch between detected CPU ({detected_match}) and system.json ({constructed_match})")

            if constructed_match is None or constructed_match.value.name != detected_match.value.name:
                mismatched_keys.append("host_processor_model_name")
            if constructed_match is None or constructed_match.value.core_count != detected_match.value.core_count:
                mismatched_keys.append("host_processor_model_name")
        return mismatched_keys

    def _verify_gpu_fields(self, json_dict: Dict[str, object], primary_accelerator: Accelerator) -> List[str]:
        """Verifies the fields in a JSON dict that pertain to the GPU (or MIG).

        Similar to _verify_cpu_fields, we use the transitive property of Matchables.

        The fields in the system JSON used to construct the GPU/MIG object are:
            - "accelerator_model_name"
            - "accelerator_memory_capacity"
            - "accelerators_per_node"

        Args:
            json_dict: A dictionary representing the system JSON file
            primary_accelerator (Accelerator): The Accelerator to verify against

        Returns:
            List[str]: A list of keys in the JSON where there were value mismatches from this system
        """
        mismatched_keys = []
        accelerator_memory_capacity = json_dict["accelerator_memory_capacity"]
        if "shared" in accelerator_memory_capacity.lower():
            accelerator_type = AcceleratorType.Integrated
            accelerator_memory_capacity = None
        else:
            accelerator_type = AcceleratorType.Discrete
            accelerator_memory_capacity = Memory.from_string(accelerator_memory_capacity)
            # MLPINF-970: There is a mismatch due to marketing terms: GB in naming actually refers to GiB
            accelerator_memory_capacity = Memory(accelerator_memory_capacity.quantity, ByteSuffix.GiB)
        accelerators_per_node = json_dict["accelerators_per_node"]  # This is always # of GPUs regardless of MIG.

        if type(primary_accelerator) is GPU:
            constructed_accelerator = GPU(json_dict["accelerator_model_name"],
                                          accelerator_type,
                                          accelerator_memory_capacity,
                                          MATCH_ANY,  # Chip power TDP not in system.json
                                          MATCH_ANY,  # Chip PCI ID not in system.json
                                          MATCH_ANY)  # Chip compute SM not in system.json
            constructed_match = KnownGPU.get_first_match(constructed_accelerator)
            detected_match = KnownGPU.get_first_match(primary_accelerator)
        else:
            # Marketing name format is "[CHIP NAME] ([NUM SLICES]x[GPC per slice]g.[MEM]gb MIG)".
            # ex. "NVIDIA A100 (1x1g.5gb MIG)"
            # See the docstring for SYSTEM_JSON_MIG_MARKETING_NAME_FORMAT for details on the match groups.
            m = SYSTEM_JSON_MIG_MARKETING_NAME_FORMAT.match(json_dict["accelerator_model_name"])
            if m is None:
                mismatched_keys.append("accelerator_model_name")
                logging.warn(f"MIG system contains misformatted accelerator_model_name in system.json")
                constructed_accelerator = None
            else:
                constructed_accelerator = MIG(f"{m.group(1)} MIG-{m.group(3)}g.{m.group(4)}gb",  # Convert to internal naming
                                              AcceleratorType.Discrete,
                                              Memory(float(m.group(4)), ByteSuffix.GiB),
                                              MATCH_ANY,  # Chip power TDP not in system.json
                                              MATCH_ANY,  # Chip PCI ID not in system.json
                                              MATCH_ANY,  # Chip compute SM not in system.json
                                              int(m.group(3)))
                accelerators_per_node *= int(m.group(2))  # Detected system counts # of MIG not GPU.
                constructed_match = KnownMIG.get_first_match(constructed_accelerator)
                detected_match = KnownMIG.get_first_match(primary_accelerator)

        assert detected_match != None, "Detected accelerator has no match. This means it was manually set in KnownSystems without being added to known_hardware. This is disallowed."
        if constructed_accelerator is not None:
            logging.debug(f"Accelerator from JSON matched: {constructed_match}")
            logging.debug(f"Detected accelerator matched: {detected_match}")
            if detected_match != constructed_match:
                logging.warn(f"Mismatch between detected accelerator ({detected_match}) and system.json ({constructed_match})")

                if constructed_match is None or constructed_match.value.name != detected_match.value.name:
                    mismatched_keys.append("accelerator_model_name")
                if constructed_match is None or constructed_match.value.vram != detected_match.value.vram:
                    mismatched_keys.append("accelerator_memory_capacity")

        if accelerators_per_node != self.accelerator_conf.layout[primary_accelerator]:
            mismatched_keys.append("accelerators_per_node")
        return mismatched_keys

    def verify_system_json(self, json_paths: Optional[List[str]] = None) -> Tuple[Dict[str, str], int]:
        """Verifies the JSON file in the systems directory and compares each field to the detected one. In addition to
        the fields verified by _verify_cpu_fields and _verify_gpu_fields, the fields that this method verifies are:
            - "host_processors_per_node" (Number of CPUs)
            - "host_memory_capacity" (Host Memory Capacity)
        See the docstrings for self._verify_cpu_fields and self._verify_gpu_fields for those related keys.

        The other fields aren't verified because they are not detected by System Detection.

        Note that this method is a noop if self.system_id is not set, since this is the value that determines the JSON
        file to compare to.

        **NOTE: For now, we are simply logging the failures as warnings, since this will be tested on each individual
        system as the L1/L2 tests run.

        Args:
            json_paths (List[str]): If set, verifies this system against the files in this list. If None, searches
                                    systems/ for matching JSON files. (Default: None)

        Returns:
            Dict[str, str]: A list of keys in the JSON where there were value mismatches from this system
            int: Number of keys that had mismatches throughout every JSON file. This is useful in the case where the
                 mismatch dictionary is something like:
                    {
                        "systems/s1.json": [],
                        "systems/s2.json": [],
                    }
                In this case, there are 0 mismatches, but the dictionary is not empty, so it cannot be checked with
                len(x) == 0.
        """
        if json_paths is None:
            json_paths = self.find_system_json_paths()
        mismatched_keys = defaultdict(list)
        for json_path in json_paths:
            logging.debug(f"Verifying system.json: {json_path}")

            with open(json_path) as f:
                d = json.load(f)

            def _check(key, actual, post_fn=None):
                """Compare values, and log if mismatched

                Args:
                    key (str): Key that caused mismatch
                    actual (object): The value that was detected
                    post_fn (Callable): If set, use post_fn(d[key]) for comparison, rather than d[key]
                """
                nonlocal mismatched_keys
                val = d[key]
                if post_fn is not None:
                    val = post_fn(val)

                if actual != val:
                    mismatched_keys[json_path].append(key)

            # Verify CPU
            mismatched_keys[json_path] += self._verify_cpu_fields(d)
            _check("host_processors_per_node", self.host_cpu_conf.chip_count())

            # Verify host mem.
            _check("host_memory_capacity",
                   MemoryConfiguration(self.host_mem_conf.host_memory_capacity, 0.1),
                   post_fn=lambda x: MemoryConfiguration(Memory.from_string(x)))

            # Verify accelerators. Again, we need to construct an accelerator object and attempt to match.
            primary_accelerator = self.accelerator_conf.get_primary_accelerator()
            if primary_accelerator is None:
                _check("accelerator_model_name", "N/A")
                _check("accelerator_memory_capacity", "N/A")
                _check("accelerators_per_node", 0)
            elif type(primary_accelerator) is Inferentia:
                _check("accelerator_model_name", "Inferentia")
                _check("accelerators_per_node", 1)
            elif type(primary_accelerator) in (GPU, MIG):
                logging.debug(primary_accelerator)
                mismatched_keys[json_path] += self._verify_gpu_fields(d, primary_accelerator)
        return mismatched_keys, sum(len(v) for k, v in mismatched_keys.items())
