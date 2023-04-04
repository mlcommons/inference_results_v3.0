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
from typing import Any, Callable, Final, List, Union

import re
import textwrap

from code.common import run_command
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.info_source import InfoSource


CPU_INFO_COMMAND: Final[str] = "lscpu"


def get_cpu_info() -> List[Dict[str, str]]:
    """Runs the CPU_INFO_COMMAND and parses the output, returning a list with single dictionary, containing the
    fields and values of CPU Info.

    Returns:
        List[Dict[str, str]]: A list of length 1, containing a dictionary mapping field_name -> value, where
        field_name is a key in the format of lscpu's output.
    """
    cpu_info = run_command(CPU_INFO_COMMAND, get_output=True, tee=False, verbose=False)
    # Get the fields from cpu_info
    cpu_fields = dict()
    for line in cpu_info:
        toks = re.split(r":\s*", line)
        if len(toks) == 2:
            cpu_fields[toks[0]] = toks[1]
    return [cpu_fields]


CPU_INFO_SOURCE: Final[InfoSource] = InfoSource(get_cpu_info)
"""InfoSource: InfoSource to use for CPU information"""


@dataclass(eq=True, frozen=True)
class CPU(Hardware):
    # etcheng: Unfortunately, typing.GenericAlias is only a feature in 3.9+, so we are forced to use Union[Matchable, T]
    # every time.
    name: Union[Matchable, str, AliasedName]
    architecture: Union[Matchable, CPUArchitecture]
    core_count: Union[Matchable, int]
    threads_per_core: Union[Matchable, int]

    @classmethod
    def detect(cls) -> CPU:
        """Grabs the CPU info and constructs a CPU object out of it. The caller must maintain CPU_INFO_SOURCE and make
        sure it is reset before calling it.

        Returns:
            CPU: A CPU object with fields retrieved from runtime data.
        """
        cpu_fields = next(CPU_INFO_SOURCE)
        return CPU(
            cpu_fields["Model name"],
            CPUArchitecture(cpu_fields["Architecture"]),
            int(cpu_fields["Core(s) per socket"]),
            int(cpu_fields["Thread(s) per core"]))

    def identifiers(self):
        return (self.name, self.architecture, self.core_count, self.threads_per_core)

    def __eq__(self, o) -> bool:
        # __eq__ needs to be explicitly defined to override the dataclass __eq__. dataclass.eq needs to be True to
        # auto-generate __hash__.
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the CPU

        Returns:
            str: 'Pretty-print' string representation of the CPU
        """
        s = f"CPU ({self.architecture}): {self.name}\n" + \
            textwrap.indent(f"{self.core_count} Cores, {self.threads_per_core} Threads/Core", " " * 4)
        return s


@dataclass(eq=True, frozen=True)
class CPUConfiguration(Hardware):
    layout: Dict[CPU, int]

    @classmethod
    def detect(cls) -> CPUConfiguration:
        """Grabs CPU info and builds a map of CPU -> count.

        Returns:
            CPUConfiguration: A CPUConfiguration object from runtime data.
        """
        CPU_INFO_SOURCE.reset()
        cpus = []
        while CPU_INFO_SOURCE.has_next():
            cpus.append(CPU.detect())  # calls CPU_INFO_SOURCE.__next__

        cpu_layout = dict()
        for i, cpu_fields in enumerate(CPU_INFO_SOURCE):
            count = int(cpu_fields["Socket(s)"])
            cpu_layout[cpus[i]] = count
        return CPUConfiguration(cpu_layout)

    def get_architecture(self):
        if len(self.layout) == 0:
            return None

        # All CPUs are assumed to have the same architecture
        return list(self.layout.keys())[0].architecture

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            if len(self.layout) != len(other.layout):
                return False

            for cpu, count in self.layout.items():
                # We actually have to iterate through each cpu in other in case of Matchables. hashes are not guaranteed
                # to be equivalent even if a.matches(b).
                found = False
                for other_cpu, other_count in other.layout.items():
                    if cpu == other_cpu:
                        found = True
                        if count != other_count:
                            return False
                        else:
                            # TODO: It is possible that multiple matches may cause issues, but since there is only 1
                            # model of CPU on our systems, and we assume homogenous CPUs, we can leave this for later.
                            break
                if not found:
                    return False
            return True
        return NotImplemented

    def __eq__(self, o) -> bool:
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the CPUConfiguration.

        Returns:
            str: 'Pretty-print' string representation of the CPUConfiguration
        """
        lines = ["CPUConfiguration:"]
        for cpu, count in self.layout.items():
            lines.append(f"{count}x " + cpu.pretty_string())
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s
