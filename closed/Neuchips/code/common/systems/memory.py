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
from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, Final, List

import math
import re
import textwrap

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.info_source import InfoSource


SYS_MEMINFO_FILE: Final[str] = "/proc/meminfo"
"""str: Defines the OS file that contains the host system memory information"""


def get_mem_info(sys_meminfo_file: str = SYS_MEMINFO_FILE) -> List[Dict[str, str]]:
    """Reads memory information from the SYS_MEMINFO_FILE. The file should be formated like DMI (i.e. like /proc/meminfo
    on Linux).

    Args:
        sys_meminfo_file (str): The path to a file containing memory information in expected format.
                                (Default: SYS_MEMINFO_FILE)

    Returns:
        List[Dict[str, str]]: A list of length 1, containing a dictionary mapping field_name -> value, where
        field_name is a key in the format of /proc/meminfo.
    """
    with open(sys_meminfo_file) as f:
        mem_info = f.read().split("\n")
    mem_fields = dict()
    for line in mem_info:
        toks = re.split(r":\s*", line)
        if len(toks) == 2:
            mem_fields[toks[0]] = toks[1]
    return [mem_fields]


MEM_INFO_SOURCE: Final[InfoSource] = InfoSource(get_mem_info)
"""InfoSource: InfoSource to use for Memory information"""


@dataclass(eq=True, frozen=True)
class MemoryConfiguration(Hardware):
    host_memory_capacity: Memory
    comparison_tolerance: float = 0.05

    @classmethod
    def detect(cls) -> MemoryConfiguration:
        """Grabs Memory info and builds a map of Memory -> count.

        Returns:
            MemoryConfiguration: A MemoryConfiguration object from runtime data.
        """
        MEM_INFO_SOURCE.reset()
        # MEM_INFO_SOURCE only has 1 element
        mem_info = next(MEM_INFO_SOURCE)
        total_mem = mem_info["MemTotal"]
        quantity_str, suffix_str = total_mem.split()
        mem = Memory(float(quantity_str), ByteSuffix[suffix_str.upper()])
        # Simplify to the largest possible unit for human readability
        mem = Memory.to_1000_base(mem.to_bytes())
        return MemoryConfiguration(mem)

    def matches(self, other) -> bool:
        if other.__class__ == self.__class__:
            return math.isclose(self.host_memory_capacity.to_bytes(), other.host_memory_capacity.to_bytes(),
                                rel_tol=self.comparison_tolerance)
        return NotImplemented

    def __eq__(self, o) -> bool:
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the MemoryConfiguration

        Returns:
            str: 'Pretty-print' string representation of the MemoryConfiguration
        """
        mem_str = self.host_memory_capacity.pretty_string()
        return f"MemoryConfiguration: {mem_str} (Matching Tolerance: {self.comparison_tolerance})"
