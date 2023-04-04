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
from typing import Any, Callable, ClassVar, Final, List, Tuple, Union

import re
import shutil
import textwrap
import os

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.info_source import InfoSource
from code.common.fix_sys_path import ScopedRestrictedImport


def get_lon_info() -> List[Dict[str, str]]:
    """Finds if the system is used for LON

    Returns:
        List[Dict[str, str]]: A list of length 1, containing a dictionary mapping field_name -> value, where
        field_name is a key for LON usage, and a value is a string showing system's role: LON_node or SUT_node
    """
    lon_role = 'Unused'
    env_lon = os.getenv('NETWORK_NODE')
    if env_lon == 'LON':
        lon_role = 'LON_node'
    elif env_lon == 'SUT':
        lon_role = 'SUT_node'
    return [{"lon_role": lon_role}]


LON_INFO_SOURCE: Final[InfoSource] = InfoSource(get_lon_info)
"""InfoSource: InfoSource to use for LON information"""

# FIXME: maybe there's new class to inherit


@dataclass(eq=True, frozen=True)
class LONConfiguration(Hardware):
    lon_role: Union[Matchable, str]

    @classmethod
    def detect(cls) -> LONConfiguration:
        """Grabs the next LON info from LON_INFO_SOURCE. The caller must maintain LON_INFO_SOURCE and make sure it is
        reset before calling it.

        Returns:
            LONConfiguration: An LONConfiguration object with fields from the top item from the InfoSource buffer
        """
        lon_fields = next(LON_INFO_SOURCE)

        return LONConfiguration(lon_fields['lon_role'])

    def identifiers(self):
        return (
            self.lon_role,
        )

    def matches(self, other) -> bool:
        """Matches this LON with 'other'.

        If other is the same class as self, compare the identifiers. 
        Returns False otherwise.

        Returns:
            bool: Equality based on the rules described above
        """
        if other.__class__ == self.__class__:
            self_id = self.identifiers()
            other_id = other.identifiers()
            return self_id == other_id
        return NotImplemented

    def __eq__(self, o) -> bool:
        # __eq__ needs to be explicitly defined to override the dataclass __eq__. dataclass.eq needs to be True to
        # auto-generate __hash__.
        return self.matches(o)

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying if the system is for LON

        Returns:
            str: 'Pretty-print' string representation of the LON
        """

        lines = [f"LON role: {self.lon_role}",
                 ]
        s = lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), " " * 4)
        return s
