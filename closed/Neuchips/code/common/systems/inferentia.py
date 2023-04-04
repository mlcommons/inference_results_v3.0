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
import requests

from code.common import run_command
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.info_source import InfoSource


def get_inferentia_info() -> List[Dict[str, str]]:
    """Runs the curl command that identifies the instance type and returns the a dictionary with the instance name.

    Returns:
        List[Dict[str, str]]: A list of length 1, containing a dictionary mapping field_name -> value, where
        field_name is a key in the format of lscpu's output.
    """
    if os.environ.get("USE_INFERENTIA") == "1":
        # AWS provides an endpoint to determine the instance type - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
        accelerator_info = requests.get('http://169.254.169.254/latest/meta-data/instance-type')
        # Get the fields from accelerator info
        accelerator_fields = dict()
        accelerator_fields["name"] = accelerator_info.text

        return [accelerator_fields]
    return []


INFERENTIA_INFO_SOURCE: Final[InfoSource] = InfoSource(get_inferentia_info)
"""InfoSource: InfoSource to use for Inferentia information"""


@dataclass(eq=True, frozen=True)
class Inferentia(Hardware):
    # etcheng: Unfortunately, typing.GenericAlias is only a feature in 3.9+, so we are forced to use Union[Matchable, T]
    # every time.
    name: Union[Matchable, str, AliasedName]

    @classmethod
    def detect(cls) -> Inferentia:
        """Grabs the instance info and constructs a Inferentia object out of it. The caller must maintain INFERENTIA_INFO_SOURCE and make
        sure it is reset before calling it.

        Returns:
            CPU: An Inferentia object with fields retrieved from runtime data.
        """
        accelerator_fields = next(INFERENTIA_INFO_SOURCE)
        return Inferentia(
            accelerator_fields["name"]
        )

    def identifiers(self):
        return (self.name,)

    def __eq__(self, o) -> bool:
        # __eq__ needs to be explicitly defined to override the dataclass __eq__. dataclass.eq needs to be True to
        # auto-generate __hash__.
        return self.matches(o)
