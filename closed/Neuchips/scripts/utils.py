#! /usr/bin/env python3
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

import os
import sys
sys.path.insert(0, os.getcwd())

import re
import sys
import shutil
import glob
import argparse
import datetime
import json
from numbers import Number
from typing import Optional


class SortingCriteria:
    Higher = True
    Lower = False


def safe_divide(numerator: Number, denominator: Number) -> Optional[float]:
    """
    Divides 2 numbers, returning None if a DivisionByZero were to occur.

    Args:
        numerator (Number):
            Value for the numerator
        denominator (Number):
            Value for the denominator

    Returns:
        Optional[float]: numerator/denominator. None if denominator is 0.
    """
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def get_system_type(system_name):
    fname = "systems/{:}.json".format(system_name)
    if not os.path.exists(fname):
        raise Exception("Could not locate system.json for {:}".format(system_name))

    with open(fname) as f:
        data = json.load(f)

    if "system_type" not in data:
        raise Exception("{:} does not have 'system_type' key".format(fname))

    return data["system_type"]
