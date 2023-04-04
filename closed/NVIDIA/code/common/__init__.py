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

import os
import sys
import json
import platform
import subprocess
import re
from glob import glob
from typing import Any, Dict, List, Optional, Set

import logging
logging.basicConfig(level=(logging.INFO if os.environ.get("VERBOSE", '0') == '0' else logging.DEBUG),
                    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

__MLPERF_INF_VERSION__ = "v3.0"
__MLPERF_INF_PAST_VERSIONS__ = ["v2.1", "v2.0", "v1.1", "v1.0", "v0.7", "v0.5"]


def run_command(cmd, get_output=False, tee=True, custom_env=None, verbose=True):
    """
    Runs a command.

    Args:
        cmd (str): The command to run.
        get_output (bool): If true, run_command will return the stdout output. Default: False.
        tee (bool): If true, captures output (if get_output is true) as well as prints output to stdout. Otherwise, does
            not print to stdout.
        verbose (bool): If True, logs the commands run and if the environment was overridden.
    """
    if verbose:
        logging.info("Running command: {:}".format(cmd))
    if not get_output:
        return subprocess.check_call(cmd, shell=True)
    else:
        output = []
        if custom_env is not None:
            if verbose:
                logging.info("Overriding Environment")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b""):
            line = line.decode("utf-8")
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line.rstrip("\n"))
        ret = p.wait()
        if ret == 0:
            return output
        else:
            raise subprocess.CalledProcessError(ret, cmd)


def args_to_string(d, blacklist=[], delimit=True, double_delimit=False):
    flags = []
    for flag in d:
        # Skip unset
        if d[flag] is None:
            continue
        # Skip blacklisted
        if flag in blacklist:
            continue
        if type(d[flag]) is bool:
            if d[flag] is True:
                flags.append("--{:}=true".format(flag))
            elif d[flag] is False:
                flags.append("--{:}=false".format(flag))
        elif type(d[flag]) in [int, float] or not delimit:
            flags.append("--{:}={:}".format(flag, d[flag]))
        else:
            if double_delimit:
                flags.append("--{:}=\\\"{:}\\\"".format(flag, d[flag]))
            else:
                flags.append("--{:}=\"{:}\"".format(flag, d[flag]))
    return " ".join(flags)


def flags_bool_to_int(d):
    for flag in d:
        if type(d[flag]) is bool:
            if d[flag]:
                d[flag] = 1
            else:
                d[flag] = 0
    return d


def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""

    val = d.get(key, default)
    return default if val is None else val


def dict_eq(d1: Dict[str, Any], d2: Dict[str, Any], ignore_keys: Optional[Set[str]] = None) -> bool:
    """Compares 2 dictionaries, returning whether or not they are equal. This function also supports ignoring keys for
    the equality check. For example, if d1 is {'a': 1, 'b': 2} and d2 is {'a': 1, 'b': 3, 'c': 1}, if ignore_keys is set
    to {'b', 'c'}, this method will return True.
    While this method supports dicts with any type of keys, it is recommended to use strings as keys.

    Args:
        d1 (Dict[str, Any]): The first dict to be compared
        d2 (Dict[str, Any]): The second dict to be compared
        ignore_keys (Set[str]): If set, will ignore keys in this set when doing the equality check

    Returns:
        bool: Whether or not d1 and d2 are equal, ignore the keys in `ignore_keys`
    """
    def filter_dict(d): return {k: v for k, v in d.items() if k not in ignore_keys}
    return filter_dict(d1) == filter_dict(d2)
