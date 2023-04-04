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

import json
from typing import Dict, Final, List, Iterable, Union

from code.common.constants import Scenario

MLPERF_LOG_PREFIX: Final[str] = ":::MLLOG"

scenario_loadgen_log_keys: Final[Dict[str, str]] = {
    Scenario.Offline: "result_samples_per_second",
    Scenario.Server: "result_scheduled_samples_per_sec",
    Scenario.SingleStream: "result_90.00_percentile_latency_ns",
    Scenario.MultiStream: "result_99.00_percentile_per_query_latency_ns",
}


def from_loadgen_by_keys(log_dir: str, keys: Iterable[str], return_list: bool = False) \
        -> Dict[str, Union[str, List[str]]]:
    """
    Gets values of certain keys from loadgen detailed logs, based on the new logging design.

    Args:
        log_dir (str):
            Directory where the mlperf log files are stored. Should contain mlperf_log_detail.txt.
        keys (Iterable):
            Collection of keys we want to query for from the Loadgen log
        return_list (bool):
            Whether or not to return all values of occurrences of a key in the Loadgen logs as a List. If False, will
            only report the latest value. Default: False.

    Returns:
        Dict[str, Union[str, List[str]]]: A Dictionary mapping keys to their values from the Loadgen detail log as
        specified. Will only contain keys specified in the `keys` argument.

    Raises:
        FileNotFoundError: When mlperf_log_defail.txt is not found in `log_dir`.
    """
    detailed_log: str = os.path.join(log_dir, "mlperf_log_detail.txt")
    with open(detailed_log) as f:
        lines: List[str] = f.read().strip().split("\n")

    log_entries: List[str] = []
    for line in lines:
        if line.startswith(MLPERF_LOG_PREFIX):
            buf = line[len(MLPERF_LOG_PREFIX) + 1:]
            log_entries.append(json.loads(buf))

    results: Dict[str, Union[str, List[str]]] = {}
    for entry in log_entries:
        key: str = entry["key"]
        if key in keys:
            if return_list:
                if key not in results:
                    results[key] = []
                results[key].append(entry["value"])
            else:
                results[key] = entry["value"]
    return results
