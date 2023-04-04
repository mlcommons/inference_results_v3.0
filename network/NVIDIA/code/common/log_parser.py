#!/usr/bin/env python3
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

import datetime
import json
import glob
import os
from typing import Dict, Final, List, Iterable, Optional, Union

from code.common.constants import Scenario
from scripts.utils import safe_divide


MLPERF_LOG_PREFIX: Final[str] = ":::MLLOG"

scenario_loadgen_log_keys: Final[Dict[str, str]] = {
    Scenario.Offline: "result_samples_per_second",
    Scenario.Server: "result_scheduled_samples_per_sec",
    Scenario.SingleStream: "result_90.00_percentile_latency_ns",
    Scenario.MultiStream: "result_99.00_percentile_per_query_latency_ns",
}


def perf_metric_from_scenario(system_id: str, scenario: str):
    """
    Return the perf metrics used for MLPerf-I comparison from the scenario and system_id
    """
    if "MaxQ" in system_id:
        if scenario in [Scenario.Offline, Scenario.Server]:
            return "qps_per_avg_watt"
        elif scenario in [Scenario.SingleStream, Scenario.MultiStream]:
            return "joules_per_stream"
        else:
            raise NotImplementedError(f"No such scenario {scenario}.")
    else:
        return scenario_loadgen_log_keys[Scenario.get_match(scenario)]


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
            # Replace some characters violating JSON standard
            log_entries.append(json.loads(buf.replace('\x00', '').replace('\u0000', '')))

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


def from_timestamp(timestamp, local_time=False):
    result = datetime.datetime.strptime(timestamp, "%m-%d-%Y %H:%M:%S.%f")
    if local_time:
        timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        result -= timezone.utcoffset(None)
    return result


def get_perf_summary(log_dir):
    """
    Returns the contents of perf_harness_summary.json as a dict with structure:

    {
        <config_name>: {
            <benchmark name>: <result string>,
            ...
        },
        ...
    }
    """
    summary_path = os.path.join(log_dir, "perf_harness_summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        results = json.load(f)
    return results


def get_acc_summary(log_dir):
    """
    Returns the contents of accuracy_summary.json as a dict with structure:

    {
        <config_name>: {
            <benchmark name>: <result string>,
            ...
        },
        ...
    }
    """
    summary_path = os.path.join(log_dir, "accuracy_summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        results = json.load(f)
    return results


def get_power_vals(lines, power_begin, power_end):
    power_vals = []
    for line in lines:
        data = line.split(",")

        if len(data) < 4:
            continue

        timestamp = data[1]
        watts = float(data[3])
        curr_time = from_timestamp(timestamp)

        if power_begin <= curr_time and curr_time <= power_end:
            power_vals.append(watts)

    return power_vals


def get_power_summary(log_dir):
    """
    Returns a list of power wattages from between power_begin and power_end for the spl.txt located in log_dir. Note
    that this does not support directories where there are multiple power harness runs in a single log_dir. Running
    multiple power harnesses in a single harness run is not advised or officially supported.
    """
    spl_path = os.path.join(log_dir, "spl.txt")
    if not os.path.exists(spl_path):
        return None

    detail_log_path = None
    if log_dir.startswith("results"):
        # In results, mlperf_log_summary would be in the same directory as spl.txt at:
        # results/<system name>/<benchmark>/<scenario>/run_1/mlperf_log_detail.txt
        detail_log_path = os.path.join(log_dir, "mlperf_log_detail.txt")
    else:
        # In a harness run log directory, mlperf_log_detail would be at:
        # build/power_logs/<timestamp>/run_1/<system name>/<benchmark>/<scenario>/mlperf_log_detail.txt
        # spl would be in: build/power_logs/<timestamp>/run_1/spl.txt
        detail_logs = glob.glob(os.path.join(log_dir, "**", "mlperf_log_detail.txt"), recursive=True)
        if len(detail_logs) == 0:
            raise RuntimeError("Could not find detail logs for power run!")
        elif len(detail_logs) > 1:
            raise RuntimeError("Power harness run contains multiple benchmark-scenario runs!")
        else:
            detail_log_path = detail_logs[0]

    if detail_log_path is None or not os.path.exists(detail_log_path):
        return None

    power_times = from_loadgen_by_keys(os.path.dirname(detail_log_path), ["power_begin", "power_end"])
    power_begin = from_timestamp(power_times["power_begin"], False)
    power_end = from_timestamp(power_times["power_end"], False)

    # Read power metrics from spl.txt
    with open(os.path.join(log_dir, "spl.txt")) as f:
        lines = f.read().split("\n")

    power_vals = get_power_vals(lines, power_begin, power_end)

    if len(power_vals) == 0:
        print("WARNING: No power samples in the window can be found. Try parsing the log again with timezone shift.")
        power_begin = from_timestamp(power_times["power_begin"], True)
        power_end = from_timestamp(power_times["power_end"], True)
        power_vals = get_power_vals(lines, power_begin, power_end)

    return power_vals


def read_loadgen_result_by_key(log_dir: str, key: str, extra_keys: Optional[Iterable[str]] = None) -> float:
    """
    Read the target stats from the loadgen results, queried by the key provided.

    Returns:
        perf_number (float): the queried value in float. 0.0 if the perf value doesn't exist
    """
    loadgen_keys = set() if extra_keys is None else set(extra_keys)
    loadgen_keys.add(key)
    result = from_loadgen_by_keys(log_dir, loadgen_keys)
    if key not in result:
        print("WARNING: Could not find perf value in file: " + entry + ". Using 0")
        perf_number = 0.0
    else:
        perf_number = float(result[key])

    return perf_number


def construct_result_log_dir(results_dir: str, system_id: str, benchmark: str, scenario: str) -> str:
    """
    Construct the path of the loadgen log given the system/benchmark/scenario as input
    Note: this is only guaranteed to work in results/ directory, since it obeys the directory structure.
    """
    return os.path.join(results_dir, system_id, benchmark, scenario, "performance", "run_1")


def extract_single_power_result(log_dir: str, system_id: str, benchmark: str, scenario: str):
    """
    Given the query entries, return the average power number calculated from the logs
    If not found, return None
    """
    if not os.path.exists(os.path.join(log_dir, "mlperf_log_detail.txt")):
        raise FileNotFoundError(f"Cannot find perf logs for {system_id}/{benchmark}/{scenario} at {log_dir}")

    avg_power = None
    power_vals = get_power_summary(log_dir)
    if power_vals != None:
        if len(power_vals) == 0:
            print("WARNING: Found power measurements in {} but no samples were within test window".format(log_dir))
        else:
            avg_power = sum(power_vals) / len(power_vals)

    return avg_power


def extract_mlperf_result_from_log(log_dir: str, system_id: str, benchmark: str, scenario: str) -> (float, str):
    """
    Given the query entries, return the perf number calculated from the keys in the log.
    The MaxP Perf number should be:
        - Offline: actual QPS 
        - Server: sceduled QPS 
        - SingleStream: 90-percentile latency
        - MultiStream: 99-percentile per-query latency
    The MaxQ perf number is:
        - Offline/Server: qps per average watt
        - SingleStream/MultiStream: joules per stream (calculated by average_power * mean_latency)
    Perf results will be converted if it needs to be in any other format (e.g. latency)

    Params:
        log_dir (str): the directory which contains the results for the particular run
        system_id (str): the system_id of the run
        benchmark (str): the benchmark of the run
        scenario (str): the scenario of the run

    Returns:
        perf (float): the requested perf number
        metric (str): the perf metric in string 
    """
    # Raise error if the log is not found
    if not os.path.exists(os.path.join(log_dir, "mlperf_log_detail.txt")):
        # FIXME: if this is first time getting the log, there is no such file to extract
        #        shouldn't it just return (None, None) if so?
        raise FileNotFoundError(f"Cannot find perf logs for {system_id}/{benchmark}/{scenario} at {log_dir}")

    metric = perf_metric_from_scenario(system_id, scenario)

    scenario_in_log = from_loadgen_by_keys(log_dir, ["requested_scenario"])["requested_scenario"]
    if scenario != scenario_in_log:
        # If the scenario requested is different from scenario in log, we need to convert the perf metric
        print(f"WARNING: requesting {scenario}, but {scenario_in_log} found in the log in {log_dir}")

    if "MaxQ" not in system_id:
        # For MaxP, we simply use the scenario metric as key to query the log
        scenario_key = scenario_loadgen_log_keys[Scenario.get_match(scenario_in_log)]
        perf_number = read_loadgen_result_by_key(log_dir, scenario_key)

        # Convert the metric if we are using the log for a different scenario
        if perf_number != 0.0 and scenario != scenario_in_log:
            print(f"WARNING: requesting {scenario}, but {scenario_in_log} found in the log in {log_dir}")
            if scenario == "Offline" and scenario_in_log == "SingleStream":
                print("WARNING: Using SingleStream logs for Offline Scenario. Converting metric to QPS")
                perf_number = 1 / (perf_number / (10 ** 9))
            else:
                raise NotImplementedError(f"ERROR: Cannot convert {scenario_in_log} in the log to {scenario}")

        return perf_number, metric
    else:
        # For MaxQ, Offline and server uses "qps_per_avg_watt"
        # singlestream and multistream uses "joules_per_stream"

        avg_power = extract_single_power_result(log_dir, system_id, benchmark, scenario)
        if scenario in [Scenario.Offline, Scenario.Server]:
            scenario_key = scenario_loadgen_log_keys[Scenario.get_match(scenario_in_log)]
            perf_number = read_loadgen_result_by_key(log_dir, scenario_key)

            if scenario != scenario_in_log:
                if scenario == "Offline" and scenario_in_log == "SingleStream":
                    print("WARNING: Using SingleStream logs for Offline Scenario. Converting metric to QPS")
                    qps = 1 / (perf_number / (10 ** 9))
                else:
                    raise NotImplementedError(f"ERROR: Cannot convert {scenario_in_log} in the log to {scenario}")
            else:
                qps = perf_number

            return safe_divide(qps, avg_power), metric
        elif scenario in [Scenario.SingleStream, Scenario.MultiStream]:
            mean_latency_key = "result_mean_latency_ns"
            mean_latency = read_loadgen_result_by_key(log_dir, mean_latency_key)

            return safe_divide(mean_latency, 10 ** 9) * avg_power, metric
        else:
            raise NotImplementedError(f"Requested scenario {scenario} is not valid.")


def get_perf_regression_ratio(prev_result: float, results_dir: str, system_id: str, benchmark: str, scenario: str):
    """
    Given the perf as input, compare against the current perf in the results dir and return the ratio of the current results.

    Returns:
        Tuple of (ratio, current_result)
    """
    log_dir = construct_result_log_dir(results_dir, system_id, benchmark, scenario)
    current_result, _ = extract_mlperf_result_from_log(log_dir, system_id, benchmark, scenario)
    if current_result is None:
        return 1.0, 0.0

    # For MaxP, we use QPS and percentile latency comparison.
    if scenario in [Scenario.Offline, Scenario.Server]:
        return safe_divide(prev_result, current_result), current_result
    elif scenario in [Scenario.SingleStream, Scenario.MultiStream]:
        return safe_divide(current_result, prev_result), current_result
    else:
        raise Exception(f"{scenario} is not a valid scenario")
