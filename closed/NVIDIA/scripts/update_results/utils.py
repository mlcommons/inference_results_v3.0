#! /usr/bin/env python3
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

import copy
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from code.common.constants import AuditTest
from code.common.utils import Tree
from code.common.log_parser import extract_mlperf_result_from_log, get_perf_regression_ratio
from scripts.utils import (
    SortingCriteria,
    SimpleLogger,
    get_system_type,
    safe_divide,
)


logger = SimpleLogger()


def find_result_candidates(log_dir: str) -> List[str]:
    """Searches in `log_dir` for subdirectories containing required Loadgen logs for a valid MLPerf Inference run:
        - mlperf_log_accuracy.json
        - mlperf_log_summary.txt
        - mlperf_log_trace.json
        - metadata.json

    Returns:
        A list of all subdirectories of `log_dir` which contain all required files for MLPerf Inference run
    """
    result_glob = os.path.join(log_dir, "**", "mlperf_log_detail.txt")
    logger.log(f"Looking for logs in {result_glob}")
    found_detail_logs = glob.glob(result_glob, recursive=True)
    logger.log(f"Found {len(found_detail_logs)} mlperf_log_detail.txt entries")

    # Verify that each mlperf_log_detail has corresponding files
    validated = []
    logger.inc_indent_level()
    for candidate_detail_log in found_detail_logs:
        basedir = os.path.dirname(candidate_detail_log)

        valid = True
        for fname in ["mlperf_log_accuracy.json", "mlperf_log_summary.txt", "mlperf_log_trace.json", "metadata.json"]:
            if not os.path.exists(os.path.join(basedir, fname)):
                logger.log(f"WARNING: {basedir} does not include mandatory file {fname}")
                valid = False
        if valid:
            validated.append(basedir)
    logger.dec_indent_level()
    return validated


def is_power_run(logdir: str, md: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Checks whether or not the run is a power submission run. If True, additionally returns if it is a power ranging
    run or performance run.

    Args:
        logdir: The MLPerf Inference run log directory to check
        md (Optional[Dict[str, Any]]): Pre-loaded metadata for an MLPerf Inference harness run. If not set, reads from
                                       `<logdir>/metadata.json`. (Default: None)

    Returns:
        bool: Whether or not this is a power run
        str: If it is a power run, 'ranging' or 'run_1', otherwise None.
    """
    if md is None:
        with open(os.path.join(logdir, "metadata.json")) as f:
            md = json.load(f)

    if md["test_mode"] == "PerformanceOnly":
        if "MaxQ" in md["config_name"]:
            # Skip ranging runs for power. On run_1 (actual test), check if ranging exists.
            if "/ranging/" in logdir:
                return True, "ranging"
            elif "/run_1/" in logdir:
                return True, "run_1"
    return False, None


def contains_valid_mlpinf_run(logdir: str, md: Optional[Dict[str, Any]] = None) -> bool:
    """Checks whether or not the run is a valid submission run.

    Args:
        logdir: The MLPerf Inference run log directory to check
        md (Optional[Dict[str, Any]]): Pre-loaded metadata for an MLPerf Inference harness run. If not set, reads from
                                       `<logdir>/metadata.json`. (Default: None)

    Returns:
        bool: Whether or not this run is valid
    """
    if md is None:
        with open(os.path.join(logdir, "metadata.json")) as f:
            md = json.load(f)

    audit_test = md.get("audit_test", None)
    if audit_test is None:
        if md["test_mode"] == "PerformanceOnly":
            valid = md.get("result_validity", "INVALID") == "VALID"
            scenario_constraint = md.get("satisfies_query_constraint", False)
            early_stopping_met = md.get("early_stopping_met", False)
            return valid and (scenario_constraint or early_stopping_met)
        elif md["test_mode"] == "AccuracyOnly":
            return md.get("accuracy_pass", False)
        return False  # Invalid test_mode
    else:  # Audit test just has a shortcut 'success' key
        return md.get("audit_success", False)


def build_candidate_tree(candidate_list: List[str]) -> Tree:
    """Helper method to parse a list of directories containing valid MLPerf Inference runs into an easily indexable
    structure. This method also discards any directories which contain logs for an INVALID MLPerf Inference run (for
    clarification, INVALID here is in the context of the validity / submittability of an MLPerf Inference result).

    Args:
        candidate_list (List[str]): A list of directories which contain all required MLPerf Inference run files

    Returns:
        Tree in the schema:
            system name
            |-> benchmark name
                |-> scenario
                    |-> AccuracyOnly
                    |-> PerformanceOnly
                    |-> power
    """
    # Build a tree of system > benchmark > scenario > runs
    candidates = Tree()
    for c in candidate_list:
        with open(os.path.join(c, "metadata.json")) as f:
            md = json.load(f)

        # Only insert if the runs were valid
        has_power_logs, power_run_type = is_power_run(c, md=md)
        if has_power_logs:
            if power_run_type == "ranging":
                continue
            elif power_run_type == "run_1":
                if c.replace("/run_1/", "/ranging/") not in candidate_list:
                    logger.log(f"No ranging log for {c}, skipping.")
                    continue

        if not contains_valid_mlpinf_run(c, md=md):
            continue

        # Conditionally add based on criteria
        system_name = md["system_name"]
        benchmark = md["benchmark_full"]
        scenario = md["scenario"]
        try:
            system_type = get_system_type(system_name)
        except Exception as e:
            logger.log(f"WARNING: Failed to get system type for '{system_name}'")
            logger.inc_indent_level()
            logger.log(f"Error: {e}")
            logger.log(f"Assume the system is in both Edge and Datacenter")
            logger.dec_indent_level()
            system_type = "datacenter,edge"

        if system_type == "edge":
            if benchmark in ["dlrm-99", "dlrm-99.9", "bert-99.9"]:
                logger.log(f"Skipping benchmark {benchmark} for edge system {system_name}")
                continue
            if scenario == "Server":
                logger.log(f"Skipping scenario {scenario} for edge system {system_name}")
                continue

        if system_type == "datacenter":
            if scenario in ["SingleStream", "MultiStream"]:
                logger.log(f"Skipping scenario {scenario} for datacenter system {system_name}")
                continue

        # Check if the system was an audit test. This will change how the candidates are selected.
        audit_test = md.get("audit_test", None)
        if audit_test is None:
            candidates.insert([md["system_name"], md["benchmark_full"], md["scenario"], md["test_mode"]], c, append=True)
            if has_power_logs:
                candidates.insert([md["system_name"], md["benchmark_full"], md["scenario"], "power"], c, append=True)
        else:
            candidates.insert([md["system_name"], md["benchmark_full"], md["scenario"], audit_test], c, append=True)
    return candidates


def find_best_candidate(candidate_list: List[str], result_dir: Optional[str] = None, regression_threshold: float = 0.97):
    """From a list of directories containing MLPerf Inference logfiles, finds the logs with the best performance results
    for submission. Performance runs will not be considered if there is no corresponding Accuracy run (and vice versa)

    Args:
        candidate_list (List[str]): A list of directories which contain all required MLPerf Inference run files
        result_dir (Optional[str]): Path to root directory of MLPerf Inference submission results to compare against. If
                                    unset, will not check for performance regressions. (Default: None)
        regression_threshold (float): Float between 0.0 and 1.0 (inclusive) representing the minimum allowed
                                                percentage of an existing MLPerf Inference result that can be
                                                considered. (Default: 0.97)

    Returns:
        Tree in the schema:
            system name
            |-> benchmark name
                |-> scenario
                    |-> AccuracyOnly
                    |-> PerformanceOnly
                    |-> power
    """
    logger.log(f"Enumerating candidates...")
    candidates = build_candidate_tree(candidate_list)
    logger.log(f"Found {len(candidates)} submission result candidates")
    for system_name, benchmarks in candidates.tree.items():
        logger.log(f"Enumerating logs for {system_name}")
        logger.inc_indent_level()
        for benchmark, scenarios in benchmarks.items():
            logger.log(f"Enumerating logs for {benchmark}")
            logger.inc_indent_level()
            for scenario, runs in scenarios.items():
                logger.log(f"Enumerating logs for {scenario}")
                logger.inc_indent_level()
                # For DLRM and 3D-UNET, we use the same logs for both high and low accuracy targets. Skip 99% if 99.9% is
                # present.
                if benchmark in ["dlrm-99", "3d-unet-99"]:
                    benchmark_highacc = benchmark + ".9"
                    if benchmark_highacc in benchmarks and scenario in benchmarks[benchmark_highacc]:
                        logger.log(f"Skipping {benchmark} {scenario} result - Using {benchmark_highacc} {scenario} instead")
                        continue

                # Check for audit tests
                for audit_test in AuditTest:
                    audit_name = audit_test.valstr()
                    count = len(runs.get(audit_name, tuple()))
                    logger.log(f"Found {count} entries for Audit Test {audit_name}")
                    if count > 1:
                        logger.log(f"Selecting most recent successful {audit_name} run")
                        runs[audit_name] = runs[audit_name][-1:]

                # Must have valid perf AND accuracy run
                if "PerformanceOnly" not in runs or "AccuracyOnly" not in runs:
                    logger.log("Must have both PerformanceOnly and AccuracyOnly. Pruning result candidates without both.")
                    runs.pop("PerformanceOnly", None)
                    runs.pop("AccuracyOnly", None)
                else:
                    num_perf = len(runs["PerformanceOnly"])
                    num_acc = len(runs["AccuracyOnly"])
                    num_power = len(runs.get("power", tuple()))
                    logger.log(f"Found {num_perf + num_acc} log files")
                    logger.log(f"{num_perf} Performance / {num_acc} Accuracy / {num_power} Power")

                    # Update perf run first. If there's regression we might not update any log.
                    # Note: Since v1.1 the perf_count has always been 1. But we keep this variable here in case we need it in the future.
                    perf_count = 1
                    if num_perf < perf_count:
                        logger.log(f"WARNING: min. {perf_count} PerformanceOnly runs required, found {num_perf}. Skipping.")
                        logger.dec_indent_level()
                        continue
                    elif num_perf > perf_count:
                        logger.log(f"Selecting {perf_count} best PerformanceOnly runs...")

                    sorted_perf_vals = get_sorted_perf_vals(runs["PerformanceOnly"], system_name, benchmark, scenario)
                    runs["PerformanceOnly"] = [k[0] for k in sorted_perf_vals][-perf_count:]
                    num_perf = perf_count

                    # Check best perf for regressions against existing results
                    if result_dir is None:
                        logger.log(f"WARNING: result_dir not specified, skipping perf regression checks.")
                    else:
                        best_perf = sorted_perf_vals[-1][1]
                        try:
                            regression_ratio, current_perf = get_perf_regression_ratio(best_perf, result_dir, system_name, benchmark, scenario)
                            if regression_ratio < regression_threshold:
                                logger.log(f"WARNING: Best found perf is {best_perf}, which is {regression_ratio} of {current_perf} (Current Perf)")
                                # TODO: Allow regressions for updated (later) software stacks
                                logger.log(f"Skipping {system_name}/{benchmark}/{scenario} due to performance regression")
                                logger.dec_indent_level()
                                continue
                        except FileNotFoundError:
                            logger.log(f"No results found under {result_dir}, skipping performance regression checks")

                    # Update accuracy run.
                    if num_acc == 0:
                        logger.log(f"WARNING: Cannot find valid accuracy run. Skipping.")
                        logger.dec_indent_level()
                        continue
                    elif num_acc > 1:
                        logger.log(f"Found {num_acc} accuracy runs - 1 required. Selecting the most recent log.")
                        runs["AccuracyOnly"] = runs["AccuracyOnly"][-1:]
                        num_acc = 1
                logger.dec_indent_level()
            logger.dec_indent_level()
        logger.dec_indent_level()
    return candidates


def get_sorted_perf_vals(log_dirs: list, system_id: str, benchmark: str, scenario: str) -> list:
    """
    Sorts performance runs via a tiebreaker criteria

    Args:
        log_dirs (list): a list of file paths which point to the loadgen logs
        system_id (str): the system name
        benchmark (str): the name of the benchmark
        scenario (str): the scenario of the runs

    Returns:
        A list of tuples (file path, perf number) sorted in the ascending order
    """
    sorting_criteria_by_metric = {
        "result_samples_per_second": SortingCriteria.Higher,
        "result_scheduled_samples_per_sec": SortingCriteria.Higher,
        "result_99.00_percentile_per_query_latency_ns": SortingCriteria.Lower,
        "result_90.00_percentile_latency_ns": SortingCriteria.Lower,
        "qps_per_avg_watt": SortingCriteria.Higher,
        "joules_per_stream": SortingCriteria.Lower,
    }

    perf_vals = []
    for log_dir in log_dirs:
        result, metric, is_strict_match = extract_mlperf_result_from_log(log_dir, system_id, benchmark, scenario, strict_match=False)
        if not is_strict_match:
            continue
        perf_vals.append((log_dir, result))

    sorted_perf_vals = sorted(perf_vals, key=lambda k: k[1],
                              reverse=(sorting_criteria_by_metric[metric] == SortingCriteria.Lower))

    return sorted_perf_vals
