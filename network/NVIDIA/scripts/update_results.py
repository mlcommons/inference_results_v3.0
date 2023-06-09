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

import argparse
import copy
import datetime
import glob
import json
import os
import re
import shutil
import sys

from code.common.constants import Scenario
from code.common.log_parser import extract_mlperf_result_from_log, extract_single_power_result, \
    from_loadgen_by_keys, get_perf_regression_ratio, scenario_loadgen_log_keys
from code.common.utils import Tree
from scripts.utils import SortingCriteria, get_system_type, safe_divide


def safe_copy(input_file, output_file, dry_run=False):
    print("Copy {:} -> {:}".format(input_file, output_file))
    if not dry_run:
        try:
            shutil.copy(input_file, output_file)
        except Exception as e:
            print("Copy failed. Error: {:}".format(e))


def safe_copytree(src_dir, dst_dir, dry_run=False):
    print("Copy {:} -> {:}".format(src_dir, dst_dir))
    if not dry_run:
        try:
            shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(src_dir, dst_dir)
        except Exception as e:
            print("Copytree failed. Error: {:}".format(e))


def get_sorted_perf_vals(perf_file_list: list, system_id: str, benchmark: str, scenario: str) -> list:
    """
    Sorts performance runs via a tiebreaker criteria

    Args:
        perf_file_list (list): a list of file paths which point to the loadgen logs
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
    for perf_file in perf_file_list:
        log_dir = os.path.dirname(perf_file)
        result, metric = extract_mlperf_result_from_log(log_dir, system_id, benchmark, scenario)
        perf_vals.append((perf_file, result))

    sorted_perf_vals = sorted(perf_vals, key=lambda k: k[1],
                              reverse=(sorting_criteria_by_metric[metric] == SortingCriteria.Lower))

    return sorted_perf_vals


def find_valid_runs(input_list, scenario):
    # Check for query constraints documented in https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#scenarios
    QUERY_METRIC_CONSTRAINTS = {
        "Offline": ("effective_samples_per_query", 24576),
        "Server": ("effective_min_query_count", 270336),
        "MultiStream": ("effective_min_query_count", 270336),
        "SingleStream": ("effective_min_query_count", 1024),
    }

    perf_list = []
    perf_power_list = []
    accu_list = []
    for input_file in input_list:
        # Check if this is Accuracy run or Performance run.
        if os.path.getsize(input_file) > 4:
            accu_list.append(input_file)
            continue

        # Check for valid perf run
        log_dir = os.path.dirname(input_file)
        scenario_key = QUERY_METRIC_CONSTRAINTS[scenario][0]
        result = from_loadgen_by_keys(log_dir, ["result_validity", "early_stopping_met", scenario_key])

        is_valid = ("result_validity" in result) and (result["result_validity"] == "VALID")
        satisfies_query_constraint = (scenario_key in result) and (float(result[scenario_key]) >= QUERY_METRIC_CONSTRAINTS[scenario][1])
        early_stopping_met = ("early_stopping_met" in result) and result["early_stopping_met"]
        if is_valid and (satisfies_query_constraint or early_stopping_met):
            perf_list.append(input_file)
            if "power" in log_dir:
                perf_power_list.append(input_file)

    return perf_list, perf_power_list, accu_list


def process_results(args, system_ids, metadata):
    time_now = str(datetime.datetime.utcnow())
    result_id = args.result_id if args.result_id is not None else "manual-{:}".format(time_now)

    for system_id in system_ids:
        try:
            system_type = get_system_type(system_id)
        except Exception as e:
            print("WARNING: Failed to get system type! Error: {:}".format(e))
            print("WARNING: Assume the system is in both Edge and Datacenter.")
            system_type = "datacenter,edge"

        for benchmark in system_ids[system_id]:
            # Skip DLRM and BERT-99.9 for Edge
            if system_type == "edge" and (benchmark.startswith("dlrm") or benchmark == "bert-99.9"):
                print("{:} is an edge system. Skipping {:}".format(system_id, benchmark))
                continue

            # Skip SSD MobileNet for datacenter
            if system_type == "datacenter" and benchmark == "ssd-mobilenet":
                print("{:} is a datacenter system. Skipping {:}".format(system_id, benchmark))
                continue

            for scenario in system_ids[system_id][benchmark]:
                # Skip Server for Edge systems
                if system_type == "edge" and scenario in {"Server"}:
                    print("{:} is an edge system. Skipping Server scenario".format(system_id))
                    continue

                # Skip Single-/Multi-Stream for Datacenter systems
                if system_type == "datacenter" and scenario in {"SingleStream", "MultiStream"}:
                    print("{:} is a datacenter system. Skipping {:} scenario".format(system_id, scenario))
                    continue

                # For DLRM and 3d-UNET, the 99.9% and 99% accuracy targets use the same engines. We use the 99.9% logs for
                # 99% submissions as well, so skip 99% if 99.9% is present.
                if benchmark in {"dlrm-99", "3d-unet-99"}:
                    benchmark_highacc = (benchmark + ".9")
                    if benchmark_highacc in system_ids[system_id] and scenario in system_ids[system_id][benchmark_highacc]:
                        print("We use the same logs for {:} and {:}. Skipping {:} in {:} scenario.".format(benchmark, benchmark_highacc, benchmark, scenario))
                        continue

                print(">>>>>>>> Processing {:}-{:}-{:} <<<<<<<<".format(system_id, benchmark, scenario))
                if args.assume_compliance:
                    # Compliance runs are more or less structured correctly already
                    runs = system_ids.get([system_id, benchmark, scenario])
                    if type(runs) is not dict:
                        raise Exception("Expected subtree under system_id/benchmark/scenario for compliance directory structure.")

                    for test_id in runs:
                        # [input_dir]/test_id/system_id/benchmark/scenario/test_id/[performance|accuracy]
                        test_root = None
                        for test_mode in runs[test_id]:
                            entry = system_ids.get([system_id, benchmark, scenario, test_id, test_mode])
                            if len(entry) != 1:
                                raise Exception("Expected unique entry. Got {:} hits instead.".format(len(entry)))
                            src_dir = entry[0]
                            dst_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, test_id, test_mode)
                            if test_root is None:
                                test_root = "/".join(src_dir.split("/")[:-1])
                            elif "/".join(src_dir.split("/")[:-1]) != test_root:
                                raise Exception("Accuracy and performance runs are not in the same directory")
                            if test_mode == "performance":
                                is_empty = len(os.listdir(src_dir)) == 0 or len(os.listdir(os.path.join(src_dir, "run_1"))) == 0
                            else:
                                is_empty = len(os.listdir(src_dir)) == 0
                            if not is_empty:
                                safe_copytree(src_dir, dst_dir, args.dry_run)
                            metadata.insert([system_id, benchmark, scenario, "compliance", test_id, test_mode, "result_export_timestamp"], time_now)
                            metadata.insert([system_id, benchmark, scenario, "compliance", test_id, test_mode, "result_id"], result_id)
                        for verify_file in ["verify_accuracy.txt", "verify_performance.txt"]:
                            src_file = os.path.join(test_root, verify_file)
                            dst_file = os.path.join(args.output_dir, system_id, benchmark, scenario, test_id, verify_file)
                            if not os.path.exists(src_file):
                                continue
                            safe_copy(src_file, dst_file, args.dry_run)
                            metadata.insert([system_id, benchmark, scenario, "compliance", test_id, verify_file, "result_export_timestamp"], time_now)
                            metadata.insert([system_id, benchmark, scenario, "compliance", test_id, verify_file, "result_id"], result_id)

                    # For DLRM and 3d-UNET, the 99.9% and 99% accuracy targets use the same engines. We use the 99.9% logs for
                    # 99% submissions as well, so copy the logs of 99.9% to 90%.
                    if benchmark in {"dlrm-99.9", "3d-unet-99.9"}:
                        benchmark_lowacc = benchmark.replace(".9", "")
                        print("Copying logs for {:} {:} to {:} {:} since they share the same engines.".format(benchmark, scenario, benchmark_lowacc, scenario))
                        src_dir = os.path.join(args.output_dir, system_id, benchmark, scenario)
                        dst_dir = os.path.join(args.output_dir, system_id, benchmark_lowacc, scenario)
                        safe_copytree(src_dir, dst_dir, args.dry_run)
                        metadata.insert([system_id, benchmark_lowacc, scenario, "compliance"], copy.deepcopy(metadata.get([system_id, benchmark, scenario, "compliance"])))

                    continue

                input_list = system_ids[system_id][benchmark][scenario]
                print("Found {:} log files".format(len(input_list)))

                perf_list, perf_power_list, accu_list = find_valid_runs(input_list, scenario)

                print("\t{:} perf logs (among them {:} has power logs)".format(len(perf_list), len(perf_power_list)))
                print("\t{:} acc logs".format(len(accu_list)))

                # Update perf run first. If there's regression we might not update any log.
                # Note: Since v1.1 the perf_count has always been 1. But we keep this variable here in case we need it in the future.
                perf_count = 1
                existing_count = metadata.get([system_id, benchmark, scenario, "performance", "count"], default=0)
                full_result_id = metadata.get([system_id, benchmark, scenario, "performance", "result_id"], default=["" for i in range(perf_count)])
                full_results_export_timestamp = metadata.get([system_id, benchmark, scenario, "performance", "results_export_timestamp"], default=["" for i in range(perf_count)])
                if len(perf_list) < perf_count:
                    print("WARNING: Cannot find enough passing perf runs. Only found {:d} runs. Skipping".format(len(perf_list)))
                    if args.abort_insufficient_runs:
                        return
                    else:
                        continue
                elif len(perf_list) > perf_count:
                    print("WARNING: Found {:d} passing perf runs, which is more than needed. Choosing the highest perf one(s).".format(len(perf_list)))

                # Sort the perf list and extract the best performing one(s)
                sorted_perf_vals = get_sorted_perf_vals(perf_list, system_id, benchmark, scenario)
                perf_list = [k[0] for k in sorted_perf_vals][-perf_count:]

                # Check if the best perf is regressing against the current results
                best_perf = sorted_perf_vals[-1][1]
                regression_ratio, current_perf = get_perf_regression_ratio(best_perf, args.output_dir, system_id, benchmark, scenario)
                if regression_ratio <= args.strict_regression_threshold:
                    print(f"WARNING: {system_id}/{benchmark}/{scenario} has {scenario_loadgen_log_keys[Scenario.get_match(scenario)]} "
                          f"of {current_perf:.0f}, but trying to update with "
                          f"{best_perf:.0f} which is only {regression_ratio:.2f} of the current perf.")
                    if not args.force_update:
                        print(f"Runs below {args.strict_regression_threshold} of the current results will not update. Skipping this result.")
                        continue
                    else:
                        print(f"Since args.force_update is set, this result will still be updated.")

                # Accuracy log count is always 1.
                metadata.insert([system_id, benchmark, scenario, "accuracy", "count"], 1)

                # Update accuracy run
                if len(accu_list) == 0:
                    print("WARNING: Cannot find valid accuracy run.")

                    if args.abort_missing_accuracy:
                        return
                else:
                    if len(accu_list) > 1:
                        print("WARNING: Found {:d} accuracy runs, which is more than needed. Empirically choose the last one.".format(len(accu_list)))
                        print(accu_list)
                    output_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, "accuracy")
                    if not args.dry_run:
                        os.makedirs(output_dir, exist_ok=True)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = accu_list[-1].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        safe_copy(input_file, output_file, args.dry_run)

                    input_file = os.path.join(os.path.dirname(accu_list[-1]), "accuracy.txt")
                    output_file = os.path.join(output_dir, "accuracy.txt")
                    safe_copy(input_file, output_file, args.dry_run)

                    # TODO: submission checker doesn't allow extra files in results/ directory yet.
                    # input_stdout_file = os.path.join(os.path.dirname(accu_list[-1]), "..", "..", "..", "stdout.txt")
                    # output_stdout_file = os.path.join(output_dir, "stdout.txt")
                    # safe_copy(input_stdout_file, output_stdout_file, args.dry_run)

                    metadata.insert([system_id, benchmark, scenario, "accuracy", "results_export_timestamp"], time_now)
                    metadata.insert([system_id, benchmark, scenario, "accuracy", "result_id"], result_id)

                # Update the performance
                starting_idx = metadata.get([system_id, benchmark, scenario, "performance", "last_updated"], default=0)
                # Starting idx is in range 1..perf_count, whereas actual indices are 0..perf_count-1. We wish the
                # first index we modify to be the one after Starting idx, so taking (N mod perf_count) works.
                starting_idx = starting_idx % perf_count

                for run_idx in range(0, len(perf_list)):
                    run_num = ((run_idx + starting_idx) % perf_count) + 1
                    output_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, "performance", "run_{:d}".format(run_num))
                    if not args.dry_run:
                        os.makedirs(output_dir, exist_ok=True)
                    for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                        input_file = perf_list[run_idx].replace("_accuracy.json", suffix)
                        output_file = os.path.join(output_dir, "mlperf_log{:}".format(suffix))
                        safe_copy(input_file, output_file, args.dry_run)

                    has_power = "power" in perf_list[run_idx]

                    # TODO: submission checker doesn't allow extra files in results/ directory yet.
                    # if has_power:
                    #     input_stdout_file = os.path.join(os.path.dirname(perf_list[run_idx]), "..", "..", "..", "..", "stdout.txt")
                    # else:
                    #     input_stdout_file = os.path.join(os.path.dirname(perf_list[run_idx]), "..", "..", "..", "stdout.txt")
                    # output_stdout_file = os.path.join(output_dir, "stdout.txt")
                    # safe_copy(input_stdout_file, output_stdout_file, args.dry_run)

                    # Handle power logs
                    if not args.ignore_power and has_power:
                        perf_dir = os.path.join(args.output_dir, system_id, benchmark, scenario, "performance")
                        input_file_dir = os.path.dirname(perf_list[run_idx])
                        power_base_dir = input_file_dir[:input_file_dir.find("run_1") - 1]
                        power_input_dir = os.path.join(power_base_dir, "power")
                        power_output_dir = os.path.join(perf_dir, "power")
                        ranging_output_dir = os.path.join(perf_dir, "ranging")
                        # Copy ranging run logs
                        if not args.dry_run:
                            os.makedirs(ranging_output_dir, exist_ok=True)
                        for suffix in ["_accuracy.json", "_detail.txt", "_summary.txt"]:
                            input_file = perf_list[run_idx].replace("run_1", "ranging").replace("_accuracy.json", suffix)
                            output_file = os.path.join(ranging_output_dir, "mlperf_log{:}".format(suffix))
                            safe_copy(input_file, output_file, args.dry_run)
                        # Copy testing run and ranging run spl.txt file
                        input_file = os.path.join(power_base_dir, "run_1", "spl.txt")
                        output_file = os.path.join(output_dir, "spl.txt")
                        safe_copy(input_file, output_file, args.dry_run)
                        input_file = os.path.join(power_base_dir, "ranging", "spl.txt")
                        output_file = os.path.join(ranging_output_dir, "spl.txt")
                        safe_copy(input_file, output_file, args.dry_run)
                        # Copy other required power files
                        if not args.dry_run:
                            os.makedirs(power_output_dir, exist_ok=True)
                        for power_file in ["client.json", "client.log", "ptd_logs.txt", "server.json", "server.log"]:
                            input_file = os.path.join(power_input_dir, power_file)
                            output_file = os.path.join(power_output_dir, power_file)
                            safe_copy(input_file, output_file, args.dry_run)

                    metadata.insert([system_id, benchmark, scenario, "performance", "last_updated"], run_num)
                    full_result_id[run_num - 1] = result_id
                    full_results_export_timestamp[run_num - 1] = time_now

                metadata.insert([system_id, benchmark, scenario, "performance", "count"], min(existing_count + len(perf_list), perf_count))
                metadata.insert([system_id, benchmark, scenario, "performance", "result_id"], full_result_id)
                metadata.insert([system_id, benchmark, scenario, "performance", "results_export_timestamp"], full_results_export_timestamp)

                # For DLRM and 3d-UNET, the 99.9% and 99% accuracy targets use the same engines. We use the 99.9% logs for
                # 99% submissions as well, so copy the logs of 99.9% to 90%.
                if benchmark in {"dlrm-99.9", "3d-unet-99.9"}:
                    benchmark_lowacc = benchmark.replace(".9", "")
                    print("Copying logs for {:} {:} to {:} {:} since they share the same engines.".format(benchmark, scenario, benchmark_lowacc, scenario))
                    src_dir = os.path.join(args.output_dir, system_id, benchmark, scenario)
                    dst_dir = os.path.join(args.output_dir, system_id, benchmark_lowacc, scenario)
                    safe_copytree(src_dir, dst_dir, args.dry_run)
                    metadata.insert([system_id, benchmark_lowacc, scenario, "accuracy"], copy.deepcopy(metadata.get([system_id, benchmark, scenario, "accuracy"])))
                    metadata.insert([system_id, benchmark_lowacc, scenario, "performance"], copy.deepcopy(metadata.get([system_id, benchmark, scenario, "performance"])))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-d",
        help="Specifies the directory containing the logs.",
        default="build/logs"
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the directory to output the results/ entries to",
        default="results"
    )
    parser.add_argument(
        "--result_id",
        help="Specifies a unique ID to use for this result",
        default=None
    )
    parser.add_argument(
        "--abort_insufficient_runs",
        help="Abort instead if there are not enough perf runs to be considered valid",
        action="store_true"
    )
    parser.add_argument(
        "--abort_missing_accuracy",
        help="Abort instead if there isn't a valid accuracy run",
        action="store_true"
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    parser.add_argument(
        "--metadata_file",
        help="File that stores metadata about these results",
        default="results_metadata.json"
    )
    parser.add_argument(
        "--add_metadata",
        help="Save a field as part of metadata to the results directory. Format period.separated.key:value",
        action="append"
    )
    parser.add_argument(
        "--assume_compliance",
        help="If set, then assume input and output directories have compliance log directory structure.",
        action="store_true"
    )
    parser.add_argument(
        "--ignore_power",
        help="If set, then ignore power logs in build/power_logs.",
        action="store_true"
    )
    parser.add_argument(
        "--input_power_dir",
        help="Specifies the directory containing the power logs.",
        default="build/power_logs"
    )
    parser.add_argument(
        "--force_update",
        help="If set, the results which regress above a certain threshold (set below) will still be updated",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--strict_regression_threshold",
        help="If force_update is not set, the results won't be updated if the new stat is lower than some percentage of the exisiting one",
        default=0.98,
    )
    return parser.parse_args()


def main():
    args = get_args()

    if args.assume_compliance:
        perf_log_glob = os.path.join(args.input_dir, "TEST*", "**", "performance")
        acc_log_glob = os.path.join(args.input_dir, "TEST*", "**", "accuracy")
        perf_logs = glob.glob(perf_log_glob, recursive=True)
        print("Found {:} compliance test perf entries".format(len(perf_logs)))
        acc_logs = glob.glob(acc_log_glob, recursive=True)
        print("Found {:} compliance test acc entries".format(len(acc_logs)))
        all_logs = perf_logs + acc_logs
    else:
        glob_to_logs = os.path.join(args.input_dir, "**", "mlperf_log_accuracy.json")
        print("Looking for logs in {:}".format(glob_to_logs))
        all_logs = glob.glob(glob_to_logs, recursive=True)
        print("Found {:} mlperf_log entries".format(len(all_logs)))
        if not args.ignore_power:
            glob_to_logs = os.path.join(args.input_power_dir, "**", "mlperf_log_accuracy.json")
            print("Looking for logs in {:}".format(glob_to_logs))
            power_logs = glob.glob(glob_to_logs, recursive=True)
            # Do not duplicate run_1 and ranging logs.
            power_logs = [i for i in power_logs if "run_1" in i]
            print("Found {:} mlperf_log entries".format(len(power_logs)))
            all_logs.extend(power_logs)

    # Loop through input_list to find all the system_ids
    system_ids = Tree()
    for entry in all_logs:
        parts = entry.split("/")
        if args.assume_compliance:
            # [input_dir]/test_id/system_id/benchmark/scenario/test_id/[performance|accuracy]
            system_id = parts[-5]
            benchmark = parts[-4]
            scenario = parts[-3]
            test_id = parts[-2]
            test_mode = parts[-1]
            system_ids.insert([system_id, benchmark, scenario, test_id, test_mode], entry, append=True)
        else:
            # normal logs: [input_dir]/<timestamp>/system_id/benchmark/scenario/*.json
            # power logs: [input_power_dir]/<timestamp>/run_1/system_id/benchmark/scenario/*.json
            system_id = parts[-4]
            benchmark = parts[-3]
            scenario = parts[-2]
            system_ids.insert([system_id, benchmark, scenario], entry, append=True)

    metadata = None
    if os.path.exists(args.metadata_file):
        with open(args.metadata_file) as f:
            metadata = json.load(f)
    metadata = Tree(starting_val=metadata)

    process_results(args, system_ids, metadata)

    # Write out custom metadata
    if args.add_metadata:
        for md in args.add_metadata:
            tmp = md.split(":")
            if len(tmp) != 2:
                print("WARNING: Invalid metadata \"{:}\"".format(md))
                continue
            keyspace = tmp[0].split(".")
            value = tmp[1]
            metadata.insert(keyspace, value)

    if not args.dry_run:
        with open(args.metadata_file, 'w') as f:
            json.dump(metadata.tree, f, indent=4, sort_keys=True)
    else:
        with open('/tmp/results_metadata.json', 'w') as f:
            json.dump(metadata.tree, f, indent=4, sort_keys=True)
            print("Metadata is dumped to temp file /tmp/results_metadata.json")

    print("Done!")


if __name__ == '__main__':
    main()
