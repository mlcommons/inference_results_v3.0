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


import os
from typing import List, Optional

from code.common.constants import AuditTest
from code.common.utils import Tree
from scripts.utils import safe_copy, safe_copytree
from scripts.update_results.utils import logger


def copy_mlpinf_run(src_dir: str, dst_dir: str, file_list: List[str], dry_run: bool = False):
    """Copies MLPerf Inference run logs from `src_dir` to `dst_dir`.

    Args:
        src_dir (str): The path of the directory containing the logs to copy
        dst_dir (str): The path to copy logs into
        file_list (List[str]): The filenames (base names) to copy from `src_dir`
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """
    os.makedirs(dst_dir, exist_ok=True)
    for fname in file_list:
        _src = os.path.join(src_dir, fname)
        _dst = os.path.join(dst_dir, fname)
        safe_copy(_src, _dst, dry_run=dry_run, logger=logger)


def copy_performance_run(src_dir: str, output_dir: str, run_num: int, has_power: bool = False, dry_run: bool = False):
    """Helper method to copy an MLPerf Inference performance run.

    Args:
        src_dir (str): The path of the directory containing the logs to copy
        dst_dir (str): The path to copy logs into
        run_num (int): The (1-indexed) index of the performance run in the MLPerf Inference submission. Since v1.0,
                       required_performance_count=1, so this should always be 1.
        has_power (bool): Whether or not the performance run in `src_dir` contains power logs (Default: False)
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """
    copy_mlpinf_run(src_dir,
                    output_dir,
                    ["mlperf_log_accuracy.json",
                     "mlperf_log_detail.txt",
                     "mlperf_log_summary.txt",
                     "metadata.json"],
                    dry_run=dry_run)
    # Copy power if it exists
    if has_power:
        # Copy spl.txt into perf run
        copy_mlpinf_run(src_dir,
                        output_dir,
                        ["spl.txt"],
                        dry_run=dry_run)
        # Copy ranging runs
        copy_mlpinf_run(src_dir.replace(f"run_{run_num}", "ranging"),
                        os.path.join(os.path.dirname(output_dir), "ranging"),
                        ["mlperf_log_accuracy.json",
                         "mlperf_log_detail.txt",
                         "mlperf_log_summary.txt",
                         "spl.txt"],
                        dry_run=dry_run)
        # Copy power logs
        # Note - 'power' directory is located in the same directory as 'run_1' and 'ranging'
        copy_mlpinf_run(os.path.join(src_dir[:src_dir.find(f"run_{run_num}") - 1], "power"),
                        os.path.join(os.path.dirname(output_dir), "power"),
                        ["client.json",
                         "client.log",
                         "ptd_logs.txt",
                         "server.json",
                         "server.log"],
                        dry_run=dry_run)


def stage_result_candidates(candidates: Tree, staging_dir: str, dry_run: bool = False):
    """Copies the required files from the result candidates into a staging directory.

    Args:
        candidates (Tree): Tree in the schema:
                           system name
                           |-> benchmark name
                               |-> scenario
                                   |-> AccuracyOnly
                                   |-> PerformanceOnly
                                   |-> power
        staging_dir (str): The directory to copy result candidate's logs into
        dry_run (bool): Whether or not to actually perform filewrites (Default: False)
    """

    logger.log(f"Copying candidate log files into {staging_dir}")
    for system_name, benchmarks in candidates.tree.items():
        logger.log(f"Copying logs for {system_name}")
        logger.inc_indent_level()
        for benchmark, scenarios in benchmarks.items():
            logger.log(f"Copying logs for {benchmark}")
            logger.inc_indent_level()
            for scenario, runs in scenarios.items():
                logger.log(f"Copying logs for {scenario}")
                if "PerformanceOnly" in runs and "AccuracyOnly" in runs:
                    # Copy perf logs
                    for i in range(len(runs["PerformanceOnly"])):
                        output_dir = os.path.join(staging_dir, system_name, benchmark, scenario, "performance", f"run_{i+1}")
                        copy_performance_run(runs["PerformanceOnly"][i],
                                             output_dir,
                                             i + 1,
                                             has_power=(runs["PerformanceOnly"][i] in runs.get("power", tuple())),
                                             dry_run=dry_run)

                    # Copy accuracy logs
                    output_dir = os.path.join(staging_dir, system_name, benchmark, scenario, "accuracy")
                    copy_mlpinf_run(runs["AccuracyOnly"][0],
                                    output_dir,
                                    ["mlperf_log_accuracy.json",
                                     "mlperf_log_detail.txt",
                                     "mlperf_log_summary.txt",
                                     "accuracy.txt",
                                     "metadata.json"],
                                    dry_run=dry_run)

                # Copy audit test logs
                for audit_test in AuditTest:
                    audit_name = audit_test.valstr()
                    if audit_name not in runs:
                        continue
                    output_dir = os.path.join(staging_dir, system_name, benchmark, scenario, audit_name)
                    # Directory structure is already correct - copy directly
                    safe_copytree(os.path.join(runs[audit_name][0], audit_name),
                                  output_dir,
                                  logger=logger,
                                  dry_run=dry_run)
                    # Copy the metadata
                    copy_mlpinf_run(runs[audit_name][0],
                                    output_dir,
                                    ["metadata.json"],
                                    dry_run=dry_run)
            logger.dec_indent_level()

            # For DLRM and 3D-UNET, our low and high accuracy target submissions are exactly the same.
            if benchmark in {"dlrm-99.9", "3d-unet-99.9"}:
                logger.log("Copying high accuracy logs for low accuracy target submission")
                input_dir = os.path.join(staging_dir, system_name, benchmark)
                output_dir = os.path.join(staging_dir, system_name, benchmark[:-2])
                safe_copytree(input_dir, output_dir, logger=logger, dry_run=dry_run)
        logger.dec_indent_level()
