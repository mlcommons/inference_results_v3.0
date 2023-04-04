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
import glob
import json
import os

import code.common.arguments as common_args
from code.common.utils import Tree
from code.common.log_parser import get_power_summary
from scripts.utils import SimpleLogger

logger = SimpleLogger(indent_size=4, prefix="")


def get_result_summaries(log_dir):
    """
    Returns a summary of the results in a particular base log directory. Returns as a dict with the following structure:

        {
            <config_name>: {
                <benchmark name>: {
                    "performance": <result string>,
                    "accuracy": <result string>,
                },
    """
    summary = Tree()
    metadata_files = glob.glob(os.path.join(log_dir, "**", "metadata.json"), recursive=True)
    for fname in metadata_files:
        with open(fname) as f:
            _dat = json.load(f)
        keyspace = [_dat["config_name"], _dat["benchmark_full"]]
        if _dat["test_mode"] == "PerformanceOnly":
            keyspace.append("performance")
        elif _dat["test_mode"] == "AccuracyOnly":
            keyspace.append("accuracy")
        summary.insert(keyspace, _dat["summary_string"])
    return summary.tree  # Return the internal dict instead of the Tree object


def main():
    log_dir = common_args.parse_args(["log_dir"])["log_dir"]

    result_summaries = get_result_summaries(log_dir)
    logger.log(f"\n{'='*24} Result summaries: {'='*24}\n")
    for config_name in result_summaries:
        logger.log(f"{config_name}:")
        logger.inc_indent_level()
        for benchmark in result_summaries[config_name]:
            logger.log(f"{benchmark}:")
            logger.inc_indent_level()
            for k, v in result_summaries[config_name][benchmark].items():
                logger.log(f"{k}: {v}")
            logger.dec_indent_level()
        logger.dec_indent_level()
        logger.log("")

    # If this is a power run, we should print out the average power
    power_vals = get_power_summary(log_dir)
    if power_vals != None:
        logger.log(f"\n{'='*24} Power results: {'='*24}\n")
        for config_name in result_summaries:
            logger.log(f"{config_name}:")
            logger.inc_indent_level()
            for benchmark in result_summaries[config_name]:
                if len(power_vals) > 0:
                    avg_power = sum(power_vals) / len(power_vals)
                    logger.log(f"{benchmark}: avg power under load: {avg_power:.2f}W with {len(power_vals)} power samples")
                else:
                    logger.log(f"{benchmark}: cannot find any power samples in the test window. Is the timezone setting correct?")
            logger.dec_indent_level()
            logger.log("")


if __name__ == "__main__":
    main()
