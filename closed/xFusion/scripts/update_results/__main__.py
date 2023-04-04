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
import os

from scripts.update_results.utils import find_result_candidates, find_best_candidate
from scripts.update_results.stage import stage_result_candidates


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        help="Specifies the directory containing the logs",
        default="build/logs"
    )
    parser.add_argument(
        "--power_log_dir",
        help="Specifies the directory containing the power logs",
        default="build/power_logs"
    )
    parser.add_argument(
        "--ignore_power",
        help="If set, ignore power logs when updating results",
        action="store_true"
    )
    parser.add_argument(
        "--staging_dir",
        help="Specifies the directory to output the results/ entries to",
        default=os.path.join("build/submission-staging/closed",
                             os.environ.get("SUBMITTER", "xFusion"),
                             "results")
    )
    parser.add_argument(
        "--results_dir",
        help="Specifies the directory committed results are stored",
        default=os.path.join("build/artifacts/closed",
                             os.environ.get("SUBMITTER", "xFusion"),
                             "results")
    )
    parser.add_argument(
        "--dry_run",
        help="Don't actually copy files, just log the actions taken.",
        action="store_true"
    )
    return parser.parse_args()


def main(args):
    logs = find_result_candidates(args.log_dir)
    if not args.ignore_power:
        logs.extend(find_result_candidates(args.power_log_dir))
    candidates = find_best_candidate(logs, result_dir=args.results_dir)

    stage_result_candidates(candidates, args.staging_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main(get_args())
