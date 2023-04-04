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

import argparse
import importlib.util
import os
import pathlib
import pprint
import re
import shutil


__doc__ = """This script exports the project into a submittable directory structure for MLPerf Inference submission.

Before you run this script:

    IF YOU ARE AN INTERNAL NVIDIA USER:
        Copy the result artifacts from the result artifacts repo into the main code repository. From closed/NVIDIA:
            cp -R build/artifacts/closed/NVIDIA/measurements . && \
                cp -R build/artifacts/closed/NVIDIA/results . && \
                cp -R build/artifacts/closed/NVIDIA/compliance .
        Repeat as necessary for other divisions, like network or open.

    IF YOU ARE AN EXTERNAL USER (i.e. submission partner):
        Copy the result artifacts from submission-staging. From closed/<submitter>:
            export SUBMITTER=<your company name>
            cp -R build/submission-staging/closed/${SUBMITTER}/measurements . && \
                cp -R build/submission-staging/closed/${SUBMITTER}/results . && \
                cp -R build/submission-staging/closed/${SUBMITTER}/compliance .

Run this script *FROM THE PROJECT ROOT DIRECTORY*, and *OUTSIDE THE CONTAINER*.

    python3 scripts/export_submission.py --divisions closed --export-to build/submission

Run the submission checker:
    From closed/<submitter>, run `make clone_loadgen` if you have not yet.

    python3 closed/<submitter>/build/inference/tools/submission/submission_checker.py --input build/submission --submitter <submitter>
"""


SUBMITTER = os.environ.get("SUBMITTER", "NVIDIA")


# Regex patterns for the file paths of the files that should be excluded.
exclude_patterns = ({
    "pycache files": r"(.*/)*__pycache__/.*",
    "compiled .pyc files": r".*\.pyc",
    "internal files": r"(.*/)*internal/.*",
    "build directory files": r"(.*/)*build/.*",
    "hidden files and directories": r"(.*/)*\..*",
    "NV internal regression tests": r"(.*/)*regression",
    "NV harness metadata.json": r"(.*/)*metadata\.json",
    "Triton inference server repo": r"(.*/)triton\-inference\-server(/.*)*",
})
exclude_patterns = {k: re.compile(v) for k, v in exclude_patterns.items()}


def filtered_scandir(dirname, filter_patterns):
    # There are a few key reasons why we choose to implement a manual walk instead of using glob:
    # 1. Python glob uses Unix path expansion, and does not support rigorous pattern exclusion
    # 2. There are quite a few files, as well as symlink redirection. It would require too high of
    #    a runtime if we first enumerated *all* files and then looped through them to exclude.
    # Avoid doing this recursively. Use a DFS approach to mimic behavior of other
    # directory enumeration implementations.

    files, symlinks = [], []
    exclusion_counts = dict()
    for k in filter_patterns:
        exclusion_counts[k] = {"files": 0, "directories": 0}
    bag = [dirname]

    while len(bag) > 0:
        curr = bag.pop()
        _toextend = []
        for f in os.scandir(curr):
            # Apply filters before continuing
            filtered = False
            for reason, pattern in filter_patterns.items():
                if pattern.fullmatch(f.path) is not None:
                    filtered = True
                    _typekey = "directories" if f.is_dir() else "files"
                    exclusion_counts[reason][_typekey] += 1
                    break

            if filtered:
                continue

            if f.is_symlink():
                symlinks.append(f.path)
            elif f.is_file(follow_symlinks=False):
                files.append(f.path)
            elif f.is_dir(follow_symlinks=False):
                _toextend.append(f.path)
            else:
                print(f"File '{f.path}' is not a file, directory, or symlink")

        bag.extend(reversed(_toextend))  # Add in reverse to maintain ordering of subdirectory in DFS

    return {
        "files": files,
        "symlinks": symlinks,
        "exclusions": exclusion_counts,
    }


def export_file(fpath, export_dir):
    dst_path = os.path.join(export_dir, fpath)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(fpath, dst_path)


def export_symlink(fpath, export_dir):
    dst_path = os.path.join(export_dir, fpath)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    src_path = pathlib.Path(fpath).resolve()
    if os.path.isdir(src_path):
        shutil.rmtree(dst_path, ignore_errors=True)
        shutil.copytree(src_path, dst_path)  # TODO: Note that this will bypass exclude patterns
    else:
        shutil.copy(src_path, dst_path)


def maybe_tqdm(L):
    # Wraps around tqdm if it is installed
    if importlib.util.find_spec("tqdm") is not None:
        from tqdm import tqdm
        return tqdm(L)
    else:
        return L


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--divisions',
                        default="closed,open,network",
                        help='Comma-separated list of divisions to check or package')
    parser.add_argument('--export-to',
                        default="build/submission",
                        help="Location to export the project to")
    # parser.add_argument('--results-from',
    #                     default="build/artifacts",
    #                     help="Relative subdirectory path from <division>/<submitter> that contain results to export")
    return parser.parse_args()


def main(args):
    subdirs = args.divisions.split(",")
    for subdir in subdirs:
        # print("Copying raw results artifacts...")
        # for artifact in ["measurements", "results", "compliance"]:
        #     src_path = os.path.join(subdir, SUBMITTER, args.results_from, subdir, SUBMITTER, artifact)
        #     dst_path = os.path.join(subdir, SUBMITTER, artifact)
        #     shutil.rmtree(dst_path, ignore_errors=True)
        #     shutil.copytree(src_path, dst_path)

        print("Filtering files for export...")
        found_files = filtered_scandir(subdir, exclude_patterns)
        valid_files = found_files["files"]
        symlinks = found_files["symlinks"]
        exclusion_counts = found_files["exclusions"]

        print(f"# files to copy: {len(valid_files)}")
        print(f"# symlinks: {len(symlinks)}")
        print(f"Skipped:")
        pprint.pprint(exclusion_counts)

        print("Exporting files...")
        for fpath in maybe_tqdm(valid_files):
            export_file(fpath, args.export_to)

        print("Exporting symlinks...")
        for slpath in maybe_tqdm(symlinks):
            export_symlink(slpath, args.export_to)

        # print("Cleanup...")
        # for artifact in ["measurements", "results", "compliance"]:
        #     src_path = os.path.join(subdir, SUBMITTER, artifact)
        #     shutil.rmtree(src_path, ignore_errors=True)
    return valid_files, symlinks


if __name__ == "__main__":
    main(parse_args())
