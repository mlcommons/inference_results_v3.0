#!/usr/bin/env python
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import hashlib
import os
import pandas as pd
import requests
import tarfile
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, verify and extract dataset files"
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        help="CSV file with urls and checksums to download.",
    )
    parser.add_argument(
        "-d", "--download_dir", type=str, help="Download destnation folder."
    )
    parser.add_argument(
        "-e",
        "--extract_dir",
        type=str,
        default=None,
        help="Extraction destnation folder. Defaults to download folder if not provided",
    )
    parser.add_argument(
        "--skip_download", action="store_true", help="Skip downloading the files"
    )
    parser.add_argument("--skip_checksum", action="store_true", help="Skip checksum")
    parser.add_argument(
        "--skip_extract", action="store_true", help="Skip extracting files"
    )
    args = parser.parse_args()
    return args


def download_file(url, dest_folder, fname, overwrite=False):
    fpath = os.path.join(dest_folder, fname)
    if os.path.isfile(fpath):
        if overwrite:
            print("Overwriting existing file")
        else:
            print("File exists, skipping download.")
            return

    tmp_fpath = fpath + ".tmp"

    r = requests.get(url, stream=True)
    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024 * 1024  # 1MB
    total_chunks = int(file_size / chunk_size)

    with open(tmp_fpath, "wb") as fp:
        content_iterator = r.iter_content(chunk_size=chunk_size)
        chunks = tqdm.tqdm(
            content_iterator, total=total_chunks, unit="MB", desc=fpath, leave=True
        )
        for chunk in chunks:
            fp.write(chunk)

    os.rename(tmp_fpath, fpath)


def md5_checksum(fpath, target_hash):
    file_hash = hashlib.new("md5", usedforsecurity=False)
    with open(fpath, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest() == target_hash


def extract(fpath, dest_folder):
    if fpath.endswith(".tar.gz"):
        mode = "r:gz"
    elif fpath.endswith(".tar"):
        mode = "r:"
    else:
        raise IOError("fpath has unknown extention: %s" % fpath)

    with tarfile.open(fpath, mode) as tar:
        members = tar.getmembers()
        for member in tqdm.tqdm(iterable=members, total=len(members), leave=True):
            tar.extract(path=dest_folder, member=member)


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv, delimiter=",")

    if not args.skip_download:
        for url in df.url:
            fname = url.split("/")[-1]
            print("==> Downloading %s:" % fname)
            download_file(url=url, dest_folder=args.download_dir, fname=fname)
    else:
        print("==> Skipping file download")

    if not args.skip_checksum:
        for index, row in df.iterrows():
            url = row["url"]
            md5 = row["md5"]
            fname = url.split("/")[-1]
            fpath = os.path.join(args.download_dir, fname)
            print("==> Verifing %s: " % fname, end="")
            ret = md5_checksum(fpath=fpath, target_hash=md5)
            if not ret:
                raise ValueError(f"Checksum for {fname} failed!")
            else:
                print(f"==> Checksum correct for {fname}")
    else:
        print("==> Skipping checksum")

    if not args.skip_extract:
        for url in df.url:
            fname = url.split("/")[-1]
            fpath = os.path.join(args.download_dir, fname)
            print("==> Decompressing %s:" % fpath)
            extract(fpath=fpath, dest_folder=args.extract_dir)
    else:
        print("==> Skipping file extraction")
