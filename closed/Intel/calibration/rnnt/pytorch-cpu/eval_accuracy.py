# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import array
import json

from models.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default=None, required=True)
    parser.add_argument("--manifest_path", type=str, default=None, required=True)
    parser.add_argument("--max_duration", type=int, default=15.0)
    args = parser.parse_args()
    return args

def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def word_error_rate(hypotheses, references):
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    for h, r in zip(hypotheses, references):
        h_list = h.split()
        r_list = r.split()
        words += len(r_list)
        scores += levenshtein(h_list, r_list)
    if words != 0:
        wer = (1.0 * scores) / words
    else:
        wer = float("inf")
    return wer, scores, words

def save_seqs(seq_list, file_name='seqs.log'):
    with open(file_name, "w") as file:
        for i, seq in enumerate(seq_list):
            file.write(f"{i}::{seq}\n")

def eval_acc(log_path, manifest_path, max_duration=15.0):
    # load ground-truth
    with open(manifest_path) as f:
        manifest = json.load(f)
    filtered_gd = [sample for sample in manifest if sample["original_duration"] <= max_duration]
    references = [sample["transcript"] for idx, sample in enumerate(filtered_gd)]
    # load hypotheses
    with open(log_path) as f:
        results = json.load(f)
    hypotheses = [None for i in range(len(results))]
    for result in results:
        seq = torch.Tensor(array.array("I", bytes.fromhex(result["data"])).tolist()).to(torch.int32)
        hypotheses[result["qsl_idx"]] = seq_to_sen(seq, seq.size(0))
    # calculate accuracy
    wer, _, _ = word_error_rate(hypotheses, references)
    print("Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100))
    save_seqs(hypotheses, "hypotheses.log")
    # save_seqs(references, "references.log")

if __name__ == "__main__":
    args = parse_args()
    eval_acc(args.log_path, args.manifest_path, args.max_duration)

