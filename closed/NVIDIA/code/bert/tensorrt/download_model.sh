#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

source code/common/file_downloads.sh

MODEL=bert

# Function to check if a file already exists before downloading
function download_if_not_exists {
    local sub_dir=$1
    local model_name=$2
    local url=$3
    local dest_filename=$4

    if [ ! -f "${MLPERF_SCRATCH_PATH}/${sub_dir}/${model_name}/${dest_filename}" ]; then
        download_file "${sub_dir}" "${model_name}" "${url}" "${dest_filename}"
    else
        echo "File ${MLPERF_SCRATCH_PATH}/${sub_dir}/${model_name}/${dest_filename} already exists. Skipping download."
    fi
}

download_if_not_exists models ${MODEL} https://zenodo.org/record/3733910/files/model.onnx?download=1 bert_large_v1_1.onnx
download_if_not_exists models ${MODEL} https://zenodo.org/record/3750364/files/bert_large_v1_1_fake_quant.onnx?download=1 bert_large_v1_1_fake_quant.onnx
download_if_not_exists models ${MODEL} https://zenodo.org/record/3750364/files/vocab.txt?download=1 vocab.txt
