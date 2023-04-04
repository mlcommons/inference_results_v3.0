#!/bin/bash
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

set -e

source code/common/file_downloads.sh

# Download the raw weights instead of pre-made JIT PyT file
download_file models retinanet-resnext50-32x4d \
    https://zenodo.org/record/6605272/files/retinanet_model_10.zip \
    retinanet.zip

md5sum ${MLPERF_SCRATCH_PATH}/models/retinanet-resnext50-32x4d/retinanet.zip | grep "2037c152a6be18e371ebec654314f7e0"
if [ $? -ne 0 ]; then
    echo "md5sum mismatch"
    exit -1
fi

unzip ${MLPERF_SCRATCH_PATH}/models/retinanet-resnext50-32x4d/retinanet.zip \
    -d ${MLPERF_SCRATCH_PATH}/models/retinanet-resnext50-32x4d

# Clone training repo
if [ ! -d "${BUILD_DIR}/training" ]; then
    git clone https://github.com/mlcommons/training.git ${BUILD_DIR}/training
fi
bash -c "cd ${BUILD_DIR}/training && git checkout a9056b8e5840d811484ad91f9fe23ed09a3f97cf"

# Run script
python3 -m code.retinanet.tensorrt.onnx_generator.create_model_onnx \
    --input ${MLPERF_SCRATCH_PATH}/models/retinanet-resnext50-32x4d/retinanet_model_10.pth
