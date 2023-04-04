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

MLPERF_SCRATCH_PATH=${MLPERF_SCRATCH_PATH:-/home/mlperf_inference_data}
MODEL_DIR=${MLPERF_SCRATCH_PATH}/models/RetinaNet
MLPERF_TRAINING_DIR=${MLPERF_TRAINING_DIR:-/tmp/training}

echo "WARNING: Model downloading script for retinanet is not ready yet. Please use provided ONNX model by NVIDIA!"
exit 0

function download_file {
    if [ ! -d ${MODEL_DIR} ]; then
        echo "Creating directory ${MODEL_DIR}"
        mkdir -p ${MODEL_DIR}
    fi
    echo "Downloading into ${MODEL_DIR}..." \
        && wget $1 -O ${MODEL_DIR}/retinanet_model_10.zip \
        && echo "Saved to ${MODEL_DIR}/retinanet_model_10.zip!"
}

download_file https://zenodo.org/record/6605272/files/retinanet_model_10.zip

if [ ! -d ${MLPERF_TRAINING_DIR}/single_stage_detector ]; then
	echo "Please clone https://github.com/mlcommons/training.git and set MLPERF_TRAINING_DIR envs to continue. Exiting..."
	exit 1
fi

if [ ! -e build/inference/vision/classification_and_detection/tools/retinanet_pytorch_to_onnx.py ]; then
	echo "Please run make clone_loadgen and make build_loadgen to update the inference repo."
	exit 1
else
	echo "Unzipping retinanet reference model..."
	cd ${MODEL_DIR} && unzip retinanet_model_10.zip && cd -
	cp build/inference/vision/classification_and_detection/tools/retinanet_pytorch_to_onnx.py ${MLPERF_TRAINING_DIR}/single_stage_detector/ssd
	cd ${MLPERF_TRAINING_DIR}/single_stage_detector/ssd && \
	python3 -m retinanet_pytorch_to_onnx --weights ${MODEL_DIR}/retinanet_model_10.pth --output ${MODEL_DIR}/resnext50_32x4d_fpn.onnx && \
	echo "Onnx model generation completed!!"
fi
