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

DATA_DIR=${DATA_DIR:-build/data}
KITS_RAW_DIR=${DATA_DIR}/KiTS19/kits19/data
INFERENCE_HASH=${INFERENCE_HASH:-`grep "INFERENCE_HASH =" Makefile | sed "s/.*= //"`}

if [ ! -s ${KITS_RAW_DIR}/case_00137/imaging.nii.gz ]
then
    echo "Cloning KITS19 repo and download RAW data into ${KITS_RAW_DIR}..." && \
    pushd ${DATA_DIR} &&\
	rm -Rf KiTS19 &&\
	mkdir -p KiTS19 &&\
	cd KiTS19 &&\
    git clone https://github.com/neheller/kits19 &&\
    cd kits19 &&\
    pip3 install -r requirements.txt &&\
    python3 -m starter_code.get_imaging &&\
	echo "Duplicating KITS19 case_00185 as case_00400..." &&\
    cp -Rf data/case_00185 data/case_00400 &&\
    popd &&\
    echo "Done."
else
    echo "Valid KITS RAW data set found in ${KITS_RAW_DIR}/, skipping download."
fi

sleep 0.1

echo "Downloading JSON files describing subset used for inference/calibration..."
wget https://raw.githubusercontent.com/mlcommons/inference/${INFERENCE_HASH}/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json -O ${DATA_DIR}/KiTS19/inference_cases.json
wget https://raw.githubusercontent.com/mlcommons/inference/${INFERENCE_HASH}/vision/medical_imaging/3d-unet-kits19/meta/calibration_cases.json -O ${DATA_DIR}/KiTS19/calibration_cases.json
echo "Done."
