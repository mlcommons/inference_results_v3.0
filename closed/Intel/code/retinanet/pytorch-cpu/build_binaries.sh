#!/usr/bin/env bash

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=retinanet-env
export WORKDIR=${CUR_DIR}/${CONDA_ENV_NAME}

echo "Working directory is ${WORKDIR}"
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export IPEX_PATH=${WORKDIR}/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch
export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
export LOADGEN_DIR=${WORKDIR}/mlperf_inference/loadgen
export OPENCV_DIR=${WORKDIR}/opencv/build
export RAPIDJSON_INCLUDE_DIR=${WORKDIR}/rapidjson/include
export GFLAGS_DIR=${WORKDIR}/gflags/build

cd ${CUR_DIR}
# ============================
echo "=== Building binaries ==="
BUILD_DIR=${CUR_DIR}/build
SRC_DIR=${CUR_DIR}/src
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib

cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
    -DLOADGEN_DIR=${LOADGEN_DIR} \
    -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
    -Dgflags_DIR=${GFLAGS_DIR} \
    -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
    -DOpenCV_DIR=${OPENCV_DIR} \
    -B${BUILD_DIR} \
    -H${SRC_DIR}

cmake --build ${BUILD_DIR} --config Release -j$(nproc)
