#!/bin/bash
set -x
# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
OPENVINO_VERSION=2022.3.0
python -m pip install --upgrade pip
python -m pip install absl-py numpy transformers openvino-dev==${OPENVINO_VERSION} "setuptools>=65.5.1" \
    intel-tensorflow wheel==0.38.1 cryptography==39.0.1

# build MLPERF
MLPERF_BRANCH=master
COMMIT_V3=f5367250115ad4febf1334b34881ab74f2e55bfe
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
MLPERF_INFERENCE_REPO=${CUR_DIR}/dependencies/mlperf-inference
git clone https://github.com/mlcommons/inference.git ${MLPERF_INFERENCE_REPO}
cd ${MLPERF_INFERENCE_REPO}
git checkout ${COMMIT_V3}
git submodule update --init third_party/pybind
cd ${MLPERF_INFERENCE_REPO}/loadgen
CFLAGS="-std=c++14" python setup.py install

pushd ${MLPERF_INFERENCE_REPO}/language/bert

set +x
