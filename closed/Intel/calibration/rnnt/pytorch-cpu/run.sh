#!/bin/bash

set -ex

: ${CONDA_ENV=${1:-'rnnt-infer'}}
: ${WORK_DIR=${2:-${PWD}/mlperf-rnnt-librispeech}}
: ${LOCAL_DATA_DIR=${WORK_DIR}/local_data}
: ${STAGE=${3:-2}}
: ${SKIP_BUILD=${4:-0}}
: ${SETUP_ONLY=${5:-'no'}} 

mkdir -p ${WORK_DIR}

if [[ ${STAGE} -le -2 ]]; then
  echo '==>Preparing conda env'
  conda create -y -n ${CONDA_ENV} python=3.8
fi

if [[ ${SKIP_BUILD} -eq 0 ]]; then
  source activate ${CONDA_ENV}
fi

if [[ ${STAGE} -le -1 ]]; then
  echo '==> Preparing env'
  ./prepare_conda_env.sh
  ./prepare_env.sh ${CONDA_ENV} ${PWD}
fi

if [[ ${STAGE} -le 0 ]]; then
  echo '==> Downloading model'
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O ${WORK_DIR}/rnnt.pt
fi

if [[ ${STAGE} -le 1 ]]; then
  echo '==> Downloading dataset'
  mkdir -p ${LOCAL_DATA_DIR}/LibriSpeech ${LOCAL_DATA_DIR}/raw
  python datasets/download_librispeech.py \
    --input_csv=configs/librispeech-inference.csv \
    --download_dir=${LOCAL_DATA_DIR}/LibriSpeech \
    --extract_dir=${LOCAL_DATA_DIR}/raw
fi

if [[ ${STAGE} -le 2 ]]; then
  echo '==> Pre-processing dataset'
  export PATH="${PWD}/third_party/bin/:${PATH}"
  export PYTHONPATH="${PWD}/models:${PYTHONPATH}"
  python datasets/convert_librispeech.py \
    --input_dir=${LOCAL_DATA_DIR}/raw/LibriSpeech/dev-clean \
    --output_dir=${LOCAL_DATA_DIR}

  python datasets/convert_librispeech.py \
    --input_dir=${LOCAL_DATA_DIR}/raw/LibriSpeech/train-clean-100 \
    --output_dir=${LOCAL_DATA_DIR} \
    --output_list=configs/calibration_files.txt
fi

if [[ ${STAGE} -le 3 ]]; then
  echo '==> Calibrating'
  ./calib_model.sh
fi

if [[ ${STAGE} -le 4 ]]; then
  echo '==> Building model'
  JIT=true WAV=true ./save_model.sh
  if [ "${SETUP_ONLY}" == "yes" ];then return 0 ;fi
fi

set +x
