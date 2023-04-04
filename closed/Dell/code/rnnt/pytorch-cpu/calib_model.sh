#!/bin/bash

set -ex

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BS=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt.pt}}
: ${MODE:=calib}
: ${DEBUG:=false}
: ${WAV:=false}

export PYTHONPATH=${PWD}:${PWD}/models:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --batch_size ${BS}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --run_mode ${MODE}"
SCRIPT_ARGS+=" --calibration"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/train-clean100-wav.json"
if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/train-clean-100-npy.pt --toml_path configs/rnnt.toml --enable_process"
else
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/train-clean-100-input.pt"
fi

[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

