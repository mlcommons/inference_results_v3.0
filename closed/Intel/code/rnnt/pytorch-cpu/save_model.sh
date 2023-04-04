#!/bin/bash

set -ex

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BS=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${INTER:=1}
: ${INTRA:=4}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt_calib.pt}}
: ${MODE:=quant}
: ${BF16:=true}
: ${WAV:=true}
: ${SAVE_JIT:=true}
: ${DEBUG:=false}

export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --batch_size ${BS}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --run_mode ${MODE}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/dev-clean-npy.pt --toml_path configs/rnnt.toml --enable_process"
else
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/dev-clean-input.pt"
fi

if [[ -n "${INTRA}" ]]; then
  echo "Use JeMalloc memory allocator"
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  echo "Use Intel OpenMP"
  export OMP_NUM_THREADS=${INTRA}
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_BLOCKTIME=1
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${LD_PRELOAD}
  EXEC_ARGS="numactl -C 0-$((${INTRA}-1)) -m $((${INTER}-1))"
fi

[ ${BF16} == "true" ] && SCRIPT_ARGS+=" --enable_bf16"
[ ${SAVE_JIT} == true ] && SCRIPT_ARGS+=" --save_jit"

[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

