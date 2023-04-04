#!/bin/bash

set -ex

: ${BS:=128}
: ${LEN:=-1}
: ${LOG_LEVEL:=10}
: ${INTER:=1}
: ${INTRA:=4}
: ${SCENARIO:=Offline}
: ${DEBUG:=false}
: ${MODE:=f32}
: ${BF16:=true}
: ${WAV:=false}
: ${ACCURACY:=true}
: ${LOAD_JIT:=false}
: ${SAVE_JIT:=false}

SUT_DIR=$(pwd)
WORK_DIR=${SUT_DIR}/mlperf-rnnt-librispeech
OUT_DIR=${SUT_DIR}/logs/${SCENARIO}

export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --scenario ${SCENARIO}"
SCRIPT_ARGS+=" --batch_size ${BS}"
SCRIPT_ARGS+=" --split_len ${LEN}"
SCRIPT_ARGS+=" --mlperf_conf ${PWD}/configs/mlperf.conf"
SCRIPT_ARGS+=" --user_conf ${PWD}/configs/user.conf"
SCRIPT_ARGS+=" --toml_path configs/rnnt.toml"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/train-clean-100-npy.pt"
SCRIPT_ARGS+=" --log_dir ${PWD}/logs/${SCENARIO}"
SCRIPT_ARGS+=" --run_mode ${MODE}"
[ ${ACCURACY} == true ] && SCRIPT_ARGS+=" --accuracy"
if [[ ${MODE} != "f32" && ${MODE} != "calib" && ${BF16} == true ]]; then
  SCRIPT_ARGS+=" --enable_bf16"
fi
# set dataset & processor
if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --infer_dataset_dir ${WORK_DIR}/dev-clean-npy.pt --enable_process"
  if [[ ${LOAD_JIT} == true ]]; then
    SCRIPT_ARGS+=" --processor_path ${WORK_DIR}/processor_jit.pt"
  fi
else
  SCRIPT_ARGS+=" --infer_dataset_dir ${WORK_DIR}/dev-clean-input.pt"
fi
# set model
if [[ ${LOAD_JIT} == true ]]; then
  SCRIPT_ARGS+=" --load_jit"
  if [[ ${MODE} == "quant" || ${MODE} == "f32" || ${MODE} == "" ]]; then
    SCRIPT_ARGS+=" --model_path ${WORK_DIR}/rnnt_${MODE}_jit.pt"
  fi
else
  if [[ ${MODE} == "quant" || ${MODE} == "fake_quant" ]]; then
    SCRIPT_ARGS+=" --model_path ${WORK_DIR}/rnnt_calib.pt"
  else
    SCRIPT_ARGS+=" --model_path ${WORK_DIR}/rnnt.pt"
  fi
fi
[ ${SAVE_JIT} == true ] && SCRIPT_ARGS+=" --save_jit"

echo "export DNNL ENV"
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

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

[ ${DEBUG} == "pdb" ] && EXEC_ARGS+=" ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS+=" gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS+=" lldb python --"
[ ${DEBUG} == true ] && EXEC_ARGS+=" python -u"
[ ${DEBUG} == false ] && EXEC_ARGS+=" python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

if [[ ${ACCURACY} == true ]]; then
  export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
  python -u eval_accuracy.py \
    --log_path=${OUT_DIR}/mlperf_log_accuracy.json \
    --manifest_path=${WORK_DIR}/local_data/wav/dev-clean-wav.json | tee ${OUT_DIR}/accuracy.txt
fi

set +x
