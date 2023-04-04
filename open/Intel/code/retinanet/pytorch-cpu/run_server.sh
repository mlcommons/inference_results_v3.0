#!/bin/bash

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_PATH=</path/to/retinanet-BNAS-int8-model.pth>"
    exit 1
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP=${CUR_DIR}/build/bin/mlperf_runner

if [ -e "mlperf_log_summary.txt" ]; then
    rm mlperf_log_summary.txt
fi

${APP} --scenario Server \
	--mode Performance \
	--mlperf_conf mlperf.conf \
	--user_conf user.conf \
	--model_name retinanet \
    --model_path ${MODEL_PATH} \
	--data_path ${DATA_DIR} \
	--num_instance 14 \
	--warmup_iters 100 \
	--cpus_per_instance 8 \
	--total_sample_count 24781 \
    --batch_size 1

if [ -e "mlperf_log_summary.txt" ]; then
    cat mlperf_log_summary.txt
fi
