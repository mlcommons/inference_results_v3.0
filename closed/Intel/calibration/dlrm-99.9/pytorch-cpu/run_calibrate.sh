#!/bin/bash

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$CPUS_PER_SOCKET
export KMP_AFFINITY="granularity=fine,compact,1,0"
export DNNL_PRIMITIVE_CACHE_CAPACITY=20971520
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export DLRM_DIR=$PWD/python/model

OUTPUT_DIR=$PWD/output/
model_path="$MODEL_DIR/dlrm_terabyte.pytorch"

python -u python/calibrate.py --model dlrm  --model-path $model_path \
       --dataset terabyte --dataset-path $DATA_DIR  --output $OUTPUT_DIR --max-ind-range=40000000 \
       --mlperf-bin-loader --calibration-batches=128000
