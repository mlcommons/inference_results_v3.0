#!/bin/bash
if [ -z $1 ]; then
  echo "Error! please input the batch_size, choice is 8 or 256"
  exit 0
fi

export BATCH_SIZE=$1
echo "Testing BATCH SIZE is " $BATCH_SIZE
if [ $1 = 8 ]; then
  NUM_INSTANCE=224
elif [ $1 = 256 ]; then
  NUM_INSTANCE=112
else
  echo "wrong batch size!"
  exit 0
fi

export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=${PWD}/models/resnet50-start-int8-model.pth
export RN50_END=${PWD}/models/resnet50-end-int8-model.pth
export RN50_FULL=${PWD}/models/resnet50-full.pth


if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${RN50_START}" ]; then
    echo "Path to resnet50_start model not set. Please set it:"
    export "RN50_START=</path/to/resnet50_start.pth>"
    exit 1
fi

if [ -z "${RN50_END}" ]; then
    echo "Path to resnet50_end model not set. Please set it:"
    export "RN50_END=</path/to/resnet50_end.pth>"
    exit 1
fi

if [ -z "${RN50_FULL}" ]; then
    echo "Path to resnet50_full model not set. Please set it:"
    export "RN50_FULL=</path/to/resnet50_full.pth>"
    exit 1
fi

CONDA_ENV_NAME=rn50-mlperf
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP=${CUR_DIR}/build/bin/mlperf_runner

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

if [ -e mlperf_log_accuracy.json ]; then
    rm mlperf_log_accuracy.json
fi

numactl -C 0-55,56-111 -m 0,1 ${APP} --scenario Offline \
	--mode Accuracy \
	--mlperf_conf ${CUR_DIR}/src/mlperf.conf \
	--user_conf ${CUR_DIR}/src/user.conf \
	--model_name resnet50 \
  --rn50-part1 ${RN50_START} \
  --rn50-part3 ${RN50_END} \
  --rn50-full-model ${RN50_FULL} \
	--data_path ${DATA_DIR} \
  --num_instance $NUM_INSTANCE \
  --warmup_iters 20 \
  --cpus_per_instance 1 \
	--total_sample_count 50000 \
  --batch_size $1

echo " ==================================="
echo "         Evaluating Accuracy        "
echo " ==================================="

if [ -e mlperf_log_accuracy.json ]; then
    python -u ${CONDA_ENV_NAME}/mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
        --mlperf-accuracy-file mlperf_log_accuracy.json \
        --imagenet-val-file ${DATA_DIR}/val_map.txt \
        --dtype int32 2>&1|tee accuracy.txt
fi

