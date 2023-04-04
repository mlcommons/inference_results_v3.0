#!/bin/bash
BASEDIR=$(dirname "$0")
MODEL_VARIANT=$(echo "$1" | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR=bert_logs/${MODEL_VARIANT}/Server/performance/run_1

CORE_COUNT=$(lscpu | awk '/^Core\(s\) per socket/{ print $4 }')
NUM_SOCKETS=$(lscpu | awk '/^Socket\(s\)/{ print $2 }')
CORE_MULT_SOCKET=$(($CORE_COUNT * $NUM_SOCKETS))

export DATA_DIR=${BERT_DATA_DIR}
export MODEL_DIR=${BERT_MODEL_DIR}
export INFERENCE_BERT_PATH=${BASEDIR}/dependencies/mlperf-inference/language/bert
export PYTHONPATH=${PYTHONPATH}:${BASEDIR}/src:${INFERENCE_BERT_PATH}
export TF_CPP_MIN_LOG_LEVEL=1
export VOCAB_FILE=${DATA_DIR}/vocab.txt
export DATASET_FILE=${DATA_DIR}/dev-v1.1.json

inf_precision=bf16
num_streams=$CORE_MULT_SOCKET

# GET INT8 MODEL SPECIFIC INFO
case $MODEL_VARIANT in
    "bert-99.9_jpqd-large")
        filename=bert-99.9_jpqd-large/openvino_model.xml
        model_name=bert-99.9_jpqd-large
        ;;

    "bert-99_jpqd-base")
        filename=bert-99_jpqd-base/openvino_model.xml
        model_name=bert-99_jpqd-base
        ;;

    "bert-99_jpqd-large")
        filename=bert-99_jpqd-large/openvino_model.xml
        model_name=bert-99_jpqd-large
        ;;

    "bert-99_jpqd-mobilebert")
        filename=bert-99_jpqd-mobilebert/openvino_model.xml
        model_name=bert-99_jpqd-mobilebert
        inf_precision=f32
        num_streams=$CORE_COUNT
        ;;

    *)
        echo "Invalid model. Available models: bert-99.9_jpqd-large, bert-99_jpqd-large, bert-99_jpqd-base, or bert-99_jpqd-mobilebert"
        exit 1

esac
mkdir -p $OUTPUT_DIR

echo "*********SERVER PERF*********"
python src/run.py --backend openvino \
    --scenario Server \
    --mlperf_conf mlperf.conf \
    --user_conf user.conf \
    --model_name ${model_name} \
    --model_path ${MODEL_DIR}/${filename} \
    --data_path ${DATA_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --inference_precision ${inf_precision} \
    --nstreams ${num_streams}

