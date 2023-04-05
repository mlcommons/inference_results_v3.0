#!/bin/bash

# BERT-99% (mixed precision).
BERT99_SINGLESTREAM_TARGET_LATENCY=12
BERT99_OFFLINE_TARGET_QPS=370
BERT99_OFFLINE_OVERRIDE_BATCH_SIZE=4096

# ResNet50.
RESNET50_SINGLESTREAM_TARGET_LATENCY=1
RESNET50_MULTISTREAM_TARGET_LATENCY=2.1
RESNET50_OFFLINE_TARGET_QPS=9700

# RetinaNet.
RETINANET_SINGLESTREAM_TARGET_LATENCY=21
RETINANET_MULTISTREAM_TARGET_LATENCY=105
RETINANET_OFFLINE_TARGET_QPS=180
RUN_CMD_SUFFIX_RETINANET="--skip_accuracy_calc"

RUN_CMD_COMMON_SUFFIX="--sleep_before_ck_benchmark_sec=120 --soc_reset"