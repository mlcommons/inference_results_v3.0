#!/bin/bash

# BERT-99% (mixed precision).
RUN_CMD_SUFFIX_BERT99="--vc=16" # TODO: Make Offline only.
BERT99_OFFLINE_OVERRIDE_BATCH_SIZE=4096
BERT99_OFFLINE_TARGET_QPS=3000
BERT99_SINGLESTREAM_TARGET_LATENCY=7.6

# ResNet50.
RUN_CMD_SUFFIX_RESNET50="--vc=16" # TODO: Make Offline only.
RESNET50_OFFLINE_TARGET_QPS=80000
RESNET50_SINGLESTREAM_TARGET_LATENCY=.34
RESNET50_MULTISTREAM_TARGET_LATENCY=.55

# RetinaNet.
RUN_CMD_SUFFIX_RETINANET="--vc=17" # TODO: Make Offline only.
RETINANET_OFFLINE_TARGET_QPS=1100
RETINANET_SINGLESTREAM_TARGET_LATENCY=15
RETINANET_MULTISTREAM_TARGET_LATENCY=27

# Use workload-specific frequency limits.
RUN_CMD_COMMON_SUFFIX_DEFAULT='--sleep_before_ck_benchmark_sec=120'
