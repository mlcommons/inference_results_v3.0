#!/bin/bash

# BERT-99% (mixed precision).
RUN_CMD_SUFFIX_BERT99="--vc=16"
BERT99_OFFLINE_OVERRIDE_BATCH_SIZE=4096
BERT99_SERVER_OVERRIDE_BATCH_SIZE=1024
BERT99_OFFLINE_TARGET_QPS=5850
BERT99_SERVER_TARGET_QPS=5375

# BERT-99.9% (FP16 precision).
RUN_CMD_SUFFIX_BERT999="--vc=13"
BERT999_OFFLINE_OVERRIDE_BATCH_SIZE=4096
BERT999_SERVER_OVERRIDE_BATCH_SIZE=1024
BERT999_OFFLINE_TARGET_QPS=2700
BERT999_SERVER_TARGET_QPS=2575

# ResNet50.
RUN_CMD_SUFFIX_RESNET50="--vc=16"
RESNET50_OFFLINE_TARGET_QPS=158000 # 159000
RESNET50_SERVER_TARGET_QPS=152750
RESNET50_MAX_WAIT=1800

# RetinaNet.
RUN_CMD_SUFFIX_RETINANET="--vc=17"
RETINANET_OFFLINE_TARGET_QPS=2060
RETINANET_SERVER_TARGET_QPS=2002 # TEST05: 2001

RUN_CMD_COMMON_SUFFIX="--sleep_before_ck_benchmark_sec=30"
