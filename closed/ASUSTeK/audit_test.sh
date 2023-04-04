#!/bin/bash

#make run_audit_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=server,offline --config_ver=default"
#sleep 30
#make run_audit_harness RUN_ARGS="--benchmarks=retinanet --scenarios=server,offline --config_ver=default"
#sleep 30
#make run_audit_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=high_accuracy"
#sleep 30
#make run_audit_harness RUN_ARGS="--benchmarks=rnnt --scenarios=offline --config_ver=default"
#sleep 30
#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=server,offline --config_ver=default,high_accuracy"
#sleep 30
make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=server,offline --config_ver=high_accuracy"
