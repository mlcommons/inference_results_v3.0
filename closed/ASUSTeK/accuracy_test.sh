#!/bin/bash
#Performance mode
#make run RUN_ARGS="--benchmarks=resnet50 --scenarios=server,offline"
#sleep 30
#make run RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=server,offline"
#sleep 30
#make run RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=high_accuracy"
#sleep 30
#make run RUN_ARGS="--benchmarks=rnnt --scenarios=server,offline --config_ver=default"
#sleep 30
#make run RUN_ARGS="--benchmarks=bert --scenarios=server,offline --config_ver=default,high_accuracy"
#sleep 30
#make run RUN_ARGS="--benchmarks=dlrm --scenarios=server,offline --config_ver=high_accuracy"
#sleep 30

#Accuracy mode 
#make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=server,offline --test_mode=AccuracyOnly"
#sleep 30
#make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=server,offline --test_mode=AccuracyOnly"
#sleep 30
#make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
#sleep 30
#make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=server,offline --config_ver=default --test_mode=AccuracyOnly"
#sleep 30
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=offline --config_ver=default,high_accuracy --test_mode=AccuracyOnly"
#sleep 30
#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=server,offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
#sleep 30
