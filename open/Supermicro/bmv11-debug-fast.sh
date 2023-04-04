#!/bin/bash
##############################################################
#  
#  
#  
#  
#  
##############################################################


#echo " DLRM Offline Test "

#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default --fast" | tee dlrm-offline-df-perf-fast.txt

#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee dlrm-offline-df-accu.txt

#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy --fast" | tee dlrm-offline-high_accu-perf-fast.txt

#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee dlrm-offline-high_accu-accu.txt

#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=triton --fast" | tee dlrm-offline-tr-perf-fast.txt

#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly" | tee dlrm-offline-tr-accu.txt

#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy_triton --fast" | tee dlrm-offline-high_accu_tr-perf-fast.txt

#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee dlrm-offline-high_accu_tr-accu.txt


#echo " RESNET50 Offline Test "

#make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default" | tee resnet50-offline-df-perf.txt

#make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee resnet50-offline-df-accu.txt


#make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=LWIS" | tee resnet50-offline-lwis-perf.txt

#make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=LWIS --test_mode=AccuracyOnly" | tee resnet50-offline-lwis-accu.txt


#echo " RETINANET Offline Test "

#make run RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=default" | tee retinanet-offline-df-perf.txt

#make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee retinanet-offline-df-accu.txt


#make run RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=triton" | tee retinanet-offline-tr-perf.txt

#make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly" | tee retinanet-offline-tr-accu.txt



echo " DLRM Server Test "

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default" | tee dlrm-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee dlrm-server-df-accu.txt
#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy --fast" | tee dlrm-server-high_accu-perf-fast.txt
#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee dlrm-server-high_accu-accu.txt


#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=triton --fast" | tee dlrm-server-tr-perf-fast.txt
#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee dlrm-server-tr-accu.txt
#make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy_triton --fast" | tee dlrm-server-high_accu_tr-perf-fast.txt
#make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee dlrm-server-high_accu_tr-accu.txt


#echo " RETINANET Server Test "

#make run RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=default" | tee retinanet-server-df-perf.txt
#make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee retinanet-server-df-accu.txt

#make run RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=triton" | tee retinanet-server-tr-perf.txt
#make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee retinanet-server-tr-accu.txt



echo " Server Test Done! "

