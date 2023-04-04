#!/bin/bash
##############################################################
#  
#  
#  
#  
#  
##############################################################


echo " 3D-Unet Offline Test "

make run RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default" | tee 3d-unet-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee 3d-unet-offline-df-accu.txt

make run RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=high_accuracy" | tee 3d-unet-offline-high_accu-perf.txt

make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee 3d-unet-offline-high_accu-accu.txt


echo " BERT Offline Test "

make run RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=default" | tee bert-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee bert-offline-df-accu.txt

make run RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy" | tee bert-offline-high_accu-perf.txt

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee bert-offline-high_accu-accu.txt

make run RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=triton" | tee bert-offline-tr-perf.txt

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly" | tee bert-offline-tr-accu.txt

make run RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy_triton" | tee bert-offline-high_accu_tr-perf.txt

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee bert-offline-high_accu_tr-accu.txt


echo " DLRM Offline Test "

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default" | tee dlrm-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee dlrm-offline-df-accu.txt

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy" | tee dlrm-offline-high_accu-perf.txt

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee dlrm-offline-high_accu-accu.txt

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=triton" | tee dlrm-offline-tr-perf.txt

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly" | tee dlrm-offline-tr-accu.txt

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy_triton" | tee dlrm-offline-high_accu_tr-perf.txt

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee dlrm-offline-high_accu_tr-accu.txt


echo " RNN-T Offline Test "

make run RUN_ARGS="--benchmarks=rnnt --scenarios=Offline --config_ver=default" | tee rnnt-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee rnnt-offline-df-accu.txt


echo " RESNET50 Offline Test "

make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default" | tee resnet50-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee resnet50-offline-df-accu.txt


make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=LWIS" | tee resnet50-offline-lwis-perf.txt

make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=LWIS --test_mode=AccuracyOnly" | tee resnet50-offline-lwis-accu.txt



echo " RETINANET Offline Test "

make run RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=default" | tee retinanet-offline-df-perf.txt

make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly" | tee retinanet-offline-df-accu.txt


make run RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=triton" | tee retinanet-offline-tr-perf.txt

make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=triton --test_mode=AccuracyOnly" | tee retinanet-offline-tr-accu.txt













echo " BERT Server Test "

make run RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=default" | tee bert-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee bert-server-df-accu.txt
make run RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy" | tee bert-server-high_accu-perf.txt
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee bert-server-high_accu-accu.txt


make run RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=triton" | tee bert-server-tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee bert-server-tr-accu.txt
make run RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy_triton" | tee bert-server-high_accu_tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee bert-server-high_accu_tr-accu.txt


echo " DLRM Server Test "

make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default" | tee dlrm-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee dlrm-server-df-accu.txt
make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy" | tee dlrm-server-high_accu-perf.txt
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy --test_mode=AccuracyOnly" | tee dlrm-server-high_accu-accu.txt


make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=triton" | tee dlrm-server-tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee dlrm-server-tr-accu.txt
make run RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy_triton" | tee dlrm-server-high_accu_tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy_triton --test_mode=AccuracyOnly" | tee dlrm-server-high_accu_tr-accu.txt


echo " RNNT Server Test "

make run RUN_ARGS="--benchmarks=rnnt --scenarios=Server --config_ver=default" | tee rnnt-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee rnnt-server-df-accu.txt


echo " RESNET50 Server Test "

make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=default" | tee resnet50-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee resnet50-server-df-accu.txt

make run RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=triton" | tee resnet50-server-tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee resnet50-server-tr-accu.txt


echo " RETINANET Server Test "

make run RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=default" | tee retinanet-server-df-perf.txt
make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=default --test_mode=AccuracyOnly" | tee retinanet-server-df-accu.txt

make run RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=triton" | tee retinanet-server-tr-perf.txt
make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=triton --test_mode=AccuracyOnly" | tee retinanet-server-tr-accu.txt



echo " Server Test Done! "

