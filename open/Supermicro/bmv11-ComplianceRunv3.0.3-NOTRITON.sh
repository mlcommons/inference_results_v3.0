#!/bin/bash
##############################################################
#  
#  
#  
#  
#  
##############################################################


#echo " 3D-Unet Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default" | tee audit-3d-unet-offline-df-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=high_accuracy" | tee audit-3d-unet-offline-high_accu-perf.txt




#echo " BERT Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=default" | tee audit-bert-offline-df-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy" | tee audit-bert-offline-high_accu-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=triton" | tee audit-bert-offline-tr-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy_triton" | tee audit-bert-offline-high_accu_tr-perf.txt




#echo " DLRM Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default" | tee audit-dlrm-offline-df-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy" | tee audit-dlrm-offline-high_accu-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=triton" | tee audit-dlrm-offline-tr-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy_triton" | tee audit-dlrm-offline-high_accu_tr-perf.txt




#echo " RNN-T Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Offline --config_ver=default" | tee audit-rnnt-offline-df-perf.txt




#echo " RESNET50 Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default" | tee audit-resnet50-offline-df-perf.txt




#make run_audit_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=LWIS" | tee audit-resnet50-offline-lwis-perf.txt





#echo " RETINANET Offline Test "

#make run_audit_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=default" | tee audit-retinanet-offline-df-perf.txt




#make run_audit_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --config_ver=triton" | tee audit-retinanet-offline-tr-perf.txt




#echo " BERT Server Test "

#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=default" | tee audit-bert-server-df-perf.txt

#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy" | tee audit-bert-server-high_accu-perf.txt



#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=triton" | tee audit-bert-server-tr-perf.txt

#make run_audit_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy_triton" | tee audit-bert-server-high_accu_tr-perf.txt









echo " RNNT Server Test "

make run_audit_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --config_ver=default" | tee audit-rnnt-server-df-perf.txt



echo " RESNET50 Server Test "

make run_audit_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=default" | tee audit-resnet50-server-df-perf.txt





echo " RETINANET Server Test "

make run_audit_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Server --config_ver=default" | tee audit-retinanet-server-df-perf.txt





#echo " DLRM Server Test "

#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default" | tee audit-dlrm-server-df-perf.txt

#make run_audit_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy" | tee audit-dlrm-server-high_accu-perf.txt





echo " Server Test Done! "

