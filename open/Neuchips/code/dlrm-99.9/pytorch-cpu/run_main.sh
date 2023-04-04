#!/bin/bash

dtype="int8"
batch_size=$(($BATCH_SIZE + 0))
if [ $# -ge 2 ]; then
    if [[ $2 == "accuracy" ]]; then
        test_type="accuracy"
    fi
    # if [[ $2 == "bf16" ]] || [[ $3 == "bf16" ]]; then
    #     dtype="bf16"
    # elif [[ $2 == "int8" ]] || [[ $3 == "int8" ]]; then
    #     dtype="int8"
    # fi
else
    test_type="performance"
fi

export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=$CPUS_PER_SOCKET
export KMP_AFFINITY="granularity=fine,compact,1,0"
#export DNNL_PRIMITIVE_CACHE_CAPACITY=20971520
#export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export DLRM_DIR=$PWD/python/model
#export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=30469645312

mode="Offline"
extra_option="--samples-per-query-offline=204800"
if [ $1 == "server" ]; then
    mode="Server"
    extra_option=""
fi

echo "Running $mode bs=$batch_size $dtype $test_type $DNNL_MAX_CPU_ISA"
./run_local.sh pytorch dlrm terabyte cpu $dtype $test_type --scenario $mode --max-ind-range=40000000 --samples-to-aggregate-quantile-file=../tools/dist_quantile.txt --max-batchsize=$batch_size $extra_option
