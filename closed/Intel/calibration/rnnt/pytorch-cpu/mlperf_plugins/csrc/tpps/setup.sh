export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so"
export OMP_NUM_THREADS=$3
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_HW_SUBSET=1s,56c,1t
# perf report
# vtune -collect hpc-performance ./xxx.exe
# perf mem record -a ./xxx.exe
./xxx.exe $1 $2 $3
