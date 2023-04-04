export NUM_SOCKETS=2      # i.e. 8
export CPUS_PER_SOCKET=96   # i.e. 28
export CPUS_PER_PROCESS=96  # which determine how much processes will be used
                            # process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=1  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                             # total-instance = instance-per-process * process-per-socket
export BATCH_SIZE=16800    #8400
#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
