export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

bash run_mlperf.sh --type=$1 \
	           --precision=int8 \
		   --user-conf=user.conf \
		   --num-instance=28 \
		   --cpus-per-instance=4 \
                   --scenario=Offline


