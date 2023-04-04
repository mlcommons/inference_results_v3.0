set -x
PATTERN='[-a-zA-Z0-9_]*='

run_type=perf

for i in "$@"
do
    case $i in
        --type=*)
            run_type=`echo $i | sed "s/${PATTERN}//"`;;
        --precision=*)
            precision=`echo $i | sed "s/${PATTERN}//"`;;
        --user-conf=*)
            user_conf=`echo $i | sed "s/${PATTERN}//"`;;
        --num-instance=*)
            num_instance=`echo $i | sed "s/${PATTERN}//"`;;
        --cpus-per-instance=*)
            cpus_per_instance=`echo $i | sed "s/${PATTERN}//"`;;
        --cpus-per-process=*)
            cpus_per_process=`echo $i | sed "s/${PATTERN}//"`;;
        --scenario=*)
            scenario=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

if [ ${run_type} = "perf" ];then
    accuracy="--mode Performance"
fi

if [ ${run_type} = "acc" ];then
    accuracy="--mode Accuracy"
fi

if [ -n "${precision}" ];then
    prec="--precision=${precision}"
fi

PYTHON_VERSION=`python -c 'import sys; print ("{}.{}".format(sys.version_info.major, sys.version_info.minor))'`
SITE_PACKAGES=`python -c 'import site; print (site.getsitepackages()[0])'`
IPEX_VERSION=`conda list |grep torch-ipex | awk '{print $2}' `
export LD_LIBRARY_PATH=$SITE_PACKAGES/torch_ipex-${IPEX_VERSION}-py$PYTHON_VERSION-linux-x86_64.egg/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so:$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

echo python3 run.py  ${accuracy} \
                --workload-name 3dunet \
                --mlperf-conf mlperf.conf \
                --user-conf $user_conf \
                --workload-config config.json \
                --num-instance $num_instance \
                --cpus-per-instance $cpus_per_instance \
                --scenario $scenario \
		--warmup 1 \
                $prec


python3 run.py  ${accuracy} \
                --workload-name 3dunet \
                --mlperf-conf mlperf.conf \
                --user-conf $user_conf \
                --workload-config config.json \
                --num-instance $num_instance \
                --cpus-per-instance $cpus_per_instance \
                --scenario $scenario \
                --warmup 1 \
                $prec

echo "Run accuracy script"
if [ ${run_type} = "acc" ];then
   mkdir -p build/postprocessed_data
   python3 accuracy_kits.py --log_file=output_logs/mlperf_log_accuracy.json 2>&1|tee output_logs/accuracy.txt
fi

set +x