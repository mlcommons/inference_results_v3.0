set -x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

accuracy=$1

sut_dir=$(pwd)
executable=${sut_dir}/build/bert_inference
mode="Server"
OUTDIR="$sut_dir/test_log"
find ${sut_dir} -maxdepth 1 -name "test_log" | xargs rm -rf
mkdir ${OUTDIR}
CONFIG="-n 14 -j 8 --test_scenario=${mode} --model_file=${sut_dir}/bert.pt --sample_file=${sut_dir}/squad.pt --mlperf_config=${sut_dir}/inference/mlperf.conf --user_config=${sut_dir}/user.conf -o ${OUTDIR} --warmup -w 1300 -u 1560 ${accuracy}"

${executable} ${CONFIG}

if [ ${accuracy} = "--accuracy" ]; then
	if [ ! -d "${DATA_PATH}" ]; then
		echo "please export the data path first!"
		exit 1
	fi

	vocab_file=`find ${DATA_PATH} -name vocab.txt` #path/to/vocab.txt
	val_data=`find ${DATA_PATH} -name dev-v1.1.json` #path/to/dev-v1.1.json
	python ./inference/language/bert/accuracy-squad.py \
		--vocab_file $vocab_file \
		--val_data $val_data \
		--log_file ./test_log/mlperf_log_accuracy.json \
		--out_file predictions.json \
		2>&1 | tee ./test_log/accuracy.txt
fi
set +x
