#!/bin/bash

CONDA_ENV_NAME=rn50-mlperf
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}


if [ -z "${CHECKPOINT}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export CHECKPOINT="
    exit 1
fi

if [ -z "${DATA_CAL_DIR}" ]; then
    echo "Path to annotations for calibration images not set. Please set it:"
    echo "export DATA_CAL_DIR="
    exit 1
fi


export ARGS="--batch-size 1 --data-path-cal ${DATA_CAL_DIR} --checkpoint-path ${CHECKPOINT} --save-dir models --calibrate-start-partition --calibrate-end-partition --calibrate-full-weights --save-full-weights --channels-last --massage"

numactl -C 0-55 -m 0 python -u main.py ${ARGS}

echo "Generating binary scales data for kernel backbone"

cp models/resnet50-int8-scales.json src/ckernels/scripts/

python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_256.cpp --batchsize 256
python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_8.cpp --batchsize 8
python src/ckernels/scripts/make_qinfo.py --infile src/ckernels/scripts/resnet50-int8-scales.json --outfile backbone_data_4.cpp --batchsize 4
mv backbone_data_* src/ckernels/src/kernel_rn50