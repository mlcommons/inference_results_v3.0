if [ -z "${CALIBRATION_DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export CALIBRATION_DATA_DIR="
    exit 1
fi

if [ -z "${MODEL_CHECKPOINT}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_CHECKPOINT="
    exit 1
fi

if [ -z "${CALIBRATION_ANNOTATIONS}" ]; then
    echo "Path to annotations for calibration images not set. Please set it:"
    echo "export CALIBRATION_ANNOTATIONS="
    exit 1
fi


NUM_CLASSES=264

export ARGS="--calibrate --cal-iters 100 --precision int8 --num-classes ${NUM_CLASSES} --batch-size 1 --quantized-weights int8-scales-${NUM_CLASSES}.json --data-path ${CALIBRATION_DATA_DIR} --annotation-file ${CALIBRATION_ANNOTATIONS} --num-iters 500 --checkpoint-path ${MODEL_CHECKPOINT} --save-trace-model --save-trace-model-path $( dirname ${MODEL_CHECKPOINT} )/retinanet-BNAS-int8-model.pth"

numactl -C 0-55 -m 0 python -u helpers/main.py ${ARGS}
