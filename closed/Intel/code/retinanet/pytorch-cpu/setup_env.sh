CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env

export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json

export DATA_DIR=${WORKLOAD_DATA}/openimages
export MODEL_PATH=${WORKLOAD_DATA}/retinanet-int8-model.pth
