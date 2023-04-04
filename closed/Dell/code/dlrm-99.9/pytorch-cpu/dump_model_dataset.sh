#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
python -u python/dump_model.py ${MODEL}/dlrm_terabyte.pytorch ${DUMP_PATH}/dlrm_model.npz
python -u python/dump_dataset.py ${DATASET} ${DUMP_PATH}
