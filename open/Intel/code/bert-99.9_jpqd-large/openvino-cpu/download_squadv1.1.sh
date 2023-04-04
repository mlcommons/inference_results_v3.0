#!/bin/bash

DATASET_PATH=${1:-squad}

mkdir -p ${DATASET_PATH}

cd ${DATASET_PATH}

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dev-v1.1.json
wget https://zenodo.org/record/3750364/files/vocab.txt?download=1 -O vocab.txt

