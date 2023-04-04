#!/bin/bash

set -ex
# Create new env and activate it
conda install -y ninja cmake jemalloc inflect libffi pandas requests toml tqdm unidecode scipy==1.9.3
conda install -c intel -y mkl mkl-include intel-openmp
conda install -c conda-forge -y llvm-openmp librosa
pip install sox

set +x
