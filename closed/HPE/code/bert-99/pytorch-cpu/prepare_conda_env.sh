#! /bin/bash

# Create new env and activate it
conda install -y python=3.8
conda install -y ninja
conda install -y cmake
conda install -c intel mkl --yes
conda install -c intel mkl-include --yes
conda install -c intel intel-openmp --yes
conda install -c conda-forge llvm-openmp --yes
conda install -c conda-forge jemalloc --yes
