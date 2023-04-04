#!/bin/bash

set -ex

CONDA_ENV=${1:-'rnnt-infer'}
HOME_DIR=${2:-${PWD}}

pushd ${HOME_DIR}

# use gcc-9 compile clang-15
echo '==> Building clang-15'
git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
git checkout llvmorg-15.0.7
mkdir build
pushd build
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX"
ninja
ninja install
popd
popd

echo '==> Building mlperf-loadgen'
git clone --recurse-submodules https://github.com/mlcommons/inference.git
pushd inference
git submodule sync && git submodule update --init --recursive
pushd loadgen
CFLAGS="-std=c++14" python setup.py install
popd
popd
cp ./inference/mlperf.conf ${HOME_DIR}/configs/.

echo '==> Building third-party'
third_party_dir=${HOME_DIR}/third_party
mkdir -p ${third_party_dir}
pushd ${third_party_dir}

wget --no-check-certificate https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O flac-1.3.2.tar.xz
tar xf flac-1.3.2.tar.xz
pushd flac-1.3.2
./configure --prefix=${third_party_dir} && make && make install
popd

wget --no-check-certificate https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O sox-14.4.2.tar.gz
tar zxf sox-14.4.2.tar.gz
pushd sox-14.4.2
LDFLAGS="-L${third_party_dir}/lib" CFLAGS="-I${third_party_dir}/include" ./configure --prefix=${third_party_dir} --with-flac && make && make install
popd
popd

echo '==> Building pytorch with mkl'
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CMAKE_LIBRARY_PATH=${CMAKE_PREFIX_PATH}/lib
export CMAKE_INCLUDE_PATH=${CMAKE_PREFIX_PATH}/include

git clone https://github.com/pytorch/pytorch.git
pushd pytorch
git checkout v1.12.0
git submodule sync && git submodule update --init --recursive
pushd third_party/ideep/mkl-dnn
git apply ${HOME_DIR}/patches/clang_mkl_dnn.patch
popd
git apply ${HOME_DIR}/patches/pytorch_official_1_12.patch
pip install -r requirements.txt
CC=clang CXX=clang++ USE_CUDA=OFF python -m pip install -e .
popd

echo '==> Preparing onednn'
pushd mlperf_plugins
git clone https://github.com/oneapi-src/oneDNN.git onednn
pushd onednn
git checkout v2.6
popd
popd

echo '==> Building mlperf_plugins, C++ loadgen & SUT'
#git submodule sync && git submodule update --init --recursive
mkdir build
pushd build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DBUILD_TPPS_INTREE=ON -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON -DCMAKE_BUILD_TYPE=Release ..
ninja
popd

# delete MKL ERROR
rm -rf ${CONDA_PREFIX}/lib/cmake/mkl/*

set +x

