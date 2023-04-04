#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Force the users to provide MLPERF_PATH var
if [[ -z "$MLPERF_PATH" ]]; then
        echo "Must provide MLPERF_PATH in environment variables" 1>&2
        exit 1
fi

# Create a symlink for python 3.8
sudo unlink /usr/bin/python && sudo ln -s /usr/bin/python3 /usr/bin/python

sudo apt install -y python3.8-dev \
  && sudo apt install -y virtualenv moreutils libnuma-dev numactl sshpass make \
  && sudo apt install -y pkg-config zip g++ unzip zlib1g zlib1g-dev ntpdate gdb \
  && sudo apt install -y --no-install-recommends clang libclang-dev libglib2.0-dev \
  && sudo apt install -y libhdf5-serial-dev hdf5-tools default-jdk \
  && sudo apt install -y protobuf-compiler libprotoc-dev \
  && sudo apt install -y zlib1g-dev zip libjpeg8-dev libhdf5-dev libtiff5-dev libffi-dev \
  && sudo ln -s /usr/lib/aarch64-linux-gnu/libclang-10.so.1 /usr/lib/aarch64-linux-gnu/libclang.so

# TODO: Fix libopenmpi-dev installation for DOS 6.0.2 RC2, need to downgrade some packages.
# Temporary workaround: sudo apititude install -y libopenmpi-dev (press in the order of N->Y->Y)
sudo apt install -y libopenmpi-dev

# matplotlib dependencies
sudo apt install -y libssl-dev libfreetype6-dev libpng-dev  \
  && sudo apt install -y libatlas3-base libopenblas-base \
  && sudo apt install -y git git-lfs && git-lfs install

# cmake, install cmake-3.18.4
# there is bug with 3.19.4, where the gflags is not linked properly
sudo apt remove -y cmake \
  && sudo rm -rf cmake-* \
  && wget https://cmake.org/files/v3.18/cmake-3.18.4.tar.gz \
  && tar -xf cmake-3.18.4.tar.gz \
  && cd cmake-3.18.4 && ./configure && sudo make -j2 install \
  && sudo ln -s /usr/local/bin/cmake /usr/bin/cmake \
  && cd /tmp && rm -rf cmake-*

# Setup CUDA and lib path for builds
export CUDA_ROOT=/usr/local/cuda
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$CUDA_ROOT/bin:/usr/bin:$PATH
export CPATH=$CUDA_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

cd /tmp \
  && wget https://bootstrap.pypa.io/get-pip.py \
  && sudo -E python3 get-pip.py --force-reinstall \
  && sudo rm /tmp/get-pip.py

# Setup python basic dependencies (takes ~5 min)
sudo -E python3 -m pip install --upgrade setuptools wheel virtualenv \
  && sudo -E python3 -m pip install Cython==0.29.23 \
  && sudo -E python3 -m pip install numpy==1.18.5 \
  && sudo rm -rf /usr/lib/python3/dist-packages/yaml /usr/lib/python3/dist-packages/PyYAML* 

# Install required dependencies (takes ~5 min)
# Pre-built wheels from https://github.com/nvzhihanj/mlperf-aarch64-wheel
# Make sure MLPerf path is set at $MLPERF_PATH
sudo -E python3 -m pip install -r $MLPERF_PATH/closed/NVIDIA/scripts/requirements_orin.txt

# Install other dependencies (takes ~2 min)
sudo -E python3 -m pip install scipy==1.6.3 \
  && sudo -E python3 -m pip install matplotlib==3.4.2 pycocotools==2.0.2 \
  && sudo -E python3 -m pip install scikit-learn@https://github.com/nvzhihanj/mlperf-aarch64-wheel/releases/download/v2.0.0/scikit_learn-0.22.1-cp38-cp38-linux_aarch64.whl \
  && sudo -E python3 -m pip install pycuda==2021.1

# Install Triton dependencies (takes ~3 min)
sudo apt install -y autoconf automake build-essential libb64-dev libre2-dev \
    libcurl4-openssl-dev libtool libboost-dev rapidjson-dev patchelf \
    libopenblas-dev software-properties-common
sudo apt install -y --allow-downgrades libopencv-dev

# Install cub 
cd /tmp \
  && sudo rm -rf cub-1.8.0.zip cub-1.8.0 /usr/include/aarch64-linux-gnu/cub \
  && wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
  && unzip cub-1.8.0.zip \
  && sudo mv cub-1.8.0/cub /usr/include/aarch64-linux-gnu/ \
  && sudo rm -rf cub-1.8.0.zip cub-1.8.0

# Install gflags
sudo rm -rf gflags \
  && git clone -b v2.2.1 https://github.com/gflags/gflags.git \
  && cd gflags \
  && mkdir build && cd build \
  && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
  && make -j8 \
  && sudo make install \
  && cd /tmp && sudo rm -rf gflags

# Install glog
sudo rm -rf glog \
  && git clone -b v0.3.5 https://github.com/google/glog.git \
  && cd glog \
  && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
  && cmake --build build \
  && sudo cmake --build build --target install \
  && cd /tmp && sudo rm -rf glog

# Pytorch
sudo -E python3 -m pip install torch@https://github.com/nvzhihanj/mlperf-aarch64-wheel/releases/download/v2.0.0/torch-1.4.0a0+b74e50a-cp38-cp38-linux_aarch64.whl \
  && sudo -E python3 -m pip install torchvision==0.2.2

# DALI
# building from scratch takes ~45 minutes
rm -rf protobuf-cpp-3.11.1.tar.gz \
 && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.1/protobuf-cpp-3.11.1.tar.gz \
 && tar -xzf protobuf-cpp-3.11.1.tar.gz \
 && rm protobuf-cpp-3.11.1.tar.gz \
 && cd protobuf-3.11.1 \
 && ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared \
 && make -j8 \
 && sudo make install \
 && sudo ldconfig \
 && cd /tmp \
 && rm -rf protobuf-3.11.1 \
 && cd /usr/local \
 && sudo rm -rf DALI \
 && sudo git clone -b release_v0.31 --recursive https://github.com/NVIDIA/DALI \
 && cd DALI \
 && sudo mkdir build \
 && cd build \
 && sudo -E cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDA_TARGET_ARCHS="87" \
    -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DBUILD_LMDB=OFF -DBUILD_NVTX=OFF -DBUILD_NVJPEG=OFF \
    -DBUILD_LIBTIFF=OFF -DBUILD_NVOF=OFF -DBUILD_NVDEC=OFF -DBUILD_LIBSND=OFF -DBUILD_NVML=OFF -DBUILD_FFTS=ON \
    -DVERBOSE_LOGS=OFF -DWERROR=OFF -DBUILD_WITH_ASAN=OFF .. \
 && sudo -E make -j8 \
 && sudo -E make install \
 && sudo -E python3 -m pip install dali/python/ \
 && sudo mv /usr/local/DALI/build/dali/python/nvidia/dali /tmp/dali \
 && sudo rm -rf /usr/local/DALI \
 && sudo mkdir -p /usr/local/DALI/build/dali/python/nvidia/ \
 && sudo mv /tmp/dali /usr/local/DALI/build/dali/python/nvidia/ \
 && cd /tmp

# Install ONNX graph surgeon, needed for 3D-Unet ONNX preprocessing.
cd /tmp \
 && rm -rf TensorRT \
 && git clone https://github.com/NVIDIA/TensorRT.git \
 && cd TensorRT \
 && git checkout release/8.0 \
 && cd tools/onnx-graphsurgeon \
 && make build \
 && sudo -E python3 -m pip install --no-deps -t /usr/local/lib/python3.8/dist-packages --force-reinstall dist/*.whl \
 && cd /tmp \
 && rm -rf TensorRT
