#!/usr/bin/env bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=rn50-mlperf
export WORKDIR=${CUR_DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    sudo rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

conda create -n ${CONDA_ENV_NAME} python=3.9 --yes

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

echo "Installiing dependencies for RN50"
python -m pip install sklearn onnx
python -m pip install dataclasses
python -m pip install opencv-python
python -m pip install absl-py
python -m pip install matplotlib Pillow pycocotools
python -m pip install setuptools==65.5.1 wheel==0.38.1 future==0.18.3

conda install typing_extensions --yes
conda config --add channels intel

conda install ninja pyyaml cmake cffi typing intel-openmp --yes

conda install -c intel mkl=2022.0.1 --yes
conda install -c intel mkl-include=2022.0.1 --yes
conda install -c conda-forge llvm-openmp --yes
conda install -c conda-forge jemalloc --yes

conda install six requests dataclasses psutil --yes



export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

#build pytorch and intel-pytorch-extension
git clone https://github.com/pytorch/pytorch.git pytorch
cd pytorch

git fetch origin pull/76869/head:opt-cat
git checkout v1.12.0-rc7
git merge opt-cat --no-edit

git submodule sync
git submodule update --init --recursive
git fetch origin pull/89925/head
git cherry-pick 78cad998e505b667d25ac42f8aaa24409f5031e1

python -m pip install -r requirements.txt
python setup.py install

cd ${WORKDIR}
git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu-dev
cd ipex-cpu-dev
git checkout v1.12.0
git submodule sync
git submodule update --init --recursive

cp ${CUR_DIR}/input_output_aligned_scales.patch .
#cp ${CUR_DIR}/runtime_ignore_dequant_check.patch .

git apply input_output_aligned_scales.patch
#git apply runtime_ignore_dequant_check.patch

python -m pip install -r requirements.txt
python setup.py install
export IPEX_PATH=${PWD}/build/Release/packages/intel_extension_for_pytorch

export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`

cd ${WORKDIR}

# Install Loadgen
echo "=== Installing loadgen ==="
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
git checkout master
git log -1
git submodule update --init --recursive third_party/pybind/
cd loadgen
mkdir build && cd build
cmake ..
make -j$(nproc)
# Build python lib
cd ..
CFLAGS="-std=c++14" python setup.py install
cd ..
cp ./mlperf.conf ${CUR_DIR}/src/mlperf.conf


export LOADGEN_DIR=${PWD}

cd ${WORKDIR}

# Build torchvision
echo "Installiing torch vision"
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
cd ${WORKDIR}

# Build OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
mkdir build && cd build
cmake ..
make -j$(nproc)

export OPENCV_DIR=${PWD}

cd ${WORKDIR}

# Download rapidjson headers
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
git checkout e4bde977

export RAPIDJSON_INCLUDE_DIR=${PWD}/include

cd ${WORKDIR}

# Build Gflags
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir build && cd build
cmake ..
make -j${nproc}

export GFLAGS_DIR=${PWD}

cd ${WORKDIR}

# Build OneDNN
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
git checkout rls-v2.6
mkdir build && cd build
cmake ..
make -j$(nproc)

export ONEDNN_DIR=${PWD}

# onednn - ckernels
cd ${CUR_DIR}/src/ckernels/ && mkdir 3rdparty
cd 3rdparty
git clone https://github.com/oneapi-src/oneDNN.git onednn
cd onednn
git checkout v2.6

