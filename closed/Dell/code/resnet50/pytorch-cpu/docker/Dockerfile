# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=rockylinux:8.7
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    numactl-devel \
    cmake \
    sudo \
    wget \
    mesa-libGL \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/yum/lists/*
    
# RUN git config --global user.name "user" && git config --global user.email "user@example.com"
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH



FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN echo "Before conda installations"
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} typing_extensions && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -y ninja pyyaml cmake cffi typing intel-openmp && \
    /opt/conda/bin/conda install -y six requests dataclasses psutil && \
    /opt/conda/bin/conda install -c intel mkl=2022.0.1 mkl-include=2022.0.1 -y && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp jemalloc -y && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG PYTORCH_VERSION=v1.12.0-rc7
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    pip install setuptools==65.5.1 wheel==0.38.1 future==0.18.3 && \
    pip install scikit-learn onnx dataclasses opencv-python absl-py matplotlib Pillow pycocotools && \
    git clone https://github.com/pytorch/pytorch.git pytorch && cd pytorch && \
    git fetch origin pull/76869/head:opt-cat && git config user.email "test@intel.com" && \
    git checkout ${PYTORCH_VERSION} && git merge opt-cat --no-edit && \
    git submodule sync && git submodule update --init --recursive && \
    git fetch origin pull/89925/head && \
    git cherry-pick 78cad998e505b667d25ac42f8aaa24409f5031e1 && git config user.email "test@intel.com" && \
    python setup.py install && cd ..

FROM dev-base as mp-base
COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
WORKDIR /opt/workdir
COPY ./code/resnet50 code/resnet50
COPY ./calibration/resnet50 calibration/resnet50
ARG IPEX_VERSION=v1.12.0
#ARG INFERENCE_COMMIT=8d83d61b5ba8308544aa5a76d311fef1e25d9584
RUN source /opt/rh/gcc-toolset-11/enable && \
    git clone https://github.com/pytorch/vision && \
    cd vision && python setup.py install && cd .. && \
    cd code/resnet50/pytorch-cpu/ && mkdir rn50-mlperf && cd rn50-mlperf && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu-dev && \
    cd ipex-cpu-dev && git checkout ${IPEX_VERSION} && \
    git submodule sync && git submodule update --init --recursive && \
    git apply /opt/workdir/code/resnet50/pytorch-cpu/input_output_aligned_scales.patch && \
    # git apply /opt/workdir/code/resnet50/pytorch-cpu/runtime_ignore_dequant_check.patch && \
    python -m pip install -r requirements.txt && \
    python setup.py install && \
    cd .. && \
    git clone https://github.com/mlcommons/inference.git mlperf_inference && \
    cp mlperf_inference/mlperf.conf /opt/workdir/code/resnet50/pytorch-cpu/src/. && \
    cd mlperf_inference && \
    git submodule update --init --recursive third_party/pybind/ && \
    cd loadgen && mkdir build && cd build && cmake .. && make -j$(nproc) && cd .. && \
    CFLAGS="-std=c++14" python setup.py install && cd ../.. && \
    git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.x && \
    mkdir build && cd build && cmake .. && make -j$(nproc) && \
    cd ../.. && \
    git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && git checkout e4bde977 && \
    cd .. && \
    git clone https://github.com/gflags/gflags.git && cd gflags && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && cd ../.. && \
    git clone https://github.com/oneapi-src/oneDNN.git && cd oneDNN && \
    git checkout rls-v2.6 && mkdir build && cd build && cmake .. && make -j$(nproc) && \
    cd ../.. && \
    cd ../ && \
    cd src/ckernels/ && mkdir 3rdparty && \
    cd 3rdparty && \
    git clone https://github.com/oneapi-src/oneDNN.git onednn && \
    cd onednn && \
    git checkout v2.6 && cd ../../../../ && \
    cd rn50-mlperf/ && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    export IPEX_PATH=${PWD}/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    export LOADGEN_DIR=${PWD}/mlperf_inference/loadgen && \
    export OPENCV_DIR=${PWD}/opencv/build && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/rapidjson/include && \
    export GFLAGS_DIR=${PWD}/gflags/build && \
    export ONEDNN_DIR=${PWD}/oneDNN && \
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIR}/lib:${ONEDNN_DIR}/build/src:${CONDA_PREFIX}/lib && \
    export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib && \
    cd ../ && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -DONEDNN_DIR=${ONEDNN_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release -j$(nproc)

ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD "/opt/conda/lib/libjemalloc.so":"/opt/conda/lib/libiomp5.so":${LD_PRELOAD}
ENV CONDA_ENV_NAME "/opt/workdir/code/resnet50/pytorch-cpu"
ENV CONDA_PREFIX "/opt/conda"
