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
    cmake \
    sudo \
    wget \
    findutils \
    gcc \
    gcc-c++ \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    perl-Data-Dumper \
    && rm -rf /var/lib/apt/lists/*
#RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN echo "alias ll='ls -l'" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    python3 -m pip install cryptography==38.0.4 && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} ninja cmake && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp jemalloc wheel==0.38.1 setuptools==65.5.1 future==0.18.3 && \
    /opt/conda/bin/conda install -c intel mkl mkl-include intel-openmp && \
    /opt/conda/bin/conda clean -ya

ARG LLVM_VERSION=llvmorg-15.0.7
ENV LD_LIBRARY_PATH "/opt/conda/lib":${LD_LIBRARY_PATH}
RUN cd /opt && git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project && git checkout ${LLVM_VERSION} && \
    mkdir build && cd build && \
    conda list |grep ninja && \
    cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX" && \
    ninja && ninja install
    

FROM conda AS build
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/bert-99/pytorch-cpu/patches patches
ENV CONDA_PREFIX "/opt/conda"
ARG PYTORCH_VERSION=v1.12.0
ARG TRANSFORMERS_COMMIT=9f4e0c23d68366985f9f584388874477ad6472d8
RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && git checkout $PYTORCH_VERSION && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/ideep/mkl-dnn && git apply /opt/workdir/patches/clang_mkl_dnn.patch && \
    cd ../../.. && git apply /opt/workdir/patches/pytorch_official_1_12.patch && \
    pip install -r requirements.txt && \
    CC=clang CXX=clang++ USE_CUDA=OFF python setup.py install && cd .. && rm -rf /opt/workdir/pyorch && \
    git clone https://github.com/huggingface/transformers.git &&  \
    cd transformers && git checkout ${TRANSFORMERS_COMMIT} && \ 
    git apply ../patches/transformers.patch && \
    python setup.py install && \
    cd .. && \ 
    rm -rf /opt/workdir/transfomers 

FROM conda as setup
COPY --from=build /opt/conda /opt/conda
ARG ONEDNN_VERSION=v2.6
WORKDIR /opt/workdir
COPY ./code/bert-99 code/bert-99
COPY ./code/run_clean.sh code/run_clean.sh
ENV CONDA_PREFIX "/opt/conda"
RUN cd code/bert-99/pytorch-cpu/ && \
    if [ -d "inference" ];then rm -rf inference ;fi && \
    git clone --recursive https://github.com/mlcommons/inference.git  && \
    cd mlperf_plugins && if [ -d "onednn" ];then rm -rf onednn ; fi && git clone https://github.com/oneapi-src/oneDNN.git onednn&& \
    cd onednn && git checkout ${ONEDNN_VERSION} && git apply ../../patches/onednnv2_6.patch && \
    cd ../../ && rm -rf /opt/conda/lib/cmake/mkl/* && mkdir build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DBUILD_TPPS_INTREE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON .. && \
    ninja && pip install boto3 tokenization


#ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.8/site-packages/lib/"
FROM dev-base as mp
COPY --from=setup /opt/conda /opt/conda
COPY --from=setup /opt/workdir /opt/workdir
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"
