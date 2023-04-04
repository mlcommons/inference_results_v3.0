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
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    cmake \
    unzip \
    wget \
    sudo \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/apt/lists/*
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc 
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -y intel-openmp mkl mkl-include mkl-service mkl_fft mkl_random --no-update-deps && \
    /opt/conda/bin/conda install -y -c conda-forge cmake jemalloc gperftools wheel==0.38.1 setuptools==65.5.1 future==0.18.3 && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
COPY ./code/3d-unet-99.9/pytorch-cpu/unet3d.patch unet3d.patch
ARG IPEX_VERSION=1.9.0-rc
ARG ONEDNN_VERSION=v2.7
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    pip install batchgenerators nibabel scipy pandas && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git && \
    cd intel-extension-for-pytorch && git checkout ${IPEX_VERSION} && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/mkl-dnn/ && git checkout ${ONEDNN_VERSION} && cd ../.. && \
    git apply ../unet3d.patch && \
    python setup.py install

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/3d-unet-99.9 code/3d-unet-99.9
COPY ./calibration/3d-unet-99.9 calibration/3d-unet-99.9
RUN source /opt/rh/gcc-toolset-11/enable && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git && \
    cd inference && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd .. && cp ./mlperf.conf /opt/workdir/code/3d-unet-99.9/pytorch-cpu/.

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
#ENV LD_LIBRARY_PATH "/opt/conda/lib":${LD_LIBRARY_PATH}
ENV CONDA_PREFIX "/opt/conda"
