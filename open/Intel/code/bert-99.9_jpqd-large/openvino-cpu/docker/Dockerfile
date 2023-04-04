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
    mesa-libGL \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    # glib2-devel tbb-devel \
    && rm -rf /var/lib/yum/lists/*
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH


FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya


FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG OPENVINO_VERSION=2022.3.0
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    python -m pip install --upgrade pip && \
    python -m pip install absl-py numpy transformers openvino-dev==${OPENVINO_VERSION} "setuptools>=65.5.1" \
        intel-tensorflow wheel==0.38.0 cryptography==39.0.1


FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
SHELL ["/bin/bash", "-c"]
WORKDIR /opt/workdir/bert-99.9_jpqd-large/openvino-cpu
COPY ./code/bert-99.9_jpqd-large/openvino-cpu/src/*.py /opt/workdir/bert-99.9_jpqd-large/openvino-cpu/src/
COPY ./code/bert-99.9_jpqd-large/openvino-cpu/*.conf /opt/workdir/bert-99.9_jpqd-large/openvino-cpu/
COPY ./code/bert-99.9_jpqd-large/openvino-cpu/*.sh /opt/workdir/bert-99.9_jpqd-large/openvino-cpu/
ARG MLPERF_BRANCH=master
ARG COMMIT_V3=f5367250115ad4febf1334b34881ab74f2e55bfe
# ENV DNNL_MAX_CPU_ISA "AVX512_CORE_AMX"
# ENV CUDA_VISIBLE_DEVICES ""
# ENV USE_GPU "0"
ENV BERT_DATA_DIR "/opt/workdir/datasets/squad"
ENV BERT_MODEL_DIR "/opt/workdir/models/bert"
ENV MLPERF_INFERENCE_REPO=/opt/workdir/bert-99.9_jpqd-large/openvino-cpu/dependencies/mlperf-inference
ENV PYTHONPATH "/opt/workdir/bert-99.9_jpqd-large/openvino-cpu/dependencies/mlperf-inference/language/bert"
RUN source /opt/rh/gcc-toolset-11/enable && \
    git clone https://github.com/mlcommons/inference.git ${MLPERF_INFERENCE_REPO} && \
    cd ${MLPERF_INFERENCE_REPO} && git checkout ${COMMIT_V3} && git submodule update --init third_party/pybind && \
    cd ${MLPERF_INFERENCE_REPO}/loadgen && \
    CFLAGS="-std=c++14" python setup.py install

