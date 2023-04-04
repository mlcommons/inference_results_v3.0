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
    perl-Data-Dumper \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    libogg \
    libvorbis \ 
    && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN echo "alias ll='ls -l'" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} ninja cmake jemalloc inflect libffi pandas requests toml tqdm unidecode scipy==1.9.3 && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp librosa wheel==0.38.1 setuptools==65.5.1 future==0.18.3 && \
    /opt/conda/bin/conda install -c intel mkl mkl-include intel-openmp && \
    /opt/conda/bin/conda clean -ya && \
    pip install sox

ARG LLVM_VERSION=llvmorg-15.0.7
ENV LD_LIBRARY_PATH "/opt/conda/lib":${LD_LIBRARY_PATH}
RUN cd /opt && git clone https://github.com/llvm/llvm-project.git && \
    cd llvm-project && git checkout ${LLVM_VERSION} && \
    mkdir build && cd build && \
    cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX" && \
    ninja && ninja install


FROM conda AS build
COPY --from=conda /opt/conda /opt/conda
#COPY --from=conda /opt/llvm-project /opt/llvm-project
WORKDIR /opt/workdir
COPY ./code/rnnt/pytorch-cpu/patches /opt/workdir/patches
ENV CONDA_PREFIX "/opt/conda"
ARG PYTORCH_VERSION=v1.12.0
RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && git checkout $PYTORCH_VERSION && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/ideep/mkl-dnn && git apply /opt/workdir/patches/clang_mkl_dnn.patch && \
    cd ../../.. && git apply /opt/workdir/patches/pytorch_official_1_12.patch && \
    pip install -r requirements.txt && \
    CC=clang CXX=clang++ USE_CUDA=OFF python setup.py install && cd .. && pip list|grep torch && rm -rf pytorch

FROM build as setup
COPY --from=build /opt/conda /opt/conda
#COPY --from=build /opt/workdir /opt/workdir
ARG ONEDNN_VERSION=v2.6
ARG third_party_dir=/opt/workdir/code/rnnt/pytorch-cpu/third_party
WORKDIR /opt/workdir
#COPY ./intel_inference_datacenter_v2-1 intel_inference_datacenter_v2-1
COPY ./code/rnnt /opt/workdir/code/rnnt
COPY ./code/run_clean.sh /opt/workdir/code/run_clean.sh
RUN if [ -d ${third_party_dir} ];then rm -rf ${third_party_dir}; fi && \
    mkdir ${third_party_dir} && cd ${third_party_dir}
RUN wget --no-check-certificate https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O flac-1.3.2.tar.xz && \
    tar xf flac-1.3.2.tar.xz && \
    pushd flac-1.3.2 && \
    ./configure --prefix=${third_party_dir} && make && make install && \
    popd
RUN wget --no-check-certificate https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O sox-14.4.2.tar.gz && \
    tar zxf sox-14.4.2.tar.gz && \
    pushd sox-14.4.2 && \
    LDFLAGS="-L${third_party_dir}/lib" CFLAGS="-I${third_party_dir}/include" ./configure --prefix=${third_party_dir} --with-flac && make && make install && \
    popd
RUN rm flac-1.3.2.tar.xz sox-14.4.2.tar.gz 
ENV CONDA_PREFIX "/opt/conda"
RUN cd /opt/workdir/code/rnnt/pytorch-cpu/ && \
    if [ -d "inference" ];then rm -rf inference; fi && \
    git clone --recursive https://github.com/mlcommons/inference.git && \
    pushd inference && git submodule sync && git submodule update --init --recursive && \
    pushd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    popd && \
    popd && \
    cp ./inference/mlperf.conf /opt/workdir/code/rnnt/pytorch-cpu/configs/. && \
    cd mlperf_plugins && if [ -d "onednn" ];then rm -rf onednn; fi && git clone https://github.com/oneapi-src/oneDNN.git onednn && \
    cd onednn && git checkout ${ONEDNN_VERSION} && \
    cd ../../ && rm -rf /opt/conda/lib/cmake/mkl/* && mkdir build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DBUILD_TPPS_INTREE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON .. && \
    ninja


FROM dev-base as mp
COPY --from=setup /opt/conda /opt/conda
COPY --from=setup /opt/workdir /opt/workdir
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"
ENV LD_LIBRARY_PATH "/opt/workdir/code/rnnt/pytorch-cpu/third_party/lib:$LD_LIBRARY_PATH"
