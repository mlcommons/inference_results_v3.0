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
	numactl \
	cmake \
	sudo \
	wget \
	procps \
	numactl-devel.x86_64 \
	gcc-toolset-11-gcc \
	gcc-toolset-11-gcc-c++ \
	&& rm -rf /var/lib/yum/lists
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc && \
	echo "source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
	chmod +x ~/miniconda.sh && \
	~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	/opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
	/opt/conda/bin/conda install -c conda-forge -y cmake libunwind gperftools wheel==0.38.1 setuptools==65.5.1 future==0.18.3 && \
	/opt/conda/bin/conda install -y intel-openmp mkl mkl-include numpy --no-update-deps && \
	/opt/conda/bin/conda install -y glog && \
	/opt/conda/bin/conda install -y pytorch=1.13.1 -c pytorch && \
	/opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
COPY ./l_HPCKit_p_2022.2.0.191.sh l_HPCKit_p_2022.2.0.191.sh
ADD ./oneDNN oneDNN
ADD ./pytorch-cpu pytorch-cpu
ENV CONDA_PREFIX /opt/conda
ENV WORKDIR /opt/workdir/code/dlrm
RUN --mount=type=cache,target=/opt/ccache \
	/bin/bash l_HPCKit_p_2022.2.0.191.sh -a -s --eula accept && \
	mkdir -p /data/mlperf_data/dlrm && \
	pip install absl-py tqdm numpy && \
	pip install -e git+https://github.com/mlperf/logging@1.1.0-rc3#egg=mlperf-logging && \
	source /opt/rh/gcc-toolset-11/enable && \
	source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh && \
	mkdir -p /opt/workdir/code/dlrm && export WORKDIR=/opt/workdir/code/dlrm && \
	cd ${WORKDIR} && \
	git clone https://github.com/mlcommons/inference.git && \
	cd inference/loadgen && \
	mkdir build && cd build && \
	CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/loadgen && \
	make -j && make install && \
	cd ${WORKDIR} && \
	mv /oneDNN ${WORKDIR} && \
	cd oneDNN && \
	git checkout v2.7 && \
	mkdir build && cd build && \
	CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/spronednn && \
	make -j && make install && \
	cd ${WORKDIR} && \
	mv /pytorch-cpu ${WORKDIR} && \
  cp ${WORKDIR}/inference/mlperf.conf /opt/workdir/code/dlrm/pytorch-cpu/. && \
	cd pytorch-cpu/src && \
	git clone https://github.com/rogersce/cnpy.git && \
	cd cnpy && \
	git checkout 4e8810b1a8637695171ed346ce68f6984e585ef4 && \
	cd ${WORKDIR}/pytorch-cpu/src && \
	mkdir build && cd build &&\
	export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:${WORKDIR}/spronednn && \
	CC=icx CXX=icpx cmake .. -DLOADGEN_DIR=${WORKDIR}/loadgen -DONEDNN_DIR=${WORKDIR}/spronednn && \
	make -j
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"
