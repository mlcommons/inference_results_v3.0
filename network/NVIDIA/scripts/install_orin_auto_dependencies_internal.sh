#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


# Install TensorRT
# As of 1/20/2022, use TRT rel-8.3 nightly
# This step will take ~10GB of disk space. Make sure you have enough left.
export CUDA_VER=11.4
cd /tmp \
    && wget https://urm.nvidia.com/artifactory/sw-tensorrt-generic/cicd/rel-8.3/L1_Nightly/38/trt_build_aarch64_d6l_cuda11.4_release_optimized.tar --user tensorrt-read-only --password "Tensorrt@123" -O TRT.tar \
    && tar -xf TRT.tar \
    && rm -f TRT.tar \
    && cd source/install/aarch64-gnu \
    && mkdir -p trt \
    && tar -xzf cuda-${CUDA_VER}/release_tarfile/TensorRT-*.Ubuntu-20.04.aarch64-gnu.cuda-${CUDA_VER}.cudnn*.tar.gz -C trt --strip-components 1 \
    && tar -xzf TensorRT-*.Ubuntu-20.04.aarch64-gnu.cuda-${CUDA_VER}.internal.tar.gz -C trt --strip-components 1 \
    && rm -rf *.deb *.gz cuda/
    && sudo cp -rv trt/lib/* /usr/lib/aarch64-linux-gnu/ \
    && sudo cp -rv trt/include/* /usr/include/aarch64-linux-gnu/ \
    && sudo cp -rv trt/bin/trtexec /usr/bin/ \
    && sudo -E python3 -m pip install trt/uff/*.whl trt/graphsurgeon/*.whl trt/python/tensorrt-*-cp38-none-linux_aarch64.whl \
    && cd ../../.. \
    && rm -rf source \

# Install cudnn dev for plugin compilation
cd /tmp \
  && wget http://cuda-repo/release-candidates/kitbundles/cudnn/v8.2_cuda_11.4/8.2.6.16/repos/d5l/arm64/libcudnn8-dev_8.2.6.16-1+cuda11.4_arm64.deb \
  && sudo dpkg -i libcudnn8-dev_8.2.6.16-1+cuda11.4_arm64.deb \
  && sudo rm libcudnn8-dev_8.2.6.16-1+cuda11.4_arm64.deb

# Install perf_runner
python3 -m pip install cffi==1.14.5 cryptography==3.3.2 \
 && python3 -m pip install perf_runner==0.10.1 nrsu==0.5.210706163335 --index-url https://sc-hw-artf.nvidia.com/api/pypi/compute-pypi/simple \
 && python3 -m pip install cnexus-perfdb-v2 --index-url https://sc-hw-artf.nvidia.com/api/pypi/compute-pypi/simple
