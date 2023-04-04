# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

pushd /tmp

# install CUDA 11.4
mkdir -p cuda
cd cuda
wget -np -nd -r http://cuda-repo/release-candidates/kitpicks/cuda-r11-4-tegra/11.4.14/011/local_installers/cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb \
  && sudo dpkg -i ./cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb \
  && sudo cp /var/cuda-repo-l4t-11-4-local/cuda-82DB0B48-keyring.gpg /usr/share/keyrings/ \
  && sudo apt update \
  && sudo apt install -y cuda-toolkit-*
cd ..
rm -rf cuda

# install cudnn 8.5
mkdir -p cudnn
cd cudnn
wget -np -nd -r http://cuda-repo/release-candidates/kitpicks/cudnn-v8-5-tegra/8.5.0.96/001/local_installers/cudnn-local-tegra-repo-ubuntu2004-8.5.0.96_1.0-1_arm64.deb \
  && sudo dpkg -i ./cudnn-local-tegra-repo-ubuntu2004-8.5.0.96_1.0-1_arm64.deb \
  && sudo cp /var/cudnn-local-tegra-repo-ubuntu2004-8.5.0.96/*-keyring.gpg /usr/share/keyrings/ \
  && sudo apt update \
  && sudo apt install libcudnn8*
cd ..
rm -rf cudnn

# install TRT 8.5
mkdir -p trt
cd trt
wget -np -nd -r http://cuda-repo/release-candidates/Libraries/TensorRT/v8.5/8.5.2.1-6ebd8d1c/11.4-r470/l4t-aarch64/deb/ \
  && sudo dpkg -i ./libnvinfer8_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-dev_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-plugin8_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-plugin-dev_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvparsers8_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvparsers-dev_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvonnxparsers8_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvonnxparsers-dev_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./python3-libnvinfer_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./python3-libnvinfer-dev_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./graphsurgeon-tf_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./uff-converter-tf_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./libnvinfer-samples_8.5.2-1+cuda11.4_all.deb \
  && sudo dpkg -i ./libnvinfer-bin_8.5.2-1+cuda11.4_arm64.deb \
  && sudo dpkg -i ./tensorrt_8.5.2.1-1+cuda11.4_arm64.deb
cd ..
rm -rf trt
popd
