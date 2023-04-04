#!/bin/bash
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

# This script will clone the python backend of Triton and perform the pre container setup for inferentia

# Set up build directory
python_build_branch=mlperf-inference-v2.0

# Clone Triton server repository
git clone --single-branch --depth=1 -b ${python_build_branch} https://github.com/triton-inference-server/python_backend.git

chmod 777 python_backend/inferentia/scripts/setup-pre-container.sh
sudo python_backend/inferentia/scripts/setup-pre-container.sh

cd -

