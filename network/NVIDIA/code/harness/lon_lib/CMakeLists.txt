# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Build options
set(CMAKE_CXX_STANDARD 17)

# Set sm versions
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_72,code=sm_72")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.0)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.4)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_87,code=sm_87")
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL 11.8)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_90,code=sm_90")
endif()

# Set the path to the IBVerbs library
unset(IBVERBS_INCLUDE_DIR CACHE)
find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h PATHS /usr/include)

# Set the path to util
unset(UTIL_INCLUDE_DIR CACHE)
find_path(UTIL_INCLUDE_DIR NAMES utils.hpp PATHS /work/code/harness/common)

add_library(lon_lib
    src/lon_ib.cpp
)

target_link_libraries(lon_lib
    ${CUDA_LIBRARIES}
    numa
    glog
    ibverbs
    nl-3
    nl-route-3
)

target_include_directories(lon_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
        ${IBVERBS_INCLUDE_DIR}
        ${UTIL_INCLUDE_DIR}
)
