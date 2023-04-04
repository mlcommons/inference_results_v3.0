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
#
# Modified by NEUCHIPS on 2023.
#

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)

# Build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# to enable timeline trace profiling, unmark below line

# Set the include directory for Minitrace header files
#add_definitions(-DNTRACE)

# Workaround for TRT header warning
#execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -g -Wno-deprecated-declarations")

project(harness LANGUAGES CXX)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Build the harness for the Triton harness and for DLRM if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

# build neuchips dlrm
execute_process(COMMAND echo "Building DLRM harness... cpu")
add_executable(harness_dlrm_n3000
    harness_dlrm_n3000/main_dlrm.cc
    harness_dlrm_n3000/dlrm_server.cc
    harness_dlrm_n3000/batch_maker.cpp )

target_link_libraries(harness_dlrm_n3000
        gflags
        glog
        ${LOADGEN_LIB}
        numa )

target_include_directories(harness_dlrm_n3000 PUBLIC
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            common
            harness_dlrm_n3000 )
    

