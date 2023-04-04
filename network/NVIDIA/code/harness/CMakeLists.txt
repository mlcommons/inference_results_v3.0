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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(mlperf-inference)

include(GNUInstallDirs)
find_package(CUDA REQUIRED)

# Build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Pass the Loadgen include directory from command line
add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})

# Workaround for TRT header warning
execute_process(COMMAND echo "Warning: setting -Wno-deprecated-declarations to avoid header warnings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

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

project(harness LANGUAGES CXX CUDA)

# Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})

# Set the path to the IBVerbs library
unset(IBVERBS_INCLUDE_DIR CACHE)
find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h PATHS /usr/include)

# Set the path to LON library
unset(LON_INCLUDE_DIR CACHE)
set(LON_INCLUDE_DIR lon_lib/include)

# Set NVTX library path
unset(NV_TOOLS_EXT_LIB CACHE)
set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Add the LON_lib subdirectory
add_subdirectory(lon_lib)

include_directories("/usr/include/libnl3")

######### LON LWIS HARNESS ########
execute_process(COMMAND echo "Building LON LWIS harness for LON node...")
add_executable(harness_lwis_lon
    harness_lwis_lon/main_lwis_lon.cc
    common/logger.cpp
    common/qdl_lon.cpp
)

target_link_libraries(harness_lwis_lon
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    numa
    ibverbs
    nl-3
    nl-route-3
    lon_lib
)

target_include_directories(harness_lwis_lon
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LON_INCLUDE_DIR}
        ${IBVERBS_INCLUDE_DIR}
        common
        harness_lwis_lon
)

execute_process(COMMAND echo "Building LON LWIS harness for SUT node...")
add_executable(harness_lwis_sut
    harness_lwis_lon/main_lwis_sut.cc
    harness_lwis_lon/lwis_lon.cpp
    harness_lwis_lon/qdl_intf_lwis.cpp
    common/logger.cpp
    common/qdl_lon.cpp
)

target_link_libraries(harness_lwis_sut
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    numa
    ibverbs
    nl-3
    nl-route-3
    lon_lib
)

target_include_directories(harness_lwis_sut
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LON_INCLUDE_DIR}
        ${IBVERBS_INCLUDE_DIR}
        common
        harness_lwis_lon
)

######### LON BERT HARNESS ########
execute_process(COMMAND echo "Building BERT LON harness for LON node...")
add_executable(harness_bert_lon
    harness_bert_lon/main_bert_lon.cc
    common/logger.cpp
    common/qdl_lon.cpp
)

target_link_libraries(harness_bert_lon
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    numa
    ibverbs
    nl-3
    nl-route-3
    lon_lib
)

target_include_directories(harness_bert_lon
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LON_INCLUDE_DIR}
        ${IBVERBS_INCLUDE_DIR}
        common
        harness_bert_lon
        harness_lwis_lon
)

execute_process(COMMAND echo "Building BERT LON harness for SUT node...")
add_executable(harness_bert_sut
    harness_bert_lon/main_bert_sut.cc
    harness_bert_lon/bert_lon_server.cc
    harness_bert_lon/bert_lon_core_vs.cc
    harness_bert_lon/qdl_intf_bert.cpp
    common/logger.cpp
    common/qdl_lon.cpp
)

target_link_libraries(harness_bert_sut
    nvinfer
    nvinfer_plugin
    gflags
    glog
    ${CUDA_LIBRARIES}
    ${LOADGEN_LIB}
    numa
    ibverbs
    nl-3
    nl-route-3
    lon_lib
)

target_include_directories(harness_bert_sut
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${LOADGEN_INCLUDE_DIR}
        ${LON_INCLUDE_DIR}
        ${IBVERBS_INCLUDE_DIR}
        common
        harness_bert_lon
        harness_lwis_lon
)