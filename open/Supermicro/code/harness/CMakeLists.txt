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

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR lwis/include)

# Set the path to the Triton library
unset(TRITON_DIR CACHE)
set(TRITON_DIR ../../build/triton-inference-server)

# Set NVTX library path
unset(NV_TOOLS_EXT_LIB CACHE)
set(NV_TOOLS_EXT_LIB ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Build the harness for the Triton harness and for DLRM if not on "aarch64" platform
execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)

if (USE_INFERENTIA)
    ######### TRITON-INFERENTIA PREREQS ##########
    execute_process(COMMAND echo "Building Triton-Inferentia harness...")
    unset(TRITON_LIB CACHE)
    find_library(TRITON_LIB NAMES libtritonserver.so PATHS ${TRITON_DIR}/out/tritonserver/install/lib)
    unset(PROTOBUF_LIB CACHE)
    find_library(PROTOBUF_LIB NAMES libprotobuf.a HINTS ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/lib)

    # Set the path to the TRITON include directories
    unset(TRITON_INCLUDE_DIRS CACHE)
    set(TRITON_INCLUDE_DIRS
        ${TRITON_DIR}/
        ${TRITON_DIR}/include/
        harness_triton_inferentia/include/
        ${TRITON_DIR}/out/tritonserver/
        ${TRITON_DIR}/out/tritonserver/install/include
        ${TRITON_DIR}/out/tritonserver/build/protobuf/include
        ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/include
        ${TRITON_DIR}/out/tritonserver/build/server
    )
    # Capture all the required protobuf source and header files (note we are excluding GRPC since we don't plan to use it)
    unset(PROTO_SRCS CACHE)
    set(PROTO_SRCS
        ${TRITON_DIR}/out/tritonserver/build/_deps/repo-core-build/triton-core/_deps/repo-common-build/protobuf/model_config.pb.cc
    )
    unset(PROTO_HDRS CACHE)
    set(PROTO_HDRS
        ${TRITON_DIR}/out/tritonserver/install/include/model_config.pb.h
    )

    # ######## TRITON-INFERENTIA HARNESS ######
    # Capture all the required source and header files
    unset(INFERENTIA_HARNESS_SERVER_SRCS CACHE)
    set(INFERENTIA_HARNESS_SERVER_SRCS
        harness_triton_inferentia/main_server.cc
        harness_triton_inferentia/src/triton_frontend_server.cpp
        harness_triton_inferentia/src/triton_frontend_helpers.cpp
        harness_triton_inferentia/src/pinned_memory_pool.cpp
        harness_triton_inferentia/src/triton_concurrent_frontend.cpp
        ${TRITON_DIR}/src/common.cc
        ${TRITON_DIR}/src/tracer.cc
    )
    unset(INFERENTIA_HARNESS_SERVER_HDRS CACHE)
    set(INFERENTIA_HARNESS_SERVER_HDRS
        ${TRITON_DIR}/src/common.h
        ${TRITON_DIR}/src/tracer.h
    )

    execute_process(COMMAND echo "Building Triton-INFERENTIA harness...")
    add_executable(harness_triton_inferentia
        ${INFERENTIA_HARNESS_SERVER_SRCS}
        ${INFERENTIA_HARNESS_SERVER_HDRS}
        ${PROTO_SRCS}
        ${PROTO_HDRS}
    )

    target_link_libraries(harness_triton_inferentia
        gflags
        glog
        ${LOADGEN_LIB}
        ${TRITON_LIB}
        ${PROTOBUF_LIB}
        pthread
        dl
        rt
        numa
        nvinfer
    )

    target_include_directories(harness_triton_inferentia
        PUBLIC
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            ${TRITON_INCLUDE_DIRS}
            harness_triton_inferentia
            common
            common_cpu
    )

else()
        # Build only CPU harness
    if (USE_CPU)

        ######### TRITON-CPU PREREQS ##########
        execute_process(COMMAND echo "Building Triton-CPU harness...")
        unset(TRITON_LIB CACHE)
        find_library(TRITON_LIB NAMES libtritonserver.so PATHS ${TRITON_DIR}/out/tritonserver/install/lib)
        unset(PROTOBUF_LIB CACHE)
        find_library(PROTOBUF_LIB NAMES libprotobuf.a HINTS ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/lib)

        # Set the path to the TRITON include directories
        unset(TRITON_INCLUDE_DIRS CACHE)
        set(TRITON_INCLUDE_DIRS
            ${TRITON_DIR}/
            ${TRITON_DIR}/include/
            ${TRITON_DIR}/out/tritonserver/
            ${TRITON_DIR}/out/tritonserver/install/include
            ${TRITON_DIR}/out/tritonserver/build/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/server
            harness_triton/include
        )
        # Capture all the required protobuf source and header files (note we are excluding GRPC since we don't plan to use it)
        unset(PROTO_SRCS CACHE)
        set(PROTO_SRCS
            ${TRITON_DIR}/out/tritonserver/build/_deps/repo-core-build/triton-core/_deps/repo-common-build/protobuf/model_config.pb.cc
        )
        unset(PROTO_HDRS CACHE)
        set(PROTO_HDRS
            ${TRITON_DIR}/out/tritonserver/install/include/model_config.pb.h
        )

        # ######## TRITON-CPU HARNESS ######
        # Capture all the required source and header files
        unset(CPU_HARNESS_SERVER_SRCS CACHE)
        set(CPU_HARNESS_SERVER_SRCS
            harness_triton_cpu/main_server.cc
            harness_triton_cpu/triton_frontend.cpp
            ${TRITON_DIR}/src/common.cc
            ${TRITON_DIR}/src/tracer.cc
        )
        unset(CPU_HARNESS_SERVER_HDRS CACHE)
        set(CPU_HARNESS_SERVER_HDRS
            ${TRITON_DIR}/src/common.h
            ${TRITON_DIR}/src/tracer.h
        )

        execute_process(COMMAND echo "Building Triton-CPU harness...")
        add_executable(harness_triton_cpu
            ${CPU_HARNESS_SERVER_SRCS}
            ${CPU_HARNESS_SERVER_HDRS}
            ${PROTO_SRCS}
            ${PROTO_HDRS}
        )

        target_link_libraries(harness_triton_cpu
            gflags
            glog
            ${LOADGEN_LIB}
            ${TRITON_LIB}
            ${PROTOBUF_LIB}
            pthread
            dl
            rt
        )

        target_include_directories(harness_triton_cpu
            PUBLIC
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                ${TRITON_INCLUDE_DIRS}
                harness_triton_cpu
                common
                common_cpu
        )
    else()
        ######### DEFAULT HARNESS ########

        if (${SOC_SM} STREQUAL "87")
            add_compile_definitions(IS_ORIN)
        endif()

	if (${IS_HOPPER})
	    add_compile_definitions(IS_HOPPER)
        endif()

        # Add the LWIS subdirectory (which will generate a static LWIS library)
        add_subdirectory(lwis)

        # Build the default harness which covers single_stream and offline scenarios on image benchmarks.
        execute_process(COMMAND echo "Building default harness...")
        add_executable(harness_default
            harness_default/main_default.cc
            common/logger.cpp
        )

        target_link_libraries(harness_default
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${CUDA_LIBRARIES}
            lwis
            ${LOADGEN_LIB}
            numa
        )

        target_include_directories(harness_default
            PUBLIC
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                common
        )

        ######### TRITON HARNESS ########
        # Find the necessary TRITON libraries
        unset(TRITON_LIB CACHE)
        find_library(TRITON_LIB NAMES libtritonserver.so PATHS ${TRITON_DIR}/out/tritonserver/install/lib)
        unset(PROTOBUF_LIB CACHE)
        find_library(PROTOBUF_LIB NAMES libprotobuf.a HINTS ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/lib)
        unset(TRITON_TENSORRT_LIB CACHE)
        find_library(TRITON_TENSORRT_LIB NAMES libtriton_tensorrt.so PATHS ${TRITON_DIR}/out/tensorrt/install/backends/tensorrt/)

        unset(TRITON_INCLUDE_DIRS CACHE)
        set(TRITON_INCLUDE_DIRS
            ${TRITON_DIR}/
            ${TRITON_DIR}/include/
            ${TRITON_DIR}/out/tritonserver/
            ${TRITON_DIR}/out/tritonserver/install/include
            ${TRITON_DIR}/out/tritonserver/build/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/server
            harness_triton/include
        )

        unset(PROTO_SRCS CACHE)
        set(PROTO_SRCS
            ${TRITON_DIR}/out/tritonserver/build/_deps/repo-core-build/triton-core/_deps/repo-common-build/protobuf/model_config.pb.cc
        )
        unset(PROTO_HDRS CACHE)
        set(PROTO_HDRS
            ${TRITON_DIR}/out/tritonserver/install/include/model_config.pb.h
        )

        # Capture all the required source and header files
        unset(HARNESS_SERVER_SRCS CACHE)
        set(HARNESS_SERVER_SRCS
            harness_triton/main_server.cc
            harness_triton/src/bert_concurrent_frontend.cpp
            harness_triton/src/dlrm_triton_concurrent_frontend.cpp
            harness_triton/src/dla_triton_frontend_helpers.cpp
            harness_triton/src/dla_concurrent_frontend.cpp
            harness_triton/src/triton_concurrent_frontend.cpp
            harness_triton/src/triton_frontend_server.cpp
            harness_triton/src/triton_frontend_helpers.cpp
            harness_triton/src/pinned_memory_pool.cpp
            common/logger.cpp
        )
        set(HARNESS_SERVER_HDRS
        )

        # Actually build the new server harness
        execute_process(COMMAND echo "Building Triton harness...")
        add_executable(harness_triton
            ${HARNESS_SERVER_SRCS}
            ${HARNESS_SERVER_HDRS}
            ${PROTO_SRCS}
            ${PROTO_HDRS}
        )

        target_link_libraries(harness_triton
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
            ${TRITON_LIB}
            ${TRITON_TENSORRT_LIB}
            ${PROTOBUF_LIB}
            numa
        )

        target_include_directories(harness_triton
            PUBLIC
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                ${TRITON_INCLUDE_DIRS}
                harness_triton
                common
                harness_dlrm
        )

        ######### TRITON UNIFIED HARNESS ########
        # Find the necessary TRITON libraries
        unset(TRITON_LIB CACHE)
        find_library(TRITON_LIB NAMES libtritonserver.so PATHS ${TRITON_DIR}/out/tritonserver/install/lib)
        unset(PROTOBUF_LIB CACHE)
        find_library(PROTOBUF_LIB NAMES libprotobuf.a HINTS ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/lib)
        unset(TRITON_TENSORRT_LIB CACHE)
        find_library(TRITON_TENSORRT_LIB NAMES libtriton_tensorrt.so PATHS ${TRITON_DIR}/out/tensorrt/install/backends/tensorrt/)

        unset(TRITON_INCLUDE_DIRS CACHE)
        set(TRITON_INCLUDE_DIRS
            ${TRITON_DIR}/
            ${TRITON_DIR}/include/
            ${TRITON_DIR}/out/tritonserver/
            ${TRITON_DIR}/out/tritonserver/install/include
            ${TRITON_DIR}/out/tritonserver/build/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/third-party/protobuf/include
            ${TRITON_DIR}/out/tritonserver/build/server
            harness_triton_unified/include
        )

        unset(PROTO_SRCS CACHE)
        set(PROTO_SRCS
            ${TRITON_DIR}/out/tritonserver/build/_deps/repo-core-build/triton-core/_deps/repo-common-build/protobuf/model_config.pb.cc
        )
        unset(PROTO_HDRS CACHE)
        set(PROTO_HDRS
            ${TRITON_DIR}/out/tritonserver/install/include/model_config.pb.h
        )

        # Capture all the required source and header files
        unset(HARNESS_SERVER_SRCS CACHE)
        set(HARNESS_SERVER_SRCS
            harness_triton_unified/src/triton_callbacks.cpp
            harness_triton_unified/src/triton_request_pool.cpp
            harness_triton_unified/src/triton_device_gpu.cpp
            harness_triton_unified/src/triton_device.cpp
            harness_triton_unified/src/triton_concurrent_sut.cpp
            harness_triton_unified/src/triton_workload.cpp
            harness_triton_unified/src/triton_helpers.cpp
            harness_triton_unified/src/pinned_memory_pool.cpp
            harness_triton_unified/main_server.cpp
            common/logger.cpp
        )
        set(HARNESS_SERVER_HDRS
        )

        # Actually build the new server harness
        execute_process(COMMAND echo "Building Triton unified harness...")
        add_executable(harness_triton_unified
            ${HARNESS_SERVER_SRCS}
            ${HARNESS_SERVER_HDRS}
            ${PROTO_SRCS}
            ${PROTO_HDRS}
        )

        target_link_libraries(harness_triton_unified
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
            ${TRITON_LIB}
            ${TRITON_TENSORRT_LIB}
            ${PROTOBUF_LIB}
            numa
        )

        target_include_directories(harness_triton_unified
            PUBLIC
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                ${TRITON_INCLUDE_DIRS}
                harness_triton_unified
                common
                harness_dlrm
        )

        # Triton Harness for multi-mig
        # Capture all the required source and header files
        if (NOT ${ARCH} MATCHES "aarch64")
            unset(HARNESS_MULTIMIG_SRCS CACHE)
            set(HARNESS_MULTIMIG_SRCS
                harness_triton_mig/main_server.cc
                harness_triton_mig/triton_frontend.cpp
                common/logger.cpp
            )
            unset(HARNESS_MULTIMIG_HDRS CACHE)
            set(HARNESS_MULTIMIG_HDRS
            )

            # Actually build the new server harness
            execute_process(COMMAND echo "Building Triton Multi-MIG harness...")
            add_executable(harness_triton_mig
                ${HARNESS_MULTIMIG_SRCS}
                ${HARNESS_MULTIMIG_HDRS}
                ${PROTO_SRCS}
                ${PROTO_HDRS}
            )

            target_link_libraries(harness_triton_mig
                nvinfer
                nvinfer_plugin
                gflags
                glog
                numa
                ${CUDA_LIBRARIES}
                ${LOADGEN_LIB}
                ${TRITON_LIB}
                ${TRITON_TENSORRT_LIB}
                ${PROTOBUF_LIB}
            )

             target_include_directories(harness_triton_mig
                 PUBLIC
                     ${CUDA_INCLUDE_DIRS}
                     ${LOADGEN_INCLUDE_DIR}
                     ${LWIS_INCLUDE_DIR}
                     ${TRITON_INCLUDE_DIRS}
                     harness_triton_mig
                     common
                     harness_dlrm
             )
         else()
             execute_process(COMMAND echo "Skipping TRITON MIG harness for ${ARCH}")
         endif()

        function(get_dali_paths DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)

        execute_process(
                COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_include_dir(), end='')"
                OUTPUT_VARIABLE DALI_INCLUDE_DIR_LOC
                RESULT_VARIABLE DALI_INCLUDE_DIR_RESULT
        )

        execute_process(
                COMMAND python3 -c "import nvidia.dali.sysconfig as dali_sc; print(dali_sc.get_lib_dir(), end='')"
                OUTPUT_VARIABLE DALI_LIB_DIR_LOC
                RESULT_VARIABLE DALI_LIB_DIR_RESULT
        )

        if (${DALI_INCLUDE_DIR_RESULT} EQUAL "1" OR ${DALI_LIB_DIR_RESULT} EQUAL "1")
            message(FATAL_ERROR "Error acquiring DALI. Verify, that you have DALI whl installed")
        endif ()
        set(DALI_INCLUDE_DIR ${DALI_INCLUDE_DIR_LOC} PARENT_SCOPE)
        set(DALI_LIB_DIR ${DALI_LIB_DIR_LOC} PARENT_SCOPE)
        set(DALI_LIBRARIES dali dali_operators dali_kernels PARENT_SCOPE)

    endfunction()

    ######### 3D-UNet-KiTS19 HARNESS ########
    execute_process(COMMAND echo "Building 3D-UNet-KiTS19 harness...")
    add_executable(harness_3dunet
        harness_3dunet/main_3dunet.cc
        harness_3dunet/lwis_3dunet.cpp
        harness_3dunet/unet3d_sw.cu
        common/logger.cpp
    )

    target_link_libraries(harness_3dunet
        nvinfer
        nvinfer_plugin
        gflags
        glog
        ${CUDA_LIBRARIES}
        ${LOADGEN_LIB}
        numa
    )

    target_include_directories(harness_3dunet
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            common
            harness_3dunet
    )

    ######### RNN-T HARNESS ########
    execute_process(COMMAND echo "Building RNN-T harness...")

    get_dali_paths(DALI_INCLUDE_DIR DALI_LIB_DIR DALI_LIBRARIES)

    message(STATUS "DALI libraries DIR: " ${DALI_LIB_DIR})
    message(STATUS "DALI include DIR: " ${DALI_INCLUDE_DIR})

    message(STATUS "DALI linked libraries: " ${DALI_LIBRARIES})

    add_executable(harness_rnnt
        harness_rnnt/main_rnnt.cc
        harness_rnnt/rnnt_kernels.cu
        common/logger.cpp
    )

    # Add gencode compilation option for SOC (i.e. Orin)
    if (${IS_SOC})
        message(STATUS "Compiling RNNT harness for SM${SOC_SM}.")
        target_compile_options(harness_rnnt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_${SOC_SM},code=sm_${SOC_SM}>)
    endif()

    target_link_directories(harness_rnnt PRIVATE ${DALI_LIB_DIR})

    target_link_libraries(harness_rnnt
        nvinfer
        nvinfer_plugin
        gflags
        glog
        ${CUDA_LIBRARIES}
        ${LOADGEN_LIB}
        ${DALI_LIBRARIES}
    )

    target_include_directories(harness_rnnt
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            ${DALI_INCLUDE_DIR}
            common
            harness_rnnt
    )

    ######### BERT HARNESS ########
    execute_process(COMMAND echo "Building BERT harness...")
    add_executable(harness_bert
        harness_bert/main_bert.cc
        harness_bert/bert_server.cc
        harness_bert/bert_core_vs.cc
        common/logger.cpp
    )

    target_link_libraries(harness_bert
        nvinfer
        nvinfer_plugin
        gflags
        glog
        ${CUDA_LIBRARIES}
        ${LOADGEN_LIB}
    )

    target_include_directories(harness_bert
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${LOADGEN_INCLUDE_DIR}
            ${LWIS_INCLUDE_DIR}
            common
            harness_bert
    )

    ######### DLRM HARNESS ########
    # SOC doesn't run DLRM
    if (NOT ${IS_SOC})
        execute_process(COMMAND echo "Building DLRM harness...")
        add_executable(harness_dlrm
            harness_dlrm/main_dlrm.cc
            harness_dlrm/dlrm_server.cc
            harness_dlrm/batch_maker.cpp
            harness_dlrm/dlrm_kernels.cu
            common/logger.cpp
        )

        target_link_libraries(harness_dlrm
            nvinfer
            nvinfer_plugin
            gflags
            glog
            ${CUDA_LIBRARIES}
            ${LOADGEN_LIB}
            ${NV_TOOLS_EXT_LIB}
            numa
        )

        target_include_directories(harness_dlrm
            PUBLIC
                ${CUDA_INCLUDE_DIRS}
                ${LOADGEN_INCLUDE_DIR}
                ${LWIS_INCLUDE_DIR}
                common
                harness_dlrm
        )
    else()
        execute_process(COMMAND echo "Skipping DLRM harness for SoC system")
    endif()
    endif()
endif()
