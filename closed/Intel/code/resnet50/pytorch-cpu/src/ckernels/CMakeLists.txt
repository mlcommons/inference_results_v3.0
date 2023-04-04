#===============================================================================
# Copyright 2020-2022 Intel Corporation
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
#===============================================================================

### General policy inherit from DNNL ###

cmake_minimum_required(VERSION 2.8.11)

if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW) # For AppleClang
endif()

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Enable RPATH on MacOS/OSX
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Do not export symbols from executables
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# Pass all flags to try_compile
if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
endif()
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# Use <PackageName>_ROOT env. variable as a prefix
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_FLAGS_RELEASEWITHASSERT "-Ofast")
# Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()


option(SC_PYTHON_BINDING "python binding" OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include("cmake/utils.cmake")

### graph-compiler specical config ###

set(PROJECT_NAME "GRAPHCOMPILER-C-Kernels")
set(PROJECT_FULL_NAME "GRAPHCOMPILER-C-Kernels")
set(PROJECT_VERSION "0.0.1")


if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

include("cmake/Doxygen.cmake")

message(STATUS "KERNEL_TYPE is ${KERNEL_TYPE}")

if ((NOT "${KERNEL_TYPE}" STREUQAL "MLP") AND (NOT "${KERNEL_TYPE}" STREQUAL "RN50"))
    message(STATUS "Invalid KERNEL_TYPE, set default to MLP")
    set(KERNEL_TYPE MLP)
endif()

if ("${KERNEL_TYPE}" STREQUAL "MLP")
    set(LIB_NAME gc_mlp)
    file(GLOB_RECURSE SC_KERNEL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_mlp/*.cpp
        )
else ()
    set(LIB_NAME gc_rn50)
    file(GLOB_RECURSE SC_KERNEL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_rn50/*.cpp
        )
endif()

set(SC_CPP_VER 11)


add_library(${LIB_NAME} STATIC ${SC_KERNEL_SOURCES})

set(SC_SKIP_SOURCES dummy)


if(NOT "${SC_CPU_RUNTIME}" MATCHES "^(OMP|TBB|SEQ)$")
    message(FATAL_ERROR "Unsupported CPU runtime: ${SC_CPU_RUNTIME}")
endif()

SET(DNNL_CPU_RUNTIME ${SC_CPU_RUNTIME})
SET(DNNL_CPU_THREADING_RUNTIME ${SC_CPU_RUNTIME})

if(${SC_CPU_RUNTIME} STREQUAL "OMP")
    set(SC_OMP_ENABLED ON)
    if(APPLE)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif()
    endif()
    find_package(OpenMP REQUIRED)
    add_definitions("-DSC_CPU_THREADPOOL=1")
    message(STATUS "OpenMP is enabled")

    if(APPLE)
        link_directories("${OPENMP_LIBRARIES}")
    endif()
endif()

if(${SC_CPU_RUNTIME} STREQUAL "TBB")
    include("cmake/TBB.cmake")
    add_definitions("-DSC_CPU_THREADPOOL=2")
endif()

if(${SC_CPU_RUNTIME} STREQUAL "SEQ")
    add_definitions("-DSC_CPU_THREADPOOL=0")
endif()

##############################
# Enable better debug info...
##############################

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-inline -fno-omit-frame-pointer -gdwarf-4")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL Clang)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-limit-debug-info")
endif()

##############################


if(NOT DEFINED GC_DNNL_PATH OR GC_DNNL_PATH STREQUAL "")
    if (DEFINED ENV{GC_DNNL_PATH})
        set(DNNL_PATH $ENV{GC_DNNL_PATH})
    else()
        set(DNNL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onednn/")
    endif()
else()
    set(DNNL_PATH ${GC_DNNL_PATH})
endif()

message(STATUS "DNNL_PATH is set to: ${DNNL_PATH}")

set(HAS_PARENT false)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/3rdparty/onednn/include
    ${DNNL_PATH}/src
    ${DNNL_PATH}/include
)

if ("${KERNEL_TYPE}" STREQUAL "MLP")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_mlp/*.hpp)
else()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_rn50/*.hpp)
endif()

option(PROFILING "enable vtune profiling" OFF)
if (PROFILING STREQUAL "ON")
    # build entir dnnl
    set(DNNL_BUILD_TESTS OFF)
    set(DNNL_BUILD_EXAMPLES OFF)
    set(DNNL_BLAS_VENDOR "NONE")
    set(DNNL_LIBRARY_TYPE STATIC)
    add_subdirectory(3rdparty/onednn)
    set(DNNL_LIBNAME dnnl)
    target_link_libraries(${LIB_NAME} PRIVATE ${DNNL_LIBNAME})
    add_dependencies(${LIB_NAME} ${DNNL_LIBNAME})
    set(DNNL_ENABLE_JIT_PROFILING ON)
else()
    # build minimal set of brgemm src for use
    add_subdirectory(src/dnnl_brgemm)
    set(DNNL_LIBNAME dnnl_brgemm)
    target_link_libraries(${LIB_NAME} PRIVATE ${DNNL_LIBNAME})
endif()


include_directories(
    ${PROJECT_BINARY_DIR}/include
)

add_definitions("-DSC_JIT_SOURCE=1")

SET(CMAKE_CXX_FLAGS "-ftemplate-depth=99999 ${CMAKE_CXX_FLAGS}")
if(NOT MSVC)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-field-initializers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-field-initializers")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -Wall -Wno-unused-variable -Wno-unused-value -Wno-unused-function -std=c99 -fPIC -Werror -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\" -dM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wall -march=native -Wno-unused-variable -Wno-unused-value -Wno-unused-function -std=c++${SC_CPP_VER} -fPIC -Werror -dM")
    # Deal with Clang < 6.0's missing-braces issue
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
        endif()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")  # clang-13 installed from brew needs this
        endif()
    endif()

    # if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #     if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 10.0)
    #         SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    #     else()
    #         message(FATAL_ERROR "Requiring GCC version > 10.0")
    #     endif()
    # endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd3931 -wd854 -wd3466 -wd2196 -wd3924 -wd597 -wd873 -wd1292")
    endif()
    append(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden")
    if(SC_OMP_ENABLED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
else()
    add_definitions(-DSC_DLL -DSC_DLL_EXPORTS)
endif()


add_subdirectory(src/runtime)
target_link_libraries(${LIB_NAME} PUBLIC graphcompiler_runtime)

if ("${KERNEL_TYPE}" STREQUAL "MLP")
    add_executable(mlpkerneltest ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlpkerneltest.cpp)
    target_link_libraries(mlpkerneltest ${LIB_NAME})
else()
  #  add_executable(test_rn50_backbonetest_bs8 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rn50_backbone_bs8.cpp)
  #  target_link_libraries(test_rn50_backbonetest_bs8 ${LIB_NAME})
  #  add_executable(test_rn50_backbonetest_bs4 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rn50_backbone_bs4.cpp)
  #  target_link_libraries(test_rn50_backbonetest_bs4 ${LIB_NAME})
  #  add_executable(test_rn50_backbonetest_bs9 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rn50_backbone_bs9.cpp)
  #  target_link_libraries(test_rn50_backbonetest_bs9 ${LIB_NAME})
  #  add_executable(test_rn50_backbonetest_bs256 ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rn50_backbone_bs256.cpp)
  #  target_link_libraries(test_rn50_backbonetest_bs256 ${LIB_NAME})
endif()

if (SC_PYTHON_BINDING STREQUAL "ON")
    file(GLOB_RECURSE SC_PYTHON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/python_binding/sc_py_binding.cpp
        )
    add_subdirectory(3rdparty/pybind11)
    pybind11_add_module(pycsc SHARED ${SC_PYTHON_SOURCES})
    set_target_properties(pycsc PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(pycsc PRIVATE ${LIB_NAME})
    add_definitions("-DSC_PYTHON_BINDING=1")
    target_compile_definitions(pycsc PUBLIC SC_DLL)
endif()

if (UNIX)
    target_link_libraries(${LIB_NAME} PUBLIC dl)
endif()

