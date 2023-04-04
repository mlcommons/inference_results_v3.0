cmake_minimum_required(VERSION 3.20)

include(root)

# set the project name
set(PROJECT_NAME "loadgen_wrapper")
project("${PROJECT_NAME}"
  VERSION 1.0
  DESCRIPTION "LoadGen SiMaAi Wrapper"
  LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(MLPERF_LOADGEN_LIB
  mlperf_loadgen
  HINTS ${MLPERF_LOADGEN_LIB_PATH}
  NO_CMAKE_FIND_ROOT_PATH
  NO_DEFAULT_PATH
  REQUIRED)

# default build is RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

link_directories(${MLPERF_LOADGE_LIB_PATH})

add_library("${PROJECT_NAME}" SHARED
  loadgen_wrapper.cc
  loadgen_query_intf.cc
  loadgen_wrapper_intf.cc
  loadgen_mem_manager.cc)


set (LOADGEN_WRAPPER_PUBLIC_HEADERS
  "image_loader.h"
  "loadgen_query_intf.h"
  "loadgen_wrapper_ctrl.h"
  "loadgen_wrapper.h"
  "loadgen_wrapper_intf.h"
  "loadgen_mem_manager.h"
  )

set_target_properties (${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER
  "${LOADGEN_WRAPPER_PUBLIC_HEADERS}"
  )

target_compile_definitions("${PROJECT_NAME}" INTERFACE USING_LOADGEN_WRAPPER_LIB)

target_include_directories ("${PROJECT_NAME}"
  PRIVATE
  ../../../3rdparty/loadgen
  ../imagenet/
  .
  )

target_link_libraries("${PROJECT_NAME}"
  ${MLPERF_LOADGEN_LIB})

# Install headers
INSTALL(TARGETS "${PROJECT_NAME}")
INSTALL(TARGETS "${PROJECT_NAME}" PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/simaai)
