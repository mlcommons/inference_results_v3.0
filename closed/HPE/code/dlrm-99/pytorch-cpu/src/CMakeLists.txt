cmake_minimum_required(VERSION 3.15)
project(DLRMInference)
find_package(gflags REQUIRED)
find_package(OpenMP REQUIRED)
find_package(dnnl  REQUIRED)
set(CMAKE_CXX_FLAGS -lnuma)
#set(CMAKE_CXX_STANDARD 17)
set(LOADGEN_LIB_DIR "${LOADGEN_DIR}/lib")
set(ONEDNN_LIB_DIR "${ONEDNN_DIR}/lib64")
add_definitions(-I"${LOADGEN_DIR}/include" -I"{ONEDNN_DIR}/include" -I"$ENV{CONDA_PREFIX}/")
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})
find_library(ONEDNN_LIB NAMES libdnnl.so PATHS ${ONEDNN_LIB_DIR})
find_library(GLOG_LIB NAMES libglog.so PATHS $ENV{CONDA_PREFIX}/lib)
add_subdirectory(cnpy)
add_compile_options(-O2 -march=sapphirerapids)
add_executable(dlrminf
  harness/dlrm_core.cpp
  harness/dlrm_main.cpp
  harness/dlrm_proc.cpp
  harness/dlrm_server.cpp
  harness/util.cpp
  harness/kmp_launcher.cpp
  )
target_include_directories(dlrminf PUBLIC include cnpy ${ONEDNN_DIR}/include)
target_link_libraries(dlrminf PUBLIC ${LOADGEN_LIB} gflags ${GLOG_LIB} cnpy ${ONEDNN_LIB} -march=sapphirerapids)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dlrminf PUBLIC OpenMP::OpenMP_CXX)
endif()
