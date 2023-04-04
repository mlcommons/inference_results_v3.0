cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(inference_sut)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0074 NEW)

# PyTorch binary dependency, Must specify
# CMAKE_PREFIX_PATH=/path/to/libtorch to make it work
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(Torch REQUIRED)

# find_package(Gperftools)

# TODO: Force IOMP, default
if(DEFINED ENV{CONDA_PREFIX})
  set(OpenMP_ROOT $ENV{CONDA_PREFIX})
else()
  set(OpenMP_ROOT /usr/local)
endif()

set(OpenMP_C_INCLUDE_DIR ${OpenMP_ROOT}/include)
set(OpenMP_CXX_INCLUDE_DIR ${OpenMP_ROOT}/include)

find_package(OpenMP REQUIRED)

add_compile_options(-Wall -gdwarf-4 -isystem ${OpenMP_CXX_INCLUDE_DIRS})

# MLPerf loadgen, plugins dependency
add_subdirectory(inference/loadgen)
add_subdirectory(mlperf_plugins)

add_executable(rnnt_inference
  csrc/torch_sut.cpp csrc/rnnt_qsl.cpp csrc/metadata.cpp
  csrc/kmp_launcher.cpp csrc/hw_topology.cpp csrc/main.cpp csrc/utils.cpp)
set_property(TARGET rnnt_inference PROPERTY CXX_STANDARD 14)

message("Extra Torch C++ flags: ${TORCH_CXX_FLAGS}")
message("Extra OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
message("Extra OpenMP C++ includes: ${OpenMP_CXX_INCLUDE_DIRS}")
message("Extra OpenMP C++ Libraries: ${OpenMP_CXX_LIBRARIES}")
# message("GPerf Tools C++ includes: ${GPERFTOOLS_INCLUDE_DIRS}")
message("Torch Inlucde directories: ${TORCH_INCLUDE_DIRS}")
message("Torch linking libraries: ${TORCH_LIBRARIES}")

# if (Gperftools_FOUND)
# message("Gperftools linking libraries: ${GPERFTOOLS_LIBRARIES}")
# endif ()

if(UNIX)
  separate_arguments(OpenMP_CXX_OPTIONS UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
else()
  separate_arguments(OpenMP_CXX_OPTIONS WINDOWS_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

# if (Gperftools_FOUND)
#   add_definitions(-D__TCMALLOC__)
#
#   if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
#     target_link_libraries(rnnt_inference PUBLIC -Wl,--no-as-needed ${GPERFTOOLS_LIBRARIES})
#   else()
#     target_link_libraries(rnnt_inference PUBLIC ${GPERFTOOLS_LIBRARIES})
#   endif()
# endif()

target_compile_options(rnnt_inference BEFORE PUBLIC ${TORCH_CXX_FLAGS} ${OpenMP_CXX_OPTIONS})
target_include_directories(rnnt_inference SYSTEM PUBLIC inference/loadgen ${TORCH_INCLUDE_DIRS})
target_link_libraries(rnnt_inference PRIVATE "${TORCH_LIBRARIES}" mlperf_loadgen ${OpenMP_CXX_LIBRARIES})

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  target_link_libraries(rnnt_inference PUBLIC -Wl,--no-as-needed mlperf_plugins)
else()
  target_link_libraries(rnnt_inference PUBLIC mlperf_plugins)
endif()
