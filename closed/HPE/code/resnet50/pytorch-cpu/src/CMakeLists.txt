cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0074 NEW)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(DEFINED ENV{CUSTOM_CXX})
    set(CMAKE_CXX_COMPILER $ENV{CUSTOM_CXX})
endif()

if(DEFINED ENV{CUSTOM_C})
    set(CMAKE_C_COMPILER $ENV{CUSTOM_C})
endif()

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
if(DEFINED ENV{CONDA_PREFIX})
  set(OpenMP_ROOT $ENV{CONDA_PREFIX})
else()
  message(FATAL_ERROR "Requires to run from conda env. Activate the relevant environment.")
  set(OpenMP_ROOT /usr/local)
endif()

set(OpenMP_C_INCLUDE_DIR ${OpenMP_ROOT}/include)
set(OpenMP_CXX_INCLUDE_DIR ${OpenMP_ROOT}/include)

#set(CMAKE_BUILD_TYPE "Release")

find_package(OpenMP REQUIRED)
find_package(PythonLibs REQUIRED)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

message("Extra Torch C++ flags: ${TORCH_CXX_FLAGS}")
message("Extra OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
message("Extra OpenMP C++ includes: ${OpenMP_CXX_INCLUDE_DIRS}")
message("Extra OpenMP C++ Libraries: ${OpenMP_CXX_LIBRARIES}")
# message("GPerf Tools C++ includes: ${GPERFTOOLS_INCLUDE_DIRS}")
message("Torch Inlucde directories: ${TORCH_INCLUDE_DIRS}")
message("Torch linking libraries: ${TORCH_LIBRARIES}")


separate_arguments(OpenMP_CXX_OPTIONS UNIX_COMMAND "${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wl,--no-as-needed -Wno-restrict -Wno-class-memaccess -Wno-unused-but-set-variable")
set(MLPERF_TARGET mlperf_runner)


add_compile_options(-Wall -isystem ${OpenMP_CXX_INCLUDE_DIRS})

#set(CONDA_PREFIX $ENV{CONDA_PREFIX})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add graph compiler kernels
add_subdirectory(ckernels)
#add_subdirectory(ckernels-old)

#set(MLPERF_TARGET mlperf_runner)
add_executable(${MLPERF_TARGET} "mlperf_runner.cpp" "dataset_utils.cpp" "sut.cpp" "backend.cpp" "kmp_launcher.cpp")

#TODO Cleanup include dirs
if (NOT(LOADGEN_DIR))
    MESSAGE(FATAL_ERROR "LOADGEN_DIR not set. Set path to loadgen header files")
endif()

if (NOT(RapidJSON_INCLUDE_DIR))
    MESSAGE(FATAL_ERROR "Requires rapidjson headers. Specify include directory with -DRapidJSON_INCLUDE_DIR")
endif()

set(COMMON_DIR "include")

set(CUSTOM_KERNELS "ckernels/src")
#set(CUSTOM_KERNELS "ckernels-old/src")

include_directories(${RapidJSON_INCLUDE_DIR} ${LOADGEN_DIR} ${OpenCV_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${IPEX_INCLUDE} ${COMMON_DIR} ${CUSTOM_KERNELS} ${PYTHON_INCLUDE_DIRS} ${ONEDNN_DIR}/include ${ONEDNN_DIR}/build/include)

target_compile_options(${MLPERF_TARGET} BEFORE PUBLIC ${TORCH_CXX_FLAGS} ${OpenMP_CXX_OPTIONS})

target_link_libraries(${MLPERF_TARGET}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${gflags_DIR}/lib/libgflags.a
    ${LOADGEN_DIR}/build/libmlperf_loadgen.a
    ${OpenMP_CXX_LIBRARIES}
    ${INTEL_EXTENSION_FOR_PYTORCH_PATH}/lib/libintel-ext-pt-cpu.so
    ${PYTHON_LIBRARIES}
    ${ONEDNN_DIR}/build/src/libdnnl.so
    stdc++fs
    gc_rn50
)

set_property(TARGET ${MLPERF_TARGET} PROPERTY CXX_STANDARD 14)