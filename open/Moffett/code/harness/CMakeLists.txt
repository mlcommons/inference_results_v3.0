cmake_minimum_required(VERSION 3.10)
project(harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# asan
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")

add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})
IF(${SERVER_MODE})
    add_definitions(-DSERVER_MODE=${SERVER_MODE})
ENDIF()
message("Server mode: ${SERVER_MODE}")
    # Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})
find_package(OpenMP)
find_package(pybind11 REQUIRED)

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR spu_backend/include)

# message(${LOADGEN_LIB_DIR})
# message(${LOADGEN_LIB})

add_subdirectory(spu_backend)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cpp
)

target_link_libraries(harness_default
    gflags
    glog
    spu_backend
    ${LOADGEN_LIB}
    pthread
    OpenMP::OpenMP_CXX
)

target_include_directories(harness_default
    PUBLIC
    ${LWIS_INCLUDE_DIR}
    ${LOADGEN_INCLUDE_DIR}
    common
)
