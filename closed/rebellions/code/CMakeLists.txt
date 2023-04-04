cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug Mode")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "Release Mode")
endif()

project(rebel_mlperf)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} main.cpp ${CPP_FILES} ${DATA_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

# rlthunk
find_package(rl-thunk CONFIG REQUIRED)
find_package(rl-thunk_fsoc CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC rl-thunk::rl-thunk)

# rebel_compiler
find_package(tvm REQUIRED HINTS $ENV{REBEL_HOME}/build/install/lib/cmake/tvm)
target_link_libraries(${PROJECT_NAME} PUBLIC tvm::tvm)

# rebel_runtime
find_package(rebel_runtime REQUIRED HINTS $ENV{REBEL_RUNTIME_HOME}/build/install/share/cmake)
target_link_libraries(${PROJECT_NAME} PUBLIC rebel_runtime::rebel_runtime)

# argparse
include(cmake/argparse.cmake)
enable_argparse(${PROJECT_NAME})

# loadgen
include(cmake/loadgen.cmake)
enable_loadgen(${PROJECT_NAME})

# Compiler options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
