cmake_minimum_required(VERSION 3.16)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PROJECT_NAME "loadgen_plugins")
project("${PROJECT_NAME}"
  VERSION 1.0
  DESCRIPTION "LoadGen Plugins"
  LANGUAGES CXX)

add_subdirectory(loadgen_plugin/loadgen)
add_subdirectory(loadgen_plugin/mlperfsrc)
add_subdirectory(loadgen_plugin/mlperfsink)
add_subdirectory(loadgen_plugin/filter)
