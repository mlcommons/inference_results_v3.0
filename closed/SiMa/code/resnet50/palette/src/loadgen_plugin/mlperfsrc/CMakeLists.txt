cmake_minimum_required(VERSION 3.16)

include (root)

# set the project name
set(PROJECT_NAME "gstsimaaimlperfsrc")

project("${PROJECT_NAME}"
  VERSION 0.1
  DESCRIPTION "SiMaAi MLPerf Src Gstremer plugin"
  LANGUAGES C CXX)

set (MLPERFSRC_LIBRARY_SOURCES
  "mlperfsrc2.cc")

find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0)
pkg_check_modules(GSTREAMER gstreamer-1.0)

if(NOT GLIB2_FOUND OR NOT GSTREAMER_FOUND )
    message(WARNING "GstRemoteOffload project is not configured due to absence of gstreamer component(s)" )
    message(WARNING "Please install GStreamer 1.14.0+ before running the sample." )
    return()
endif()

set(GLIBS ${GLIBS} glib-2.0 gobject-2.0 gio-2.0 gstcontroller-1.0 gstbase-1.0 gstreamer-1.0)

add_definitions(-DVERSION=\"1.18.16\")
add_definitions(-DGST_LICENSE=\"APACHE\")
add_definitions(-DGST_PACKAGE_NAME=\"GStreamer\ SiMa.ai\ Topk\ Plug-in\")
add_definitions(-DGST_PACKAGE_ORIGIN=\"https://bitbucket.org/sima-ai/gst-plugins-sima\")
add_definitions(-DPACKAGE=\"gst-plugins-sima\")

message(${LOADGEN_WRAPPER_LIB_PATH})

find_library(MLPERF_LOADGEN_LIB
  mlperf_loadgen
  HINTS ${MLPERF_LOADGEN_LIB_PATH}
  NO_CMAKE_FIND_ROOT_PATH
  NO_DEFAULT_PATH
  REQUIRED)

add_library(${PROJECT_NAME}
  SHARED
  ${MLPERFSRC_LIBRARY_SOURCES})

include(GNUInstallDirs)

target_include_directories ("${PROJECT_NAME}"
  PRIVATE
  .
  )

target_include_directories( ${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
  ${GLIB2_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${LOADGEN_WRAPPER_INCLUDE_DIRS}
  ${IMAGENET_INCLUDE_DIRS}
  ${MLPERF_LOADGEN_INCLUDE_DIRS})

find_library(GLIB2_LIBRARY glib-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GOBJECT2_LIBRARY gobject-2.0 PATHS ${GLIB2_LIBRARY_DIRS} )
find_library(GSTBASE_LIBRARY gstbase-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )
find_library(GST_LIBRARY gstreamer-1.0 PATHS ${GSTREAMER_LIBRARY_DIRS} )

target_link_libraries(${PROJECT_NAME}
  PUBLIC  ${GLIB2_LIBRARY} ${GOBJECT2_LIBRARY} ${GSTBASE_LIBRARY} ${GST_LIBRARY}
  loadgen_wrapper
  gstsimaallocator
  simaaimem)

INSTALL(TARGETS "${PROJECT_NAME}"  DESTINATION ${CMAKE_INSTALL_LIBDIR})
INSTALL(TARGETS "${PROJECT_NAME}"  DESTINATION ${CMAKE_INSTALL_LIBDIR}/gstreamer-1.0)
