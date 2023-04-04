
set(SOLA_RT_INC /usr/local/sola/include)
set(SOLA_RT_LIB /usr/local/sola/lib)
# set(SOLA_RT_INC /home/moffett/workspace/mfruntime/mfrt)
# set(SOLA_RT_LIB /home/moffett/workspace/mfruntime/bazel-bin)

find_library(SOLA NAMES sola HINTS ${SOLA_RT_LIB})

add_library(spu_backend
	src/server.cpp
	src/timer.cpp
	src/numa.cpp
	src/mf_sola.cpp
	src/model_executor.cpp
	src/multithreading.cpp
)

target_link_libraries(spu_backend
    glog
    ${SOLA}
	hwloc
)

target_include_directories(spu_backend
    PUBLIC
	${LOADGEN_INCLUDE_DIR}
	${SOLA_RT_INC}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
)
