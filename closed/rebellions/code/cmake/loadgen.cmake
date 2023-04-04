include(FetchContent)

function(enable_loadgen TARGET_NAME)
    FetchContent_Declare(mlperf_loadgen
        GIT_REPOSITORY https://github.com/mlcommons/inference.git
        GIT_TAG c4a19872d9e7ba2fe2d5b8a4c3d3c02e82233785
        SOURCE_SUBDIR loadgen)

    FetchContent_GetProperties(mlperf_loadgen)

    if(NOT mlperf_loadgen_POPULATED)
        FetchContent_MakeAvailable(mlperf_loadgen)
    endif()

    add_dependencies(${TARGET_NAME} mlperf_loadgen)
    target_include_directories(${TARGET_NAME} PRIVATE ${mlperf_loadgen_SOURCE_DIR}/loadgen)
    target_link_libraries(${TARGET_NAME} PRIVATE mlperf_loadgen)
endfunction()