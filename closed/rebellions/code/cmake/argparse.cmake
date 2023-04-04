include(FetchContent)

function (enable_argparse TARGET_NAME)
    FetchContent_Declare(argparse
                         GIT_REPOSITORY https://github.com/p-ranav/argparse.git
                         GIT_TAG v2.3)

    FetchContent_GetProperties(argparse)

    if(NOT argparse_POPULATED)
        FetchContent_MakeAvailable(argparse)
    endif()

    target_include_directories(${TARGET_NAME} PRIVATE ${argparse_SOURCE_DIR}/include)
endfunction()
