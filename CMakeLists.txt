cmake_minimum_required(VERSION 3.18)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

include(CMakeCommon.txt)

set(PROJECT_NAME d2p2)
project(${PROJECT_NAME})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(THIRDPARTY_ROOT ${PROJECT_ROOT}/thirdparty)

include_directories(AFTER ${PROJECT_ROOT})
include_directories(AFTER ${THIRDPARTY_ROOT})

set(OUTPUT_DIRECTORY "${PROJECT_ROOT}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

set(ISPC_TARGET "avx2-i32x8")
set(ISPC_ARCH "x86-64")
set(ISPC_DEVICE "core-avx2")

set(ISPC_BIN "${THIRDPARTY_ROOT}/ispc-v1.19.0-windows/bin/ispc.exe")
set(ISPC_EXENAME "")
if(WIN32)
endif(WIN32)

set(HEADERS
    ${PROJECT_ROOT}/d2p2/d2p2.h
    ${PROJECT_ROOT}/d2p2/tensor.h
    ${PROJECT_ROOT}/d2p2/function.h
    ${PROJECT_ROOT}/d2p2/layer.h
    ${PROJECT_ROOT}/d2p2/util.h)

set(ISPC_SOURCES
    ${PROJECT_ROOT}/src/ispc/matrix.ispc)


if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_UNICODE /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd")
    if(MSVC_VERSION VERSION_LESS_EQUAL "1900")
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++latest")
    else()
        set(DEFAULT_CXX_FLAGS "${DEFAULT_CXX_FLAGS} /Zc:__cplusplus /std:c++20")
    endif()

    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Zi /Ob0 /Od /RTC1 /Gy /GR- /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Oi /GL /GR- /DNDEBUG /EHsc-")

    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /SUBSYSTEM:CONSOLE")

elseif(UNIX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++20 -march=x86-64-v3")
elseif(APPLE)
endif()

add_subdirectory(devel)
add_subdirectory(test)

