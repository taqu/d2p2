cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME devel)
project(${PROJECT_NAME})

set(DEVEL_SOURCES
    ${PROJECT_ROOT}/devel/main.cpp
    ${PROJECT_ROOT}/src/d2p2.cpp
    ${PROJECT_ROOT}/src/tensor.cpp
    ${PROJECT_ROOT}/src/function.cpp
    ${PROJECT_ROOT}/src/layer.cpp
    ${PROJECT_ROOT}/src/util.cpp)

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${DEVEL_SOURCES})
source_group("src/ispc" FILES ${ISPC_SOURCES})
add_ispcs(ISPC_OBJS ${ISPC_SOURCES})
add_executable(${PROJECT_NAME} ${DEVEL_SOURCES} ${HEADERS} ${ISPC_OBJS} ${ISPC_SOURCES})

if(MSVC)
elseif(UNIX)
elseif(APPLE)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_ROOT}")

