cmake_minimum_required(VERSION 2.8)

# Define project name
project(cux)
set(PROJECT_NAME cux)

# Include
SET(INCLUDE_LIST 
    ${PROJECT_SOURCE_DIR}/../include
)
message(STATUS "include: ${INCLUDE_LIST}")
include_directories(${INCLUDE_LIST})

# Source
file(GLOB_RECURSE SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
    "${PROJECT_SOURCE_DIR}/*.cu"
    "${PROJECT_SOURCE_DIR}/operator/*.cpp"
    "${PROJECT_SOURCE_DIR}/operator/*.cu"
    "${PROJECT_SOURCE_DIR}/util/*.cpp"
    "${PROJECT_SOURCE_DIR}/util/*.cu"
)
message(STATUS "src: ${SRC_LIST}")

cuda_add_library(${PROJECT_NAME} STATIC ${SRC_LIST})  #SHARED

set_target_properties(${PROJECT_NAME} PROPERTIES output_name "utils") 
target_link_libraries(${PROJECT_NAME})

# Set output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../build)
message(STATUS "LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")