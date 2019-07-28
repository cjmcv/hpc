cmake_minimum_required(VERSION 2.8)

# Define project name
project(example)
set(PROJECT_NAME example)

# Include
SET(INCLUDE_LIST ${CUX_INCLUDE_DIR})
message(STATUS "include: ${INCLUDE_LIST}")
include_directories(${INCLUDE_LIST})

# Source
file(GLOB_RECURSE EXAMPLE_SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
)
message(STATUS "test src: ${EXAMPLE_SRC_LIST}")

# Build
#add_executable(${PROJECT_NAME} ${EXAMPLE_SRC_LIST})
cuda_add_executable(${PROJECT_NAME} ${EXAMPLE_SRC_LIST})

# Depends on project cux.
target_link_libraries(${PROJECT_NAME} cux)

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})