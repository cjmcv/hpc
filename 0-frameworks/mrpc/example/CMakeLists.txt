cmake_minimum_required(VERSION 2.8)

# Include
SET(EXAMPLE_INCLUDE_DIR ${MRPC_INCLUDE_DIR} ${3RDPARTY_INCLUDE_DIR})
message(STATUS "include: ${EXAMPLE_INCLUDE_DIR}")
include_directories(${EXAMPLE_INCLUDE_DIR})

# Build
add_executable(example_client "${PROJECT_SOURCE_DIR}/example/test_client.cpp")
add_executable(example_server "${PROJECT_SOURCE_DIR}/example/test_server.cpp")

# Depends on project mrpc_lib.
target_link_libraries(example_client mrpc_lib)
target_link_libraries(example_server mrpc_lib)

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})