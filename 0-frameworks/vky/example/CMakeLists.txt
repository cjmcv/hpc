cmake_minimum_required(VERSION 2.8)

# Define project name
project(example)
set(PROJECT_NAME example)

# Include
SET(EXAMPLE_INCLUDE_DIR
    ${VKY_INCLUDE_DIR}
    ${VULKAN_HEADERS_INSTALL_DIR}  # TODO: delete it.
)
message(STATUS "Example include: ${EXAMPLE_INCLUDE_DIR}")
include_directories(${EXAMPLE_INCLUDE_DIR})

# Source
file(GLOB_RECURSE EXAMPLE_SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
)
message(STATUS "Example src: ${EXAMPLE_SRC_LIST}")

# Build
add_executable(${PROJECT_NAME} ${EXAMPLE_SRC_LIST})

# Depends on project cux.
target_link_libraries(${PROJECT_NAME} vky)

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})