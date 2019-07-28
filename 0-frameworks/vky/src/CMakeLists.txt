cmake_minimum_required(VERSION 2.8)

# Define project name
project(vky)
set(PROJECT_NAME vky)

# Include
SET(SRC_INCLUDE_DIR
    ${VKY_INCLUDE_DIR}
    ${VULKAN_HEADERS_INSTALL_DIR}
)
message(STATUS "src include: ${SRC_INCLUDE_DIR}")
include_directories(${SRC_INCLUDE_DIR})

# Source
file(GLOB_RECURSE SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
    # Include
    "${VKY_INCLUDE_DIR}/*.h"
)
message(STATUS "src: ${SRC_LIST}")

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})  #SHARED

set_target_properties(${PROJECT_NAME} PROPERTIES output_name "utils") 
target_link_libraries(${PROJECT_NAME} ${VULKAN_LIBS})

# Set output path
set(LIBRARY_OUTPUT_PATH ${OUTPUT_DIR})
message(STATUS "LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")