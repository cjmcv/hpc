cmake_minimum_required(VERSION 2.8)

# Define project name
project(mrpc_lib)
set(PROJECT_NAME mrpc_lib)

# Include 
SET(LIB_INCLUDE_DIR ${MRPC_INCLUDE_DIR} ${3RDPARTY_INCLUDE_DIR})
include_directories(${LIB_INCLUDE_DIR})
message(STATUS "include: ${LIB_INCLUDE_DIR}")

# Source
file(GLOB_RECURSE SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
    "${PROJECT_SOURCE_DIR}/*.cc"
    # Include.
    "${MRPC_INCLUDE_DIR}/*.hpp"
    "${MRPC_INCLUDE_DIR}/*.h"
)
message(STATUS "src: ${SRC_LIST}")

#set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
#message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})  #SHARED

set_target_properties(${PROJECT_NAME} PROPERTIES output_name "utils") 
target_link_libraries(${PROJECT_NAME})# ${CUDA_LIBRARIES}

# Set output path
set(LIBRARY_OUTPUT_PATH ${OUTPUT_DIR})
message(STATUS "LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")