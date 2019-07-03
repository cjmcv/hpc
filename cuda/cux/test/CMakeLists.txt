cmake_minimum_required(VERSION 2.8)

# Define project name
project(cux_test)
set(PROJECT_NAME cux_test)

# Include
SET(INCLUDE_LIST 
    ${PROJECT_SOURCE_DIR}/../include
)
message(STATUS "include: ${INCLUDE_LIST}")
include_directories(${INCLUDE_LIST})

# Source
file(GLOB_RECURSE TEST_SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
)
message(STATUS "test src: ${TEST_SRC_LIST}")

# Build
#add_executable(${PROJECT_NAME} ${TEST_SRC_LIST})
cuda_add_executable(${PROJECT_NAME} ${TEST_SRC_LIST})

# Link libraries
if(WIN32)
  set(LIBS_LIST ${PROJECT_SOURCE_DIR}/../build/${CMAKE_BUILD_TYPE}/cux.lib)
else(WIN32)
  set(LIBS_LIST ${PROJECT_SOURCE_DIR}/../build/libcux.so)
endif(WIN32)

target_link_libraries(${PROJECT_NAME} ${LIBS_LIST})

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../build)