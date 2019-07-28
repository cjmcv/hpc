# Define project name
project(unit_tests)
set(PROJECT_NAME unit_tests)

# Include
SET(INCLUDE_LIST 
    ${CUX_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
)
message(STATUS "include: ${INCLUDE_LIST}")
include_directories(${INCLUDE_LIST})

# Source
file(GLOB_RECURSE TESTS_SRC_LIST 
    "${PROJECT_SOURCE_DIR}/*.cpp"
)
message(STATUS "testS src: ${TESTS_SRC_LIST}")

# Build
cuda_add_executable(${PROJECT_NAME} ${TESTS_SRC_LIST})

# Depends on project cux.
target_link_libraries(${PROJECT_NAME} cux ${GTEST_LIBRARY})

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})