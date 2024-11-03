cmake_minimum_required(VERSION 3.2)

# Define project name
project(pai_vk)
set(PROJECT_NAME pai_vk)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# Release by default
set(CMAKE_BUILD_TYPE "Release")  # Debug / Release

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Set for .so
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g") 
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Include
SET(INCLUDE_DIR
    ${VULKAN_WIN_SDK}/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
)
message(STATUS "include: ${INCLUDE_DIR}")
include_directories(${INCLUDE_DIR})
	
# Source GLOB_RECURSE
file(GLOB SRC_LIST 
    "${CMAKE_CURRENT_SOURCE_DIR}/../${MAIN_SRC}" # PROJECT_SOURCE_DIR
    # Include
    # "${INCLUDE_DIR}/*.h"
)
message(STATUS "src: ${SRC_LIST}")

# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/kernel/vulkan)
# list(APPEND ECAS_OBJECTS_TO_LINK $<TARGET_OBJECTS:ECAS_KERNEL_VULKAN>)

# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/backend/engine/vulkan)
# list(APPEND ECAS_OBJECTS_TO_LINK $<TARGET_OBJECTS:ECAS_ENGINE_VULKAN>)

if (VULKAN_WIN_SDK) # for windows x86 sdk
    list(APPEND LIBS_TO_LINK ${VULKAN_WIN_SDK}/Lib/vulkan-1.lib)
    message(STATUS "VULKAN_WIN_SDK: ${VULKAN_WIN_SDK}")
else()
    list(APPEND LIBS_TO_LINK -lvulkan)
endif()

# add_library(ECAS_ENGINE_VULKAN OBJECT ${SRC_LIST})

# add_library(${PROJECT_NAME} STATIC  ${SRC_LIST})  #SHARED
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS_TO_LINK})

# Set output path
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIR})