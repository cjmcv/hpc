cmake_minimum_required(VERSION 2.8)

# Note: This cmakelists have only been verified on the Windows platform.

# Define project name
project(mrpc)
set(PROJECT_NAME mrpc)

add_definitions(-DUSE_ASIO)

include(CheckCXXCompilerFlag)

# Release by default
set(CMAKE_BUILD_TYPE "Release")  # Debug

if(WIN32)
  # MSVC
  CHECK_CXX_COMPILER_FLAG("/std:c++17" COMPILER_SUPPORTS_CXX17)
  if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MTd /Ob2 /Od /ZI")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
else(WIN32) # TODO
  # GNU
  CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
  if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Set for .so
  set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g") 
  set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 
endif(WIN32)

set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)

############################################
# source
set(3RDPARTY_BASE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(ASIO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/asio/asio/include)

set(3RDPARTY_INCLUDE_DIR ${3RDPARTY_BASE_INCLUDE_DIR} ${ASIO_INCLUDE_DIR})

set(MRPC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mrpc)
add_subdirectory(mrpc)

############################################
# example
add_subdirectory(example)