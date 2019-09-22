cmake_minimum_required(VERSION 2.8)

# Note: This cmakelists have only been verified on the Windows platform.

# Options
option(CUX_UNIT_TESTS  "Build with unit tests."  ON)
option(CUX_WIN_CPU_SIMD_OPTIONS  "Enables CPU SIMD instructions."  ON)

# Add definitions by some options
#if(CUX_WIN_CPU_SIMD_OPTIONS)
#  add_definitions(-DCUX_X86_SIMD)
#endif()

# Version.
set(CUDA_VERSION "10.0" CACHE STRING "CUDA version to build against")

# Define project name
project(main_cux)
set(PROJECT_NAME main_cux)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Release by default
set(CMAKE_BUILD_TYPE "Release")  # Debug

if(WIN32)
  # MSVC  
  if(CUX_WIN_CPU_SIMD_OPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MTd /Ob2 /Od /ZI")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
else(WIN32) # TODO
  # GNU
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Set for .so
  set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g") 
  set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 
endif(WIN32)

# OpenMP Support
CHECK_CXX_COMPILER_FLAG("/openmp" MSVC_OPENMP_SUPPORT)
if (MSVC_OPENMP_SUPPORT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif()  
CHECK_CXX_COMPILER_FLAG("-fopenmp" GCC_OPENMP_SUPPORT)
if (GCC_OPENMP_SUPPORT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

  
# Set cuda.
find_package(CUDA ${CUDA_VERSION} REQUIRED EXACT)
set(CUDA_NVCC_FLAG_DEBUG "-O0 -g -G")
set(CUDA_NVCC_FLAG_RELEASE "-O3 -Wall")
# by default we assume compute cabability 3.5 and 5.2. If you change this change it in
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero

set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)

############################################
# 3rdparty
if(CUX_UNIT_TESTS)
  set(GOOGLETEST_VERSION 1.9.0)
  set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest/googletest")
  set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
  set(GTEST_LIBRARIES gtest gtest_main)
  set(GTEST_MAIN_LIBRARY gtest_main)
  set(GTEST_LIBRARY gtest)
  add_subdirectory(${GTEST_ROOT})
endif()

############################################
# cux source
set(CUX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)

############################################
# example
add_subdirectory(example)

############################################
# unit test
if(CUX_UNIT_TESTS)
  add_subdirectory(tests)
endif()