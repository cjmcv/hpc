cmake_minimum_required(VERSION 2.8)

# Options
#option(USE_OP_TEST  "Build with operator test"  ON) # Should be the same with option in libdlex_cnn.so

# Add definitions by some options
#if(USE_OP_TEST)
#    add_definitions(-DUSE_OP_TEST)
#endif()
set(CUDA_VERSION "9.0" CACHE STRING "CUDA version to build against")

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MDd /Ob2 /Od")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /D_ITERATOR_DEBUG_LEVEL=0")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /D_ITERATOR_DEBUG_LEVEL=0")
else(WIN32)
  # GNU
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # Set for .so
  set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g") 
  set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 
endif(WIN32)

# Set cuda.
find_package(CUDA ${CUDA_VERSION} REQUIRED EXACT)
set(CUDA_NVCC_FLAG_DEBUG "-O0 -g -G")
set(CUDA_NVCC_FLAG_RELEASE "-O3 -Wall")
# by default we assume compute cabability 3.5 and 5.2. If you change this change it in
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero

############################################
# Subdir
add_subdirectory(src)
add_subdirectory(test)