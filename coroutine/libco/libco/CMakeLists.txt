cmake_minimum_required(VERSION 2.8)
project(libco)

# This for mac osx only
set(CMAKE_MACOSX_RPATH 0)

# Set cflags
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -g -fno-strict-aliasing -O2 -Wall -export-dynamic -Wall -pipe  -D_GNU_SOURCE -D_REENTRANT -fPIC -Wno-deprecated -m64)

# Use c and asm
enable_language(C ASM)

# include
set(INCLUDE_LIST ${PROJECT_SOURCE_DIR}
                 ${PROJECT_SOURCE_DIR}/include) 

include_directories(${INCLUDE_LIST})

# Add source files
set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/co_epoll.cpp
                 ${PROJECT_SOURCE_DIR}/src/co_hook_sys_call.cpp
                 ${PROJECT_SOURCE_DIR}/src/co_routine.cpp
                 ${PROJECT_SOURCE_DIR}/src/coctx.cpp
                 ${PROJECT_SOURCE_DIR}/src/coctx_swap.S)

# Add static and shared library target
add_library(colib_static STATIC ${SOURCE_FILES})

# Set library output name
set_target_properties(colib_static PROPERTIES OUTPUT_NAME colib)
set_target_properties(colib_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
