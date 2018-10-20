#ifndef CJMCV_CUDA_UTIL_HPP_
#define CJMCV_CUDA_UTIL_HPP_

#include <iostream>
#include <CL/cl.h>

namespace cjmcv_ocl_util {

////////////////
// Macro.
////////////////
#define OCL_CHECK(condition) \
  do { \
    cl_int error = condition; \
    if (error != CL_SUCCESS) { \
      fprintf(stderr, "OCL_CHECK error in line %d of file %s : %s \n", \
              __LINE__, __FILE__, cjmcv_ocl_util::GetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);
////////////////

// Functions.
////////////////
// Get the error string by error code.
const char* GetErrorString(cl_int error);

// Load a program file and prepend the preamble to the code.
char* LoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

// Print the name and version of the platform.
void PrintPlatBasicInfo(cl_platform_id &platform);

} //namespace cjmcv_ocl_util

#endif //CJMCV_CUDA_UTIL_HPP_