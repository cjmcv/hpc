#ifndef CJMCV_OCL_UTIL_HPP_
#define CJMCV_OCL_UTIL_HPP_

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

// Print the name and version of the platform.
void PrintPlatBasicInfo(cl_platform_id &platform);

// Get a multiple that's rounded up.
size_t GetRoundUpMultiple(size_t dividend, size_t divisor);

void PrintCommandElapsedTime(cl_event ev);

////////////////
// Structure.
////////////////


////////////////
// Class.
////////////////

// KernelLoader: It is used to get the kernel functions from the program file.
class KernelLoader {
public:
  KernelLoader();
  // Load 
  bool Load(const char *source_file);
  void UnLoad();
  bool CreateProgram(const cl_context &context);
  bool GetKernel(const char *kernel_name, cl_kernel *kernel);

private:
  char* LoadProgSource(const char* file_name, const char* preamble, size_t* final_length);

private:
  cl_int err_code_;
  // Byte size of kernel code
  // Buffer to hold source for compilation
  size_t program_length_;
  char* program_source_;

  cl_program program_;
};


} //namespace cjmcv_ocl_util

#endif //CJMCV_OCL_UTIL_HPP_