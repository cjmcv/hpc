#include <iostream>
#include "ocl_util.h"

namespace cjmcv_ocl_util {

const char* GetErrorString(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_OUT_OF_RESOURCES";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case CL_INVALID_PROPERTY:
    return "CL_INVALID_PROPERTY";
  case CL_INVALID_IMAGE_DESCRIPTOR:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case CL_INVALID_COMPILER_OPTIONS:
    return "CL_INVALID_COMPILER_OPTIONS";
  case CL_INVALID_LINKER_OPTIONS:
    return "CL_INVALID_LINKER_OPTIONS";
  case CL_INVALID_DEVICE_PARTITION_COUNT:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
  }

  return "Unknown opencl status";
}

//  Loads a Program file and prepends the preamble to the code.
char* LoadProgSource(const char* file_name, const char* preamble, size_t* final_length) {
  // locals 
  FILE* file_stream = NULL;
  size_t source_length;

  // open the OpenCL source code file
#ifdef _WIN32   // Windows version
  if (fopen_s(&file_stream, file_name, "rb") != 0) {
    printf("Can not open the file : %s.\n", file_name);
    return NULL;
  }
#else           // Linux version
  file_stream = fopen(file_name, "rb");
  if (file_stream == 0) {
    printf("Can not open the file : %s.\n", file_name);
    return NULL;
  }
#endif

  size_t preamble_length = strlen(preamble);

  // get the length of the source code
  fseek(file_stream, 0, SEEK_END);
  source_length = ftell(file_stream);
  fseek(file_stream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in
  char* source_string = (char *)malloc(source_length + preamble_length + 1);
  memcpy(source_string, preamble, preamble_length);
  if (fread((source_string)+preamble_length, source_length, 1, file_stream) != 1) {
    fclose(file_stream);
    free(source_string);
    return 0;
  }

  // close the file and return the total length of the combined (preamble + source) string
  fclose(file_stream);
  if (final_length != 0) {
    *final_length = source_length + preamble_length;
  }
  source_string[source_length + preamble_length] = '\0';

  return source_string;
}

// Print the name and version of the platform.
void PrintPlatBasicInfo(cl_platform_id &platform) {
  size_t ext_size;
  OCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS,
    0, NULL, &ext_size));
  char *name = (char*)malloc(ext_size);
  OCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME,
    ext_size, name, NULL));

  char *version = (char*)malloc(ext_size);
  OCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION,
    ext_size, version, NULL));

  printf("The name of the platform is <%s> with version <%s>.\n", name, version);

  free(name);
  free(version);
}

} //namespace cjmcv_ocl_util