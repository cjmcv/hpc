/*!
* \brief Utility.
*/

#ifndef CUX_UTIL_H_
#define CUX_UTIL_H_

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include "util/half.h"

namespace cux {

////////////////
// Enumeration.
////////////////
enum TypeFlag {
  FLOAT32 = 0,
  INT32 = 1,
  FLOAT16 = 2, 
  INT8 = 3,
  TYPES_NUM = 4  // Used to mark the number of elements in the enumeration TypeFlag.
};

enum OpRunMode {
  ON_HOST,
  ON_DEVICE
};

enum Shape {
  NUMBER,
  CHANNELS,
  HEIGHT,
  WIDTH
};

enum DataFetchMode {
  NO_PUSH,        // Get current data without pushing data across devices.
  PUSH,           // If you want to take data from CPU, and there's data in GPU, it pushes data from GPU to CPU.
  PUSH_IF_EMPTY   // Add push condition: the data to be fetched is empty.
};

////////////////
// Function.
////////////////
const char* CublasGetErrorString(cublasStatus_t error);

////////////////
// Class.
////////////////
class StrProcessor {
public:
  static std::string FetchSubStr(std::string &src_str, std::string start_str, std::string end_str) {
    int start_idx = src_str.find(start_str) + start_str.length();
    int end_idx = src_str.find(end_str, start_idx);
    return src_str.substr(start_idx, end_idx - start_idx);
  }
};

////////////////
// Macro.
////////////////
// Check for cuda error messages.
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s : %s \n", \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

// Check for cublas error messages.
#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
	  if(status != CUBLAS_STATUS_SUCCESS) {	\
		  fprintf(stderr, "CUBLAS_CHECK error in line %d of file %s : %s \n", \
              __LINE__, __FILE__, cux::CublasGetErrorString(status) ); \
		  exit(EXIT_FAILURE);	\
	  } \
  } while (0)

// Log
#define CUXLOG_ERR(format, ...) \
  do { \
    fprintf(stderr,"[ERROR]: (%s: %d)"##format"\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::abort(); \
  } while(0)
#define CUXLOG_WARN(format, ...) fprintf(stdout,"[WARN]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_INFO(format, ...) fprintf(stdout,"[INFO]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_COUT(format, ...) fprintf(stdout,"> "##format"\n", ##__VA_ARGS__);

#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<cux::half>

#define TYPE_SWITCH(type, DType, ...)               \
  switch (type) {                                   \
  case cux::TypeFlag::FLOAT32:                     \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::FLOAT16:                     \
    {                                               \
      typedef cux::half DType;                      \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::INT32:                       \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::INT8:                        \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    CUXLOG_ERR("Unknown type enum %d", type);       \
  }
////////////////
// Struct.
////////////////
struct Device {
  int id;
  cudaDeviceProp prop;
};

template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = cux::TypeFlag::FLOAT32;
};
template<>
struct DataType<cux::half> {
  static const int kFlag = cux::TypeFlag::FLOAT16;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = cux::TypeFlag::INT32;
};
template<>
struct DataType<uint32_t> {
  static const int kFlag = cux::TypeFlag::INT32;
};
template<>
struct DataType<int8_t> {
  static const int kFlag = cux::TypeFlag::INT8;
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = cux::TypeFlag::INT8;
};

////
// TODO: 3rdparty: 1.使用数据库，做参数查询，性能数据备份.
//
} // cux.
#endif //CUX_UTIL_H_
