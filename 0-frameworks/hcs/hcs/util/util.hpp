/*!
* \brief Util.
*/

#ifndef HCS_UTIL_H_
#define HCS_UTIL_H_

#include <iostream>
#include "../common.hpp"

namespace hcs {

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

// Get type from type flag.
#define TYPE_SWITCH(type, DType, ...)               \
  switch (type) {                                   \
  case hcs::TypeFlag::FLOAT32:                      \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case hcs::TypeFlag::INT32:                        \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case hcs::TypeFlag::INT8:                         \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    printf("Unknown type enum %d", type);           \
  }

// Get type flag from type.
template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = hcs::TypeFlag::FLOAT32;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = hcs::TypeFlag::INT32;
};
template<>
struct DataType<uint32_t> {
  static const int kFlag = hcs::TypeFlag::INT32;
};
template<>
struct DataType<int8_t> {
  static const int kFlag = hcs::TypeFlag::INT8;
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = hcs::TypeFlag::INT8;
};
} // hcs.
#endif //HCS_UTIL_H_
