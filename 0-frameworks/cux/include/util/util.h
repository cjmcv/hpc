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
enum Code {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
};

enum TypeFlag {
  kFloat32 = 0,
  kFloat16 = 1, 
  kInt32 = 2,
  kInt8 = 3,
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
#define CUXLOG_ERR(format, ...) fprintf(stderr,"[ERROR]: "##format"\n", ##__VA_ARGS__); std::abort();
#define CUXLOG_WARN(format, ...) fprintf(stdout,"[WARN]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_INFO(format, ...) fprintf(stdout,"[INFO]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_COUT(format, ...) fprintf(stdout,"> "##format"\n", ##__VA_ARGS__);

//#define INSTANTIATE_CLASS(classname) \
//  char gInstantiationGuard##classname; \
//  template class classname<float>; \
//  template class classname<half>

#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>

#define TYPE_SWITCH(type, DType, ...)               \
  switch (type) {                                   \
  case cux::TypeFlag::kFloat32:                     \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::kFloat16:                     \
    {                                               \
      typedef cux::half DType;                      \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::kInt32:                       \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case cux::TypeFlag::kInt8:                        \
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

struct DataTypeSum {
  static const int kNum = 4;
};

template<typename DType>
struct DataType;
template<>
struct DataType<float> {
  static const int kFlag = cux::TypeFlag::kFloat32;
};
template<>
struct DataType<cux::half> {
  static const int kFlag = cux::TypeFlag::kFloat16;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = cux::TypeFlag::kInt32;
};
template<>
struct DataType<uint32_t> {
  static const int kFlag = cux::TypeFlag::kInt32;
};
template<>
struct DataType<int8_t> {
  static const int kFlag = cux::TypeFlag::kInt8;
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = cux::TypeFlag::kInt8;
};

// TODO: 2. 升级Array4D：静态动态内存、异步拷贝、对齐。。
//       4. 内存池（低优先级）
//       5. CPU端异常处理/告警机制/错误码
//       7. Layout渐变的效率分析？
//       8. 检索查看所有gpu设备。deviceQuery - finish
//       9. cmake添加新筛选器？
//       10. 分析cmake出来的debug和cuda的demo工程的debug的耗时差异。
//       11. Array4D添加半精度；- Finish
//       11. gemm cublas半精度;
////
// TODO: 1. 算法与cublas对应；命名统一、功能统一 - Finish
//       2. 运算子分成有输入和输出的，以及单一输入即输出（如转置，在自己的内存操作）的两种。
//
//       3. demo：1）多组数据连续处理（预取），2）多操作混搭组合成公式做运算
//       4. Prefetcher, 预取器，预取数据到GPU，隐藏IO延时
//       5. BlockingQueue, 堵塞缓冲队列，用于服务预取器Prefetcher，缓存预取的数据
//       6. InnerThread, 内部线程，为堵塞队列替换数据，共同服务于预取器Prefetcher
//
//       7. 在demo中，由用户自定义OP.
//       8. 使用模板控制Op的数据类型（可能需要针对每一种类型重写kernel）- Finish
//       9. DataType根据模板的类型获得类型的变量，从而切换该调用的kernel
//       10. other: 图任务自动调度框架。自己定义op，及其依赖关系。
////
// TODO: 3rdparty: 均以宏定义覆盖，可手动选择不使用
//                 1.使用gtest，添加单元测试模块: 性能测试/多版本核函数结果验证/异常出入判断 - Finish
//                 2.使用cublas，添加到Op中作为测试基准.-Finish
//                 https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html
//                 https://github.com/mratsim/Arraymancer/issues/112
//                 3.使用cub，封装显存管理模块.
//                 4.使用数据库，做参数查询，性能数据备份.
//                 5.python接口封装，前置任务->生成dll，导出多个必须的接口，才由python对这些接口做封装。
//
} // cux.
#endif //CUX_UTIL_H_
