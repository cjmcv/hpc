/*!
* \brief Utility.
*/

#ifndef CUX_UTIL_HPP_
#define CUX_UTIL_HPP_

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>

namespace cux {

////////////////
// Enumeration.
////////////////
enum RunMode {
  ON_HOST,
  ON_DEVICE
};

enum CuxShape {
  NUMBER,
  CHANNELS,
  HEIGHT,
  WIDTH
};

// TODO: CPU端异常处理/告警机制
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

// Log
#define CUXLOG_ERR(format, ...) fprintf(stderr,"[ERROR]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_WARN(format, ...) fprintf(stdout,"[WARN]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_INFO(format, ...) fprintf(stdout,"[INFO]: "##format"\n", ##__VA_ARGS__);
#define CUXLOG_COUT(format, ...) fprintf(stdout,"> "##format"\n", ##__VA_ARGS__);

////////////////
// Class.
////////////////

// Timer for gpu.
class GpuTimer {
public:
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  inline void Start() { cudaEventRecord(start_, NULL); }
  inline void Stop() { cudaEventRecord(stop_, NULL); }
  inline float MilliSeconds() {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

protected:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

// Timer for cpu.
class CpuTimer {
public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;

  inline void Start() { start_time_ = clock::now(); }
  inline void Stop() { stop_time_ = clock::now(); }
  inline float NanoSeconds() {
    return (float)std::chrono::duration_cast<ns>(stop_time_ - start_time_).count();
  }

  // Returns the elapsed time in milliseconds.
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }

  // Returns the elapsed time in microseconds.
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }

  // Returns the elapsed time in seconds.
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
  std::chrono::time_point<clock> start_time_;
  std::chrono::time_point<clock> stop_time_;
};

// 
class StrProcessor {
public:
  static std::string FetchSubStr(std::string &src_str, std::string start_str, std::string end_str) {
    int start_idx = src_str.find(start_str) + start_str.length();
    int end_idx = src_str.find(end_str, start_idx);
    return src_str.substr(start_idx, end_idx - start_idx);
  }
};

// TODO: 1. 性能测试模块，含IO和kernel等 - Finish
//       2. 信息打印 / 日志输出模块. - Finish - TODO: 升级
//       3. 内存池 / 显存池（低优先级）
//
// TODO: 1. Prefetcher, 预取器，预取数据到GPU，隐藏IO延时
//       2. BlockingQueue, 堵塞缓冲队列，用于服务预取器Prefetcher，缓存预取的数据
//       3. InnerThread, 内部线程，为堵塞队列替换数据，共同服务于预取器Prefetcher
//
// TODO: 3rdparty: 均以宏定义覆盖，可手动选择不使用
//                 1.使用gtest，添加单元测试模块: 性能测试/多版本核函数结果验证/异常出入判断
//                 2.使用cublas，添加到Op中作为分时基准.
//                 3.使用cub，封装显存管理模块.
//                 4.使用数据库，做参数查询，性能数据备份.
//                 5.python接口封装?
//       
} // cux.
#endif //CUX_UTIL_HPP_