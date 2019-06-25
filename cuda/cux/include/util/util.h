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

template <typename Dtype>
class ResultChecker {
public:
  ResultChecker() :prev_data_(nullptr), len_() {}
  ~ResultChecker() {
    if (prev_data_ != nullptr) {
      delete[]prev_data_;
      len_ = 0;
    }
  }

  bool CheckArray(const Dtype *in, const int len, const int id) {
    if (id == 0) {
      SetBenchmarkData(in, len);
      return true;
    }
    float diff = 0.0;
    for (int i = 0; i < len; i++) {
      Dtype t = prev_data_[i] - in[i];
      diff += (t >= 0 ? t : -t);
    }
    if (diff < DBL_MIN) {
      CUXLOG_INFO("Pass: V0 vs V%d -> (diff: %f, first number: %f, %f)", 
        id, diff, (float)prev_data_[0], (float)in[0]);
      return true;
    }
    else {
      CUXLOG_WARN("Fail: V0 vs V%d -> (diff: %f, first number: %f, %f)",
        id, diff, (float)prev_data_[0], (float)in[0]);
      return false;
    }
  }

private:
  void SetBenchmarkData(const Dtype *in, const int len) {
    if (prev_data_ == nullptr) {
      prev_data_ = new Dtype[len];
      len_ = len;
    }
    else if (len_ != len) {
      delete[]prev_data_;
      prev_data_ = new Dtype[len];
      len_ = len;
    }
    memcpy(prev_data_, in, sizeof(Dtype) * len);
  }

private:
  Dtype *prev_data_;
  int len_;
};
// TODO: 1. 结果检查，单独写一个函数将所有核函数运行一遍，结果一致则输出pass。
//       2. 升级CuxData：静态动态内存、异步拷贝、对齐. 添加标记位，避免重复拷贝。
//       3. 性能测试：添加占用率数据。
//       4. 内存池 / 显存池（低优先级）？
//       5. CPU端异常处理/告警机制/错误码
//       6. Layout推荐
////
// TODO: 1. 算法与cublas对应；命名统一、功能统一
//       2. 运算子分成有输入和输出的，以及单一输入即输出（如转置，在自己的内存操作）的两种。
//
//       3. demo：1）多组数据连续处理（预取），2）多操作混搭组合成公式做运算
//       4. Prefetcher, 预取器，预取数据到GPU，隐藏IO延时
//       5. BlockingQueue, 堵塞缓冲队列，用于服务预取器Prefetcher，缓存预取的数据
//       6. InnerThread, 内部线程，为堵塞队列替换数据，共同服务于预取器Prefetcher
//
//       7. 在demo中，由用户自定义OP.
////
// TODO: 3rdparty: 均以宏定义覆盖，可手动选择不使用
//                 1.使用gtest，添加单元测试模块: 性能测试/多版本核函数结果验证/异常出入判断
//                 2.使用cublas，添加到Op中作为测试基准.
//                 3.使用cub，封装显存管理模块.
//                 4.使用数据库，做参数查询，性能数据备份.
//                 5.python接口封装?
//
} // cux.
#endif //CUX_UTIL_HPP_