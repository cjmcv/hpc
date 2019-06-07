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
  OnHost,
  OnDevice
};

////////////////
// Macro.
////////////////
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s : %s \n", \
              __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

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
    return std::chrono::duration_cast<ns>(stop_time_ - start_time_).count();
  }

  // Returns the elapsed time in milliseconds.
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }

  //brief Returns the elapsed time in microseconds.
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }

  //Returns the elapsed time in seconds.
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
  std::chrono::time_point<clock> start_time_;
  std::chrono::time_point<clock> stop_time_;
};

// TODO: Benchmark. Check IO / kernel...
} // cux.
#endif //CUX_UTIL_HPP_