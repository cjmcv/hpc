/*!
* \brief Utility.
*/

#ifndef CUX_TIME_H_
#define CUX_TIME_H_

#include <iostream>
#include <chrono>

#include "util/util.h"

namespace cux {

// Timer for gpu.
class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }
  inline void Start() { CUDA_CHECK(cudaEventRecord(start_, NULL)); }
  inline void Stop() { CUDA_CHECK(cudaEventRecord(stop_, NULL)); }
  inline float MilliSeconds() {
    float elapsed;
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
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

/////////////////////////////////////////////////
//  auto func = [&]()
//  -> float {
//    timer.Start();
//    cux::QueryDevices();
//    timer.Stop();
//    return timer.MilliSeconds();
//  };
//  ret = func();
#define GET_TIME_DIFF(timer, ...)     \
  [&]() -> float {                    \
    timer.Start();                    \
    {__VA_ARGS__}                     \
    timer.Stop();                     \
    return timer.MilliSeconds();      \
  }();

} // cux.
#endif //CUX_TIME_H_
