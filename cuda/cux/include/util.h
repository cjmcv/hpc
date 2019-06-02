/*!
* \brief Utility functions.
*/

#ifndef CUX_UTIL_HPP_
#define CUX_UTIL_HPP_

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "time.h"

namespace cux {

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
// Structure.
////////////////

// Timer for cuda.
struct GpuTimer {
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void Start() {
    cudaEventRecord(start_, NULL);
  }
  void Stop() {
    cudaEventRecord(stop_, NULL);
  }
  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

  cudaEvent_t start_;
  cudaEvent_t stop_;
};
} // cux.
#endif //CUX_UTIL_HPP_