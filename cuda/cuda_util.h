/*!
* \brief Utility functions.
*/

#ifndef CJMCV_CUDA_UTIL_HPP_
#define CJMCV_CUDA_UTIL_HPP_

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "time.h"

namespace cjmcv_cuda_util {

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

////////////////
// Function.
////////////////

// 
int InitEnvironment(const int dev_id) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return 1;
  }
  fprintf(stderr, "GPU Device %d: \"%s\" with compute capability %d.%d with %d multi-processors.\n\n", 
    dev_id, device_prop.name, device_prop.major, device_prop.minor, device_prop.multiProcessorCount);

  return 0;
}

void CleanUpEnvironment() {
  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  CUDA_CHECK(cudaDeviceReset());
}

template <typename T>
void PrintArray(std::string str, T *h_in, int num_items) {
  std::cout << str.c_str();
  for (int i = 0; i < num_items; i++) {
    std::cout << h_in[i] << ",";
  }
  std::cout << std::endl;
}

}
#endif //CJMCV_CUDA_UTIL_HPP_