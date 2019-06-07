/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_HPP_
#define CUX_EXECUTOR_HPP_

#include "util.h"

namespace cux {

class Executor {
public:
  Executor(){}

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

};


} // cux.
#endif //CUX_EXECUTOR_HPP_