/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_HPP_
#define CUX_EXECUTOR_HPP_

#include "util.h"

namespace cux {

class Executor {
public:
  Executor() {
    op_ = new VectorDotProduct();
  }
  ~Executor() {
    delete op_;
  }

  int InitEnvironment(const int dev_id, const bool is_show_info = true) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
    if (device_prop.computeMode == cudaComputeModeProhibited) {
      printf("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 1;
    }
    if (is_show_info) {
      printf("********************** GPU Information ************************\n");
      printf("*\n");
      printf("* GPU Device %d: \"%s\". \n", dev_id, device_prop.name);
      printf("* [device] Compute capability: %d.%d. \n", device_prop.major, device_prop.minor);
      printf("* [device] Multi-processors count: %d. \n", device_prop.multiProcessorCount);
      printf("* [device] Global memory available on device in bytes: %zd. \n", device_prop.totalGlobalMem);
      printf("* [device] Constant memory available on device in bytes: %zd. \n", device_prop.totalConstMem); 
      printf("*\n");
      printf("* [grid] Maximum size of each dimension of a grid: (%d, %d, %d). \n",
        device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
      printf("*\n");
      printf("* [block] Shared memory available per block in bytes: %zd. \n", device_prop.sharedMemPerBlock);
      printf("* [block] 32-bit registers available per block: %d. \n", device_prop.regsPerBlock);
      printf("* [block] Maximum number of threads per block: %d. \n", device_prop.maxThreadsPerBlock);
      printf("* [block] Maximum size of each dimension of a block: (%d, %d, %d). \n", 
        device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
      printf("*\n");
      printf("* [thread] Warp size in threads: %d. \n", device_prop.warpSize);
      printf("*\n");
      printf("***************************************************************\n");
    }
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

  int SetOpIoParams(const std::vector< CuxData<float>* > &input, 
                    const std::vector< CuxData<float>* > &output, 
                    const void *params) {
    return op_->SetIoParams(input, output, params);
  }
  void SetDebugParams(const int loop) {
    op_->SetLoops(10);
  }

  void Run(const RunMode mode) {
    if (mode == RunMode::ON_HOST) {
      op_->RunOnHost();
      op_->PrintCpuRunTime();
      op_->PrintResult();
    }
    else {
      op_->RunOnDevice();
      op_->PrintGpuRunTime();
      op_->PrintResult();
    }
  }
private:
  // TODO: 1. 将VectorDotProduct改成Operator，可手动切换具体的Operator.
  //       2. 添加OpFactory，用于注册和生产Op.
  //       3. GPU信息查询. - Finish.
  VectorDotProduct *op_;
};


} // cux.
#endif //CUX_EXECUTOR_HPP_