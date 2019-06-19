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
    CUXLOG_INFO("Initialize CUDA Ennviroment.");

    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
    if (device_prop.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
      return 1;
    }
    if (is_show_info) {
      CUXLOG_COUT("********************** GPU Information ************************");
      CUXLOG_COUT("*");
      CUXLOG_COUT("* GPU Device %d: \"%s\". ", dev_id, device_prop.name);
      CUXLOG_COUT("* (device) Compute capability: %d.%d. ", device_prop.major, device_prop.minor);
      CUXLOG_COUT("* (device) Multi-processors count: %d. ", device_prop.multiProcessorCount);
      CUXLOG_COUT("* (device) Global memory available on device in bytes: %zd. ", device_prop.totalGlobalMem);
      CUXLOG_COUT("* (device) Constant memory available on device in bytes: %zd. ", device_prop.totalConstMem);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (grid) Maximum size of each dimension of a grid: (%d, %d, %d). ",
        device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (block) Shared memory available per block in bytes: %zd. ", device_prop.sharedMemPerBlock);
      CUXLOG_COUT("* (block) 32-bit registers available per block: %d. ", device_prop.regsPerBlock);
      CUXLOG_COUT("* (block) Maximum number of threads per block: %d. ", device_prop.maxThreadsPerBlock);
      CUXLOG_COUT("* (block) Maximum size of each dimension of a block: (%d, %d, %d). ",
        device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (thread) Warp size in threads: %d. ", device_prop.warpSize);
      CUXLOG_COUT("*");
      CUXLOG_COUT("***************************************************************");
    }
    return 0;
  }

  void CleanUpEnvironment() {
    CUXLOG_INFO("Cleanup CUDA Ennviroment.");
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
    }
    else {
      op_->RunOnDevice();
    }  
    op_->PrintElapsedTime(mode);
  }
private:
  // TODO: 1. 将VectorDotProduct改成Operator，可手动切换具体的Operator.
  //       2. 添加OpFactory，用于注册和生产Op.
  //       3. GPU信息查询. - Finish.
  VectorDotProduct *op_;
};


} // cux.
#endif //CUX_EXECUTOR_HPP_