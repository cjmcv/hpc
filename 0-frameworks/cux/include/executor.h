/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_H_
#define CUX_EXECUTOR_H_

#include "util/util.h"
#include "util/op_factory.h"
#include "util/launch_config.h"
#include "operator.h"
#include "operator/dot_product.h"
#include "operator/gemm.h"

namespace cux {

static int InitEnvironment() {
  CUXLOG_INFO("Initialize Environment.");

  OpFactory<float>::GetInstance().RegisterOpClass("dot_product", VectorDotProduct<float>::Creator);
  OpFactory<float>::GetInstance().RegisterOpClass("gemm", GEMM<float>::Creator);

  CUXLOG_COUT("* Registered Op: %s.", OpFactory<float>::GetInstance().PrintList().c_str());
  return 0;
}

static void CleanUpEnvironment() {
  CUXLOG_INFO("Cleanup Environment.");
  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  CUDA_CHECK(cudaDeviceReset());
}

static void QueryDevices() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  CUXLOG_COUT("-------------< Device Information >--------------");
  CUXLOG_COUT("-- The number of compute-capable devices: %d.", device_count);
  CUXLOG_COUT("-------------------------------------------------");
  for (int id = 0; id < device_count; ++id) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, id);
    CUXLOG_COUT("***");
    CUXLOG_COUT("* GPU Device %d: \"%s\". ", id, device_prop.name);
    CUXLOG_COUT("* (device) Compute capability: %d.%d. ", device_prop.major, device_prop.minor);
    CUXLOG_COUT("* (device) Number of multiprocessors on device: %d. ", device_prop.multiProcessorCount);
    CUXLOG_COUT("* (device) Maximum resident threads per multiprocessor: %d. ", device_prop.maxThreadsPerMultiProcessor);
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
    CUXLOG_COUT("***");
  }
  CUXLOG_COUT("-------------------------------------------------");
}

class Executor {
public:
  Executor() :op_(nullptr), launch_config_(nullptr) {}

  int Initialize(const int dev_id) {
    device_.id = dev_id;

    CUDA_CHECK(cudaSetDevice(device_.id));
    CUDA_CHECK(cudaGetDeviceProperties(&device_.prop, device_.id));
    if (device_.prop.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
      return -1;
    }

    launch_config_ = new LaunchConfig(&device_);
    return 0;
  }

  void Clear() {
    if (op_ != nullptr) {
      delete op_;
      op_ = nullptr;
    }
    if (launch_config_ != nullptr) {
      delete launch_config_;
      launch_config_ = nullptr;
    }
  }

  void SelectOp(std::string op_name, std::string params) {
    op_ = OpFactory<float>::GetInstance().CreateOpByType(op_name, params);
  }

  int SetOpIoData(const std::vector< Array4D* > &input, 
                  const std::vector< Array4D* > &output) {
    return op_->SetIoData(input, output);
  }

  void SetOpParams() {
    OpParams params;
    params.launch_config = launch_config_;
    op_->SetOpParams(params);
  }

  void Run(const OpRunMode mode) {
    if (mode == OpRunMode::ON_HOST) {
      op_->RunOnHost();
    }
    else {
      op_->RunOnDevice();
    }  
    op_->PrintElapsedTime(mode);
  }

private:
  Device device_;
  LaunchConfig *launch_config_;
  Operator<float> *op_;
};

} // cux.
#endif //CUX_EXECUTOR_H_
