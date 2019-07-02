/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_HPP_
#define CUX_EXECUTOR_HPP_

#include "util/util.h"
#include "util/op_factory.h"
#include "util/launch_config.h"
// TODO: Hide it.
#include "operator.h"
#include "operator/dot_product.h"
#include "operator/gemm.h"

namespace cux {

static int InitEnvironment() {
  CUXLOG_INFO("Initialize Ennviroment.");

  OpFactory::GetInstance().RegisterOpClass("dot_product", VectorDotProduct::Creator);
  OpFactory::GetInstance().RegisterOpClass("gemm", GEMM::Creator);

  CUXLOG_COUT("* Registered Op: < %s>", OpFactory::GetInstance().PrintList().c_str());
  return 0;
}

static void CleanUpEnvironment() {
  CUXLOG_INFO("Cleanup Ennviroment.");
  // Reset the device and exit
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  CUDA_CHECK(cudaDeviceReset());
}

class Executor {
public:
  Executor() :op_(nullptr), launch_config_(nullptr) {}

  int Initialize(const int dev_id, const bool is_show_info = true) {
    device_.id = dev_id;

    CUDA_CHECK(cudaSetDevice(device_.id));
    CUDA_CHECK(cudaGetDeviceProperties(&device_.prop, device_.id));
    if (device_.prop.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
      return -1;
    }
    launch_config_ = new LaunchConfig(&device_);

    if (is_show_info) {
      CUXLOG_COUT("********************** Information ************************");
      CUXLOG_COUT("*");
      CUXLOG_COUT("* GPU Device %d: \"%s\". ", device_.id, device_.prop.name);
      CUXLOG_COUT("* (device) Compute capability: %d.%d. ", device_.prop.major, device_.prop.minor);
      CUXLOG_COUT("* (device) Multi-processors count: %d. ", device_.prop.multiProcessorCount);
      CUXLOG_COUT("* (device) Global memory available on device in bytes: %zd. ", device_.prop.totalGlobalMem);
      CUXLOG_COUT("* (device) Constant memory available on device in bytes: %zd. ", device_.prop.totalConstMem);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (grid) Maximum size of each dimension of a grid: (%d, %d, %d). ",
        device_.prop.maxGridSize[0], device_.prop.maxGridSize[1], device_.prop.maxGridSize[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (block) Shared memory available per block in bytes: %zd. ", device_.prop.sharedMemPerBlock);
      CUXLOG_COUT("* (block) 32-bit registers available per block: %d. ", device_.prop.regsPerBlock);
      CUXLOG_COUT("* (block) Maximum number of threads per block: %d. ", device_.prop.maxThreadsPerBlock);
      CUXLOG_COUT("* (block) Maximum size of each dimension of a block: (%d, %d, %d). ",
        device_.prop.maxThreadsDim[0], device_.prop.maxThreadsDim[1], device_.prop.maxThreadsDim[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (thread) Warp size in threads: %d. ", device_.prop.warpSize);
      CUXLOG_COUT("*");
      CUXLOG_COUT("***********************************************************");
    }
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
    op_ = OpFactory::GetInstance().CreateOpByType(op_name, params);
  }

  int SetOpIoData(const std::vector< CuxData<float>* > &input, 
                  const std::vector< CuxData<float>* > &output) {
    return op_->SetIoData(input, output);
  }

  void SetOpParams(const int loop_cn) {
    OpParams params;
    params.launch_config = launch_config_;
    params.loop_cn = loop_cn;
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
  Operator *op_;
};

} // cux.
#endif //CUX_EXECUTOR_HPP_