/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_HPP_
#define CUX_EXECUTOR_HPP_

#include "util/util.h"
#include "util/op_factory.h"
// TODO: Hide it.
#include "operator.h"
#include "operator/dot_product.h"
#include "operator/gemm.h"

namespace cux {

class Executor {
public:
  Executor() {
    op_ = nullptr;
  }
  ~Executor() {
    if (op_ != nullptr) {
      delete op_;
      op_ = nullptr;
    }
  }

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

  int SetDevice(const int dev_id, const bool is_show_info = true) {
    device_id_ = dev_id;

    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop_, device_id_));
    if (device_prop_.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
      return -1;
    }
    if (is_show_info) {
      CUXLOG_COUT("********************** Information ************************");
      CUXLOG_COUT("*");
      CUXLOG_COUT("* GPU Device %d: \"%s\". ", device_id_, device_prop_.name);
      CUXLOG_COUT("* (device) Compute capability: %d.%d. ", device_prop_.major, device_prop_.minor);
      CUXLOG_COUT("* (device) Multi-processors count: %d. ", device_prop_.multiProcessorCount);
      CUXLOG_COUT("* (device) Global memory available on device in bytes: %zd. ", device_prop_.totalGlobalMem);
      CUXLOG_COUT("* (device) Constant memory available on device in bytes: %zd. ", device_prop_.totalConstMem);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (grid) Maximum size of each dimension of a grid: (%d, %d, %d). ",
        device_prop_.maxGridSize[0], device_prop_.maxGridSize[1], device_prop_.maxGridSize[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (block) Shared memory available per block in bytes: %zd. ", device_prop_.sharedMemPerBlock);
      CUXLOG_COUT("* (block) 32-bit registers available per block: %d. ", device_prop_.regsPerBlock);
      CUXLOG_COUT("* (block) Maximum number of threads per block: %d. ", device_prop_.maxThreadsPerBlock);
      CUXLOG_COUT("* (block) Maximum size of each dimension of a block: (%d, %d, %d). ",
        device_prop_.maxThreadsDim[0], device_prop_.maxThreadsDim[1], device_prop_.maxThreadsDim[2]);
      CUXLOG_COUT("*");
      CUXLOG_COUT("* (thread) Warp size in threads: %d. ", device_prop_.warpSize);
      CUXLOG_COUT("*");
      CUXLOG_COUT("***********************************************************");
    }
    return 0;
  }
  void SelectOp(std::string op_name, std::string params) {
    // TODO: 使用Operator作为未搜索到的op_name的情况执行，并告警
    op_ = OpFactory::GetInstance().CreateOpByType(op_name, params);
  }

  int SetOpIoData(const std::vector< CuxData<float>* > &input, 
                  const std::vector< CuxData<float>* > &output) {
    return op_->SetIoData(input, output);
  }

  void SetDebugParams(const int loop) {
    op_->SetLoops(10);
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
  int device_id_;
  cudaDeviceProp device_prop_;
  Operator *op_;
};


} // cux.
#endif //CUX_EXECUTOR_HPP_