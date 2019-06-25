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

  static int InitEnvironment(const int dev_id, const bool is_show_info = true) {
    RegisterOps();

    CUXLOG_INFO("Initialize CUDA Ennviroment.");

    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, dev_id));
    if (device_prop.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
      return 1;
    }
    if (is_show_info) {
      CUXLOG_COUT("********************** Information ************************");
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
      CUXLOG_COUT("* Registered Op: < %s>", OpFactory::GetInstance().PrintList().c_str());
      CUXLOG_COUT("*");
      CUXLOG_COUT("***********************************************************");
    }

    return 0;
  }

  static void CleanUpEnvironment() {
    CUXLOG_INFO("Cleanup CUDA Ennviroment.");
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_CHECK(cudaDeviceReset());
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
  static void RegisterOps() {
    OpFactory::GetInstance().RegisterOpClass("dot_product", VectorDotProduct::Creator);
    OpFactory::GetInstance().RegisterOpClass("gemm", GEMM::Creator);
  }

private:
  Operator *op_;
};


} // cux.
#endif //CUX_EXECUTOR_HPP_