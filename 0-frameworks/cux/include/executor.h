/*!
* \brief Executor. cux's outermost scheduler.
*/

#ifndef CUX_EXECUTOR_H_
#define CUX_EXECUTOR_H_

#include "util/util.h"
#include "util/launch_config.h"
#include "operator/operator.h"
#include "operator/op_factory.h"
#include "operator/nrm2.h"
#include "operator/dot_product.h"
#include "operator/gemm.h"

namespace cux {

static void InitEnvironment() {
  CUXLOG_INFO("Initialize Environment.");

  OpFactory::GetInstance().RegisterOpClass("dot", Dot::Creator);
  OpFactory::GetInstance().RegisterOpClass("nrm2", Nrm2::Creator);
  OpFactory::GetInstance().RegisterOpClass("gemm", Gemm::Creator);

  CUXLOG_COUT("* Registered Op: %s.", OpFactory::GetInstance().PrintList().c_str());
  CUXLOG_COUT("");
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
  Executor() :op_(nullptr), op_assistor_(nullptr) {}

  void Initialize(const int dev_id) {
    device_.id = dev_id;

    CUDA_CHECK(cudaSetDevice(device_.id));
    CUDA_CHECK(cudaGetDeviceProperties(&device_.prop, device_.id));
    if (device_.prop.computeMode == cudaComputeModeProhibited) {
      CUXLOG_ERR("Device (%d) is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().", dev_id);
    }

    op_assistor_ = new OpAssistor(&device_);
    if (op_assistor_ == nullptr)
      CUXLOG_ERR("Failed to new an OpAssistor.");
  }

  void Clear() {
    if (op_ != nullptr) {
      delete op_;
      op_ = nullptr;
    }
    if (op_assistor_ != nullptr) {
      delete op_assistor_;
      op_assistor_ = nullptr;
    }
  }

  // Create op by op name.
  void SelectOp(std::string op_name, std::string params) {
    op_ = OpFactory::GetInstance().CreateOpByType(op_name, op_assistor_, params);
  }

  // Add user-defined kernel to a existing op.
  void AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
    if (op_ == nullptr) {
      CUXLOG_ERR("The operator has not been selected yet. Please select an operator first.");
    }
    op_->AddPlugin(kernel_if, mode);
  }

  // Bind and fill input and output data for this executor.
  void BindAndFill(const std::vector< Array4D* > &inputs,
                   const std::vector< Array4D* > &outputs,
                   int min_value, int max_value, int decimal_pose) {
    // Bind Array.
    inputs_.assign(inputs.begin(), inputs.end());
    outputs_.assign(outputs.begin(), outputs.end());
    
    // Fill.
    for (int i = 0; i < inputs_.size(); i++)
      inputs_[i]->Fill(min_value, max_value, decimal_pose, TypeFlag::FLOAT32, OpRunMode::ON_HOST);
    for (int i = 0; i < outputs_.size(); i++)
      outputs_[i]->Fill(min_value, max_value, decimal_pose, TypeFlag::FLOAT32, OpRunMode::ON_HOST);
    
    // Data synchronization across types.
    std::vector<int> type_flags;
    op_->ExtractDataTypes(type_flags);

    TYPE_SWITCH(TypeFlag::FLOAT32, FP32, {
      for (int type = 0; type < type_flags.size(); type++) {
        if (type == TypeFlag::FLOAT32 || type_flags[type] == 0) {
          continue;
        }
        TYPE_SWITCH(type, DstType, {
          for (int i = 0; i < inputs_.size(); i++)
            inputs_[i]->PrecsCpuCvt<FP32, DstType>();
          for (int i = 0; i < outputs_.size(); i++)
            outputs_[i]->PrecsCpuCvt<FP32, DstType>();
        });
      }
    });
  }

  // Run with the binding arrays.
  void Run(const OpRunMode mode) {
    if (mode == OpRunMode::ON_HOST) {
      op_->RunOnHost(inputs_, outputs_);
    }
    else {
      op_->RunOnDevice(inputs_, outputs_);
    }
  }

private:
  Device device_;
  OpAssistor *op_assistor_;
  Operator *op_;

  std::vector< Array4D* > inputs_;
  std::vector< Array4D* > outputs_;
};

} // cux.
#endif //CUX_EXECUTOR_H_
