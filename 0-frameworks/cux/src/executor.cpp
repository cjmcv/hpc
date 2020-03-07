/*!
* \brief Executor.
*/

#include "executor.h"

namespace cux {

// Executor.
void Executor::Initialize(const int dev_id) {
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

void Executor::Clear() {
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
void Executor::SelectOp(std::string op_name, std::string params) {
  op_ = OpFactory::GetInstance().CreateOpByType(op_name, op_assistor_, params);
}

// Add user-defined kernel to a existing op.
void Executor::AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
  if (op_ == nullptr) {
    CUXLOG_ERR("The operator has not been selected yet. Please select an operator first.");
  }
  op_->AddPlugin(kernel_if, mode);
}

// Bind and fill input and output data for this executor.
void Executor::BindAndFill(const std::vector< Array4D* > &inputs,
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
void Executor::Run(const OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST) {
    op_->RunOnHost(inputs_, outputs_);
  }
  else {
    op_->RunOnDevice(inputs_, outputs_);
  }
}

} // cux.
