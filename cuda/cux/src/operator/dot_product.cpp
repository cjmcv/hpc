#include "operator/dot_product.h"

namespace cux {

////////////////
// CPU Kernel.
void VectorDotProductHostV0(const float *vec_a, const float *vec_b, const int len, float &res) {
  res = 0;
  for (int i = 0; i < len; i++) {
    res += vec_a[i] * vec_b[i];
  }
}

void VectorDotProduct::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: Vector Dot Product.");
  CUXLOG_COUT("* Function: sum += a[i] * b[i]");
  CUXLOG_COUT("* Inputs:  [Two] CuxData with one vector each. ");
  CUXLOG_COUT("* Outputs: [One] CuxData with one element.");
  CUXLOG_COUT("* Params:  [None].");
  CUXLOG_COUT("**************************************************");
}

Operator *VectorDotProduct::Creator(std::string &params_str) {
  return new VectorDotProduct();
}

int VectorDotProduct::SetIoData(const std::vector< CuxData<float>* > &input,
                                const std::vector< CuxData<float>* > &output) {
  // Check the dimensions.
  if (input.size() != 2 || output.size() != 1) {
    CUXLOG_ERR("Error: The dimensions of the input parameters do not match.");
    Help();
    // TODO: Error code.
    return -1;
  }

  in_a_ = input[0];
  in_b_ = input[1];
  out_ = output[0];

  return 0;
}
////////////////////////////////////////////////
// cpp version: 965ms
// Normal version in cpu as a reference
void VectorDotProduct::RunOnHost() {
  CpuTimer cpu_timer;

  // Warp.
  const float *vec_a = in_a_->GetCpuData(PUSH_IF_EMPTY);
  const float *vec_b = in_b_->GetCpuData(PUSH_IF_EMPTY);
  const int len = in_a_->num_element();
  float *result = out_->GetCpuData(NO_PUSH);
  
  // Run.
  cpu_time_kernel_record_.clear();
  cpu_timer.Start();
  for (int i = 0; i < op_params_.loop_cn; i++) {
    *result = 0;
    VectorDotProductHostV0(vec_a, vec_b, len, *result);
  }
  cpu_timer.Stop();
  cpu_time_kernel_record_.push_back(cpu_timer.MilliSeconds() / op_params_.loop_cn);

  checker_.CheckArray(out_->GetCpuData(PUSH), out_->num_element(), 0);
  checker_.CheckArray(out_->GetCpuData(PUSH), out_->num_element(), -1);

  CUXLOG_COUT("result: %f.", *out_->GetCpuData(NO_PUSH));
}

//////////////////
// cuda version.
void VectorDotProduct::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  if (cublas_handle_ == nullptr) {
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
      CUXLOG_ERR("Cannot create Cublas handle. Cublas won't be available.");
      return;
    }
  }

  // Input.
  gpu_timer.Start();
  const float *vec_a = in_a_->GetGpuData(PUSH_IF_EMPTY);
  const float *vec_b = in_b_->GetGpuData(PUSH_IF_EMPTY);
  const int len = in_a_->num_element();
  float *result = out_->GetGpuData(NO_PUSH);
  gpu_timer.Stop();
  gpu_time_in_record_ = gpu_timer.MilliSeconds();

  // Prepare launch config for kernels.
  PrepareLaunchConfig(len);

  // Warm up.
  gpu_timer.Start();
  VectorDotProductDevice(0, len, vec_a, vec_b, result);
  gpu_timer.Stop();
  gpu_time_warnup_record_ = gpu_timer.MilliSeconds();

  // Run.
  gpu_time_kernel_record_.clear();
  for (int ki = 0; ki < gpu_kernel_cnt_; ki++) {
    gpu_timer.Start();
    for (int i = 0; i < op_params_.loop_cn; i++) {
      cudaMemset(result, 0, sizeof(float));
      VectorDotProductDevice(ki, len, vec_a, vec_b, result);
    }
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds() / op_params_.loop_cn);

    // Output, Only record the first time.
    if (ki == 0) {
      gpu_timer.Start();
      out_->GetCpuData(PUSH);
      gpu_timer.Stop();
      gpu_time_out_record_ = gpu_timer.MilliSeconds();
    }
    checker_.CheckArray(out_->GetCpuData(PUSH), out_->num_element(), ki);
  }
}

}