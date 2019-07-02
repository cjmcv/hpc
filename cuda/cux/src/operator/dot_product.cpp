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

  CUXLOG_COUT("result: %f.", *out_->GetCpuData(NO_PUSH));
}

}