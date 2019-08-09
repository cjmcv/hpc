#include "operator/dot_product.h"

namespace cux {

// CPU version 0: 11 ms
// Normal version in cpu as a reference
void VectorDotProductHostV0(int len, const float *vec_a, const float *vec_b, float *res) {
  float temp = 0;
  for (int i = 0; i < len; i++) {
    temp += vec_a[i] * vec_b[i];
  }
  *res = temp;
}

// CPU version 1: 3 ms
// SIMD.
void VectorDotProductHostV1(int len, const float *vec_a, const float *vec_b, float *res) {
  float result = 0;

  // Using 8 as the base number to call simd.
  if (len > 8) {
    __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < len - 7; i += 8) {
      __m128 a0 = _mm_loadu_ps(vec_a + i);
      __m128 a1 = _mm_loadu_ps(vec_a + i + 4);

      __m128 b0 = _mm_loadu_ps(vec_b + i);
      __m128 b1 = _mm_loadu_ps(vec_b + i + 4);

      sum = _mm_add_ps(sum, _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1)));
    }
    result = sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
  }

  // Calculate the remaining part.
  for (int i = len / 8 * 8; i < len; i++) {
    result += vec_a[i] * vec_b[i];
  }
  *res = result;
}

//////////////////////
void VectorDotProduct::CpuKernelsSetup() {
  cpu_kernels_.clear();
  // Kernel v0.
  {
    auto func = [&](int len, const void *vec_a, const void *vec_b, void *res) -> void {
      VectorDotProductHostV0(len, (float *)vec_a, (float *)vec_b, (float *)res);
    };

    DotProductCpuKernelIF *kernel = new DotProductCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "Normal";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v1.
  {
    auto func = [&](int len, const void *vec_a, const void *vec_b, void *res) -> void {
      VectorDotProductHostV1(len, (float *)vec_a, (float *)vec_b, (float *)res);
    };

    DotProductCpuKernelIF *kernel = new DotProductCpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "SIMD";

    cpu_kernels_.push_back(kernel);
  }
}

//////////////////////
Operator *VectorDotProduct::Creator(OpAssistor *op_assistor, std::string &params_str) {
  return new VectorDotProduct(op_assistor);
}

void VectorDotProduct::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: Vector Dot Product.");
  CUXLOG_COUT("* Function: sum += a[i] * b[i]");
  CUXLOG_COUT("* Inputs:  [Two] Array4D with one vector each. ");
  CUXLOG_COUT("* Outputs: [One] Array4D with one element.");
  CUXLOG_COUT("* Params:  [None].");
  CUXLOG_COUT("**************************************************");
}

void VectorDotProduct::IoCheckAndSet(const std::vector< Array4D* > &input,
                                     const std::vector< Array4D* > &output) {
  // Check the dimensions.
  if (input.size() != 2 || output.size() != 1) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }
  if (input[0]->shape()[Shape::WIDTH] != input[1]->shape()[Shape::WIDTH] ||
    output[0]->shape()[Shape::WIDTH] != 1) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }

  in_a_ = input[0];
  in_b_ = input[1];
  out_ = output[0];
}

void VectorDotProduct::AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST)
    cpu_kernels_.push_back((DotProductCpuKernelIF*)kernel_if);
  else
    gpu_kernels_.push_back((DotProductGpuKernelIF*)kernel_if);

  ResetByKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
}

void VectorDotProduct::ExtractDataTypes(std::vector<int>& type_flags) {
  type_flags.clear();
  type_flags.resize(TYPES_NUM);
  for (int i = 0; i < type_flags.size(); i++) {
    type_flags[i] = 0;
  }
  for (int i = 0; i < cpu_kernels_.size(); i++) {
    type_flags[cpu_kernels_[i]->type_flag] = 1;
  }
  for (int i = 0; i < gpu_kernels_.size(); i++) {
    type_flags[gpu_kernels_[i]->type_flag] = 1;
  }
}
////////////////////////////////////////////////
// cpu version.
void VectorDotProduct::RunOnHost(const std::vector< Array4D* > &input,
                                 const std::vector< Array4D* > &output) {
  CUXLOG_COUT("VectorDotProduct -> CPU: ");
  IoCheckAndSet(input, output);

  const int len = in_a_->num_element();
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    DotProductCpuKernelIF *kernel = cpu_kernels_[ki];

    // Input.
    const void *vec_a, *vec_b;
    void *result;
    cpu_timer_record_[ki].input = GET_TIME_DIFF(cpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        vec_a = in_a_->GetCpuData<T>(PUSH_IF_EMPTY);
        vec_b = in_b_->GetCpuData<T>(PUSH_IF_EMPTY);
        result = out_->GetCpuData<T>(NO_PUSH);
      };);
    );
    // Run.
    cpu_timer_record_[ki].run = GET_TIME_DIFF(cpu_timer_,
      kernel->func(len, vec_a, vec_b, result);
    );
    // Output.
    TYPE_SWITCH(kernel->type_flag, T,
      assistor_->checker()->CheckArray(out_->GetCpuData<T>(PUSH), out_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_HOST, ki, cpu_kernels_[ki]);
  }
}

//////////////////
// cuda version.
void VectorDotProduct::RunOnDevice(const std::vector< Array4D* > &input,
                                   const std::vector< Array4D* > &output) {
  CUXLOG_COUT("VectorDotProduct -> GPU:");
  IoCheckAndSet(input, output);

  const int len = in_a_->num_element();
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    DotProductGpuKernelIF *kernel = gpu_kernels_[ki];
    Config1D config = kernel->get_config(len);

    // Record the occupancy for profiling.
    QueryPotentialOccupancy(kernel->kernel_address, ki,
                            config.threads_per_block, 
                            config.shared_memory_size);
    // Input.
    const void *vec_a, *vec_b;
    void *result;
    gpu_timer_record_[ki].input = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        vec_a = in_a_->GetGpuData<T>(PUSH_IF_EMPTY);
        vec_b = in_b_->GetGpuData<T>(PUSH_IF_EMPTY);
        result = out_->GetGpuData<T>(NO_PUSH); 
      };);
    );
    // Warm up.
    gpu_timer_record_[ki].warnup = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, len, vec_a, vec_b, result);
    );
    // Run.
    TYPE_SWITCH(kernel->type_flag, T, cudaMemset(result, 0, sizeof(T)););
    gpu_timer_record_[ki].run = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, len, vec_a, vec_b, result);
    );
    // Output.
    gpu_timer_record_[ki].output = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, { out_->GetCpuData<T>(PUSH); });
    );
    // Check.
    TYPE_SWITCH(kernel->type_flag, T,
      assistor_->checker()->CheckArray(out_->GetCpuData<T>(PUSH), out_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_DEVICE, ki, gpu_kernels_[ki]);
  }
}

}