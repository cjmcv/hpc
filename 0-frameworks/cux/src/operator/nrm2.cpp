#include "operator/nrm2.h"

namespace cux {

// Kernel V0
// Normal version in cpu as a reference
void Nrm2HostV0(int n, float *x, float *result) {
  if (n <= 0) { *result = 0.0; return; };
  if (n == 1) { *result = std::abs(x[0]); return; };

  float s = 0.0;
  for (int i = 0; i < n; i++) {
    float v = std::abs(x[i]);
    s += v * v;
  }
  *result = std::sqrt(s);
}

void Nrm2HostV1(int n, float *x, float *result) {
  if (n <= 0) { *result = 0.0; return; };
  if (n == 1) { *result = std::abs(x[0]); return; };
  
  float acc = 0.0;
  float absx[8] = {0};
  __m256 acc8 = _mm256_setzero_ps();
  for (int i = 0; i < n - 7; i += 8) {
    absx[0] = std::abs(x[i]);
    absx[1] = std::abs(x[i + 1]);
    absx[2] = std::abs(x[i + 2]);
    absx[3] = std::abs(x[i + 3]);
    absx[4] = std::abs(x[i + 4]);
    absx[5] = std::abs(x[i + 5]);
    absx[6] = std::abs(x[i + 6]);
    absx[7] = std::abs(x[i + 7]);

    __m256 absx8 = _mm256_loadu_ps(absx);
    acc8 = _mm256_fmadd_ps(absx8, absx8, acc8);
  }
  _mm256_storeu_ps(absx, acc8);
  for (int i = 0; i < 8; i++) {
    acc += absx[i];
  }
  *result = std::sqrt(acc);
}

//////////////////////
void Nrm2::CpuKernelsSetup() {
  cpu_kernels_.clear();
  // Kernel v0.
  {
    auto func = [&](int n, const void *x, void *result) -> void {
      Nrm2HostV0(n, (float *)x, (float *)result);
    };

    Nrm2CpuKernelIF *kernel = new Nrm2CpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "Normal";

    cpu_kernels_.push_back(kernel);
  }
  // Kernel v1.
  {
    auto func = [&](int n, const void *x, void *result) -> void {
      Nrm2HostV1(n, (float *)x, (float *)result);
    };

    Nrm2CpuKernelIF *kernel = new Nrm2CpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->func = func;
    kernel->describe_info = "SIMD";

    cpu_kernels_.push_back(kernel);
  }
}

//////////////////////
Operator *Nrm2::Creator(OpAssistor *op_assistor, std::string &params_str) {
  return new Nrm2(op_assistor);
}

void Nrm2::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: Euclidean Norm.");
  CUXLOG_COUT("* Function: ");
  CUXLOG_COUT("* Inputs:  [One] Array4D with one vector. ");
  CUXLOG_COUT("* Outputs: [One] Array4D with one element.");
  CUXLOG_COUT("* Params:  [None].");
  CUXLOG_COUT("**************************************************");
}

void Nrm2::IoCheckAndSet(const std::vector< Array4D* > &input,
                         const std::vector< Array4D* > &output) {
  // Check the dimensions.
  if (input.size() != 1 || output.size() != 1) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }
  if (output[0]->shape()[Shape::WIDTH] != 1) {
    Help();
    CUXLOG_ERR("The dimensions of the data do not match.");
  }

  in_ = input[0];
  out_ = output[0];
}

void Nrm2::AddPlugin(KernelInterface *kernel_if, OpRunMode mode) {
  if (mode == OpRunMode::ON_HOST)
    cpu_kernels_.push_back((Nrm2CpuKernelIF*)kernel_if);
  else
    gpu_kernels_.push_back((Nrm2GpuKernelIF*)kernel_if);

  ResetByKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
}

void Nrm2::ExtractDataTypes(std::vector<int>& type_flags) {
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
void Nrm2::RunOnHost(const std::vector< Array4D* > &input,
                     const std::vector< Array4D* > &output) {
  CUXLOG_COUT("Nrm2 -> CPU: ");
  IoCheckAndSet(input, output);

  const int len = in_->num_element();
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    Nrm2CpuKernelIF *kernel = cpu_kernels_[ki];

    // Input.
    const void *x;
    void *result;
    cpu_timer_record_[ki].input = GET_TIME_DIFF(cpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        x = in_->GetCpuData<T>(PUSH_IF_EMPTY);
        result = out_->GetCpuData<T>(NO_PUSH);
      };);
    );
    // Run.
    cpu_timer_record_[ki].run = GET_TIME_DIFF(cpu_timer_,
      kernel->func(len, x, result);
    );
    // Output.
    TYPE_SWITCH(kernel->type_flag, T,
      assistor_->checker()->CheckArray(out_->GetCpuData<T>(PUSH), 
        out_->num_element(), 1.0 / in_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < cpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_HOST, ki, cpu_kernels_[ki]);
  }
}

//////////////////
// cuda version.
void Nrm2::RunOnDevice(const std::vector< Array4D* > &input,
                       const std::vector< Array4D* > &output) {
  CUXLOG_COUT("Nrm2 -> GPU:");
  IoCheckAndSet(input, output);

  const int len = in_->num_element();
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    Nrm2GpuKernelIF *kernel = gpu_kernels_[ki];
    Config1D config = kernel->get_config(len);

    // Record the occupancy for profiling.
    QueryPotentialOccupancy(kernel->kernel_address, ki,
                            config.threads_per_block, 
                            config.shared_memory_size);
    // Input.
    const void *x;
    void *result;
    gpu_timer_record_[ki].input = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, {
        x = in_->GetGpuData<T>(PUSH_IF_EMPTY);
        result = out_->GetGpuData<T>(NO_PUSH); 
      };);
    );
    // Warm up.
    gpu_timer_record_[ki].warnup = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, len, x, result);
    );
    // Run.
    TYPE_SWITCH(kernel->type_flag, T, cudaMemset(result, 0, sizeof(T)););
    gpu_timer_record_[ki].run = GET_TIME_DIFF(gpu_timer_,
      kernel->func(config, len, x, result);
    );
    // Output.
    gpu_timer_record_[ki].output = GET_TIME_DIFF(gpu_timer_,
      TYPE_SWITCH(kernel->type_flag, T, { out_->GetCpuData<T>(PUSH); });
    );
    // Check.
    TYPE_SWITCH(kernel->type_flag, T,
      assistor_->checker()->CheckArray(out_->GetCpuData<T>(PUSH), 
        out_->num_element(), 1.0 / in_->num_element(), ki);
    );
  }
  // Show.
  for (int ki = 0; ki < gpu_kernels_.size(); ki++) {
    PrintRecordedInfo(OpRunMode::ON_DEVICE, ki, gpu_kernels_[ki]);
  }
}

}