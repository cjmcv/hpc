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

template <typename Dtype>
std::string &VectorDotProduct<Dtype>::GetHostKernelsInfo(int kernel_id) {
  static std::string info[2] = { "Normal", "SIMD" };
  if (kernel_id < 0 || kernel_id >= 2) {
    CUXLOG_ERR("GetDeviceKernelsInfo -> Device Kernel id (%d) not found.", kernel_id);
  }
  return info[kernel_id];
}

template <typename Dtype>
void VectorDotProduct<Dtype>::VectorDotProductHost(int kernel_id, int len,
                                                   const Dtype *vec_a, const Dtype *vec_b,
                                                   Dtype *res) {
  switch (kernel_id) {
  case 0:
    VectorDotProductHostV0(len, vec_a, vec_b, res);
    break;
  case 1:
    VectorDotProductHostV1(len, vec_a, vec_b, res);
    break;
  default:
    CUXLOG_ERR("Host Kernel id (%d) not found.", kernel_id);
  }
}

template <typename Dtype>
void VectorDotProduct<Dtype>::Help() const {
  CUXLOG_COUT("***************** Op Helper ********************");
  CUXLOG_COUT("* Name: Vector Dot Product.");
  CUXLOG_COUT("* Function: sum += a[i] * b[i]");
  CUXLOG_COUT("* Inputs:  [Two] Array4D with one vector each. ");
  CUXLOG_COUT("* Outputs: [One] Array4D with one element.");
  CUXLOG_COUT("* Params:  [None].");
  CUXLOG_COUT("**************************************************");
}

template <typename Dtype>
Operator<Dtype> *VectorDotProduct<Dtype>::Creator(std::string &params_str) {
  return new VectorDotProduct();
}

template <typename Dtype>
int VectorDotProduct<Dtype>::SetIoData(const std::vector< Array4D* > &input,
                                       const std::vector< Array4D* > &output) {
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
template <typename Dtype>
void VectorDotProduct<Dtype>::RunOnHost() {
  CpuTimer cpu_timer;

  // Warp.
  const float *vec_a = in_a_->GetCpuData<float>(PUSH_IF_EMPTY);
  const float *vec_b = in_b_->GetCpuData<float>(PUSH_IF_EMPTY);
  const int len = in_a_->num_element();
  float *result = out_->GetCpuData<float>(NO_PUSH);
  
  // Run.
  cpu_time_kernel_record_.clear();
  for (int ki = 0; ki < cpu_kernel_cnt_; ki++) {
    *result = 0;

    cpu_timer.Start();
    VectorDotProductHost(ki, len, vec_a, vec_b, result);
    cpu_timer.Stop();  
    cpu_time_kernel_record_.push_back(cpu_timer.MilliSeconds());

    checker_.CheckArray(out_->GetCpuData<float>(PUSH), out_->num_element(), ki);
  }

  CUXLOG_COUT("result: %f.", *out_->GetCpuData<float>(NO_PUSH));
}

//////////////////
// cuda version.
template <typename Dtype>
std::string &VectorDotProduct<Dtype>::GetDeviceKernelsInfo(int kernel_id) {
  static std::string info[4] = { "Shared memory",
    "Shared memory / Loop unrolling",
    "Shared memory / Loop unrolling",
    "Cublas" };
  if (kernel_id < 0 || kernel_id >= 4) {
    CUXLOG_ERR("GetDeviceKernelsInfo -> Device Kernel id (%d) not found.", kernel_id);
  }
  return info[kernel_id];
}

template <typename Dtype>
void VectorDotProduct<Dtype>::RunOnDevice() {
  // Time recorder.
  GpuTimer gpu_timer;

  // Input.
  gpu_timer.Start();
  const float *vec_a = in_a_->GetGpuData<float>(PUSH_IF_EMPTY);
  const float *vec_b = in_b_->GetGpuData<float>(PUSH_IF_EMPTY);
  const int len = in_a_->num_element();
  float *result = out_->GetGpuData<float>(NO_PUSH);
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
    cudaMemset(result, 0, sizeof(float));

    gpu_timer.Start();
    VectorDotProductDevice(ki, len, vec_a, vec_b, result);
    gpu_timer.Stop();
    gpu_time_kernel_record_.push_back(gpu_timer.MilliSeconds());

    // Output, Only record the first time.
    if (ki == 0) {
      gpu_timer.Start();
      out_->GetCpuData<float>(PUSH);
      gpu_timer.Stop();
      gpu_time_out_record_ = gpu_timer.MilliSeconds();
    }
    checker_.CheckArray(out_->GetCpuData<float>(PUSH), out_->num_element(), ki);
  }
}
INSTANTIATE_CLASS(VectorDotProduct);
}