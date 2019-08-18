#include "operator/nrm2.h"

namespace cux {

extern __global__ void DotDeviceV2(const int len, const float *vec_a, const float *vec_b, float *res);

__global__ void Sqrt(const int n, float *x) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < n;
    i += blockDim.x * gridDim.x) {
    x[i] = __fsqrt_rn(x[i]);
  }
}

// Fuse DotDeviceV2 and Sqrt.
void Nrm2DeviceV0(Config1D config, int n, const float *x, float *result) {
  DotDeviceV2 << <config.blocks_per_grid,
    config.threads_per_block,
    config.shared_memory_size >> >
    (n, x, x, result);

  Sqrt << <1,1,1>> >(1, result);
}

void Nrm2::GpuKernelsSetup() {
  gpu_kernels_.clear();
  // Kernel v0
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      config.threads_per_block = 1024;
      config.blocks_per_grid = (len + config.threads_per_block - 1) / config.threads_per_block;
      config.shared_memory_size = config.threads_per_block * sizeof(float);
      return config;
    };
    auto func = [&](Config1D config, int n, const void *x, void *result) -> void {
      Nrm2DeviceV0(config, n, (float *)x, (float *)result);
    };

    Nrm2GpuKernelIF *kernel = new Nrm2GpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Shared memory / Loop unrolling";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = DotDeviceV2;

    gpu_kernels_.push_back(kernel);
  }
  // Kernel v1.
  {
    auto get_config = [&](int len) -> Config1D {
      Config1D config;
      return config;
    };
    auto func = [&](Config1D config, int n, const void *x, void *result) -> void {
      // CUBLAS_POINTER_MODE_DEVICE: Return data on device -> res is a pointer for device.
      // CUBLAS_POINTER_MODE_HOST: On host.
      CUBLAS_CHECK(cublasSetPointerMode(assistor_->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));
      CUBLAS_CHECK(cublasSnrm2(assistor_->cublas_handle(), n, (float *)x, 1, (float *)result));
    };

    Nrm2GpuKernelIF *kernel = new Nrm2GpuKernelIF();
    kernel->type_flag = TypeFlag::FLOAT32;
    kernel->describe_info = "Cublas";
    kernel->get_config = get_config;
    kernel->func = func;
    kernel->config_kernel = nullptr;

    gpu_kernels_.push_back(kernel);
  }
}

}