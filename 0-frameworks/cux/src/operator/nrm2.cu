#include "operator/nrm2.h"

namespace cux {

void Nrm2::GpuKernelsSetup() {
  gpu_kernels_.clear();
  // Kernel v0.
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
    kernel->kernel_address = nullptr;

    gpu_kernels_.push_back(kernel);
  }
}

}