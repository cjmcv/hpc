/*!
* \brief Operator.
*/

#ifndef CUX_KERNEL_INTERFACE_H_
#define CUX_KERNEL_INTERFACE_H_

#include <functional>
#include "util/util.h"
#include "util/launch_config.h"

namespace cux {

////////////////////////
// Base
struct KernelInterface {
  TypeFlag type_flag;
  std::string describe_info;
};

////////////////////////
// Dot
//   Lambda functions cannot be assigned in this case.
// So use std::function instead.
// typedef void (*KernelFuncPtr)();
struct DotCpuKernelIF :KernelInterface {
  std::function<void(int len, const void *vec_a, const void *vec_b, void *res)> func;
};

struct DotGpuKernelIF :KernelInterface {
  std::function<Config1D(int len)> get_config;
  std::function<void(Config1D, int len, const void *vec_a, const void *vec_b, void *res)> func;
  // The kernel to be configured.
  void *config_kernel;
};

////////////////////////
// Nrm2
struct Nrm2CpuKernelIF :KernelInterface {
  std::function<void(int n, const void *x, void *result)> func;
};

struct Nrm2GpuKernelIF :KernelInterface {
  std::function<Config1D(int len)> get_config;
  std::function<void(Config1D, int n, const void *x, void *result)> func;

  void *config_kernel;
};

////////////////////////
// Gemm
struct GemmCpuKernelIF :KernelInterface {
  std::function<void(const int M, const int N,
    const int K, const float alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc)> func;
};

struct GemmGpuKernelIF :KernelInterface {
  std::function<Config2D(int M, int N)> get_config;
  std::function<void(Config2D,
    const int M, const int N,
    const int K, const float alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc)> func;

  void *config_kernel;
};

} // cux.

#endif //CUX_KERNEL_INTERFACE_H_
