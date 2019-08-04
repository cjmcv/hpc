/*!
* \brief Operator.
*/

#ifndef CUX_KERNEL_INTERFACE_H_
#define CUX_KERNEL_INTERFACE_H_

#include <functional>
#include "util/util.h"

namespace cux {

struct KernelInterface {
  TypeFlag type_flag;
  std::string describe_info;
};

////////////////////////
////////////////////////
// VectorDotProduct
struct DotProductCpuKernelIF :KernelInterface {
  std::function<void(int len, const void *vec_a, const void *vec_b, void *res)> func;
};

struct DotProductGpuKernelIF :KernelInterface {
  std::function<Config1D(int len)> get_config;
  std::function<void(Config1D, int len, const void *vec_a, const void *vec_b, void *res)> func;

  void *kernel_address;
};

////////////////////////
// Gemm
struct GemmKernelParam {
  float alpha = 1.0;
  float beta = 0.0;

  GemmKernelParam& operator=(const GemmKernelParam& in) {
    alpha = in.alpha;
    beta = in.beta;
    return *this;
  }
};

//// Lambda functions cannot be assigned in this case.
//// So use std::function instead.
//typedef void (*KernelFuncPtr)();
struct GemmCpuKernelIF :KernelInterface {
  GemmKernelParam params;

  std::function<void(const int M, const int N,
    const int K, const float alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc)> func;
};

struct GemmGpuKernelIF :KernelInterface {
  GemmKernelParam params;

  std::function<Config2D(int M, int N)> get_config;
  std::function<void(Config2D,
    const int M, const int N,
    const int K, const float alpha,
    const void *A, const int lda,
    const void *B, const int ldb,
    const float beta,
    void *C, const int ldc)> func;

  void *kernel_address;
};


} // cux.

#endif //CUX_KERNEL_INTERFACE_H_
