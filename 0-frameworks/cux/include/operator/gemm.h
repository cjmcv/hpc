/*!
* \brief gemm: C(M, N) = A(M, K) * B(K, N). -> (height, width)
*/

#ifndef CUX_GEMM_H_
#define CUX_GEMM_H_

#include <functional>

#include "util/util.h"
#include "operator.h"

namespace cux {

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
struct GemmCpuKernel :OpKernel {
  GemmKernelParam params;

  std::function<void(const int M, const int N,
                    const int K, const float alpha,
                    const void *A, const int lda,
                    const void *B, const int ldb,
                    const float beta,
                    void *C, const int ldc)> func;
};

struct GemmGpuKernel :OpKernel {  
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

class Gemm : public Operator {
public:
  Gemm(OpAssistor *assistor, GemmKernelParam &params) :Operator(assistor) {
    CpuKernelsSetup(params);
    GpuKernelsSetup(params);
    gpu_kernel_occupancys_.resize(gpu_kernels_.size());
    gpu_kernel_active_blocks_.resize(gpu_kernels_.size());
  }
  ~Gemm() {
    for (int i = 0; i < cpu_kernels_.size(); i++) {
      delete cpu_kernels_[i];
    }
    for (int i = 0; i < gpu_kernels_.size(); i++) {
      delete gpu_kernels_[i];
    }
  }
  static Operator *Gemm::Creator(OpAssistor *assistor, std::string &params_str);

  void Help() const;
  int SetIoData(const std::vector< Array4D* > &input,
                const std::vector< Array4D* > &output);
  void RunOnHost();
  void RunOnDevice();

private:
  void CpuKernelsSetup(GemmKernelParam &params);
  void GpuKernelsSetup(GemmKernelParam &params);

private:
  Array4D *A_;
  Array4D *B_;
  Array4D *C_;

  std::vector<GemmCpuKernel *> cpu_kernels_;
  std::vector<GemmGpuKernel *> gpu_kernels_;
};
} // cux.

#endif //CUX_GEMM_H_
