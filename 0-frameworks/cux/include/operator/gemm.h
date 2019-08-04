/*!
* \brief gemm: C(M, N) = A(M, K) * B(K, N). -> (height, width)
*/

#ifndef CUX_GEMM_H_
#define CUX_GEMM_H_

#include <functional>

#include "util/util.h"
#include "operator.h"

namespace cux {

class Gemm : public Operator {
public:
  Gemm(OpAssistor *assistor, GemmKernelParam &params) :Operator(assistor) {
    CpuKernelsSetup(params);
    GpuKernelsSetup(params);
    gpu_kernel_occupancys_.resize(gpu_kernels_.size());
    gpu_kernel_active_blocks_.resize(gpu_kernels_.size());
    cpu_timer_record_.resize(cpu_kernels_.size());
    gpu_timer_record_.resize(gpu_kernels_.size());
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

  std::vector<GemmCpuKernelIF *> cpu_kernels_;
  std::vector<GemmGpuKernelIF *> gpu_kernels_;
};
} // cux.

#endif //CUX_GEMM_H_
