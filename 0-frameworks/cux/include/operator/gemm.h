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

class Gemm : public Operator {
public:
  Gemm(OpAssistor *assistor, GemmKernelParam &params) :Operator(assistor) {
    params_ = params;
    CpuKernelsSetup();
    GpuKernelsSetup();
    ResetKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
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
  void AddPlugin(KernelInterface *kernel_if, OpRunMode mode);

  void RunOnHost();
  void RunOnDevice();

private:
  void CpuKernelsSetup();
  void GpuKernelsSetup();

private:
  Array4D *A_;
  Array4D *B_;
  Array4D *C_;

  GemmKernelParam params_;

  std::vector<GemmCpuKernelIF *> cpu_kernels_;
  std::vector<GemmGpuKernelIF *> gpu_kernels_;
};
} // cux.

#endif //CUX_GEMM_H_
