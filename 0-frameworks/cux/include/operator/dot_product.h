/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_H_
#define CUX_DOT_PRODUCT_H_

#include <functional>

#include "util/util.h"
#include "operator.h"

namespace cux {

class VectorDotProduct : public Operator {
public:
  VectorDotProduct(OpAssistor *assistor) :Operator(assistor) {
    CpuKernelsSetup();
    GpuKernelsSetup();
    gpu_kernel_occupancys_.resize(gpu_kernels_.size());
    gpu_kernel_active_blocks_.resize(gpu_kernels_.size());
    cpu_timer_record_.resize(cpu_kernels_.size());
    gpu_timer_record_.resize(gpu_kernels_.size());
  }
  ~VectorDotProduct() {
    for (int i = 0; i < cpu_kernels_.size(); i++) {
      delete cpu_kernels_[i];
    }
    for (int i = 0; i < gpu_kernels_.size(); i++) {
      delete gpu_kernels_[i];
    }
  }
  static Operator *VectorDotProduct::Creator(OpAssistor *op_assistor, std::string &params_str);
  
  void Help() const;
  int SetIoData(const std::vector< Array4D* > &input,
                const std::vector< Array4D* > &output);
  void RunOnHost();
  void RunOnDevice();

private:
  void CpuKernelsSetup();
  void GpuKernelsSetup();

private:
  Array4D *in_a_; // Don not hold, needn't be released here.
  Array4D *in_b_;
  Array4D *out_;

  std::vector<DotProductCpuKernelIF *> cpu_kernels_;
  std::vector<DotProductGpuKernelIF *> gpu_kernels_;
};
} // cux.

#endif //CUX_DOT_PRODUCT_H_
