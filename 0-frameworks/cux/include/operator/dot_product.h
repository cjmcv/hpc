/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_H_
#define CUX_DOT_PRODUCT_H_

#include <functional>

#include "util/util.h"
#include "operator.h"

namespace cux {

struct DotProductCpuKernel :OpKernel {
  std::function<void(int len, const void *vec_a, const void *vec_b, void *res)> func;
};

struct DotProductGpuKernel :OpKernel { 
  std::function<Config1D(int len)> get_config;
  std::function<void(Config1D, int len, const void *vec_a, const void *vec_b, void *res)> func;
  
  void *kernel_address;
};


class VectorDotProduct : public Operator {
public:
  VectorDotProduct(Device *device) :Operator(device) {
    CpuKernelsSetup();
    GpuKernelsSetup();
    gpu_kernel_occupancys_.resize(gpu_kernels_.size());
    gpu_kernel_active_blocks_.resize(gpu_kernels_.size());
  }
  ~VectorDotProduct() {
    for (int i = 0; i < cpu_kernels_.size(); i++) {
      delete cpu_kernels_[i];
    }
    for (int i = 0; i < gpu_kernels_.size(); i++) {
      delete gpu_kernels_[i];
    }
  }
  static Operator *VectorDotProduct::Creator(Device *device, std::string &params_str);
  
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

  std::vector<DotProductCpuKernel *> cpu_kernels_;
  std::vector<DotProductGpuKernel *> gpu_kernels_;
};
} // cux.

#endif //CUX_DOT_PRODUCT_H_
