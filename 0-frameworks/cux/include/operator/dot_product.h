/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_H_
#define CUX_DOT_H_

#include <functional>

#include "util/util.h"
#include "operator.h"

namespace cux {

class Dot : public Operator {
public:
  Dot(OpAssistor *assistor) :Operator(assistor) {
    CpuKernelsSetup();
    GpuKernelsSetup();
    ResetByKernelNum(cpu_kernels_.size(), gpu_kernels_.size());
  }
  ~Dot() {
    for (int i = 0; i < cpu_kernels_.size(); i++) {
      delete cpu_kernels_[i];
    }
    for (int i = 0; i < gpu_kernels_.size(); i++) {
      delete gpu_kernels_[i];
    }
  }
  static Operator *Dot::Creator(OpAssistor *op_assistor, std::string &params_str);
  
  void Help() const;
  void AddPlugin(KernelInterface *kernel_if, OpRunMode mode);
  void ExtractDataTypes(std::vector<int>& type_flags);

  void RunOnHost(const std::vector< Array4D* > &input,
                 const std::vector< Array4D* > &output);
  void RunOnDevice(const std::vector< Array4D* > &input,
                   const std::vector< Array4D* > &output);

private:
  void IoCheckAndSet(const std::vector< Array4D* > &input,
                     const std::vector< Array4D* > &output);
  void CpuKernelsSetup();
  void GpuKernelsSetup();

private:
  Array4D *in_a_;
  Array4D *in_b_;
  Array4D *out_;

  std::vector<DotCpuKernelIF *> cpu_kernels_;
  std::vector<DotGpuKernelIF *> gpu_kernels_;
};
} // cux.

#endif //CUX_DOT_H_
