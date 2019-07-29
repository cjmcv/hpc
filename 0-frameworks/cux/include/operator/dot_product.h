/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_H_
#define CUX_DOT_PRODUCT_H_

#include "util/util.h"
#include "operator.h"

namespace cux {

template <typename Dtype>
class VectorDotProduct : public Operator<Dtype> {
public:
  VectorDotProduct() :Operator(2, 4) {
    config_1d_.resize(gpu_kernel_cnt_);
  }
  static Operator *VectorDotProduct::Creator(std::string &params_str);
  
  void Help() const;
  int SetIoData(const std::vector< Array4D* > &input,
                const std::vector< Array4D* > &output);
  void RunOnHost();
  void RunOnDevice();

private:
  std::string &GetHostKernelsInfo(int kernel_id);
  std::string &GetDeviceKernelsInfo(int kernel_id);

  void VectorDotProductHost(int kernel_id, int len,
                            const Dtype *vec_a, const Dtype *vec_b,
                            Dtype *res);
  void VectorDotProductDevice(int kernel_id, int len,
                              const Dtype *vec_a, const Dtype *vec_b,
                              Dtype *res);

  void PrepareLaunchConfig(int len);

private:
  Array4D *in_a_;
  Array4D *in_b_;
  Array4D *out_;

  std::vector<Config1D> config_1d_;
};
} // cux.

#endif //CUX_DOT_PRODUCT_H_
