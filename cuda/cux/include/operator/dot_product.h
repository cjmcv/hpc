/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_H_
#define CUX_DOT_PRODUCT_H_

#include "util/util.h"
#include "operator.h"

namespace cux {

class VectorDotProduct : public Operator {
public:
  VectorDotProduct() :Operator(2, 4) {
    config_1d_.resize(gpu_kernel_cnt_);
  }
  static Operator *VectorDotProduct::Creator(std::string &params_str);
  
  void Help() const;
  int SetIoData(const std::vector< CuxData<float>* > &input,
                const std::vector< CuxData<float>* > &output);
  void RunOnHost();
  void RunOnDevice();

  std::string &GetHostKernelsInfo(int kernel_id);
  std::string &GetDeviceKernelsInfo(int kernel_id);

  void VectorDotProductHost(int kernel_id, int len,
                            const float *vec_a, const float *vec_b,
                            float *res);
  void VectorDotProductDevice(int kernel_id, int len,
                              const float *vec_a, const float *vec_b,
                              float *res);

  void PrepareLaunchConfig(int len);

private:
  CuxData<float> *in_a_;
  CuxData<float> *in_b_;
  CuxData<float> *out_;

  std::vector<Config1D> config_1d_;
};
} // cux.

#endif //CUX_DOT_PRODUCT_H_
