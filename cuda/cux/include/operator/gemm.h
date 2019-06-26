/*!
* \brief gemm: C(M, N) = A(M, K) * B(K, N). -> (height, width)
*/

#ifndef CUX_GEMM_HPP_
#define CUX_GEMM_HPP_

#include "util/util.h"
#include "operator.h"

namespace cux {

struct GEMMOpParam {
  float alpha_ = 1.0;
  // TODO: Use beta in kernels.
  float beta_ = 0.0;

  GEMMOpParam& operator=(const GEMMOpParam& cls) {
    alpha_ = cls.alpha_;
    beta_ = cls.beta_;
    return *this;
  }
};

class GEMM : public Operator {
public:
  GEMM(GEMMOpParam &params) :params_(params) {}
  static Operator *GEMM::Creator(std::string &params_str);

  void Help() const;
  int SetIoData(const std::vector< CuxData<float>* > &input,
                const std::vector< CuxData<float>* > &output);
  void RunOnHost();
  void RunOnDevice();

private:
  CuxData<float> *A_;
  CuxData<float> *B_;
  CuxData<float> *C_;

  GEMMOpParam params_;
};
} // cux.

#endif //CUX_GEMM_HPP_