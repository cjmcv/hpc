/*!
* \brief gemm: C(M, N) = A(M, K) * B(K, N). -> (height, width)
*/

#ifndef CUX_GEMM_HPP_
#define CUX_GEMM_HPP_

#include "util/util.h"
#include "operator.h"

namespace cux {

struct GEMMOpParam : public OpParam {
  float alpha_ = 1.0;
  // TODO: Use beta in kernels.
  float beta_ = 0.0;
};

class GEMM : public Operator {
public:
  GEMM() {}

  void Help() const;

  int SetIoParams(const std::vector< CuxData<float>* > &input,
                  const std::vector< CuxData<float>* > &output,
                  const OpParam *params);
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