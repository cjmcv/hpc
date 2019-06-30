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
  float beta_ = 0.0;

  GEMMOpParam& operator=(const GEMMOpParam& in) {
    alpha_ = in.alpha_;
    beta_ = in.beta_;
    return *this;
  }
};

class GEMM : public Operator {
public:
  GEMM(GEMMOpParam &params) :params_(params), Operator(2, 3) {}
  static Operator *GEMM::Creator(std::string &params_str);

  void Help() const;
  int SetIoData(const std::vector< CuxData<float>* > &input,
                const std::vector< CuxData<float>* > &output);
  void RunOnHost();
  void RunOnDevice();

  void GEMMHost(const int kernel_id, 
                const int M, const int N,
                const int K, const float ALPHA,
                const float *A, const int lda,
                const float *B, const int ldb,
                const float beta,
                float *C, const int ldc);

  void GEMMDevice(const int kernel_id, 
                  const dim3 &blocks_per_grid, 
                  const dim3 &threads_per_block,
                  const int M, const int N, 
                  const int K, const float ALPHA,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  const float beta,
                  float *C, const int ldc);

private:
  CuxData<float> *A_;
  CuxData<float> *B_;
  CuxData<float> *C_;

  GEMMOpParam params_;
};
} // cux.

#endif //CUX_GEMM_HPP_