/*!
* \brief gemm: C(M, N) = A(M, K) * B(K, N). -> (height, width)
*/

#ifndef CUX_GEMM_H_
#define CUX_GEMM_H_

#include "util/util.h"
#include "operator.h"

namespace cux {

struct GEMMKernelParam {
  float alpha = 1.0;
  float beta = 0.0;

  GEMMKernelParam& operator=(const GEMMKernelParam& in) {
    alpha = in.alpha;
    beta = in.beta;
    return *this;
  }
};

template <typename Dtype>
class GEMM : public Operator<Dtype> {
public:
  GEMM(GEMMKernelParam &params) :kernel_params_(params), Operator(3, 3) {
    config_2d_.resize(gpu_kernel_cnt_);
  }
  static Operator *GEMM::Creator(std::string &params_str);

  void Help() const;
  int SetIoData(const std::vector< CuxData<Dtype>* > &input,
                const std::vector< CuxData<Dtype>* > &output);
  void RunOnHost();
  void RunOnDevice();

private:
  std::string &GetHostKernelsInfo(int kernel_id);
  std::string &GetDeviceKernelsInfo(int kernel_id);

  void GEMMHost(const int kernel_id, 
                const int M, const int N,
                const int K, const float ALPHA,
                const Dtype *A, const int lda,
                const Dtype *B, const int ldb,
                const float beta,
                Dtype *C, const int ldc);

  void GEMMDevice(const int kernel_id,
                  const int M, const int N, 
                  const int K, const float ALPHA,
                  const Dtype *A, const int lda,
                  const Dtype *B, const int ldb,
                  const float beta,
                  Dtype *C, const int ldc);

  void PrepareLaunchConfig(int N, int M);

private:
  CuxData<Dtype> *A_;
  CuxData<Dtype> *B_;
  CuxData<Dtype> *C_;

  GEMMKernelParam kernel_params_;
  std::vector<Config2D> config_2d_;
};
} // cux.

#endif //CUX_GEMM_H_
