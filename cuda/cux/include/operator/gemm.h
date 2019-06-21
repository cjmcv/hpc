/*!
* \brief gemm.
*/

#ifndef CUX_GEMM_HPP_
#define CUX_GEMM_HPP_

#include "util.h"
#include "operator.h"

namespace cux {

class GEMM : public Operator {
public:
  GEMM() {}

  void Help() const;

  int SetIoParams(const std::vector< CuxData<float>* > &input,
                  const std::vector< CuxData<float>* > &output,
                  const void *params);
  void RunOnHost();
  void RunOnDevice();

private:
  CuxData<float> *A_;
  CuxData<float> *B_;
  CuxData<float> *C_;
};
} // cux.

#endif //CUX_GEMM_HPP_