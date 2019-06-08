/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_HPP_
#define CUX_DOT_PRODUCT_HPP_

#include "util.h"
#include "operator.h"

namespace cux {

class VectorDotProduct : public Operator {
public:
  VectorDotProduct() {}

  void RunOnHost(const float *vec_a, const float *vec_b, const int len, float &result);
  void RunOnDevice(const float *d_vec_a, const float *d_vec_b, const int len, float &d_result);
};
} // cux.

#endif //CUX_DOT_PRODUCT_HPP_