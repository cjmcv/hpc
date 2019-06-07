/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_HPP_
#define CUX_DOT_PRODUCT_HPP_

#include "util.h"
#include "operator.h"

namespace cux {

// TODO: 1. Encapsulated into a class.
//       2. Called by Executor.
//       3. Has the same input parameters.
//       4. Kernel can be manually switched.
class VectorDotProduct : public Operator {
public:
  VectorDotProduct() {}

  void RunOnHost(const float *vec_a, const float *vec_b, const int len, float &result);
  void RunOnDevice(const float *d_vec_a, const float *d_vec_b, const int len, float &d_result);
};
} // cux.

#endif //CUX_DOT_PRODUCT_HPP_