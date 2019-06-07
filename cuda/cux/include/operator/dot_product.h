/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_HPP_
#define CUX_DOT_PRODUCT_HPP_

#include "util.h"

namespace cux {

// TODO: 1. Encapsulated into a class.
//       2. Called by Executor.
//       3. Has the same input parameters.
//       4. Kernel can be manually switched.
float VectorDotProductCPU(const float *vec_a, const float *vec_b, const int len);

float VectorDotProductCUDA(const int loops, const float *vec_a, const float *vec_b, const int len, float &result);

} // cux.

#endif //CUX_DOT_PRODUCT_HPP_