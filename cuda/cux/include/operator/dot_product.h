/*!
* \brief Dot Product.
*/

#ifndef CUX_DOT_PRODUCT_HPP_
#define CUX_DOT_PRODUCT_HPP_

#include "util.h"

namespace cux {

float VectorDotProductCPU(const float *vec_a, const float *vec_b, const int len);

float VectorDotProductCUDA(const int loops, const float *vec_a, const float *vec_b, const int len, float &result);

} // cux.

#endif //CUX_DOT_PRODUCT_HPP_