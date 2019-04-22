/*!
* \brief Vector dot product: result = SUM(A * B).
*/
#include <iostream>
#include "time.h"

#include "xmmintrin.h"

// Initialize the input data.
void GenVector(const int len, float *vec) {
	for(int i=0; i<len; i++)
    vec[i] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
}

float VectorDotNormal(const float *vec_a, const float *vec_b, const int len) {
	float result = 0;
	for(int i=0; i<len; i++) {
    result += vec_a[i] * vec_b[i];
	}
	return result;
}

float VectorDotSIMD(const float *vec_a, const float *vec_b, const int len) {
  float result = 0;

  // Using 8 as the base number to call simd.
  if (len > 8) {
	  __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < len - 7; i += 8) {
		  __m128 a0 = _mm_loadu_ps(vec_a + i);
		  __m128 a1 = _mm_loadu_ps(vec_a + i + 4);

		  __m128 b0 = _mm_loadu_ps(vec_b + i);
		  __m128 b1 = _mm_loadu_ps(vec_b + i + 4);

		  sum = _mm_add_ps(sum, _mm_add_ps(_mm_mul_ps(a0, b0), _mm_mul_ps(a1, b1)));
	  }
	  result = sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
  }

  // Calculate the remaining part.
  for (int i = len / 8 * 8; i < len; i++) {
    result += vec_a[i] * vec_b[i];
  }
	return result;
}

int main() {
	int len = 12345689;
	float *vec_a = new float[len];
	float *vec_b = new float[len];

	srand(0);
	GenVector(len, vec_a);
	GenVector(len, vec_b);

  time_t stime;

  stime = clock();
	float result_normal = VectorDotNormal(vec_a, vec_b, len);
  std::cout << "Normal ->  time: " << clock() - stime << ", result: " << result_normal << std::endl;

  stime = clock();
  float result_simd = VectorDotSIMD(vec_a, vec_b, len);
  std::cout << "SIMD ->  time: " << clock() - stime << ", result: " << result_simd << std::endl;

  delete[] vec_a;
  delete[] vec_b;

  return 0;
}