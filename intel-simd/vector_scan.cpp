/*!
* \brief Scan. Prefix Sum.
* \example: input: 1,2,3,4
*           operation: Add
*           ouput: 1,3,6,10 (out[i]=sum(in[0:i]))
*/

#include <iostream>
#include "time.h"

#include "xmmintrin.h"

// Initialize the input data.
void GenVector(const int len, float *vec) {
  for (int i = 0; i < len; i++)
    vec[i] = 2;//(float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
}

// Normal version in cpu as a reference
void VectorScanNormal(const float *vec_in, const int len, float *vec_out) {
  vec_out[0] = vec_in[0];
  for (int i = 1; i<len; i++) {
    vec_out[i] = vec_in[i] + vec_out[i - 1];
  }
}

// 1, Change the type of x to __m128i for calling the shifts function.
// 2, Shift a right by imm8 bytes while shifting in zeros, and store the results in dst.
//    4 * 8 = 32 = one float data.
// 3, Change the type of x back to __m128.
// \example (1 2 3 4) + (0 1 2 3) = (1 3 5 7)
//          (1 3 5 7) + (0 0 1 3) = (1 3 6 10)
inline __m128 ScanM128(__m128 x) {
  x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
  x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
  return x;
}

void VectorScanSIMDv1(const float *vec_in, const int len, float *vec_out) {
  __m128 offset = _mm_setzero_ps();
  for (int i = 0; i < len; i += 4) {
    __m128 x = _mm_loadu_ps(vec_in + i);
    __m128 y = ScanM128(x);
    y = _mm_add_ps(y, offset);
    _mm_store_ps(vec_out + i, y);
    // offset = _mm_set1_ps(scan_out.m128_f32[3]);
    // Selete the third element to form a new m128.
    offset = _mm_shuffle_ps(y, y, _MM_SHUFFLE(3, 3, 3, 3));
  }
}

int main() {
  const int loops = 100;
  const int len = 1000000;
  float *vec_in = new float[len];
  float *vec_out = new float[len];

  srand(0);
  GenVector(len, vec_in);

  time_t stime;

  stime = clock();
  for(int i=0; i<loops; i++)
    VectorScanNormal(vec_in, len, vec_out);
  std::cout << "Normal ->  time: " << clock() - stime << ", result: " << vec_out[len - 1] << std::endl;

  memset(vec_out, 0, sizeof(float) * len);

  stime = clock();
  for (int i = 0; i < loops; i++)
    VectorScanSIMDv1(vec_in, len, vec_out);
  std::cout << "SIMD ->  time: " << clock() - stime << ", result: " << vec_out[len - 1] << std::endl;

  delete[] vec_in;
  delete[] vec_out;

  system("pause");
  return 0;
}