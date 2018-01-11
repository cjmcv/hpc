/*!
* \brief Matrix Multiplication.
*/
#include<iostream>
#include "time.h"

#include "xmmintrin.h"

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
    }
  }
}

// Normal method, version 1.
// (1900ms)
void MatrixMulNormalv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float temp = 0;
      for (k = 0; k < K; ++k) {
        temp += A[i*lda + k] * B[k*ldb + j];
      }
      C[i*ldc + j] = temp;
    }
  }
}


// Normal method, version 2.
// (348ms)
void MatrixMulNormalv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA*A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART*B[k*ldb + j];
      }
    }
  }
}

// m128.
// The speed is consistent with Normal method while opening /O2. (379ms)
void MatrixMulSSEv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {  
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      __m128 apart = _mm_set_ps1(ALPHA*A[i*lda + k]);
      for (j = 0; j < N - 3; j += 4) {
        __m128 b = _mm_loadu_ps(B + k*ldb + j);
        __m128 c = _mm_loadu_ps(C + i*ldc + j);
        c = _mm_add_ps(c, _mm_mul_ps(apart, b));
        _mm_storeu_ps(C + i*ldc + j, c);
      }
      for (;j < N; j++) {
        C[i*ldc + j] += apart.m128_f32[0] * B[k*ldb + j];
      }
    }
  }
}

// Use m256 instead, and replace _mm_add_ps(c, _mm_mul_ps(apart, b)) with _mm256_fmadd_ps.
// Faster than v1. (179ms)
void MatrixMulSSEv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      __m256 apart = _mm256_set1_ps(ALPHA*A[i*lda + k]);
      for (j = 0; j < N - 7; j += 8) {
        __m256 b = _mm256_loadu_ps(B + k*ldb + j);
        __m256 c = _mm256_loadu_ps(C + i*ldc + j);
        c = _mm256_fmadd_ps(apart, b, c); // apart * b + c
        _mm256_storeu_ps(C + i*ldc + j, c);
      }
      for (;j < N; j++) {
        C[i*ldc + j] += apart.m256_f32[0] * B[k*ldb + j];
      }
    }
  }
}

// Loop unrolling
// Even slower than v2. (255ms)
void MatrixMulSSEv3(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      __m256 apart = _mm256_set1_ps(ALPHA*A[i*lda + k]);
      for (j = 0; j < N - 15; j += 16) {
        __m256 b0 = _mm256_loadu_ps(B + k*ldb + j);
        __m256 b1 = _mm256_loadu_ps(B + k*ldb + j + 8);
        __m256 c0 = _mm256_loadu_ps(C + i*ldc + j);
        __m256 c1 = _mm256_loadu_ps(C + i*ldc + j + 8);
        c0 = _mm256_fmadd_ps(apart, b0, c0);
        c1 = _mm256_fmadd_ps(apart, b1, c1);
        _mm256_storeu_ps(C + i*ldc + j, c0);
        _mm256_storeu_ps(C + i*ldc + j + 8, c1);
      }
      for (;j < N; j++) {
        C[i*ldc + j] += apart.m256_f32[0] * B[k*ldb + j];
      }
    }
  }
}

// Can not be faster while using aligned memory.(195ms)
//#define TEST_GET_ARRAY_DIRECTLY 1
void MatrixMulSSEv4(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
#ifdef TEST_GET_ARRAY_DIRECTLY
  __m256 *aligned_c = (__m256 *)C;
#endif
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      __m256 apart = _mm256_set1_ps(ALPHA*A[i*lda + k]);
      for (j = 0; j < N - 15; j += 16) {
#ifdef TEST_GET_ARRAY_DIRECTLY  // Slower.
        __m256 b0 = _mm256_loadu_ps(B + k*ldc + j);
        __m256 b1 = _mm256_loadu_ps(B + k*ldc + j + 8);
        const int c_offset = i*ldc / 8;
        aligned_c[c_offset + j / 8] = _mm256_fmadd_ps(apart, b0, aligned_c[c_offset + j / 8]);
        aligned_c[c_offset + j / 8 + 1] = _mm256_fmadd_ps(apart, b1, aligned_c[c_offset + j / 8 + 1]);
#else
        __m256 b0 = _mm256_loadu_ps(B + k*ldc + j);
        __m256 b1 = _mm256_loadu_ps(B + k*ldc + j + 8);
        __m256 c0 = _mm256_load_ps(C + i*ldc + j);
        __m256 c1 = _mm256_load_ps(C + i*ldc + j + 8);
        c0 = _mm256_fmadd_ps(apart, b0, c0);
        c1 = _mm256_fmadd_ps(apart, b1, c1);
        _mm256_store_ps(C + i*ldc + j, c0);
        _mm256_store_ps(C + i*ldc + j + 8, c1);
#endif
      }
      for (;j < N; j++) {
        C[i*ldc + j] += apart.m256_f32[0] * B[k*ldb + j];
      }
    }
  }
}

// Just for checking the result.
float GetMean(const float* mat, const int height, const int width) {
  int num = height * width;
  float total = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      total += mat[i*width + j];
    }
  }
  return total / num;
}

void MatrixPrint(const float* mat, const int height, const int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << mat[i*width + j] << ",";
    }
    std::cout << std::endl;
  }
}

int main() {
  int height_a = 500, width_a = 405;
  int height_b = 405, width_b = 120;
  if (width_a != height_b) {
    std::cout << "input params are invalid." << std::endl;
    return 1;
  }
  int height_ret = height_a, width_ret = width_b;

  float *mat_a = new float[height_a * width_a];
  float *mat_b = new float[height_b * width_b];
  float *mat_ret_normal_v1 = new float[height_ret * width_ret];
  float *mat_ret_normal_v2 = new float[height_ret * width_ret];
  float *mat_ret_sse_v1 = new float[height_ret * width_ret];
  float *mat_ret_sse_v2 = new float[height_ret * width_ret];
  float *mat_ret_sse_v3 = new float[height_ret * width_ret];
  float *mat_ret_sse_v4 = (float *)_aligned_malloc(height_ret * width_ret * sizeof(float), 32);

  srand(0);
  GenMatrix(height_a, width_a, mat_a);
  GenMatrix(height_b, width_b, mat_b);

  time_t stime;
  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormalv1(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_normal_v1, width_ret);
  }
  std::cout << "Normalv1 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_normal_v1, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormalv2(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_normal_v2, width_ret);
  }
  std::cout << "Normalv2 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_normal_v2, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSEv1(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_sse_v1, width_ret);
  }
  std::cout << "SSEv1 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_sse_v1, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSEv2(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_sse_v2, width_ret);
  }
  std::cout << "SSEv2 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_sse_v2, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSEv3(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_sse_v3, width_ret);
  }
  std::cout << "SSEv3 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_sse_v3, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSEv4(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_sse_v4, width_ret);
  }
  std::cout << "SSEv4 ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_sse_v4, height_ret, width_ret) << std::endl;

  //std::cout << "\n Org result" << std::endl;
  //MatrixPrint(mat_ret_normal, height_ret, width_ret);

  //std::cout << "\n SSE result \n" << std::endl;
  //MatrixPrint(mat_ret_sse, height_ret, width_ret);

  delete[] mat_a;
  delete[] mat_b;
  delete[] mat_ret_normal_v1;
  delete[] mat_ret_normal_v2;
  delete[] mat_ret_sse_v1;
  delete[] mat_ret_sse_v2;
  delete[] mat_ret_sse_v3;
  _aligned_free(mat_ret_sse_v4);

  return 0;
}