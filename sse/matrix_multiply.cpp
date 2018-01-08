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

void MatrixMulNormal(const int M, const int N, const int K, const float ALPHA,
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

void MatrixMulSSE(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {  
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      __m128 apart = _mm_set_ps1(ALPHA*A[i*lda + k]);
      for (j = 0; j < N - 3; j += 4) {
        __m128 b = _mm_loadu_ps(B + k*ldc + j);
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
  int height_a = 500, width_a = 40;
  int height_b = 40, width_b = 1200;
  if (width_a != height_b) {
    std::cout << "input params are invalid." << std::endl;
    return 1;
  }
  int height_ret = height_a, width_ret = width_b;

  float *mat_a = new float[height_a * width_a];
  float *mat_b = new float[height_b * width_b];
  float *mat_ret_normal = new float[height_ret * width_ret];
  float *mat_ret_sse = new float[height_ret * width_ret];

  srand(0);
  GenMatrix(height_a, width_a, mat_a);
  GenMatrix(height_b, width_b, mat_b);

  time_t stime;
  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormal(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_normal, width_ret);
  }
  std::cout << "Normal ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_normal, height_ret, width_ret) << std::endl;

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSSE(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_sse, width_ret);
  }
  std::cout << "SSE ->  time: " << clock() - stime << ", mean value: " << GetMean(mat_ret_sse, height_ret, width_ret) << std::endl;

  //std::cout << "\n Org result" << std::endl;
  //MatrixPrint(mat_ret_normal, height_ret, width_ret);

  //std::cout << "\n SSE result \n" << std::endl;
  //MatrixPrint(mat_ret_sse, height_ret, width_ret);

  delete[] mat_a;
  delete[] mat_b;
  delete[] mat_ret_normal;
  delete[] mat_ret_sse;

  return 0;
}