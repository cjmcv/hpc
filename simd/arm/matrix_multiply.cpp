/*!
* \brief Matrix Multiplication.
*/
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <time.h>
#include <arm_neon.h>

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      mat[i*width + j] = i + j;
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
      printf("%f,", mat[i*width + j]);
    }
    printf("\n");
  }
}

//////////////////////////////////////////////
// No simd, version 1.
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

// No simd, version 2. 
void MatrixMulNormalv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA * A[i*lda + k];
      for (j = 0; j < N; ++j) {
        C[i*ldc + j] += A_PART * B[k*ldb + j];
      }
    }
  }
}

// https://github.com/cjmcv/hpc/blob/master/simd/x86/matrix_multiply.cpp
void MatrixMulSIMDv1(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float apart0 = ALPHA * A[i*lda + k];
      float32x4_t apart = vdupq_n_f32(apart0);
      for (j = 0; j < N - 3; j += 4) {
        float32x4_t b = vld1q_f32(B + k * ldb + j);
        float32x4_t c = vld1q_f32(C + i * ldc + j);
        c = vaddq_f32(c, vmulq_f32(apart, b));
        vst1q_f32(C + i * ldc + j, c);
      }
      for (; j < N; j++) {
        C[i*ldc + j] += apart0 * B[k*ldb + j];
      }
    }
  }
}

void MatrixMulSIMDv2(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float apart0 = ALPHA * A[i*lda + k];
      float32x4_t apart = vdupq_n_f32(apart0);
      for (j = 0; j < N - 7; j += 8) {
        float32x4_t b0 = vld1q_f32(B + k * ldb + j);
        float32x4_t b1 = vld1q_f32(B + k * ldb + j + 8);
        float32x4_t c0 = vld1q_f32(C + i * ldc + j);
        float32x4_t c1 = vld1q_f32(C + i * ldc + j + 8);
        c0 = vmlaq_f32(c0, apart, b0); // apart * b + c
        c1 = vmlaq_f32(c1, apart, b1);
        vst1q_f32(C + i * ldc + j, c0);
        vst1q_f32(C + i * ldc + j + 8, c1);
      }
      for (; j < N; j++) {
        C[i*ldc + j] += apart0 * B[k*ldb + j];
      }
    }
  }
}

void MatrixMulSIMDv3(const int M, const int N, const int K, const float ALPHA,
  const float *A, const int lda,
  const float *B, const int ldb,
  float *C, const int ldc) {
  int i, j, k;
  memset(C, 0, sizeof(float) * ldc * M);
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float apart0 = ALPHA * A[i*lda + k];
      float32x4_t apart = vdupq_n_f32(apart0);
      for (j = 0; j < N - 7; j += 8) {
        __builtin_prefetch(B + k * ldb + j + 16, 0, 1);
        __builtin_prefetch(C + i * ldc + j + 16, 0, 1);

        float32x4_t b0 = vld1q_f32(B + k * ldb + j);
        float32x4_t b1 = vld1q_f32(B + k * ldb + j + 8);
        float32x4_t c0 = vld1q_f32(C + i * ldc + j);
        float32x4_t c1 = vld1q_f32(C + i * ldc + j + 8);
        c0 = vmlaq_f32(c0, apart, b0); // apart * b + c
        c1 = vmlaq_f32(c1, apart, b1);
        vst1q_f32(C + i * ldc + j, c0);
        vst1q_f32(C + i * ldc + j + 8, c1);
      }
      for (; j < N; j++) {
        C[i*ldc + j] += apart0 * B[k*ldb + j];
      }
    }
  }
}

int main() {
  int height_a = 64, width_a = 32;
  int height_b = 32, width_b = 128;
  if (width_a != height_b) {
    printf("input params are invalid.");
    return 1;
  }

  int height_ret = height_a, width_ret = width_b;
  float *mat_a = (float *)malloc(height_a * width_a * sizeof(float));
  float *mat_b = (float *)malloc(height_b * width_b * sizeof(float));
  float *mat_ret_normal_v1 = (float *)malloc(height_ret * width_ret * sizeof(float));
  float *mat_ret_normal_v2 = (float *)malloc(height_ret * width_ret * sizeof(float));
  float *mat_ret_simd_v1 = (float *)malloc(height_ret * width_ret * sizeof(float));
  float *mat_ret_simd_v2 = (float *)malloc(height_ret * width_ret * sizeof(float));
  float *mat_ret_simd_v3 = (float *)malloc(height_ret * width_ret * sizeof(float));
  //float *mat_ret_simd_v4 = (float *)_aligned_malloc(height_ret * width_ret * sizeof(float), 32);
  GenMatrix(height_a, width_a, mat_a);
  GenMatrix(height_b, width_b, mat_b);

  time_t stime;
  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormalv1(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_normal_v1, width_ret);
  }
  printf("Normalv1 -> time: %d, mean value: %f\n", clock() - stime, GetMean(mat_ret_normal_v1, height_ret, width_ret));

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulNormalv2(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_normal_v2, width_ret);
  }
  printf("Normalv2 -> time: %d, mean value: %f\n", clock() - stime, GetMean(mat_ret_normal_v1, height_ret, width_ret));

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSIMDv1(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_simd_v1, width_ret);
  }
  printf("SIMDv1 -> time: %d, mean value: %f\n", clock() - stime, GetMean(mat_ret_normal_v1, height_ret, width_ret));

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSIMDv2(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_simd_v2, width_ret);
  }
  printf("SIMDv2 -> time: %d, mean value: %f\n", clock() - stime, GetMean(mat_ret_normal_v1, height_ret, width_ret));

  stime = clock();
  for (int i = 0; i < 100; i++) {
    MatrixMulSIMDv3(height_ret, width_ret, width_a, 1.0, mat_a, width_a, mat_b, width_b, mat_ret_simd_v3, width_ret);
  }
  printf("SIMDv3 -> time: %d, mean value: %f\n", clock() - stime, GetMean(mat_ret_normal_v1, height_ret, width_ret));

  free(mat_a);
  free(mat_b);
  free(mat_ret_normal_v1);
  free(mat_ret_normal_v2);
  free(mat_ret_simd_v1);
  free(mat_ret_simd_v2);
  free(mat_ret_simd_v3);
  //_aligned_free(mat_ret_simd_v4);
  return 0;
}