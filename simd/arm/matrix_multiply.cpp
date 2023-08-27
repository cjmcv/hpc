/*!
* \brief Matrix Multiplication.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory.h>
#include <time.h>
#include <arm_neon.h>

#define HEIGHT_A 480  // M
#define WIDTH_A 320  // K
#define HEIGHT_B 320  // K
#define WIDTH_B 720   // N
#define HEIGHT_C HEIGHT_A
#define WIDTH_C WIDTH_B

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
// version 1. 常规普通写法
// Cache miss: A K/p * N * M
//             B K * N * M
//             C K * N * M
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

//////////////////////////////////////////////
// version 2. 调整for循环层级顺序，提高cache命中率
// Cache miss: A K/p * M
//             B N/p * K * M
//             C N/p * K * M
void MatrixMulNormalv2(const int M, const int N, const int K, const float ALPHA,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART * B[k*ldb + j];
            }
        }
    }
}

//////////////////////////////////////////////
// version 3. 调整for循环层级顺序，提高cache命中率
// Cache miss: A K/p * M * N/T （多乘一个N/T）
//             B K * M * N/T 
//             C M * N/T       （少乘一个K）
// T 应在cache line范围内（即为p），
//   则B的jj循环内cache命中，k++时miss
//   C的jj循环内cache命中，k++时数据不用换，仍命中
void MatrixMulNormalv3(const int M, const int N, const int K, const float ALPHA,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      float *C, const int ldc) {
    int i, j, jj, k;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 128;
    for (j = 0; j < N; j += T) {
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                float A_PART = ALPHA * A[i * lda + k];
                for (jj = j; jj < std::min(j + T, N); ++jj) {
                    C[i * ldc + jj] += A_PART * B[k * ldb + jj];
                }
            }
        }
    }
}

//////////////////////////////////////////////
// version 3. 调整for循环层级顺序，提高cache命中率
// Cache miss: A K/p * M * N/T （多乘一个N/T）
//             B K * M * N/T 
//             C M * N/T       （少乘一个K）
// T 应在cache line范围内（即为p），
//   则B的jj循环内cache命中，k++时miss
//   C的jj循环内cache命中，k++时数据不用换，仍命中
void MatrixMulNormalv4(const int M, const int N, const int K, const float ALPHA,
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      float *C, const int ldc) {
    int i, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 128;
    for (k = 0; k < K; k += T) {
        for (j = 0; j < N; j += T) {
            for (i = 0; i < M; ++i) {
                for (kk = k; kk < std::min(k + T, K); ++kk) {
                    float A_PART = ALPHA * A[i * lda + kk];
                    for (jj = j; jj < std::min(j + T, N); ++jj) {
                        C[i * ldc + jj] += A_PART * B[kk * ldb + jj];
                    }
                }
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
                float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
                float32x4_t c0 = vld1q_f32(C + i * ldc + j);
                float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
                c0 = vmlaq_f32(c0, apart, b0); // apart * b + c
                c1 = vmlaq_f32(c1, apart, b1);
                vst1q_f32(C + i * ldc + j, c0);
                vst1q_f32(C + i * ldc + j + 4, c1);
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
                __builtin_prefetch(B + k * ldb + j + 8, 0, 1);
                __builtin_prefetch(C + i * ldc + j + 8, 0, 1);

                float32x4_t b0 = vld1q_f32(B + k * ldb + j);
                float32x4_t b1 = vld1q_f32(B + k * ldb + j + 4);
                float32x4_t c0 = vld1q_f32(C + i * ldc + j);
                float32x4_t c1 = vld1q_f32(C + i * ldc + j + 4);
                c0 = vmlaq_f32(c0, apart, b0); // apart * b + c
                c1 = vmlaq_f32(c1, apart, b1);
                vst1q_f32(C + i * ldc + j, c0);
                vst1q_f32(C + i * ldc + j + 4, c1);
            }
            for (; j < N; j++) {
                C[i*ldc + j] += apart0 * B[k*ldb + j];
            }
        }
    }
}

#define TEST_MODULE(func)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(float)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 1; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, 1.0, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C); \
        }                                                                                          \
        printf("%s -> time: %f s, mean value: %f\n",                                               \
               #func, double(clock() - stime)/CLOCKS_PER_SEC, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)


int main() {

    float *mat_a = (float *)malloc(HEIGHT_A * WIDTH_A * sizeof(float));
    float *mat_b = (float *)malloc(HEIGHT_B * WIDTH_B * sizeof(float));
    float *mat_c = (float *)malloc(HEIGHT_C * WIDTH_C * sizeof(float));

    //float *mat_ret_simd_v4 = (float *)_aligned_malloc(HEIGHT_C * WIDTH_C * sizeof(float), 32);
    GenMatrix(HEIGHT_A, WIDTH_A, mat_a);
    GenMatrix(HEIGHT_B, WIDTH_B, mat_b);

    TEST_MODULE(MatrixMulNormalv1);
    TEST_MODULE(MatrixMulNormalv2);
    TEST_MODULE(MatrixMulNormalv3);
    TEST_MODULE(MatrixMulNormalv4);
    TEST_MODULE(MatrixMulSIMDv1);
    TEST_MODULE(MatrixMulSIMDv2);
    TEST_MODULE(MatrixMulSIMDv3);

    /*
        MatrixMulNormalv1 -> time: 0.441107 s, mean value: 121543.968750
        MatrixMulNormalv2 -> time: 0.249376 s, mean value: 121543.968750
        MatrixMulNormalv3 -> time: 0.249129 s, mean value: 121543.968750
        MatrixMulSIMDv1 -> time: 0.253189 s, mean value: 121543.968750
        MatrixMulSIMDv2 -> time: 0.260855 s, mean value: 121543.968750
        MatrixMulSIMDv3 -> time: 0.281071 s, mean value: 121543.968750
    */
      
    free(mat_a);
    free(mat_b);
    free(mat_c);

    //_aligned_free(mat_ret_simd_v4);
    return 0;
}