/*!
* \brief Matrix Multiplication.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory.h>
#include <time.h>

#include "pocket-ai/prof/cpu_selector.hpp"
#include "pocket-ai/prof/peak_perf_detector.hpp"

#include <arm_neon.h>

// 生成随机数 -10000 到 10000 的随机数，并限制精度到小数点后1位
void GenMatrix(const int height, const int width, float *mat) {
    std::srand(static_cast<unsigned int>(time(nullptr)));
    float upper_bound = 10000;
    float lower_bound = -10000;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = lower_bound + static_cast<float>(rand()) / (RAND_MAX / (upper_bound - lower_bound)); // i + j;
            
            int temp = mat[i*width + j]; // * 10;
            mat[i*width + j] = temp; // / 10.f;
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
// 假定只有一个cache，且相对够大
// version 1. 常规普通写法
// Cache miss: A K/P * N * M
//             B K * N * M
//             C N/P * M
void GemmV1(const int M, const int N, const int K,
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
// Cache miss: A K/P * M
//             B N/P * K * M
//             C N/P * K * M
void GemmV2(const int M, const int N, const int K,
            const float *A, const int lda,
            const float *B, const int ldb,
            float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
        }
    }
}

//////////////////////////////////////////////
// version 3. 调整for循环层级顺序，提高cache命中率
// Cache miss: A K/P * M * N/T       （多乘一个N/T）
//             B T/P * K * M * N/T 
//             C T/P * M * N/T       （少乘一个K）
// T 应在cache line范围内（即为p），
//   则B的jj循环内cache命中，k++时miss
//   C的jj循环内cache命中，k++时数据不用换，仍命中
void GemmV3(const int M, const int N, const int K, 
            const float *A, const int lda,
            const float *B, const int ldb,
            float *C, const int ldc) {
    int i, j, jj, k;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 128;
    for (j = 0; j < N; j += T) {
        for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                for (jj = j; jj < std::min(j + T, N); ++jj) {
                    C[i * ldc + jj] += A[i * lda + k] * B[k * ldb + jj];
                }
            }
        }
    }
}

//////////////////////////////////////////////
// version 4.
// L1 Cache miss: A T/P * M * N/T * K/T
//                B T/P * T * N/T * K/T   (M的遍历不影响kk和jj, 少乘一个M)
//                C T/P * M * N/T * K/T   (多乘一个K/T)
void GemmV4(const int M, const int N, const int K,
            const float *A, const int lda,
            const float *B, const int ldb,
            float *C, const int ldc) {
    int i, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 80;
    for (k = 0; k < K; k += T) {
        for (j = 0; j < N; j += T) {
            for (i = 0; i < M; ++i) {
                for (kk = k; kk < std::min(k + T, K); ++kk) {
                    for (jj = j; jj < std::min(j + T, N); ++jj) {
                        C[i * ldc + jj] += A[i * lda + kk] * B[kk * ldb + jj];
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////
// version 5.
//
// cache line为64B(16个float)，L1 cache大小为16KB~64KB
// 令T为80
// A: 最内循环kk，miss T/P=80/16=5次
// 次内循环ii，每自增1都会有一次miss，则为T/P * T
// 循环j，与ii和kk无关，忽略
// 循环ik，每次都会miss，则T/P * T * K/T * M/T （少乘一个N/T）
//
// B: 最内循环jj，miss T/P
// 次内循环kk，每自增1都会有一次miss，则为T/P * T 
// 循环ii，每自增1时，因(jj,kk)块内的数据均已经在cache（80*80*4=25KB假定在容量内）中，不会发生miss。
// 循环ijk，每次会都miss，则T/P * T * N/T * K/T * M/T;（多乘一个M/T）
//
// C: 最内循环jj，miss T/P
// 次内循环kk，与C当前所需数据无关，忽略
// 循环ii，每自增1都会有一次miss，则为T/P * T
// 循环ijk，每次会都miss，则T/P * T * N/T * K/T * M/T; （不变）
//
// 当处于高效的情况下，T为80, 块内所需内存的80*80*4=25K，再乘以3个块，即75KB，已超过L1的大小，
// 推测v5效率高于v4的原因在于L2
void GemmV5(const int M, const int N, const int K,
            const float *A, const int lda,
            const float *B, const int ldb,
            float *C, const int ldc) {
    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 80;
    for (i = 0; i < M; i += T) {
        for (k = 0; k < K; k += T) {
            for (j = 0; j < N; j += T) {
                for (ii = i; ii < std::min(i + T, M); ++ii) {
                    for (kk = k; kk < std::min(k + T, K); ++kk) {
                        for (jj = j; jj < std::min(j + T, N); ++jj) {
                            C[ii * ldc + jj] += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}


typedef void (*UKernelFunc)(const int mstart, const int mend, 
                            const int nstart, const int nend, 
                            const int kstart, const int kend, 
                            const float *A, const int lda,
                            const float *B, const int ldb,
                            float *C, const int ldc);

void GemmTile(const int M, const int N, const int K,
              const float *A, const int lda,
              const float *B, const int ldb,
              float *C, const int ldc, UKernelFunc ukernel) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);

    int T = 80;
    for (i = 0; i < M; i += T) {
        for (k = 0; k < K; k += T) {
            for (j = 0; j < N; j += T) {
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        k, std::min(k + T, K),
                        A, lda, B, ldb, C, ldc);
            }
        }
    }
}


// ukernel v6，基于同v5拆分函数便于分析
void UKernelV6(const int mstart, const int mend, 
                   const int nstart, const int nend, 
                   const int kstart, const int kend, 
                   const float *A, const int lda,
                   const float *B, const int ldb,
                   float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (k = kstart; k < kend; ++k) {
            for (j = nstart; j < nend; j++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// ukernel v7，将最内层常规改写neon的形式，速度比v6慢了近1倍
// 可见编译器对其进行优化的程度要远优于当前的这种写法，手动neon改写反而给编译器带来了一定的约束，造成了负效果。
// v2写法存在的问题，1）A矩阵的加载一次只加载一个；2）内层循环两个加载和一个写回只搭配了一个mla。
//                  neon寄存器的加载和写回代价较大，也考虑到A53是双发射流水线，内层循环太短，会打断流水。
void UKernelV7(const int mstart, const int mend, 
                   const int nstart, const int nend, 
                   const int kstart, const int kend, 
                   const float *A, const int lda,
                   const float *B, const int ldb,
                   float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (k = kstart; k < kend; ++k) {
            float32x4_t va = vdupq_n_f32(A[i*lda + k]);
            for (j = nstart; j < nend - 3; j += 4) {
                float32x4_t vb0 = vld1q_f32(B + k * ldb + j);
                float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
                vc0 = vmlaq_f32(vc0, va, vb0);
                // vc0 = vfmaq_laneq_f32(vc0, vb0, va, 0); // 结果会有轻微差异
                vst1q_f32(C + i * ldc + j, vc0);
            }
            for (; j < nend; ++j) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// v8，在v7的基础上，观察vc由ij组成，与k无关，则可以尝试展开k
//     基于k复用vc的读写，C矩阵的加载和写回节省了3/4。
//     实测比v1相近，即相当于编译器simd优化达到的程度。
void UKernelV8(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const float *A, const int lda,
               const float *B, const int ldb,
               float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (k = kstart; k < kend - 3; k += 4) {
            float32x4_t va = vld1q_f32(A + i*lda + k);
            for (j = nstart; j < nend - 7; j += 8) {
                float32x4_t vb0k0 = vld1q_f32(B + (k+0) * ldb + j);
                float32x4_t vb1k0 = vld1q_f32(B + (k+0) * ldb + j+4);

                float32x4_t vb0k1 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j+4);
                
                float32x4_t vb0k2 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb1k2 = vld1q_f32(B + (k+2) * ldb + j+4);

                float32x4_t vb0k3 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb1k3 = vld1q_f32(B + (k+3) * ldb + j+4);

                float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
                float32x4_t vc1 = vld1q_f32(C + i * ldc + j+4);

                vc0 = vfmaq_laneq_f32(vc0, vb0k0, va, 0);  // vc0[i] = vc0[i] + vb0k0[i] * va[0]
                vc0 = vfmaq_laneq_f32(vc0, vb0k1, va, 1);  // vc0[i] = vc0[i] + vb1k0[i] * va[1]
                vc0 = vfmaq_laneq_f32(vc0, vb0k2, va, 2);    
                vc0 = vfmaq_laneq_f32(vc0, vb0k3, va, 3);        

                vc1 = vfmaq_laneq_f32(vc1, vb1k0, va, 0);  // vc1[i] = vc1[i] + vb0k1[i] * va[0]
                vc1 = vfmaq_laneq_f32(vc1, vb1k1, va, 1);
                vc1 = vfmaq_laneq_f32(vc1, vb1k2, va, 2);
                vc1 = vfmaq_laneq_f32(vc1, vb1k3, va, 3);

                vst1q_f32(C + i * ldc + j, vc0);
                vst1q_f32(C + i * ldc + j + 4, vc1);
            }
            for (; j < nend; ++j) {
                for (int kk=0; kk<4; kk++) {
                    C[i * ldc + j] += A[i * lda + k+kk] * B[(k+kk) * ldb + j];
                }
            }
        }
        for (; k < kend; ++k) {
            for (j = nstart; j < nend; j++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// v9，在v8的基础上，观察到B矩阵的neon寄存器加载不涉及i，即可以由i复用，对i进行展开，提高计算访存比
void UKernelV9(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        for (k = kstart; k < kend -3; k += 4) {
            float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
            float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
            float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
            float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);
            for (j = nstart; j < nend - 7; j += 8) {
                float32x4_t vb0k0 = vld1q_f32(B + (k+0) * ldb + j);
                float32x4_t vb1k0 = vld1q_f32(B + (k+0) * ldb + j+4);

                float32x4_t vb0k1 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j+4);
                
                float32x4_t vb0k2 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb1k2 = vld1q_f32(B + (k+2) * ldb + j+4);

                float32x4_t vb0k3 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb1k3 = vld1q_f32(B + (k+3) * ldb + j+4);

                float32x4_t vc0i0 = vld1q_f32(C + (i+0) * ldc + j);
                float32x4_t vc1i0 = vld1q_f32(C + (i+0) * ldc + j+4);

                float32x4_t vc0i1 = vld1q_f32(C + (i+1) * ldc + j);
                float32x4_t vc1i1 = vld1q_f32(C + (i+1) * ldc + j+4);

                float32x4_t vc0i2 = vld1q_f32(C + (i+2) * ldc + j);
                float32x4_t vc1i2 = vld1q_f32(C + (i+2) * ldc + j+4);

                float32x4_t vc0i3 = vld1q_f32(C + (i+3) * ldc + j);
                float32x4_t vc1i3 = vld1q_f32(C + (i+3) * ldc + j+4);


                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);        

                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);

                vst1q_f32(C + (i+0) * ldc + j, vc0i0);
                vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);

                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);        

                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vst1q_f32(C + (i+1) * ldc + j, vc0i1);
                vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        

                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vst1q_f32(C + (i+2) * ldc + j, vc0i2);
                vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        

                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);

                vst1q_f32(C + (i+3) * ldc + j, vc0i3);
                vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);
            }
            for (; j < nend; ++j) {
                for (int ii=0; ii<4; ii++) {
                    for (int kk=0; kk<4; kk++) {
                        C[(i+ii) * ldc + j] += A[(i+ii) * lda + k+kk] * B[(k+kk) * ldb + j];
                    }                    
                }
            }
        }
        for (; k < kend; ++k) {
            for (int ii=0; ii<4; ii++) {
                for (j = nstart; j < nend; j++) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[k * ldb + j];
                }                
            }
        }
    }
    for (; i < mend; ++i) {
        for (k = kstart; k < kend; ++k) {
            for (j = nstart; j < nend; j++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// 基于v9
// 考虑到neon寄存器的数据读取和回写是有一定代价的，观察到C矩阵在最内层循环有回写的操作。
// 然后C矩阵的读写只涉及到了三层循环中的i和j，是否可以把kk循环放回到最内层，以减少C回写次数, 提高计算访存比。
// 但是会引入之前的老问题，B矩阵的最内层循环访存不连续，cache miss加重。
void UKernelV10(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = vld1q_f32(C + (i+0) * ldc + j);
            float32x4_t vc1i0 = vld1q_f32(C + (i+0) * ldc + j+4);

            float32x4_t vc0i1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc1i1 = vld1q_f32(C + (i+1) * ldc + j+4);

            float32x4_t vc0i2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc1i2 = vld1q_f32(C + (i+2) * ldc + j+4);

            float32x4_t vc0i3 = vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc1i3 = vld1q_f32(C + (i+3) * ldc + j+4);

            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);

                float32x4_t vb0k0 = vld1q_f32(B + (k+0) * ldb + j);
                float32x4_t vb1k0 = vld1q_f32(B + (k+0) * ldb + j+4);

                float32x4_t vb0k1 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j+4);
                
                float32x4_t vb0k2 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb1k2 = vld1q_f32(B + (k+2) * ldb + j+4);

                float32x4_t vb0k3 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb1k3 = vld1q_f32(B + (k+3) * ldb + j+4);


                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);        
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);        
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);

            for (; k < kend; k++) {
                for (int ii=0; ii<4; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[k * ldb + (j+jj)];
                    }
                }
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[k * ldb + j];
                }
            }
        }
    }
    for (; i < mend; ++i) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// 试验单独处理分块边界，简化边界处理逻辑
void UKernelV10P(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = vld1q_f32(C + (i+0) * ldc + j);
            float32x4_t vc1i0 = vld1q_f32(C + (i+0) * ldc + j+4);

            float32x4_t vc0i1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc1i1 = vld1q_f32(C + (i+1) * ldc + j+4);

            float32x4_t vc0i2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc1i2 = vld1q_f32(C + (i+2) * ldc + j+4);

            float32x4_t vc0i3 = vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc1i3 = vld1q_f32(C + (i+3) * ldc + j+4);

            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);

                float32x4_t vb0k0 = vld1q_f32(B + (k+0) * ldb + j);
                float32x4_t vb1k0 = vld1q_f32(B + (k+0) * ldb + j+4);

                float32x4_t vb0k1 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j+4);
                
                float32x4_t vb0k2 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb1k2 = vld1q_f32(B + (k+2) * ldb + j+4);

                float32x4_t vb0k3 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb1k3 = vld1q_f32(B + (k+3) * ldb + j+4);


                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);        
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);        
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);
        }
    }

    // 独立分离
    int k_remain = kstart + (kend - kstart) / 4 * 4;
    for (i = mstart; i < mend-3; i += 4) {
        for (j = nstart; j < nend - 7; j += 8) {
            for (k = k_remain; k < kend; k++) {
                for (int ii=0; ii<4; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[k * ldb + (j+jj)];
                    }
                }
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[k * ldb + j];
                }
            }
        }
    }
    for (; i < mend; ++i) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// 基于UKernelV10, 采用repack的方式，将内存B矩阵的访问转为连续内存访问，提高缓存命中率。
// 搭配PackTB2BC使用。
// Note: A矩阵的访问也不连续，专门去packA，代价可能比较大，但如果使用im2col时，可考虑顺便对A矩阵做pack操作。
void UKernelV11(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int bid,
                float *C, const int ldc) {
                    
    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = vld1q_f32(C + (i+0) * ldc + j);
            float32x4_t vc1i0 = vld1q_f32(C + (i+0) * ldc + j+4);

            float32x4_t vc0i1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc1i1 = vld1q_f32(C + (i+1) * ldc + j+4);

            float32x4_t vc0i2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc1i2 = vld1q_f32(C + (i+2) * ldc + j+4);

            float32x4_t vc0i3 = vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc1i3 = vld1q_f32(C + (i+3) * ldc + j+4);

            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);

                float32x4_t vb0k0 = vld1q_f32(B + lbid + 0); 
                float32x4_t vb1k0 = vld1q_f32(B + lbid + 4);
                float32x4_t vb0k1 = vld1q_f32(B + lbid + 8);
                float32x4_t vb1k1 = vld1q_f32(B + lbid + 12);

                float32x4_t vb0k2 = vld1q_f32(B + lbid + 16); 
                float32x4_t vb1k2 = vld1q_f32(B + lbid + 20);
                float32x4_t vb0k3 = vld1q_f32(B + lbid + 24);
                float32x4_t vb1k3 = vld1q_f32(B + lbid + 28);
                lbid += 32;

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);        
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);        
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);

            for (; k < kend; k++) {
                for (int ii=0; ii<4; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[lbid+jj];
                    }
                }
                lbid+=8;
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[lbid+k];
                }
            }
            lbid += kend-kstart;
        }
    }
    for (; i < mend; ++i) {
        int lbid = bid;
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[lbid++];
            }
        }
    }
}

// 基于UKernelV11, C矩阵的访问涉及i和j，在外两层循环，初始为0，不需要 vld1q_f32，直接写0;
void UKernelV12(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int bid,
                float *C, const int ldc) {

    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = zero; // vld1q_f32(C + (i+0) * ldc + j);
            float32x4_t vc1i0 = zero; // vld1q_f32(C + (i+0) * ldc + j+4);

            float32x4_t vc0i1 = zero; // vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc1i1 = zero; // vld1q_f32(C + (i+1) * ldc + j+4);

            float32x4_t vc0i2 = zero; // vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc1i2 = zero; // vld1q_f32(C + (i+2) * ldc + j+4);

            float32x4_t vc0i3 = zero; // vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc1i3 = zero; // vld1q_f32(C + (i+3) * ldc + j+4);

            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);

                // __builtin_prefetch(B + lbid + 256);   
                float32x4_t vb0k0 = vld1q_f32(B + lbid + 0); 
                float32x4_t vb1k0 = vld1q_f32(B + lbid + 4);
                float32x4_t vb0k1 = vld1q_f32(B + lbid + 8);
                float32x4_t vb1k1 = vld1q_f32(B + lbid + 12);

                float32x4_t vb0k2 = vld1q_f32(B + lbid + 16);
                float32x4_t vb1k2 = vld1q_f32(B + lbid + 20);
                float32x4_t vb0k3 = vld1q_f32(B + lbid + 24);
                float32x4_t vb1k3 = vld1q_f32(B + lbid + 28);
                lbid += 32;

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);        
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);

            for (; k < kend; k++) {
                for (int ii=0; ii<4; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[lbid+jj];
                    }
                }
                lbid+=8;
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[lbid+k];
                }
            }
            lbid += kend-kstart;
        }
    }
    for (; i < mend; ++i) {
        int lbid = bid;
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[lbid++];
            }
        }
    }
}

// 基于UKernelV12, 尝试进一步展开i循环， 因B矩阵不涉及i，所以pack不用改。
//                 可提高内层循环的计算访存比，多个va的四次读取，多了32次fmla。
//                 而此时有16个va，8个vb，8个vc，则使用的向量寄存器达到了32个，于v8而言，已用满。
void UKernelV13(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int bid,
                float *C, const int ldc) {

    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend-7; i += 8) {
        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = zero;
            float32x4_t vc1i0 = zero;
            float32x4_t vc0i1 = zero;
            float32x4_t vc1i1 = zero;
            float32x4_t vc0i2 = zero;
            float32x4_t vc1i2 = zero;
            float32x4_t vc0i3 = zero;
            float32x4_t vc1i3 = zero;

            float32x4_t vc0i4 = zero;
            float32x4_t vc1i4 = zero;
            float32x4_t vc0i5 = zero;
            float32x4_t vc1i5 = zero;
            float32x4_t vc0i6 = zero;
            float32x4_t vc1i6 = zero;
            float32x4_t vc0i7 = zero;
            float32x4_t vc1i7 = zero;

            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);
                float32x4_t vai4 = vld1q_f32(A + (i+4)*lda + k);
                float32x4_t vai5 = vld1q_f32(A + (i+5)*lda + k);
                float32x4_t vai6 = vld1q_f32(A + (i+6)*lda + k);
                float32x4_t vai7 = vld1q_f32(A + (i+7)*lda + k);

                // __builtin_prefetch(B + lbid + 256);   
                float32x4_t vb0k0 = vld1q_f32(B + lbid + 0); 
                float32x4_t vb1k0 = vld1q_f32(B + lbid + 4);
                float32x4_t vb0k1 = vld1q_f32(B + lbid + 8);
                float32x4_t vb1k1 = vld1q_f32(B + lbid + 12);
                float32x4_t vb0k2 = vld1q_f32(B + lbid + 16);
                float32x4_t vb1k2 = vld1q_f32(B + lbid + 20);
                float32x4_t vb0k3 = vld1q_f32(B + lbid + 24);
                float32x4_t vb1k3 = vld1q_f32(B + lbid + 28);
                lbid += 32;
                // 
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
                //
                //
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k0, vai4, 0);
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k1, vai4, 1);
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k2, vai4, 2);    
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k3, vai4, 3);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k0, vai4, 0);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k1, vai4, 1);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k2, vai4, 2);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k3, vai4, 3);
                
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k0, vai5, 0);
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k1, vai5, 1);
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k2, vai5, 2);    
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k3, vai5, 3);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k0, vai5, 0); 
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k1, vai5, 1);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k2, vai5, 2);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k3, vai5, 3);

                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k0, vai6, 0);
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k1, vai6, 1);
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k2, vai6, 2);    
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k3, vai6, 3);        
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k0, vai6, 0); 
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k1, vai6, 1);
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k2, vai6, 2);
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k3, vai6, 3);

                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k0, vai7, 0);
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k1, vai7, 1);
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k2, vai7, 2);    
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k3, vai7, 3);        
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k0, vai7, 0); 
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k1, vai7, 1);
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k2, vai7, 2);
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k3, vai7, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);
            vst1q_f32(C + (i+4) * ldc + j, vc0i4);
            vst1q_f32(C + (i+4) * ldc + j + 4, vc1i4);
            vst1q_f32(C + (i+5) * ldc + j, vc0i5);
            vst1q_f32(C + (i+5) * ldc + j + 4, vc1i5);
            vst1q_f32(C + (i+6) * ldc + j, vc0i6);
            vst1q_f32(C + (i+6) * ldc + j + 4, vc1i6);
            vst1q_f32(C + (i+7) * ldc + j, vc0i7);
            vst1q_f32(C + (i+7) * ldc + j + 4, vc1i7);

            for (; k < kend; k++) {
                for (int ii=0; ii<8; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[lbid+jj];
                    }
                }
                lbid += 8;
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<8; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[lbid+k];
                }
            }
            lbid += kend-kstart;
        }
    }
    for (; i < mend; ++i) {
        int lbid = bid;
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[lbid++];
            }
        }
    }
}

// 基于 UKernelV13，通过gcc -O3 -S -c gemm.cpp得到gemm.s
// 搜索 UKernelV13，得到该函数的汇编代码.
// 以Begin function为起点，End function为终点
// 从
// 	.globl	_Z10UKernelV13iiiiiiPKfiS0_iPfi // -- Begin function _Z10UKernelV13iiiiiiPKfiS0_iPfi
// 	.p2align	2
// 	.type	_Z10UKernelV13iiiiiiPKfiS0_iPfi,@function
// _Z10UKernelV13iiiiiiPKfiS0_iPfi:        // @_Z10UKernelV13iiiiiiPKfiS0_iPfi
// 到
// .Lfunc_end19:
// 	.size	_Z10UKernelV13iiiiiiPKfiS0_iPfi, .Lfunc_end19-_Z10UKernelV13iiiiiiPKfiS0_iPfi
//                                         // -- End function
// 修改_Z10UKernelV13iiiiiiPKfiS0_iPfi函数名为新函数名UKernelV13Asm，函数参数即原V13的格式，
// cpp 调用的地方用 extern "C" 包含住。
extern "C" void UKernelV13Asm(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc);

// 基于UKernelV13, 观察内层循环中，B的读取存在i上的跨行访问，cache命中率稍低。
//                 进而考虑对A也进行pack操作。
void UKernelV14(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int aid,
                const float *B, const int bid,
                float *C, const int ldc) {

    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend-7; i += 8) {
        int oaid = i * (kend - kstart);
        int lbid = bid; // bid 跟j/k层循环有关，bid由外面的j指定，内部两层循环j和k的数据连续，i跳变时重复，则这里指定。
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = zero;
            float32x4_t vc1i0 = zero;
            float32x4_t vc0i1 = zero;
            float32x4_t vc1i1 = zero;
            float32x4_t vc0i2 = zero;
            float32x4_t vc1i2 = zero;
            float32x4_t vc0i3 = zero;
            float32x4_t vc1i3 = zero;

            float32x4_t vc0i4 = zero;
            float32x4_t vc1i4 = zero;
            float32x4_t vc0i5 = zero;
            float32x4_t vc1i5 = zero;
            float32x4_t vc0i6 = zero;
            float32x4_t vc1i6 = zero;
            float32x4_t vc0i7 = zero;
            float32x4_t vc1i7 = zero;

            int laid = oaid; // bid 只跟i/k三层循环有关，内层k跳变时数据重复。
            for (k = kstart; k < kend -3; k += 4) {
                float32x4_t vai0 = vld1q_f32(A + laid + 0); // i0_k0k1k2k3 i1_k4k5k6k7
                float32x4_t vai1 = vld1q_f32(A + laid + 4);
                float32x4_t vai2 = vld1q_f32(A + laid + 8);
                float32x4_t vai3 = vld1q_f32(A + laid + 12);
                float32x4_t vai4 = vld1q_f32(A + laid + 16);
                float32x4_t vai5 = vld1q_f32(A + laid + 20);
                float32x4_t vai6 = vld1q_f32(A + laid + 24);
                float32x4_t vai7 = vld1q_f32(A + laid + 28);
                laid += 32;

                // __builtin_prefetch(B + lbid + 256);   
                float32x4_t vb0k0 = vld1q_f32(B + lbid + 0); 
                float32x4_t vb1k0 = vld1q_f32(B + lbid + 4);
                float32x4_t vb0k1 = vld1q_f32(B + lbid + 8);
                float32x4_t vb1k1 = vld1q_f32(B + lbid + 12);
                float32x4_t vb0k2 = vld1q_f32(B + lbid + 16);
                float32x4_t vb1k2 = vld1q_f32(B + lbid + 20);
                float32x4_t vb0k3 = vld1q_f32(B + lbid + 24);
                float32x4_t vb1k3 = vld1q_f32(B + lbid + 28);
                lbid += 32;
                // 
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, vai0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k1, vai0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k2, vai0, 2);    
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k3, vai0, 3);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, vai0, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k1, vai0, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k2, vai0, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k3, vai0, 3);
                
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k0, vai1, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, vai1, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k2, vai1, 2);    
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k3, vai1, 3);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k0, vai1, 0); 
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, vai1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k2, vai1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k3, vai1, 3);

                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k0, vai2, 0);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k1, vai2, 1);
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k2, vai2, 2);    
                vc0i2 = vfmaq_laneq_f32(vc0i2, vb0k3, vai2, 3);        
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k0, vai2, 0); 
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k1, vai2, 1);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k2, vai2, 2);
                vc1i2 = vfmaq_laneq_f32(vc1i2, vb1k3, vai2, 3);

                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k0, vai3, 0);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k1, vai3, 1);
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k2, vai3, 2);    
                vc0i3 = vfmaq_laneq_f32(vc0i3, vb0k3, vai3, 3);        
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k0, vai3, 0); 
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k1, vai3, 1);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k2, vai3, 2);
                vc1i3 = vfmaq_laneq_f32(vc1i3, vb1k3, vai3, 3);
                //
                //
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k0, vai4, 0);
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k1, vai4, 1);
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k2, vai4, 2);    
                vc0i4 = vfmaq_laneq_f32(vc0i4, vb0k3, vai4, 3);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k0, vai4, 0);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k1, vai4, 1);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k2, vai4, 2);
                vc1i4 = vfmaq_laneq_f32(vc1i4, vb1k3, vai4, 3);
                
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k0, vai5, 0);
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k1, vai5, 1);
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k2, vai5, 2);    
                vc0i5 = vfmaq_laneq_f32(vc0i5, vb0k3, vai5, 3);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k0, vai5, 0); 
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k1, vai5, 1);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k2, vai5, 2);
                vc1i5 = vfmaq_laneq_f32(vc1i5, vb1k3, vai5, 3);

                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k0, vai6, 0);
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k1, vai6, 1);
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k2, vai6, 2);    
                vc0i6 = vfmaq_laneq_f32(vc0i6, vb0k3, vai6, 3);        
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k0, vai6, 0); 
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k1, vai6, 1);
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k2, vai6, 2);
                vc1i6 = vfmaq_laneq_f32(vc1i6, vb1k3, vai6, 3);

                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k0, vai7, 0);
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k1, vai7, 1);
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k2, vai7, 2);    
                vc0i7 = vfmaq_laneq_f32(vc0i7, vb0k3, vai7, 3);        
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k0, vai7, 0); 
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k1, vai7, 1);
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k2, vai7, 2);
                vc1i7 = vfmaq_laneq_f32(vc1i7, vb1k3, vai7, 3);
            }
            vst1q_f32(C + (i+0) * ldc + j, vc0i0);
            vst1q_f32(C + (i+0) * ldc + j + 4, vc1i0);
            vst1q_f32(C + (i+1) * ldc + j, vc0i1);
            vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i+2) * ldc + j, vc0i2);
            vst1q_f32(C + (i+2) * ldc + j + 4, vc1i2);
            vst1q_f32(C + (i+3) * ldc + j, vc0i3);
            vst1q_f32(C + (i+3) * ldc + j + 4, vc1i3);
            vst1q_f32(C + (i+4) * ldc + j, vc0i4);
            vst1q_f32(C + (i+4) * ldc + j + 4, vc1i4);
            vst1q_f32(C + (i+5) * ldc + j, vc0i5);
            vst1q_f32(C + (i+5) * ldc + j + 4, vc1i5);
            vst1q_f32(C + (i+6) * ldc + j, vc0i6);
            vst1q_f32(C + (i+6) * ldc + j + 4, vc1i6);
            vst1q_f32(C + (i+7) * ldc + j, vc0i7);
            vst1q_f32(C + (i+7) * ldc + j + 4, vc1i7);

            //// Pack A的顺序
            // for (; k < kend; k++) {
            //     for (int ii=0; ii<8; ii++) {
            //         nA[aid++] = A[(i+ii) * lda + k]; // k不足的部分： k0_i0i1i2i3i4i5i6i7, k1_i0i1i2i3i4i5i6i7
            //     }
            // }
            for (; k < kend; k++) {
                for (int ii=0; ii<8; ii++) {
                    for (int jj=0; jj<8; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[laid] * B[lbid+jj];
                    }
                    laid++;
                }
                lbid += 8;
            }
        }

        //// 参考v13的代码
        // for (; j < nend; j++) {
        //     for (int ii=0; ii<8; ii++) {
        //         for (k = kstart; k < kend; ++k) {
        //             C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[lbid+k];
        //         }
        //     }
        //     lbid += kend-kstart;
        // }
        // TODO: 这里数据不对，如将N改为可被4整除，则这里不会调用，数据正确。
        for (; j < nend; j++) {
            int laid = oaid;
            for (k = kstart; k < kend-3; k += 4) {
                // float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                // float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                // float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                // float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);
                // float32x4_t vai4 = vld1q_f32(A + (i+4)*lda + k);
                // float32x4_t vai5 = vld1q_f32(A + (i+5)*lda + k);
                // float32x4_t vai6 = vld1q_f32(A + (i+6)*lda + k);
                // float32x4_t vai7 = vld1q_f32(A + (i+7)*lda + k);

                for (int ii=0; ii<8; ii++) {
                    C[(i+ii) * ldc + j] += A[laid++] * B[lbid+k];
                    C[(i+ii) * ldc + j] += A[laid++] * B[lbid+k];
                    C[(i+ii) * ldc + j] += A[laid++] * B[lbid+k];
                    C[(i+ii) * ldc + j] += A[laid++] * B[lbid+k];
                }
            }
            for (; k < kend; k ++) {
                for (int ii=0; ii<8; ii++) {
                    C[(i+ii) * ldc + j] += A[laid++] * B[lbid+k];
                }
            }
            lbid += kend-kstart;
        }
    }
    //// Pack A的顺序
    // for (; i < mend; ++i) {
    //     for (k = kstart; k < kend; ++k) {
    //         nA[aid++] = A[i * lda + k];
    //     }
    // }
    for (; i < mend; ++i) {
        int oaid = i * (kend - kstart);
        int lbid = bid;
        for (j = nstart; j < nend; j++) {
            int laid = oaid;
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[laid++] * B[lbid++];
            }
        }
    }
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
/// V20 B矩阵转置版本，分block后的原始内层实现
void UKernelV20(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; i++) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// v21 基本版neno实现
void UKernelV21(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend; i ++) {
        for (j = nstart; j < nend - 3; j += 4) {
            for (k = kstart; k < kend - 3; k += 4) {
                // C[i * ldc + (j+0)] += A[i * lda + k+0] * B[(j+0) * ldb + k+0];
                // C[i * ldc + (j+0)] += A[i * lda + k+1] * B[(j+0) * ldb + k+1];
                // C[i * ldc + (j+0)] += A[i * lda + k+2] * B[(j+0) * ldb + k+2];
                // C[i * ldc + (j+0)] += A[i * lda + k+3] * B[(j+0) * ldb + k+3];

                // C[i * ldc + (j+1)] += A[i * lda + k+0] * B[(j+1) * ldb + k+0];
                // C[i * ldc + (j+1)] += A[i * lda + k+1] * B[(j+1) * ldb + k+1];
                // C[i * ldc + (j+1)] += A[i * lda + k+2] * B[(j+1) * ldb + k+2];
                // C[i * ldc + (j+1)] += A[i * lda + k+3] * B[(j+1) * ldb + k+3];

                // C[i * ldc + (j+2)] += A[i * lda + k+0] * B[(j+2) * ldb + k+0];
                // C[i * ldc + (j+2)] += A[i * lda + k+1] * B[(j+2) * ldb + k+1];
                // C[i * ldc + (j+2)] += A[i * lda + k+2] * B[(j+2) * ldb + k+2];
                // C[i * ldc + (j+2)] += A[i * lda + k+3] * B[(j+2) * ldb + k+3];

                // C[i * ldc + (j+3)] += A[i * lda + k+0] * B[(j+3) * ldb + k+0];
                // C[i * ldc + (j+3)] += A[i * lda + k+1] * B[(j+3) * ldb + k+1];
                // C[i * ldc + (j+3)] += A[i * lda + k+2] * B[(j+3) * ldb + k+2];
                // C[i * ldc + (j+3)] += A[i * lda + k+3] * B[(j+3) * ldb + k+3];

                float32x4_t va0 = vld1q_f32(A + i*lda + k);
                
                float32x4_t vb0j0 = vld1q_f32(B + (j+0) * ldb + k);
                float32x4_t vb1j0 = vld1q_f32(B + (j+1) * ldb + k);
                float32x4_t vb2j0 = vld1q_f32(B + (j+2) * ldb + k);
                float32x4_t vb3j0 = vld1q_f32(B + (j+3) * ldb + k);

                float32x4_t va0b0 = vmulq_f32(va0, vb0j0);
                float32x4_t va0b1 = vmulq_f32(va0, vb1j0);
                float32x4_t va0b2 = vmulq_f32(va0, vb2j0);
                float32x4_t va0b3 = vmulq_f32(va0, vb3j0);

                C[i*ldc + j+0] += vgetq_lane_f32(va0b0, 0) + vgetq_lane_f32(va0b0, 1) + vgetq_lane_f32(va0b0, 2) + vgetq_lane_f32(va0b0, 3);
                C[i*ldc + j+1] += vgetq_lane_f32(va0b1, 0) + vgetq_lane_f32(va0b1, 1) + vgetq_lane_f32(va0b1, 2) + vgetq_lane_f32(va0b1, 3);
                C[i*ldc + j+2] += vgetq_lane_f32(va0b2, 0) + vgetq_lane_f32(va0b2, 1) + vgetq_lane_f32(va0b2, 2) + vgetq_lane_f32(va0b2, 3);
                C[i*ldc + j+3] += vgetq_lane_f32(va0b3, 0) + vgetq_lane_f32(va0b3, 1) + vgetq_lane_f32(va0b3, 2) + vgetq_lane_f32(va0b3, 3);
            }
            for (; k < kend; ++k) {
                for (int jj=0; jj<4; jj++) {
                    C[i * ldc + j+jj] += A[i * lda + k] * B[(j+jj) * ldb + k];                    
                }
            }
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// 基于V21优化
// 1. 将k进一步展开，可多一个向量加操作。
// 2. 使用vuzpq_f32将结果转置，可少执行多次vgetq_lane_f32
//    vuzpq_f32: 参考matrix_transpose
// 下一步: C[0]=A[i]*B[i]需要额外的转置操作很费时，
//         回到V10中使用vfmaq_laneq_f32 C[i] = C[i] + B[i] * A[0], 可以省掉转置操作vuzpq_f32，需要在pack时将B转置回来。
// 得到 GemmTilePackTBL2UKernelV10，耗时得到降低。回到普通矩阵乘法优化上，基于UKernelV10继续优化。
void UKernelV22(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; i ++) {
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
            for (k = kstart; k < kend - 7; k += 8) {
                float32x4_t va0 = vld1q_f32(A + i*lda + k);
                float32x4_t va1 = vld1q_f32(A + i*lda + k+4);

                float32x4_t vb0j0 = vld1q_f32(B + j * ldb + k);
                float32x4_t vb1j0 = vld1q_f32(B + (j+1) * ldb + k);
                float32x4_t vb2j0 = vld1q_f32(B + (j+2) * ldb + k);
                float32x4_t vb3j0 = vld1q_f32(B + (j+3) * ldb + k);

                float32x4_t vb0j1 = vld1q_f32(B + j * ldb + k+4);
                float32x4_t vb1j1 = vld1q_f32(B + (j+1) * ldb + k+4);
                float32x4_t vb2j1 = vld1q_f32(B + (j+2) * ldb + k+4);
                float32x4_t vb3j1 = vld1q_f32(B + (j+3) * ldb + k+4);

                float32x4_t va0b0 = vmulq_f32(va0, vb0j0);
                float32x4_t va0b1 = vmulq_f32(va0, vb1j0);
                float32x4_t va0b2 = vmulq_f32(va0, vb2j0);
                float32x4_t va0b3 = vmulq_f32(va0, vb3j0);

                float32x4_t vab0 = vfmaq_f32(va0b0, va1, vb0j1);
                float32x4_t vab1 = vfmaq_f32(va0b1, va1, vb1j1);
                float32x4_t vab2 = vfmaq_f32(va0b2, va1, vb2j1);
                float32x4_t vab3 = vfmaq_f32(va0b3, va1, vb3j1);
                
                // 这四行使用下面8行代替
                // C[i*ldc + j+0] += vgetq_lane_f32(vab0, 0) + vgetq_lane_f32(vab0, 1) + vgetq_lane_f32(vab0, 2) + vgetq_lane_f32(vab0, 3);
                // C[i*ldc + j+1] += vgetq_lane_f32(vab1, 0) + vgetq_lane_f32(vab1, 1) + vgetq_lane_f32(vab1, 2) + vgetq_lane_f32(vab1, 3);
                // C[i*ldc + j+2] += vgetq_lane_f32(vab2, 0) + vgetq_lane_f32(vab2, 1) + vgetq_lane_f32(vab2, 2) + vgetq_lane_f32(vab2, 3);
                // C[i*ldc + j+3] += vgetq_lane_f32(vab3, 0) + vgetq_lane_f32(vab3, 1) + vgetq_lane_f32(vab3, 2) + vgetq_lane_f32(vab3, 3);

                float32x4x2_t t0 = vuzpq_f32(vab0, vab1);
                float32x4x2_t t1 = vuzpq_f32(vab2, vab3);
                float32x4x2_t s0 = vuzpq_f32(t0.val[0], t1.val[0]);
                float32x4x2_t s1 = vuzpq_f32(t0.val[1], t1.val[1]);

                float32x4_t sum0 = vaddq_f32(s0.val[0], s1.val[0]);
                float32x4_t sum1 = vaddq_f32(s0.val[1], s1.val[1]);
                float32x4_t sum2 = vaddq_f32(sum0, sum1);

                vc0 = vaddq_f32(vc0, sum2);
            }
            vst1q_f32(C + i * ldc + j, vc0);
            for (; k < kend; ++k) {
                for (int jj=0; jj<4; jj++) {
                    C[i * ldc + j+jj] += A[i * lda + k] * B[(j+jj) * ldb + k];                    
                }
            }
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// 对原本转置的B矩阵，转置回来变成正常的非转置gemm，回归到UKernelV10的实现。
void PackTB2B(const bool is_transposed_b, const int T, const int N, const int K, const float *B, const int ldb, float *nB, int *nldb) {

    // 如果是B本身转置过的, 将其转置回去, 正常的非转置gemm
    if (is_transposed_b) {
        *nldb = N;
        for (size_t j = 0; j < N; j ++) {
            for (size_t k = 0; k < K; k ++) {
                nB[k*(*nldb)+j] = B[j*ldb+k]; // *(B++) = src[k*ldb+j];
            }
        }
    }
    else {
        *nldb = N;
        memcpy(nB, B, sizeof(float) * N * K);
    }
}

// 在PackTB2B的基础上使B访存连续
// 改写流程，将外层循环拷贝至此，去掉i层循环，将j层循环改为oj，
// ukernel部分直接将UKernelV10拷贝至此，去掉i循环，将B矩阵的访问用一个局部变量进行递增索引。
void PackTB2BC(const bool is_transposed_b, const int T, const int N, const int K, const float *B, const int ldb, float *nB, int *nldb) {

    float *tB = nullptr;
    *nldb = N;
    // 如果是B本身转置过的, 将其转置回去, 变为正常gemm
    if (is_transposed_b) {
        tB = new float[N*K];
        for (size_t j = 0; j < N; j ++) {
            for (size_t k = 0; k < K; k ++) {
                tB[k*(*nldb)+j] = B[j*ldb+k]; // *(B++) = src[k*ldb+j];
            }
        }
    }
    else {
        tB = (float *)B;
    }

    int bid = 0;
    for (int oj = 0; oj < N; oj += T) {
        // ukernel(oi, std::min(i + T, M),
        //         oj, std::min(j + T, N),
        //         0, K,
        //         A, lda, nB, bid, C, ldc);
        int nstart = oj;
        int nend = std::min(oj + T, N);
        int kstart = 0;
        int kend = K;

        int j, k;
        for (j = nstart; j < nend - 7; j += 8) {
            for (k = kstart; k < kend - 3; k += 4) {
                //// 计算访问顺序
                // float32x4_t vb0k0 = vld1q_f32(B + (k + 0) * ldb + j);
                // float32x4_t vb1k0 = vld1q_f32(B + (k + 0) * ldb + j + 4);
                // float32x4_t vb0k1 = vld1q_f32(B + (k + 1) * ldb + j);
                // float32x4_t vb1k1 = vld1q_f32(B + (k + 1) * ldb + j + 4);
                // float32x4_t vb0k2 = vld1q_f32(B + (k + 2) * ldb + j);
                // float32x4_t vb1k2 = vld1q_f32(B + (k + 2) * ldb + j + 4);
                // float32x4_t vb0k3 = vld1q_f32(B + (k + 3) * ldb + j);
                // float32x4_t vb1k3 = vld1q_f32(B + (k + 3) * ldb + j + 4);
                // // 普通实现
                // for (int r=0; r<8; r++)  nB[bid++] = tB[(k+0) * (*nldb) + j+r];
                // for (int r=0; r<8; r++)  nB[bid++] = tB[(k+1) * (*nldb) + j+r];
                // for (int r=0; r<8; r++)  nB[bid++] = tB[(k+2) * (*nldb) + j+r];
                // for (int r=0; r<8; r++)  nB[bid++] = tB[(k+3) * (*nldb) + j+r];
                //// 改用memcpy，耗时差不多
                memcpy(nB + bid, &tB[(k + 0) * (*nldb) + j], 32); // 32 = sizeof(float) * 8
                memcpy(nB + bid + 8, &tB[(k + 1) * (*nldb) + j], 32);
                memcpy(nB + bid + 16, &tB[(k + 2) * (*nldb) + j], 32);
                memcpy(nB + bid + 24, &tB[(k + 3) * (*nldb) + j], 32);
                bid += 32;
            }
            for (; k < kend; k++) {
                for (int jj = 0; jj < 8; jj++) {
                    nB[bid++] = tB[k * (*nldb) + (j + jj)];
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                nB[bid++] = tB[k * (*nldb) + j];
            }
        }
    }

    if (is_transposed_b) {
        delete[] tB;
    }
}

// is_transposed_b 表示B矩阵是否已经被转置了
typedef void (*UPackBFunc)(const bool is_transposed_b, const int T, const int N, const int K, 
                           const float *B, const int ldb, float *nB, int *nldb);

void GemmTilePackTBL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, 
                  UPackBFunc upack, bool is_transposed_b, UKernelFunc ukernel, bool is_b_continuous) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
    int nldb;
    float *nB = (float *)malloc(N * K * sizeof(float));
    upack(is_transposed_b, T, N, K, B, ldb, nB, &nldb);

    // B矩阵根据访问顺序做调整，使其在连续内存上连续访问，此时需要将b的当前访问下标传递进去
    if (is_b_continuous) {
        int bid = 0;
        for (i = 0; i < M; i += T) {
            for (j = 0; j < N; j += T) {
                bid = j * K;                  // 观察PackTB2BC函数中，oj层+1，bid会+K.
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        0, K,
                        A, lda, nB, bid, C, ldc);
            }
        }
    }
    else {
        for (i = 0; i < M; i += T) {
            for (j = 0; j < N; j += T) {
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        0, K,
                        A, lda, nB, nldb, C, ldc);
            }
        }        
    }
    free(nB);
}

void PackA(const int T, const int M, const int K, const float *A, float *nA) {
    int aid = 0;
    for (int oi = 0; oi < M; oi += T) {
        // ukernel(oi, std::min(i + T, M),
        //         oj, std::min(j + T, N),
        //         0, K,
        //         A, lda, nB, bid, C, ldc);
        int mstart = oi;
        int mend = std::min(oi + T, M);
        int kstart = 0;
        int kend = K;
        int lda = K;

        int i, k;
        for (i = mstart; i < mend-7; i += 8) {
            for (k = kstart; k < kend -3; k += 4) {
                // float32x4_t vai0 = vld1q_f32(A + (i+0)*lda + k);
                // float32x4_t vai1 = vld1q_f32(A + (i+1)*lda + k);
                // float32x4_t vai2 = vld1q_f32(A + (i+2)*lda + k);
                // float32x4_t vai3 = vld1q_f32(A + (i+3)*lda + k);
                // float32x4_t vai4 = vld1q_f32(A + (i+4)*lda + k);
                // float32x4_t vai5 = vld1q_f32(A + (i+5)*lda + k);
                // float32x4_t vai6 = vld1q_f32(A + (i+6)*lda + k);
                // float32x4_t vai7 = vld1q_f32(A + (i+7)*lda + k);
                memcpy(nA + aid,      &A[(i+0)*lda + k], 16); // 16 = sizeof(float) * 4
                memcpy(nA + aid + 4,  &A[(i+1)*lda + k], 16);
                memcpy(nA + aid + 8,  &A[(i+2)*lda + k], 16);
                memcpy(nA + aid + 12, &A[(i+3)*lda + k], 16);
                memcpy(nA + aid + 16, &A[(i+4)*lda + k], 16);
                memcpy(nA + aid + 20, &A[(i+5)*lda + k], 16);
                memcpy(nA + aid + 24, &A[(i+6)*lda + k], 16);
                memcpy(nA + aid + 28, &A[(i+7)*lda + k], 16);
                aid += 32;
            }
            for (; k < kend; k++) {
                for (int ii=0; ii<8; ii++) {
                    nA[aid++] = A[(i+ii) * lda + k];
                }
            }
        }
        for (; i < mend; ++i) {
            for (k = kstart; k < kend; ++k) {
                nA[aid++] = A[i * lda + k];
            }
        }
    }
}


void GemmTilePackATBL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, 
                  UPackBFunc upack, bool is_transposed_b, UKernelFunc ukernel, bool is_b_continuous) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
    int nldb;
    float *nB = (float *)malloc(N * K * sizeof(float));
    upack(is_transposed_b, T, N, K, B, ldb, nB, &nldb);

    float *nA = (float *)malloc(M * K * sizeof(float));
    PackA(T, M, K, A, nA);
    // B矩阵根据访问顺序做调整，使其在连续内存上连续访问，此时需要将b的当前访问下标传递进去
    if (is_b_continuous) {
        int bid = 0;
        int aid = 0;
        for (i = 0; i < M; i += T) {
            aid = i * K;                  // i+1时，A开始访问下一行
            for (j = 0; j < N; j += T) {
                bid = j * K;              // j+1时，B开始访问下一行
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        0, K,
                        nA, aid, nB, bid, C, ldc);
            }
        }
    }
    else {
        for (i = 0; i < M; i += T) {
            for (j = 0; j < N; j += T) {
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        0, K,
                        nA, lda, nB, nldb, C, ldc);
            }
        }
    }
    free(nB);
}

// 对B矩阵进行专置操作
void NormalTranspose(const int N, const int K, const float *B, const int ldb, float *nB, int *nldb) {
    *nldb = K;
    for (size_t j = 0; j < N; j ++) {
        for (size_t k = 0; k < K; k ++) {
            nB[j*(*nldb)+k] = B[k*ldb+j]; // *(B++) = src[k*ldb+j];
        }
    }
}

// 尝试使ukernel内的数据在L2中得到覆盖
// cpu越弱，分块可能小一点好，使cache能覆盖住，但不是绝对，需要尝试
// 如果K太大了，需要对K分块。
void GemmTileL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 16;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    A, lda, B, ldb, C, ldc);
        }
    }
}



/// 基于V25，观察到C的累加也不需要赋零，把k的第一次计算拆分出来即可
/// 无效果，编译后的汇编代码差不多
void UKernelPBV26(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        float *a0 = (float *)A + i * lda;
        float *a1 = (float *)A + (i+1) * lda;
        float *a2 = (float *)A + (i+2) * lda;
        float *a3 = (float *)A + (i+3) * lda;
        float *a4 = (float *)A + (i+4) * lda;
        float *a5 = (float *)A + (i+5) * lda;
        float *a6 = (float *)A + (i+6) * lda;
        float *a7 = (float *)A + (i+7) * lda;

        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t va0 = vld1q_f32(a0 + kstart);
            float32x4_t va1 = vld1q_f32(a1 + kstart);
            float32x4_t va2 = vld1q_f32(a2 + kstart);
            float32x4_t va3 = vld1q_f32(a3 + kstart);
            float32x4_t va4 = vld1q_f32(a4 + kstart);
            float32x4_t va5 = vld1q_f32(a5 + kstart);
            float32x4_t va6 = vld1q_f32(a6 + kstart);
            float32x4_t va7 = vld1q_f32(a7 + kstart);

            float32x4_t vb0j0 = vld1q_f32(B + lbid);
            float32x4_t vb1j0 = vld1q_f32(B + lbid + 4);
            float32x4_t vb2j0 = vld1q_f32(B + lbid + 8);
            float32x4_t vb3j0 = vld1q_f32(B + lbid + 12);
            lbid += 16;

            float32x4_t vc0 = vmulq_laneq_f32(vb0j0, va0, 0);
            vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
            vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
            vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

            float32x4_t vc1 = vmulq_laneq_f32(vb0j0, va1, 0);
            vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
            vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
            vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

            float32x4_t vc2 = vmulq_laneq_f32(vb0j0, va2, 0);
            vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
            vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
            vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

            float32x4_t vc3 = vmulq_laneq_f32(vb0j0, va3, 0);
            vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
            vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
            vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

            float32x4_t vc4 = vmulq_laneq_f32(vb0j0, va4, 0);
            vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
            vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
            vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

            float32x4_t vc5 = vmulq_laneq_f32(vb0j0, va5, 0);
            vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
            vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
            vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

            float32x4_t vc6 = vmulq_laneq_f32(vb0j0, va6, 0);
            vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
            vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
            vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

            float32x4_t vc7 = vmulq_laneq_f32(vb0j0, va7, 0);
            vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
            vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
            vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);

            for (k = kstart + 4; k < kend - 3; k += 4) {
                va0 = vld1q_f32(a0 + k);
                va1 = vld1q_f32(a1 + k);
                va2 = vld1q_f32(a2 + k);
                va3 = vld1q_f32(a3 + k);
                va4 = vld1q_f32(a4 + k);
                va5 = vld1q_f32(a5 + k);
                va6 = vld1q_f32(a6 + k);
                va7 = vld1q_f32(a7 + k);
                
                vb0j0 = vld1q_f32(B + lbid); 
                vb1j0 = vld1q_f32(B + lbid + 4);
                vb2j0 = vld1q_f32(B + lbid + 8);
                vb3j0 = vld1q_f32(B + lbid + 12);
                lbid += 16;

                vc0 = vfmaq_laneq_f32(vc0, vb0j0, va0, 0);
                vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
                vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
                vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

                vc1 = vfmaq_laneq_f32(vc1, vb0j0, va1, 0);
                vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
                vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
                vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

                vc2 = vfmaq_laneq_f32(vc2, vb0j0, va2, 0);
                vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
                vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
                vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

                vc3 = vfmaq_laneq_f32(vc3, vb0j0, va3, 0);
                vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
                vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
                vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

                vc4 = vfmaq_laneq_f32(vc4, vb0j0, va4, 0);
                vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
                vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
                vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

                vc5 = vfmaq_laneq_f32(vc5, vb0j0, va5, 0);
                vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
                vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
                vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

                vc6 = vfmaq_laneq_f32(vc6, vb0j0, va6, 0);
                vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
                vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
                vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

                vc7 = vfmaq_laneq_f32(vc7, vb0j0, va7, 0);
                vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
                vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
                vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);
            }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
    }
}

// 基于V25和V25Asm，发现ldr扎堆在一起
// 查A55优化手册，ldr的q格式的加载无法进行双发射，只有d格式的才行。
// 而且ldr最大吞吐量是1，所以可以用fmla跟ldr进行组合双发射，以隐藏ldr的耗时。
//
// https://zhuanlan.zhihu.com/p/517371998
// ldr q0, [%[b_ptr]] 是 从ptr加载16B到v0寄存器
// // 优化后，ldr与fmla一起双发射
// ldr d0, [%[b_ptr]] 是从b_ptr加载8B到v0寄存器的低8B
// ldr x0, [%[b_ptr], #8] 是从b_ptr+8加载8B到x0寄存器
// ins v0.d[1], x0 是从x0寄存器加载8B到v0寄存器的高8B
//
// 但看V25生成的汇编代码，观察其内层循环AB矩阵数据的加载为：
// ldp	q21, q20, [x14, #-32]
// ldr	q22, [x28, x7]
// 如x28需要从固定的改为浮动，访问后直接自身递增偏移，而不再需要x7这样的中间变量做偏移。
// 如首部将B付给pB指针，随后pB自身做偏移； 同理A也一样。
// 这样就可以如 ldr x0, [x28, #8]那样取8B，否则则需要[x28, x7]偏移后，又在偏移8B.

// 根据上面内容，尝试
// 1. 将vld1q_f32改成vld1_f32，即将Q格式的ldr改为D格式.随之用vfmaq_lane_f32代替vfmaq_laneq_f32
// 2. 按循环内先用后取（取下一次）的方式，将乘加和ldr在C++的层面上进行组合。
// 未起到优化作用！！
void UKernelPBV27(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {
                    
    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        float *a0 = (float *)A + i * lda;
        float *a1 = (float *)A + (i+1) * lda;
        float *a2 = (float *)A + (i+2) * lda;
        float *a3 = (float *)A + (i+3) * lda;
        float *a4 = (float *)A + (i+4) * lda;
        float *a5 = (float *)A + (i+5) * lda;
        float *a6 = (float *)A + (i+6) * lda;
        float *a7 = (float *)A + (i+7) * lda;

        // int lbid = bid;
        float *pB = (float *)B + bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;

            float32x2_t va00 = vld1_f32(a0 + kstart);
            float32x2_t va01 = vld1_f32(a0 + kstart + 2);
            float32x2_t va10 = vld1_f32(a1 + kstart);
            float32x2_t va11 = vld1_f32(a1 + kstart + 2);
            float32x2_t va20 = vld1_f32(a2 + kstart);
            float32x2_t va21 = vld1_f32(a2 + kstart + 2);
            float32x2_t va30 = vld1_f32(a3 + kstart); 
            float32x2_t va31 = vld1_f32(a3 + kstart + 2); 
            float32x2_t va40 = vld1_f32(a4 + kstart);
            float32x2_t va41 = vld1_f32(a4 + kstart + 2);
            float32x2_t va50 = vld1_f32(a5 + kstart);
            float32x2_t va51 = vld1_f32(a5 + kstart + 2);
            float32x2_t va60 = vld1_f32(a6 + kstart);
            float32x2_t va61 = vld1_f32(a6 + kstart + 2);
            float32x2_t va70 = vld1_f32(a7 + kstart);
            float32x2_t va71 = vld1_f32(a7 + kstart + 2);

            float32x4_t vb0j0 = vld1q_f32(pB);
            float32x4_t vb1j0 = vld1q_f32(pB + 4);
            float32x4_t vb2j0 = vld1q_f32(pB + 8);
            float32x4_t vb3j0 = vld1q_f32(pB + 12);
            pB += 16;

            for (k = kstart + 4; k < kend - 3; k += 4) {

                vc0 = vfmaq_lane_f32(vc0, vb0j0, va00, 0); // vfmaq_lane_f32中lane用32x2，而vfmaq_laneq_f32用32x4
                vc1 = vfmaq_lane_f32(vc1, vb0j0, va10, 0);              
                vc0 = vfmaq_lane_f32(vc0, vb1j0, va00, 1);
                va00 = vld1_f32(a0 + k);
                vc0 = vfmaq_lane_f32(vc0, vb2j0, va01, 0);
                vc0 = vfmaq_lane_f32(vc0, vb3j0, va01, 1);
                va01 = vld1_f32(a0 + k + 2);
                

                vc1 = vfmaq_lane_f32(vc1, vb1j0, va10, 1);
                va10 = vld1_f32(a1 + k);
                vc1 = vfmaq_lane_f32(vc1, vb2j0, va11, 0);
                vc1 = vfmaq_lane_f32(vc1, vb3j0, va11, 1);
                va11 = vld1_f32(a1 + k + 2);

                vc2 = vfmaq_lane_f32(vc2, vb0j0, va20, 0);
                vc2 = vfmaq_lane_f32(vc2, vb1j0, va20, 1);
                va20 = vld1_f32(a2 + k);
                vc2 = vfmaq_lane_f32(vc2, vb2j0, va21, 0);
                vc2 = vfmaq_lane_f32(vc2, vb3j0, va21, 1);
                va21 = vld1_f32(a2 + k + 2);
                
                vc3 = vfmaq_lane_f32(vc3, vb0j0, va30, 0);
                vc3 = vfmaq_lane_f32(vc3, vb1j0, va30, 1);
                va30 = vld1_f32(a3 + k);
                vc3 = vfmaq_lane_f32(vc3, vb2j0, va31, 0);
                vc3 = vfmaq_lane_f32(vc3, vb3j0, va31, 1);
                va31 = vld1_f32(a3 + k + 2);

                vc4 = vfmaq_lane_f32(vc4, vb0j0, va40, 0);
                vc4 = vfmaq_lane_f32(vc4, vb1j0, va40, 1);
                va40 = vld1_f32(a4 + k);                
                vc4 = vfmaq_lane_f32(vc4, vb2j0, va41, 0);
                vc4 = vfmaq_lane_f32(vc4, vb3j0, va41, 1);
                va41 = vld1_f32(a4 + k + 2);

                vc5 = vfmaq_lane_f32(vc5, vb0j0, va50, 0);
                vc5 = vfmaq_lane_f32(vc5, vb1j0, va50, 1);
                va50 = vld1_f32(a5 + k);                
                vc5 = vfmaq_lane_f32(vc5, vb2j0, va51, 0);
                vc5 = vfmaq_lane_f32(vc5, vb3j0, va51, 1);
                va51 = vld1_f32(a5 + k + 2);

                vc6 = vfmaq_lane_f32(vc6, vb0j0, va60, 0);
                vc6 = vfmaq_lane_f32(vc6, vb1j0, va60, 1);
                va60 = vld1_f32(a6 + k);                
                vc6 = vfmaq_lane_f32(vc6, vb2j0, va61, 0);
                vc6 = vfmaq_lane_f32(vc6, vb3j0, va61, 1);
                va61 = vld1_f32(a6 + k + 2);

                vc7 = vfmaq_lane_f32(vc7, vb0j0, va70, 0);
                vc7 = vfmaq_lane_f32(vc7, vb1j0, va70, 1);
                va70 = vld1_f32(a7 + k);                
                vc7 = vfmaq_lane_f32(vc7, vb2j0, va71, 0);
                vc7 = vfmaq_lane_f32(vc7, vb3j0, va71, 1);
                va71 = vld1_f32(a7 + k + 2);

                vb0j0 = vld1q_f32(pB); 
                vb1j0 = vld1q_f32(pB + 4);
                vb2j0 = vld1q_f32(pB + 8);
                vb3j0 = vld1q_f32(pB + 12);
                pB += 16;
            }
            // 最后一次不需要加载
            {
                vc0 = vfmaq_lane_f32(vc0, vb0j0, va00, 0);
                vc0 = vfmaq_lane_f32(vc0, vb1j0, va00, 1);
                vc0 = vfmaq_lane_f32(vc0, vb2j0, va01, 0);
                vc0 = vfmaq_lane_f32(vc0, vb3j0, va01, 1);

                vc1 = vfmaq_lane_f32(vc1, vb0j0, va10, 0);
                vc1 = vfmaq_lane_f32(vc1, vb1j0, va10, 1);
                vc1 = vfmaq_lane_f32(vc1, vb2j0, va11, 0);
                vc1 = vfmaq_lane_f32(vc1, vb3j0, va11, 1);

                vc2 = vfmaq_lane_f32(vc2, vb0j0, va20, 0);
                vc2 = vfmaq_lane_f32(vc2, vb1j0, va20, 1);
                vc2 = vfmaq_lane_f32(vc2, vb2j0, va21, 0);
                vc2 = vfmaq_lane_f32(vc2, vb3j0, va21, 1);

                vc3 = vfmaq_lane_f32(vc3, vb0j0, va30, 0);
                vc3 = vfmaq_lane_f32(vc3, vb1j0, va30, 1);
                vc3 = vfmaq_lane_f32(vc3, vb2j0, va31, 0);
                vc3 = vfmaq_lane_f32(vc3, vb3j0, va31, 1);

                vc4 = vfmaq_lane_f32(vc4, vb0j0, va40, 0);
                vc4 = vfmaq_lane_f32(vc4, vb1j0, va40, 1);
                vc4 = vfmaq_lane_f32(vc4, vb2j0, va41, 0);
                vc4 = vfmaq_lane_f32(vc4, vb3j0, va41, 1);

                vc5 = vfmaq_lane_f32(vc5, vb0j0, va50, 0);
                vc5 = vfmaq_lane_f32(vc5, vb1j0, va50, 1);
                vc5 = vfmaq_lane_f32(vc5, vb2j0, va51, 0);
                vc5 = vfmaq_lane_f32(vc5, vb3j0, va51, 1);

                vc6 = vfmaq_lane_f32(vc6, vb0j0, va60, 0);
                vc6 = vfmaq_lane_f32(vc6, vb1j0, va60, 1);
                vc6 = vfmaq_lane_f32(vc6, vb2j0, va61, 0);
                vc6 = vfmaq_lane_f32(vc6, vb3j0, va61, 1);

                vc7 = vfmaq_lane_f32(vc7, vb0j0, va70, 0);
                vc7 = vfmaq_lane_f32(vc7, vb1j0, va70, 1);
                vc7 = vfmaq_lane_f32(vc7, vb2j0, va71, 0);
                vc7 = vfmaq_lane_f32(vc7, vb3j0, va71, 1);
            }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
    }
}

// V25 内嵌汇编 直接翻译版
// https://blog.alex.balgavy.eu/a-practical-guide-to-gcc-inline-assembly/
void UKernelPBV25MixAsm(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {
                    
    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        float *a0 = (float *)A + i * lda;
        float *a1 = (float *)A + (i+1) * lda;
        float *a2 = (float *)A + (i+2) * lda;
        float *a3 = (float *)A + (i+3) * lda;
        float *a4 = (float *)A + (i+4) * lda;
        float *a5 = (float *)A + (i+5) * lda;
        float *a6 = (float *)A + (i+6) * lda;
        float *a7 = (float *)A + (i+7) * lda;

        // int lbid = bid; 
        float *pB = (float *)B + bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;

            for (k = kstart; k < kend - 3; k += 4) {

                float *pA0 = a0 + k;
                float *pA1 = a1 + k;
                float *pA2 = a2 + k;
                float *pA3 = a3 + k;
                float *pA4 = a4 + k;
                float *pA5 = a5 + k;
                float *pA6 = a6 + k;
                float *pA7 = a7 + k;

                asm volatile(
                    // "prfm   pldl1keep, [%0, #512]   \n"
                    "ldr    q0, [%18] \n" // va0
                    "ldr    q1, [%19] \n"
                    "ldr    q2, [%20] \n"
                    "ldr    q3, [%21] \n"
                    "ldr    q4, [%22] \n"
                    "ldr    q5, [%23] \n"
                    "ldr    q6, [%24] \n"
                    "ldr    q7, [%25] \n"
                    "ldp	q8, q9, [%8], #32 \n" // q8 = vb0j0, q9 = vb1j0
                    "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0
                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%0.4s, v9.4s, v0.s[1] \n"
                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%0.4s, v11.4s, v0.s[3] \n"

                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"

                    "fmla	%3.4s, v8.4s, v3.s[0] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"

                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%4.4s, v10.4s, v4.s[2] \n"
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"

                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"

                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"

                    "fmla	%7.4s, v8.4s, v7.s[0] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pB),                   
                      "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3),  // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                      "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 22/23/24/25
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");

                // float32x4_t va0 = vld1q_f32(a0 + k);
                // float32x4_t va1 = vld1q_f32(a1 + k);
                // float32x4_t va2 = vld1q_f32(a2 + k);
                // float32x4_t va3 = vld1q_f32(a3 + k);
                // float32x4_t va4 = vld1q_f32(a4 + k);
                // float32x4_t va5 = vld1q_f32(a5 + k);
                // float32x4_t va6 = vld1q_f32(a6 + k);
                // float32x4_t va7 = vld1q_f32(a7 + k);
                
                // float32x4_t vb0j0 = vld1q_f32(pB); 
                // float32x4_t vb1j0 = vld1q_f32(pB + 4);
                // float32x4_t vb2j0 = vld1q_f32(pB + 8);
                // float32x4_t vb3j0 = vld1q_f32(pB + 12);
                // pB += 16;

                // vc0 = vfmaq_laneq_f32(vc0, vb0j0, va0, 0);
                // vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
                // vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
                // vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

                // vc1 = vfmaq_laneq_f32(vc1, vb0j0, va1, 0);
                // vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
                // vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
                // vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

                // vc2 = vfmaq_laneq_f32(vc2, vb0j0, va2, 0);
                // vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
                // vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
                // vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

                // vc3 = vfmaq_laneq_f32(vc3, vb0j0, va3, 0);
                // vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
                // vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
                // vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

                // vc4 = vfmaq_laneq_f32(vc4, vb0j0, va4, 0);
                // vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
                // vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
                // vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

                // vc5 = vfmaq_laneq_f32(vc5, vb0j0, va5, 0);
                // vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
                // vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
                // vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

                // vc6 = vfmaq_laneq_f32(vc6, vb0j0, va6, 0);
                // vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
                // vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
                // vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

                // vc7 = vfmaq_laneq_f32(vc7, vb0j0, va7, 0);
                // vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
                // vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
                // vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);
            }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
    }
}

// 基于V25MixAsm，手动调流水
// 1. 拆分%0/1/2/3的累计的写后读依赖；
// 2. 将内层循环拆分成前中后三段，前段只加载，中段先计算后加载，后段只计算。
//    目的是把核心的中段部分的计算不依赖于数据加载，使数据加载可以穿插在计算中。
void UKernelPBV25MixAsmOpt(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {
                    
    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        float *a0 = (float *)A + i * lda;
        float *a1 = (float *)A + (i+1) * lda;
        float *a2 = (float *)A + (i+2) * lda;
        float *a3 = (float *)A + (i+3) * lda;
        float *a4 = (float *)A + (i+4) * lda;
        float *a5 = (float *)A + (i+5) * lda;
        float *a6 = (float *)A + (i+6) * lda;
        float *a7 = (float *)A + (i+7) * lda;

        // int lbid = bid; 
        float *pB = (float *)B + bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;

            float *pA0 = a0 + kstart;
            float *pA1 = a1 + kstart;
            float *pA2 = a2 + kstart;
            float *pA3 = a3 + kstart;
            float *pA4 = a4 + kstart;
            float *pA5 = a5 + kstart;
            float *pA6 = a6 + kstart;
            float *pA7 = a7 + kstart;

            asm volatile(
                "ldr    q0, [%2] \n" // va0
                "ldr    q1, [%3] \n"
                "ldr    q2, [%4] \n"
                "ldr    q3, [%5] \n"
                "ldr    q4, [%6] \n"
                "ldr    q5, [%7] \n"
                "ldr    q6, [%8] \n"
                "ldr    q7, [%9] \n"
                "ldp	q8, q9, [%0], #32 \n"   // q8 = vb0j0, q9 = vb1j0
                "ldp	q10, q11, [%0], #32 \n" // q10 = vb2j0, q11 = vb3j0

                : "=r"(pB)
                : "0"(pB), "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3), // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                  "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 22/23/24/25
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
            
            // for (k = kstart + 4; k < kend - 3; k += 4) {
            for (k = kstart + 4; k < kend - 3; k += 4) {
                float *pA0 = a0 + k;
                float *pA1 = a1 + k;
                float *pA2 = a2 + k;
                float *pA3 = a3 + k;
                float *pA4 = a4 + k;
                float *pA5 = a5 + k;
                float *pA6 = a6 + k;
                float *pA7 = a7 + k;

                asm volatile(
                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%3.4s, v8.4s, v3.s[0] \n"
                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%7.4s, v8.4s, v7.s[0] \n"

                    "fmla	%0.4s, v9.4s, v0.s[1] \n"
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"

                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n"
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"
                    "fmla	%4.4s, v10.4s, v4.s[2] \n"      
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n"

                    "fmla	%0.4s, v11.4s, v0.s[3] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    // "prfm   pldl1keep, [%0, #512]   \n"
                    "ldr    q0, [%18] \n" // va0
                    "ldr    q1, [%19] \n"
                    "ldr    q2, [%20] \n"
                    "ldr    q3, [%21] \n"
                    "ldr    q4, [%22] \n"
                    "ldr    q5, [%23] \n"
                    "ldr    q6, [%24] \n"
                    "ldr    q7, [%25] \n"
                    "ldp	q8, q9, [%8], #32 \n" // q8 = vb0j0, q9 = vb1j0
                    "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0
                    // "ldr	q8, [%8] \n" // q8 = vb0j0, q9 = vb1j0
                    // "ldr	q9, [%8, #16] \n" 
                    // "ldr	q10, [%8, #32] \n" 
                    // "ldr	q11, [%8, #48] \n" 
                    // "add	%8, %8, #64 \n" 

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pB),                   
                      "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3),  // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                      "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 22/23/24/25
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
            }

            // 最后一次不需要加载
            {
                asm volatile(

                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%3.4s, v8.4s, v3.s[0] \n"
                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%7.4s, v8.4s, v7.s[0] \n"

                    "fmla	%0.4s, v9.4s, v0.s[1] \n"
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"

                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n"
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"
                    "fmla	%4.4s, v10.4s, v4.s[2] \n"      
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n"

                    "fmla	%0.4s, v11.4s, v0.s[3] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pB)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
            }


            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
    }
}

// 基于UKernelPBV25MixAsmOpt，将中段循环的ldr-Q进行拆分，得到ldr-d，于fmla进行组合
// 未起作用，原因在于手上的手机芯片的A53中，ldr-D和fmla无法形成双发射（在perf.cpp中实验出）
void UKernelPBV25MixAsmOptV2(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {
                    
    float32x4_t zero = vdupq_n_f32(0);

    int i, j, k, k1;
    for (i = mstart; i < mend - 7; i += 8) {
        float *a0 = (float *)A + i * lda;
        float *a1 = (float *)A + (i+1) * lda;
        float *a2 = (float *)A + (i+2) * lda;
        float *a3 = (float *)A + (i+3) * lda;
        float *a4 = (float *)A + (i+4) * lda - 4;
        float *a5 = (float *)A + (i+5) * lda - 4;
        float *a6 = (float *)A + (i+6) * lda - 4;
        float *a7 = (float *)A + (i+7) * lda - 4;

        // int lbid = bid; 
        float *pB = (float *)B + bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;

            float *pA0 = a0 + kstart;
            float *pA1 = a1 + kstart;
            float *pA2 = a2 + kstart;
            float *pA3 = a3 + kstart;
            float *pA4 = a4 + kstart;
            float *pA5 = a5 + kstart;
            float *pA6 = a6 + kstart;
            float *pA7 = a7 + kstart;

            asm volatile(
                "ldr    q0, [%2] \n" // va0
                "ldr    q1, [%3] \n"
                "ldr    q2, [%4] \n"
                "ldr    q3, [%5] \n"

                // "ldr    q4, [%6] \n"
                // "ldr    q5, [%7] \n"
                // "ldr    q6, [%8] \n"
                // "ldr    q7, [%9] \n"
                // "ldp	q8, q9, [%0], #32 \n"   // q8 = vb0j0, q9 = vb1j0
                // "ldp	q10, q11, [%0], #32 \n" // q10 = vb2j0, q11 = vb3j0

                : "=r"(pB)
                : "0"(pB), "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3) // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                : "cc", "memory", "v0", "v1", "v2", "v3");
            
            for (k = kstart + 4; k < kend - 3; k += 4) {
                float *pA0 = a0 + k;
                float *pA1 = a1 + k;
                float *pA2 = a2 + k;
                float *pA3 = a3 + k;
                float *pA4 = a4 + k;
                float *pA5 = a5 + k;
                float *pA6 = a6 + k;
                float *pA7 = a7 + k;

                asm volatile(
                    "ldp	q8, q9, [%8], #32 \n"   // q8 = vb0j0, q9 = vb1j0
                    "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0
                    "prfm	PLDL1KEEP, [%8, #640] \n"

                    "ldr    q4, [%22] \n"
                    // "ldr	d4, [%22] \n" 
                    // "ldr    x4, [%22, #8] \n" 
                    // "ins    v4.d[1], x4 \n" 

                    "ldr    q5, [%23] \n"
                    // "ldr	d5, [%23] \n" 
                    // "ldr    x5, [%23, #8] \n" 
                    // "ins    v5.d[1], x5 \n" 

                    "ldr    q6, [%24] \n"
                    // "ldr	d6, [%24] \n" 
                    // "ldr    x6, [%24, #8] \n" 
                    // "ins    v6.d[1], x6 \n"

                    "ldr    q7, [%25] \n"
                    // "ldr	d7, [%25] \n" 
                    // "ldr    x7, [%25, #8] \n" 
                    // "ins    v7.d[1], x7 \n" 

                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%3.4s, v8.4s, v3.s[0] \n"

                    "fmla	%0.4s, v9.4s, v0.s[1] \n"                    
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    
                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n" 
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"

                    "fmla	%0.4s, v11.4s, v0.s[3] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"

                    "ldr    q0, [%18] \n" // va0
                    // "ldr	d0, [%18] \n" 
                    // "ldr    x20, [%18, #8] \n" 
                    // "ins    v0.d[1], x20 \n" 

                    "ldr    q1, [%19] \n"
                    // "ldr	d1, [%19] \n" 
                    // "ldr    x21, [%19, #8] \n" 
                    // "ins    v1.d[1], x21 \n" 

                    "ldr    q2, [%20] \n"
                    // "ldr	d2, [%20] \n" 
                    // "ldr    x22, [%20, #8] \n" 
                    // "ins    v2.d[1], x22 \n" 

                    "ldr    q3, [%21] \n"
                    // "ldr	d3, [%21] \n" 
                    // "ldr    x23, [%21, #8] \n" 
                    // "ins    v3.d[1], x23 \n" 

                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%7.4s, v8.4s, v7.s[0] \n"

                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"                 

                    "fmla	%4.4s, v10.4s, v4.s[2] \n"
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n" 
                    
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    // "prfm   pldl1keep, [%0, #512]   \n"
                    // "ldr    q0, [%18] \n" // va0
                    // "ldr    q1, [%19] \n"
                    // "ldr    q2, [%20] \n"
                    // "ldr    q3, [%21] \n"
                    // "ldr    q4, [%22] \n"
                    // "ldr    q5, [%23] \n"
                    // "ldr    q6, [%24] \n"
                    // "ldr    q7, [%25] \n"

                    // "ldp	q8, q9, [%8], #32 \n" // q8 = vb0j0, q9 = vb1j0
                    // "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0

                    // "ldr	q8, [%8] \n" // q8 = vb0j0, q9 = vb1j0
                    // "ldr	q9, [%8, #16] \n" 
                    // "ldr	q10, [%8, #32] \n" 
                    // "ldr	q11, [%8, #48] \n" 
                    // "add	%8, %8, #64 \n" 

                    // "ldr	d8, [%8] \n" 
                    // "ldr x8, [%8, #8] \n" 
                    // "ins v8.d[1], x8 \n" 

                    // "ldr	d9, [%8, #16] \n" 
                    // "ldr x9, [%8, #24] \n" 
                    // "ins v9.d[1], x9 \n" 

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pB),                   
                      "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3),  // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                      "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 22/23/24/25
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                      "x20", "x21", "x22", "x23");

            //  asm volatile(
            //         "ldp	q8, q9, [%8], #32 \n"   // q8 = vb0j0, q9 = vb1j0
            //         "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0
            //         // "prfm	PLDL1KEEP, [%8, #640] \n"

            //         "ldr    q4, [%22] \n"
            //         "ldr    q5, [%23] \n"
            //         "ldr    q6, [%24] \n"
            //         "ldr    q7, [%25] \n"
            //         "fmla	%0.4s, v8.4s, v0.s[0] \n"
            //         "fmla	%1.4s, v8.4s, v1.s[0] \n"
            //         "fmla	%2.4s, v8.4s, v2.s[0] \n"
            //         "fmla	%3.4s, v8.4s, v3.s[0] \n"

            //         "fmla	%0.4s, v9.4s, v0.s[1] \n"
            //         "fmla	%1.4s, v9.4s, v1.s[1] \n"
            //         "fmla	%2.4s, v9.4s, v2.s[1] \n"
            //         "fmla	%3.4s, v9.4s, v3.s[1] \n"

            //         "fmla	%0.4s, v10.4s, v0.s[2] \n"
            //         "fmla	%1.4s, v10.4s, v1.s[2] \n" 
            //         "fmla	%2.4s, v10.4s, v2.s[2] \n"
            //         "fmla	%3.4s, v10.4s, v3.s[2] \n"

            //         "fmla	%0.4s, v11.4s, v0.s[3] \n"
            //         "fmla	%1.4s, v11.4s, v1.s[3] \n"
            //         "fmla	%2.4s, v11.4s, v2.s[3] \n"
            //         "fmla	%3.4s, v11.4s, v3.s[3] \n"


            //         "ldr    q0, [%18] \n" // va0
            //         "ldr    q1, [%19] \n"
            //         "ldr    q2, [%20] \n"
            //         "ldr    q3, [%21] \n"
            //         "fmla	%4.4s, v8.4s, v4.s[0] \n"
            //         "fmla	%5.4s, v8.4s, v5.s[0] \n"
            //         "fmla	%6.4s, v8.4s, v6.s[0] \n"
            //         "fmla	%7.4s, v8.4s, v7.s[0] \n"

            //         "fmla	%4.4s, v9.4s, v4.s[1] \n"
            //         "fmla	%5.4s, v9.4s, v5.s[1] \n"
            //         "fmla	%6.4s, v9.4s, v6.s[1] \n"
            //         "fmla	%7.4s, v9.4s, v7.s[1] \n"                 

            //         "fmla	%4.4s, v10.4s, v4.s[2] \n"
            //         "fmla	%5.4s, v10.4s, v5.s[2] \n"
            //         "fmla	%6.4s, v10.4s, v6.s[2] \n"
            //         "fmla	%7.4s, v10.4s, v7.s[2] \n" 
                    
            //         "fmla	%4.4s, v11.4s, v4.s[3] \n"
            //         "fmla	%5.4s, v11.4s, v5.s[3] \n"
            //         "fmla	%6.4s, v11.4s, v6.s[3] \n"
            //         "fmla	%7.4s, v11.4s, v7.s[3] \n"

            //         // "ldr    q0, [%18] \n" // va0
            //         // "ldr    q1, [%19] \n"
            //         // "ldr    q2, [%20] \n"
            //         // "ldr    q3, [%21] \n"
            //         // "ldr    q4, [%22] \n"
            //         // "ldr    q5, [%23] \n"
            //         // "ldr    q6, [%24] \n"
            //         // "ldr    q7, [%25] \n"

            //         : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
            //           "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
            //           "=r"(pB)
            //         : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
            //           "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
            //           "8"(pB),                   
            //           "r"(pA0), "r"(pA1), "r"(pA2), "r"(pA3),  // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
            //           "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 22/23/24/25
            //         : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
            //           "x20", "x21", "x22", "x23");
            }

            // 最后一次不需要加载
            {
                // float *pA0 = a0 + kstart;
                // float *pA1 = a1 + kstart;
                // float *pA2 = a2 + kstart;
                // float *pA3 = a3 + kstart;
                float *pA4 = a4 + k;
                float *pA5 = a5 + k;
                float *pA6 = a6 + k;
                float *pA7 = a7 + k;

                asm volatile(
                    "ldp	q8, q9, [%8], #32 \n"   // q8 = vb0j0, q9 = vb1j0
                    "ldp	q10, q11, [%8], #32 \n" // q10 = vb2j0, q11 = vb3j0

                    "ldr    q4, [%18] \n"
                    "ldr    q5, [%19] \n"
                    "ldr    q6, [%20] \n"
                    "ldr    q7, [%21] \n"

                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%3.4s, v8.4s, v3.s[0] \n"
                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%7.4s, v8.4s, v7.s[0] \n"

                    "fmla	%0.4s, v9.4s, v0.s[1] \n"
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"

                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n"
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"
                    "fmla	%4.4s, v10.4s, v4.s[2] \n"      
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n"

                    "fmla	%0.4s, v11.4s, v0.s[3] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pB),                   
                      "r"(pA4), "r"(pA5), "r"(pA6), "r"(pA7)  // 输出9个，前面输入有9个，即这里从18开始，18/19/20/21
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
            }


            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
    }
}


// 基于GemmTilePackBL2，将nB矩阵从堆放到栈中，能得到轻微加速(0.087100s -> 0.086598s)
void GemmTilePackBL2V2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {
    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
     
    int nldb;
    float nB[N*K];
    PackB4x4V2(T, N, K, B, ldb, nB);

    int bid = 0;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            bid = j * K;
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    A, lda, nB, bid, C, ldc);
        }
    }
}

// V30, 基于UKernelPBV25对A也进行Pack 使其内层连续, 但实测packA的代价高于A的访存收益，导致整体耗时反而轻微增加
void UKernelPABV30(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int aid,
                 const float *B, const int bid, 
                 float *C, const int ldc) {
                    
    int i, j, k;
    float32x4_t zero = vdupq_n_f32(0);
    for (i = mstart; i < mend - 7; i += 8) {
        int iaid = i * kend;
        int lbid = bid; // bid 不能把i循环包含在内，所以每次这里刷新
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;  

            int laid = iaid; // aid 不能把j循环包含在内，只算i的，所以每次这里刷新
            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + laid);
                float32x4_t va1 = vld1q_f32(A + laid + 4);
                float32x4_t va2 = vld1q_f32(A + laid + 8);
                float32x4_t va3 = vld1q_f32(A + laid + 12);
                float32x4_t va4 = vld1q_f32(A + laid + 16);
                float32x4_t va5 = vld1q_f32(A + laid + 20);
                float32x4_t va6 = vld1q_f32(A + laid + 24);
                float32x4_t va7 = vld1q_f32(A + laid + 28);
                laid += 32;
                
                float32x4_t vb0j0 = vld1q_f32(B + lbid); 
                float32x4_t vb1j0 = vld1q_f32(B + lbid + 4);
                float32x4_t vb2j0 = vld1q_f32(B + lbid + 8);
                float32x4_t vb3j0 = vld1q_f32(B + lbid + 12);
                lbid += 16;

                vc0 = vfmaq_laneq_f32(vc0, vb0j0, va0, 0);
                vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
                vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
                vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

                vc1 = vfmaq_laneq_f32(vc1, vb0j0, va1, 0);
                vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
                vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
                vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

                vc2 = vfmaq_laneq_f32(vc2, vb0j0, va2, 0);
                vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
                vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
                vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

                vc3 = vfmaq_laneq_f32(vc3, vb0j0, va3, 0);
                vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
                vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
                vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

                vc4 = vfmaq_laneq_f32(vc4, vb0j0, va4, 0);
                vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
                vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
                vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

                vc5 = vfmaq_laneq_f32(vc5, vb0j0, va5, 0);
                vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
                vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
                vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

                vc6 = vfmaq_laneq_f32(vc6, vb0j0, va6, 0);
                vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
                vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
                vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

                vc7 = vfmaq_laneq_f32(vc7, vb0j0, va7, 0);
                vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
                vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
                vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);
            }
            // for (; k < kend; ++k) {
            //     C[i * ldc + j] += A[laid++] * B[lbid++];
            // }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
        // for (; j < nend; ++j) {
        //     int laid = i * kend;
        //     for (k = kstart; k < kend; k++) {
        //         C[i * ldc + j] += A[laid++] * B[lbid++];
        //     }
        // }
    }
    // for (; i < mend; ++i) {
    //     int lbid = bid;
    //     for (j = nstart; j < nend; j++) {
    //         int laid = i * kend;
    //         for (k = kstart; k < kend; ++k) {
    //             C[i * ldc + j] += A[laid++] * B[lbid++];
    //         }
    //     }
    // }
}

void UKernelPABV30MixASM(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int aid,
                 const float *B, const int bid, 
                 float *C, const int ldc) {
                    
    int i, j, k;
    float32x4_t zero = vdupq_n_f32(0);
    for (i = mstart; i < mend - 7; i += 8) {
        int iaid = i * kend;
        float *pA = (float *)A + iaid;
        float *pB = (float *)B + bid; // bid 不能把i循环包含在内，所以每次这里刷新
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = zero;
            float32x4_t vc1 = zero;
            float32x4_t vc2 = zero;
            float32x4_t vc3 = zero;
            float32x4_t vc4 = zero;
            float32x4_t vc5 = zero;
            float32x4_t vc6 = zero;
            float32x4_t vc7 = zero;  

            pA = (float *)A + iaid;// aid 不能把j循环包含在内，只算i的，所以每次这里刷新
            for (k = kstart; k < kend - 3; k += 4) {

                asm volatile(
                    // 
                    // "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%8], #64\n"
                    // "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%8], #64\n"

                    // "ldr	q0, [%8], #16 \n" 
                    // "ldr	q1, [%8], #16 \n"
                    // "ldr	q2, [%8], #16 \n" 
                    // "ldr	q3, [%8], #16 \n" 
                    // "ldr	q4, [%8], #16 \n" 
                    // "ldr	q5, [%8], #16 \n" 
                    // "ldr	q6, [%8], #16 \n" 
                    // "ldr	q7, [%8], #16 \n"

                    "ldp	q0, q1, [%8], #32 \n" 
                    "ldp	q2, q3, [%8], #32 \n" 
                    "ldp	q4, q5, [%8], #32 \n" 
                    "ldp	q6, q7, [%8], #32 \n"
                    "prfm   pldl1keep, [%8, #640]   \n"

                    "ldp	q8, q9, [%9], #32 \n" // q8 = vb0j0, q9 = vb1j0
                    "ldp	q10, q11, [%9], #32 \n" // q10 = vb2j0, q11 = vb3j0
                    "prfm   pldl1keep, [%9, #640]   \n"

                    "fmla	%0.4s, v8.4s, v0.s[0] \n"
                    "fmla	%1.4s, v8.4s, v1.s[0] \n"
                    "fmla	%2.4s, v8.4s, v2.s[0] \n"
                    "fmla	%3.4s, v8.4s, v3.s[0] \n"
                    "fmla	%4.4s, v8.4s, v4.s[0] \n"
                    "fmla	%5.4s, v8.4s, v5.s[0] \n"
                    "fmla	%6.4s, v8.4s, v6.s[0] \n"
                    "fmla	%7.4s, v8.4s, v7.s[0] \n"

                    "fmla	%0.4s, v9.4s, v0.s[1] \n"
                    "fmla	%1.4s, v9.4s, v1.s[1] \n"
                    "fmla	%2.4s, v9.4s, v2.s[1] \n"
                    "fmla	%3.4s, v9.4s, v3.s[1] \n"
                    "fmla	%4.4s, v9.4s, v4.s[1] \n"
                    "fmla	%5.4s, v9.4s, v5.s[1] \n"
                    "fmla	%6.4s, v9.4s, v6.s[1] \n"
                    "fmla	%7.4s, v9.4s, v7.s[1] \n"

                    "fmla	%0.4s, v10.4s, v0.s[2] \n"
                    "fmla	%1.4s, v10.4s, v1.s[2] \n"
                    "fmla	%2.4s, v10.4s, v2.s[2] \n"
                    "fmla	%3.4s, v10.4s, v3.s[2] \n"
                    "fmla	%4.4s, v10.4s, v4.s[2] \n"      
                    "fmla	%5.4s, v10.4s, v5.s[2] \n"
                    "fmla	%6.4s, v10.4s, v6.s[2] \n"
                    "fmla	%7.4s, v10.4s, v7.s[2] \n"

                    "fmla	%0.4s, v11.4s, v0.s[3] \n"
                    "fmla	%1.4s, v11.4s, v1.s[3] \n"
                    "fmla	%2.4s, v11.4s, v2.s[3] \n"
                    "fmla	%3.4s, v11.4s, v3.s[3] \n"
                    "fmla	%4.4s, v11.4s, v4.s[3] \n"
                    "fmla	%5.4s, v11.4s, v5.s[3] \n"
                    "fmla	%6.4s, v11.4s, v6.s[3] \n"
                    "fmla	%7.4s, v11.4s, v7.s[3] \n"

                    : "=w"(vc0), "=w"(vc1), "=w"(vc2), "=w"(vc3),  // 输出：用w传neon寄存器，用r传普通寄存器
                      "=w"(vc4), "=w"(vc5), "=w"(vc6), "=w"(vc7),  // 取别名 [inptr0] "+r"(inptr0)
                      "=r"(pA), "=r"(pB)
                    : "0"(vc0), "1"(vc1), "2"(vc2), "3"(vc3),  // 输入
                      "4"(vc4), "5"(vc5), "6"(vc6), "7"(vc7),  // 输出里有，所以与输出同号 
                      "8"(pA), "9"(pB)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");

                // float32x4_t va0 = vld1q_f32(pA);
                // float32x4_t va1 = vld1q_f32(pA + 4);
                // float32x4_t va2 = vld1q_f32(pA + 8);
                // float32x4_t va3 = vld1q_f32(pA + 12);
                // float32x4_t va4 = vld1q_f32(pA + 16);
                // float32x4_t va5 = vld1q_f32(pA + 20);
                // float32x4_t va6 = vld1q_f32(pA + 24);
                // float32x4_t va7 = vld1q_f32(pA + 28);
                // pA += 32;
                
                // float32x4_t vb0j0 = vld1q_f32(pB); 
                // float32x4_t vb1j0 = vld1q_f32(pB + 4);
                // float32x4_t vb2j0 = vld1q_f32(pB + 8);
                // float32x4_t vb3j0 = vld1q_f32(pB + 12);
                // pB += 16;

                // vc0 = vfmaq_laneq_f32(vc0, vb0j0, va0, 0);
                // vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
                // vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
                // vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

                // vc1 = vfmaq_laneq_f32(vc1, vb0j0, va1, 0);
                // vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
                // vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
                // vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

                // vc2 = vfmaq_laneq_f32(vc2, vb0j0, va2, 0);
                // vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
                // vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
                // vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

                // vc3 = vfmaq_laneq_f32(vc3, vb0j0, va3, 0);
                // vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
                // vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
                // vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

                // vc4 = vfmaq_laneq_f32(vc4, vb0j0, va4, 0);
                // vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
                // vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
                // vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

                // vc5 = vfmaq_laneq_f32(vc5, vb0j0, va5, 0);
                // vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
                // vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
                // vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

                // vc6 = vfmaq_laneq_f32(vc6, vb0j0, va6, 0);
                // vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
                // vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
                // vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

                // vc7 = vfmaq_laneq_f32(vc7, vb0j0, va7, 0);
                // vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
                // vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
                // vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);
            }
            // for (; k < kend; ++k) {
            //     C[i * ldc + j] += A[laid++] * B[lbid++];
            // }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
            vst1q_f32(C + (i+4) * ldc + j, vc4);
            vst1q_f32(C + (i+5) * ldc + j, vc5);
            vst1q_f32(C + (i+6) * ldc + j, vc6);
            vst1q_f32(C + (i+7) * ldc + j, vc7);
        }
        // for (; j < nend; ++j) {
        //     int laid = i * kend;
        //     for (k = kstart; k < kend; k++) {
        //         C[i * ldc + j] += A[laid++] * B[lbid++];
        //     }
        // }
    }
    // for (; i < mend; ++i) {
    //     int lbid = bid;
    //     for (j = nstart; j < nend; j++) {
    //         int laid = i * kend;
    //         for (k = kstart; k < kend; ++k) {
    //             C[i * ldc + j] += A[laid++] * B[lbid++];
    //         }
    //     }
    // }
}



// v26，基于v25。考虑到pack都是在循环前统一进行的，即pack的时候数据加载进cache，在计算时仍然需要从内存加载进cache。
//      两次进cache的过程可以省略掉一次. (矩阵较大时不适用？因为会超过L1的范围（64B）)
void GemmTileL2PackABV31(const int M, const int N, const int K,
                        const float *A, const int lda,
                        const float *B, const int ldb,
                        float *C, const int ldc) {

    int i, ii, j, jj, k, kk;
    float32x4_t zero = vdupq_n_f32(0);
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;

    // float *nB = (float *)malloc(N * K * sizeof(float));
    // float *nA = (float *)malloc(M * K * sizeof(float));
    float nB[N*K];
    float nA[M*K];
    // PackB4x4V2(T, N, K, B, ldb, nB);
    // PackA8x4V2(T, M, K, A, lda, nA);
    int aid = 0;
    int bid = 0;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            bid = j * K;
            if (i == 0) {
                // PackB
                int obid = bid;
                for (jj = j; jj < std::min(j + T, N) - 3; jj += 4) {
                    for (k = 0; k < K - 3; k += 4) {
                        int kldb = k * ldb;
                        int kldbp = kldb + ldb;
                        int kldbpp = kldbp + ldb;
                        int kldbppp = kldbpp + ldb;

                        nB[obid++] = B[kldb + jj];
                        nB[obid++] = B[kldbp + jj];
                        nB[obid++] = B[kldbpp + jj];
                        nB[obid++] = B[kldbppp + jj];

                        nB[obid++] = B[kldb + (jj + 1)];
                        nB[obid++] = B[kldbp + (jj + 1)];
                        nB[obid++] = B[kldbpp + (jj + 1)];
                        nB[obid++] = B[kldbppp + (jj + 1)];

                        nB[obid++] = B[kldb + (jj + 2)];
                        nB[obid++] = B[kldbp + (jj + 2)];
                        nB[obid++] = B[kldbpp + (jj + 2)];
                        nB[obid++] = B[kldbppp + (jj + 2)];

                        nB[obid++] = B[kldb + (jj + 3)];
                        nB[obid++] = B[kldbp + (jj + 3)];
                        nB[obid++] = B[kldbpp + (jj + 3)];
                        nB[obid++] = B[kldbppp + (jj + 3)];
                    }
                }
            }
            if (j == 0) {
                for (ii = i; ii < std::min(i + T, M) - 7; ii += 8) {
                    for (k = 0; k < K - 3; k += 4) {
                        float *pA = (float *)A + ii*lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];

                        pA += lda;
                        nA[aid++] = pA[k];
                        nA[aid++] = pA[k+1];
                        nA[aid++] = pA[k+2];
                        nA[aid++] = pA[k+3];
                    }
                }
            }
            for (ii = i; ii < std::min(i + T, M) - 7; ii += 8) {
                int iaid = ii * K;
                int oaid = ii * K;
                int lbid = bid; // bid 不能把i循环包含在内，所以每次这里刷新
                int obid = bid;
                for (jj = j; jj < std::min(j + T, N) - 3; jj += 4) {
                    float32x4_t vc0 = zero;
                    float32x4_t vc1 = zero;
                    float32x4_t vc2 = zero;
                    float32x4_t vc3 = zero;
                    float32x4_t vc4 = zero;
                    float32x4_t vc5 = zero;
                    float32x4_t vc6 = zero;
                    float32x4_t vc7 = zero;  

                    int laid = iaid; // aid 不能把j循环包含在内，只算i的，所以每次这里刷新
                    for (k = 0; k < K - 3; k += 4) {
                        float32x4_t va0 = vld1q_f32(nA + laid);
                        float32x4_t va1 = vld1q_f32(nA + laid + 4);
                        float32x4_t va2 = vld1q_f32(nA + laid + 8);
                        float32x4_t va3 = vld1q_f32(nA + laid + 12);
                        float32x4_t va4 = vld1q_f32(nA + laid + 16);
                        float32x4_t va5 = vld1q_f32(nA + laid + 20);
                        float32x4_t va6 = vld1q_f32(nA + laid + 24);
                        float32x4_t va7 = vld1q_f32(nA + laid + 28);
                        laid += 32;

                        float32x4_t vb0j0 = vld1q_f32(nB + lbid); 
                        float32x4_t vb1j0 = vld1q_f32(nB + lbid + 4);
                        float32x4_t vb2j0 = vld1q_f32(nB + lbid + 8);
                        float32x4_t vb3j0 = vld1q_f32(nB + lbid + 12);
                        lbid += 16;

                        vc0 = vfmaq_laneq_f32(vc0, vb0j0, va0, 0);
                        vc0 = vfmaq_laneq_f32(vc0, vb1j0, va0, 1);
                        vc0 = vfmaq_laneq_f32(vc0, vb2j0, va0, 2);
                        vc0 = vfmaq_laneq_f32(vc0, vb3j0, va0, 3);

                        vc1 = vfmaq_laneq_f32(vc1, vb0j0, va1, 0);
                        vc1 = vfmaq_laneq_f32(vc1, vb1j0, va1, 1);
                        vc1 = vfmaq_laneq_f32(vc1, vb2j0, va1, 2);
                        vc1 = vfmaq_laneq_f32(vc1, vb3j0, va1, 3);

                        vc2 = vfmaq_laneq_f32(vc2, vb0j0, va2, 0);
                        vc2 = vfmaq_laneq_f32(vc2, vb1j0, va2, 1);
                        vc2 = vfmaq_laneq_f32(vc2, vb2j0, va2, 2);
                        vc2 = vfmaq_laneq_f32(vc2, vb3j0, va2, 3);

                        vc3 = vfmaq_laneq_f32(vc3, vb0j0, va3, 0);
                        vc3 = vfmaq_laneq_f32(vc3, vb1j0, va3, 1);
                        vc3 = vfmaq_laneq_f32(vc3, vb2j0, va3, 2);
                        vc3 = vfmaq_laneq_f32(vc3, vb3j0, va3, 3);

                        vc4 = vfmaq_laneq_f32(vc4, vb0j0, va4, 0);
                        vc4 = vfmaq_laneq_f32(vc4, vb1j0, va4, 1);
                        vc4 = vfmaq_laneq_f32(vc4, vb2j0, va4, 2);
                        vc4 = vfmaq_laneq_f32(vc4, vb3j0, va4, 3);

                        vc5 = vfmaq_laneq_f32(vc5, vb0j0, va5, 0);
                        vc5 = vfmaq_laneq_f32(vc5, vb1j0, va5, 1);
                        vc5 = vfmaq_laneq_f32(vc5, vb2j0, va5, 2);
                        vc5 = vfmaq_laneq_f32(vc5, vb3j0, va5, 3);

                        vc6 = vfmaq_laneq_f32(vc6, vb0j0, va6, 0);
                        vc6 = vfmaq_laneq_f32(vc6, vb1j0, va6, 1);
                        vc6 = vfmaq_laneq_f32(vc6, vb2j0, va6, 2);
                        vc6 = vfmaq_laneq_f32(vc6, vb3j0, va6, 3);

                        vc7 = vfmaq_laneq_f32(vc7, vb0j0, va7, 0);
                        vc7 = vfmaq_laneq_f32(vc7, vb1j0, va7, 1);
                        vc7 = vfmaq_laneq_f32(vc7, vb2j0, va7, 2);
                        vc7 = vfmaq_laneq_f32(vc7, vb3j0, va7, 3);
                    }
                    vst1q_f32(C + ii * ldc + jj, vc0);
                    vst1q_f32(C + (ii+1) * ldc + jj, vc1);
                    vst1q_f32(C + (ii+2) * ldc + jj, vc2);
                    vst1q_f32(C + (ii+3) * ldc + jj, vc3);
                    vst1q_f32(C + (ii+4) * ldc + jj, vc4);
                    vst1q_f32(C + (ii+5) * ldc + jj, vc5);
                    vst1q_f32(C + (ii+6) * ldc + jj, vc6);
                    vst1q_f32(C + (ii+7) * ldc + jj, vc7);
                }
            }
        }
    }
    // free(nB);
    // free(nA);
}

extern "C" void UKernelPABV30Asm(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc);

void GemmTilePackABL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {

    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
     
    float *nB = (float *)malloc(N * K * sizeof(float));
    float *nA = (float *)malloc(M * K * sizeof(float));
    PackB4x4V2(T, N, K, B, ldb, nB);
    PackA8x4V2(T, M, K, A, lda, nA);

    int aid = 0;
    int bid = 0;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            bid = j * K;
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    nA, aid, nB, bid, C, ldc);
        }
    }

    free(nB);
    free(nA);
}

void GemmTilePackABL2V2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {

    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
     
    float nB[N*K];
    float nA[M*K];
    PackB4x4V2(T, N, K, B, ldb, nB);
    PackA8x4V2(T, M, K, A, lda, nA);

    int aid = 0;
    int bid = 0;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            bid = j * K;
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    nA, aid, nB, bid, C, ldc);
        }
    }
}

#define TEST_MODULE(func)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(float)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %f\n",                                               \
               #func, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

#define TEST_MODULE_UKERNEL(func, kernel)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(float)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C, kernel); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %f\n",                                               \
               #func#kernel, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

// is_transposed_b: 当前输入的B矩阵是否已经被转置
// is_continuous_b: kernel是否按B连续访问的方式进行访问。
#define TEST_MODULE_PACK_UKERNEL(func, pack, is_transposed_b, kernel, is_continuous_b)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(float)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C, pack, is_transposed_b, kernel, is_continuous_b); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %f\n",                                               \
               #func#kernel, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

int main() {
    // pai::prof::CpuSelector cpu_selector;
    // cpu_selector.FetchCpuInfo(true);
    // // cpu_selector.BindCoreWithId(2);
    // cpu_selector.BindCoreWithFreq(true);

    // pai::prof::PeakPerfDetector ppf;
    // ppf.RunFmla(12);
    /////////////////////////////////////

    int HEIGHT_A = 800;  // M
    int WIDTH_A = 600;   // K
    int HEIGHT_B = 600;  // K
    int WIDTH_B = 700;   // N
    
    int HEIGHT_C = HEIGHT_A;
    int WIDTH_C = WIDTH_B;

    // gemm输出 MxN 个点, 每个点需要K次乘和K-1次加, 则计算量为 M*N*(2*K-1) flop, 简化可按2MNK算
    // g是10e9, 则乘以10e-9 是将 flop 转为 gflop
    float GFLOP = HEIGHT_C * WIDTH_C * WIDTH_A * 2 * 1e-9;
    
    float *mat_a = (float *)malloc(HEIGHT_A * WIDTH_A * sizeof(float));
    float *mat_b = (float *)malloc(HEIGHT_B * WIDTH_B * sizeof(float));
    float *mat_c = (float *)malloc(HEIGHT_C * WIDTH_C * sizeof(float));

    GenMatrix(HEIGHT_A, WIDTH_A, mat_a);
    GenMatrix(HEIGHT_B, WIDTH_B, mat_b);

    // 普通矩阵乘，C=AxB
    // TEST_MODULE(GemmV1);
    TEST_MODULE(GemmV2);  // 调整for循环层级顺序，提高cache命中率
    TEST_MODULE(GemmV3);  // 分块1层
    TEST_MODULE(GemmV4);  // 分块2层
    TEST_MODULE(GemmV5);  // 分块3层
    TEST_MODULE_UKERNEL(GemmTile, UKernelV6); // 同v5
    TEST_MODULE_UKERNEL(GemmTile, UKernelV7); // 将最内层常规改写neon的形式，速度比v6慢了近1倍
    TEST_MODULE_UKERNEL(GemmTile, UKernelV8); // 尝试展开k，基于k复用vc的读写，C矩阵的加载和写回节省了3/4。
    TEST_MODULE_UKERNEL(GemmTile, UKernelV9); // 对i进行展开，提高计算访存比
    TEST_MODULE_UKERNEL(GemmTile, UKernelV10); // 调整循环顺序，减少C矩阵读写次数；但也导致B矩阵的最内层循环访存不连续，cache miss加重。
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2B, false, UKernelV10, false); // 与下面的 GemmTilePackTBL2UKernelV10 对应
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, false, UKernelV11, true); 
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, false, UKernelV13, true); 
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, false, UKernelV13Asm, true); 
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, false, UKernelV15, true); 

    float *old_mat_b = mat_b;
    int new_ldb;   
    float *BT = (float *)malloc(HEIGHT_B * WIDTH_B * sizeof(float));
    NormalTranspose(WIDTH_B, HEIGHT_B, mat_b, WIDTH_B, BT, &new_ldb);
    mat_b = BT;
    WIDTH_B = new_ldb;

    // // B矩阵转置的矩阵乘，C=AxB^T，在im2col+gemm中gemm的B矩阵是转置后的
    TEST_MODULE_UKERNEL(GemmTile, UKernelV20);
    TEST_MODULE_UKERNEL(GemmTile, UKernelV21);
    TEST_MODULE_UKERNEL(GemmTile, UKernelV22);
    TEST_MODULE_UKERNEL(GemmTileL2, UKernelV22);
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2B, true, UKernelV10, false); // 将原本转置的B转置回去，重新变成普通矩阵乘
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2B, true, UKernelV10P, false); // 试验单独处理分块边界
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, true, UKernelV11, true); // 采用repack的方式，将内存B矩阵的访问转为连续内存访问，提高缓存命中率
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, true, UKernelV12, true); // C矩阵的访问涉及i和j，在外两层循环，初始为0，不需要 vld1q_f32，直接写0
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, true, UKernelV13, true); // 进一步展开i，提高内层计算访存比
    TEST_MODULE_PACK_UKERNEL(GemmTilePackATBL2, PackTB2BC, true, UKernelV14, true); // 进一步Pack A, 收益不大；有bug, N维度上的边界数据没排对, 如N能被4整除，结果是对的


    // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV25MixAsm);
    // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV25MixAsmOpt);
    // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV25MixAsmOptV2);

    // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV26);    // 没啥用
    // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV27);

    // TEST_MODULE_UKERNEL(GemmTilePackABL2, UKernelPABV30); // PackA和B
    // TEST_MODULE_UKERNEL(GemmTilePackABL2V2, UKernelPABV30);  // AB用栈，与UKernelPBV25对标
    // TEST_MODULE(GemmTileL2PackABV31); // 将pack藏入到kernel中
    // TEST_MODULE_UKERNEL(GemmTilePackABL2V2, UKernelPABV30Asm);
    // TEST_MODULE_UKERNEL(GemmTilePackABL2V2, UKernelPABV30MixASM);
    // // TEST_MODULE_UKERNEL(GemmTilePackBL2V2, UKernelPBV25MixAsm);

    // TEST_MODULE_UKERNEL(GemmTilePackBL2V3, UKernelPBV25Asm);

    /*
        测试设备: Cortex A77 2.6GHz armv8.2 https://blog.csdn.net/qq_45683435/article/details/103218558
        L1 cache 每个核心配备 64KB 的 L1 指令缓存和 64KB 的 L1 数据缓存，即总共 128KB 的 L1 缓存。
        L1 cache line 大小为 64B (64字节)
        L2 cache 256KB 或 512KB 
        L2 cache line 大小为 64B
        L3 cache 多核共享4M

        64B == 16个4(float)
        16KB == 4096个4(float) == 64 * 64个float

        GemmV2 -> time: ( 0.330006 ) s ( 1.861784 ) gflops, mean value: 532273.000000
        GemmV3 -> time: ( 0.142212 ) s ( 4.320311 ) gflops, mean value: 532273.000000
        GemmV4 -> time: ( 0.118872 ) s ( 5.168585 ) gflops, mean value: 532273.000000
        GemmV5 -> time: ( 0.114907 ) s ( 5.346933 ) gflops, mean value: 532273.000000
        GemmTileUKernelV6 -> time: ( 0.122490 ) s ( 5.015920 ) gflops, mean value: 532273.000000
        GemmTileUKernelV7 -> time: ( 0.200787 ) s ( 3.059959 ) gflops, mean value: 532273.000000
        GemmTileUKernelV8 -> time: ( 0.075784 ) s ( 8.107253 ) gflops, mean value: 532273.625000
        GemmTileUKernelV9 -> time: ( 0.046159 ) s ( 13.310514 ) gflops, mean value: 532273.625000
        GemmTileUKernelV10 -> time: ( 0.037164 ) s ( 16.532129 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV10 -> time: ( 0.094283 ) s ( 6.516552 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV11 -> time: ( 0.036345 ) s ( 16.904664 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV13 -> time: ( 0.033845 ) s ( 18.153347 ) gflops, mean value: 532273.625000
        GemmTileUKernelV20 -> time: ( 0.500990 ) s ( 1.226372 ) gflops, mean value: 532273.000000
        GemmTileUKernelV21 -> time: ( 0.507616 ) s ( 1.210364 ) gflops, mean value: 532271.250000
        GemmTileUKernelV22 -> time: ( 0.107756 ) s ( 5.701771 ) gflops, mean value: 532269.250000
        GemmTileL2UKernelV22 -> time: ( 0.098150 ) s ( 6.259807 ) gflops, mean value: 532269.250000
        GemmTilePackTBL2UKernelV10 -> time: ( 0.096987 ) s ( 6.334870 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV10P -> time: ( 0.096322 ) s ( 6.378605 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV11 -> time: ( 0.038658 ) s ( 15.893218 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV12 -> time: ( 0.038875 ) s ( 15.804502 ) gflops, mean value: 532273.625000
        GemmTilePackTBL2UKernelV13 -> time: ( 0.037643 ) s ( 16.321760 ) gflops, mean value: 532273.625000
        GemmTilePackATBL2UKernelV14 -> time: ( 0.037861 ) s ( 16.227781 ) gflops, mean value: 532273.625000

    //  
        GemmV1 -> time: 7.522371 s, mean value: 92430664.000000
        GemmV2 -> time: 0.847263 s, mean value: 92430664.000000
        GemmV3 -> time: 0.404122 s, mean value: 92430664.000000
        GemmV4 -> time: 0.329029 s, mean value: 92430664.000000
        GemmV5 -> time: 0.274101 s, mean value: 92430664.000000
        GemmTileUKernelV6 -> time: 0.275483 s, mean value: 92430664.000000
        GemmTileUKernelV7 -> time: 0.515028 s, mean value: 92430664.000000
        GemmTileUKernelV8 -> time: 0.272625 s, mean value: 92430664.000000
        GemmTileUKernelV9 -> time: 0.141661 s, mean value: 92430664.000000
        GemmTileUKernelV10 -> time: 0.204835 s, mean value: 92430664.000000
        GemmTileUKernelV20 -> time: 0.470878 s, mean value: 92430664.000000
        GemmTilePackTUKernelV21 -> time: 0.326319 s, mean value: 92430664.000000
        GemmTilePackTUKernelV22 -> time: 0.154177 s, mean value: 92430664.000000
        GemmTilePackTL2UKernelV22 -> time: 0.129545 s, mean value: 92430664.000000
        GemmTilePackTL2UKernelPTV23 -> time: 0.124593 s, mean value: 92430664.000000
        GemmTilePackBL2UKernelPBV24 -> time: 0.089169 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV24 -> time: 0.087821 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV25 -> time: 0.076751 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV25Asm -> time: 0.076478 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV25MixAsm -> time: 0.141486 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV25MixAsmOpt -> time: 0.085762 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV25MixAsmOptV2 -> time: 0.083344 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV26 -> time: 0.080930 s, mean value: 92430664.000000
        GemmTilePackBL2V2UKernelPBV27 -> time: 0.081412 s, mean value: 92430664.000000
        GemmTilePackABL2UKernelPABV30 -> time: 0.088672 s, mean value: 92430664.000000
        GemmTilePackABL2V2UKernelPABV30 -> time: 0.085463 s, mean value: 92430664.000000
        GemmTileL2PackABV31 -> time: 0.084739 s, mean value: 92430664.000000
        GemmTilePackABL2V2UKernelPABV30Asm -> time: 0.079307 s, mean value: 92430664.000000
        GemmTilePackABL2V2UKernelPABV30MixASM -> time: 0.077714 s, mean value: 92430664.000000
    */
      
    free(mat_a);
    free(mat_b);
    free(mat_c);

    return 0;
}