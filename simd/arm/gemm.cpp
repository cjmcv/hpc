/*!
* \brief Matrix Multiplication.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory.h>
#include <time.h>

#include <arm_neon.h>

#define HEIGHT_A 512  // M
#define WIDTH_A 384  // K
#define HEIGHT_B 384  // K
#define WIDTH_B 640   // N
// 计算量为 K次乘加得到输出的一个点，需要输出M*N个点，即 2 K M N = 0.252 GFLOP
// 0.252 计算量 * 2次 => 0.504 / 0.077s 耗时 = 6.55 GFLOP/s
#define HEIGHT_C HEIGHT_A
#define WIDTH_C WIDTH_B

void* aligned_malloc(size_t required_bytes, size_t alignment = 32) {
    int offset = alignment - 1 + sizeof(void*);
    void* p1 = (void*)malloc(required_bytes + offset);
    if (p1 == NULL)
        return NULL;

    void** p2 = (void**)( ( (size_t)p1 + offset ) & ~(alignment - 1) );
    p2[-1] = p1;
    return p2;
}
 
void aligned_free(void *p2) {
    void* p1 = ((void**)p2)[-1];
    free(p1);
}

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
                float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
                float32x4_t vc1 = vld1q_f32(C + i * ldc + j + 4);

                float32x4_t vb0k0 = vld1q_f32(B + k * ldb + j);
                float32x4_t vb0k1 = vld1q_f32(B + k * ldb + j + 4);

                float32x4_t vb1k0 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j + 4);

                float32x4_t vb2k0 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb2k1 = vld1q_f32(B + (k+2) * ldb + j + 4);

                float32x4_t vb3k0 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb3k1 = vld1q_f32(B + (k+3) * ldb + j + 4);

                vc0 = vfmaq_laneq_f32(vc0, vb0k0, va, 0);
                vc1 = vfmaq_laneq_f32(vc1, vb0k1, va, 0);
                vc0 = vfmaq_laneq_f32(vc0, vb1k0, va, 1);
                vc1 = vfmaq_laneq_f32(vc1, vb1k1, va, 1);
                vc0 = vfmaq_laneq_f32(vc0, vb2k0, va, 2);
                vc1 = vfmaq_laneq_f32(vc1, vb2k1, va, 2);
                vc0 = vfmaq_laneq_f32(vc0, vb3k0, va, 3);
                vc1 = vfmaq_laneq_f32(vc1, vb3k1, va, 3);

                vst1q_f32(C + i * ldc + j, vc0);
                vst1q_f32(C + i * ldc + j + 4, vc1);
            }
            for (; j < nend; ++j) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
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
    for (i = mstart; i < mend - 3; i += 4) {
        for (k = kstart; k < kend - 3; k += 4) {
            float32x4_t va0 = vld1q_f32(A + i*lda + k);
            float32x4_t va1 = vld1q_f32(A + (i+1)*lda + k);
            float32x4_t va2 = vld1q_f32(A + (i+2)*lda + k);
            float32x4_t va3 = vld1q_f32(A + (i+3)*lda + k);
            for (j = nstart; j < nend - 7; j += 8) {
                float32x4_t vc0i0 = vld1q_f32(C + i * ldc + j);
                float32x4_t vc0i1 = vld1q_f32(C + i * ldc + j + 4);
                float32x4_t vc1i0 = vld1q_f32(C + (i+1) * ldc + j);
                float32x4_t vc1i1 = vld1q_f32(C + (i+1) * ldc + j + 4);
                float32x4_t vc2i0 = vld1q_f32(C + (i+2) * ldc + j);
                float32x4_t vc2i1 = vld1q_f32(C + (i+2) * ldc + j + 4);
                float32x4_t vc3i0 = vld1q_f32(C + (i+3) * ldc + j);
                float32x4_t vc3i1 = vld1q_f32(C + (i+3) * ldc + j + 4);

                float32x4_t vb0k0 = vld1q_f32(B + k * ldb + j);
                float32x4_t vb0k1 = vld1q_f32(B + k * ldb + j + 4);
                float32x4_t vb1k0 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j + 4);
                float32x4_t vb2k0 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb2k1 = vld1q_f32(B + (k+2) * ldb + j + 4);
                float32x4_t vb3k0 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb3k1 = vld1q_f32(B + (k+3) * ldb + j + 4);

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, va0, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, va0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb1k0, va0, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb1k1, va0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb2k0, va0, 2);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb2k1, va0, 2);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb3k0, va0, 3);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb3k1, va0, 3);

                vc1i0 = vfmaq_laneq_f32(vc1i0, vb0k0, va1, 0);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb0k1, va1, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, va1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, va1, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb2k0, va1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb2k1, va1, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb3k0, va1, 3);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb3k1, va1, 3);

                vc2i0 = vfmaq_laneq_f32(vc2i0, vb0k0, va2, 0);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb0k1, va2, 0);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb1k0, va2, 1);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb1k1, va2, 1);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb2k0, va2, 2);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb2k1, va2, 2);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb3k0, va2, 3);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb3k1, va2, 3);

                vc3i0 = vfmaq_laneq_f32(vc3i0, vb0k0, va3, 0);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb0k1, va3, 0);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb1k0, va3, 1);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb1k1, va3, 1);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb2k0, va3, 2);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb2k1, va3, 2);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb3k0, va3, 3);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb3k1, va3, 3);

                vst1q_f32(C + i * ldc + j, vc0i0);
                vst1q_f32(C + i * ldc + j + 4, vc0i1);
                vst1q_f32(C + (i+1) * ldc + j, vc1i0);
                vst1q_f32(C + (i+1) * ldc + j + 4, vc1i1);
                vst1q_f32(C + (i+2) * ldc + j, vc2i0);
                vst1q_f32(C + (i+2) * ldc + j + 4, vc2i1);
                vst1q_f32(C + (i+3) * ldc + j, vc3i0);
                vst1q_f32(C + (i+3) * ldc + j + 4, vc3i1);
            }
            for (; j < nend; ++j) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
        for (; k < kend; ++k) {
            for (j = nstart; j < nend; j++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
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
    for (i = mstart; i < mend - 3; i += 4) {
        for (j = nstart; j < nend - 7; j += 8) {
            float32x4_t vc0i0 = vld1q_f32(C + i * ldc + j);
            float32x4_t vc0i1 = vld1q_f32(C + i * ldc + j + 4);
            float32x4_t vc1i0 = vld1q_f32(C + (i + 1) * ldc + j);
            float32x4_t vc1i1 = vld1q_f32(C + (i + 1) * ldc + j + 4);
            float32x4_t vc2i0 = vld1q_f32(C + (i + 2) * ldc + j);
            float32x4_t vc2i1 = vld1q_f32(C + (i + 2) * ldc + j + 4);
            float32x4_t vc3i0 = vld1q_f32(C + (i + 3) * ldc + j);
            float32x4_t vc3i1 = vld1q_f32(C + (i + 3) * ldc + j + 4);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i*lda + k);
                float32x4_t va1 = vld1q_f32(A + (i+1)*lda + k);
                float32x4_t va2 = vld1q_f32(A + (i+2)*lda + k);
                float32x4_t va3 = vld1q_f32(A + (i+3)*lda + k);

                float32x4_t vb0k0 = vld1q_f32(B + k * ldb + j);
                float32x4_t vb0k1 = vld1q_f32(B + k * ldb + j + 4);
                float32x4_t vb1k0 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb1k1 = vld1q_f32(B + (k+1) * ldb + j + 4);
                float32x4_t vb2k0 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb2k1 = vld1q_f32(B + (k+2) * ldb + j + 4);
                float32x4_t vb3k0 = vld1q_f32(B + (k+3) * ldb + j);
                float32x4_t vb3k1 = vld1q_f32(B + (k+3) * ldb + j + 4);

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, va0, 0);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb0k1, va0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb1k0, va0, 1);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb1k1, va0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb2k0, va0, 2);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb2k1, va0, 2);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb3k0, va0, 3);
                vc0i1 = vfmaq_laneq_f32(vc0i1, vb3k1, va0, 3);

                vc1i0 = vfmaq_laneq_f32(vc1i0, vb0k0, va1, 0);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb0k1, va1, 0);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb1k0, va1, 1);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb1k1, va1, 1);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb2k0, va1, 2);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb2k1, va1, 2);
                vc1i0 = vfmaq_laneq_f32(vc1i0, vb3k0, va1, 3);
                vc1i1 = vfmaq_laneq_f32(vc1i1, vb3k1, va1, 3);

                vc2i0 = vfmaq_laneq_f32(vc2i0, vb0k0, va2, 0);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb0k1, va2, 0);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb1k0, va2, 1);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb1k1, va2, 1);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb2k0, va2, 2);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb2k1, va2, 2);
                vc2i0 = vfmaq_laneq_f32(vc2i0, vb3k0, va2, 3);
                vc2i1 = vfmaq_laneq_f32(vc2i1, vb3k1, va2, 3);

                vc3i0 = vfmaq_laneq_f32(vc3i0, vb0k0, va3, 0);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb0k1, va3, 0);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb1k0, va3, 1);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb1k1, va3, 1);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb2k0, va3, 2);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb2k1, va3, 2);
                vc3i0 = vfmaq_laneq_f32(vc3i0, vb3k0, va3, 3);
                vc3i1 = vfmaq_laneq_f32(vc3i1, vb3k1, va3, 3);
            }
            for (; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }

            vst1q_f32(C + i * ldc + j, vc0i0);
            vst1q_f32(C + i * ldc + j + 4, vc0i1);
            vst1q_f32(C + (i + 1) * ldc + j, vc1i0);
            vst1q_f32(C + (i + 1) * ldc + j + 4, vc1i1);
            vst1q_f32(C + (i + 2) * ldc + j, vc2i0);
            vst1q_f32(C + (i + 2) * ldc + j + 4, vc2i1);
            vst1q_f32(C + (i + 3) * ldc + j, vc3i0);
            vst1q_f32(C + (i + 3) * ldc + j + 4, vc3i1);
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
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

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
/// V20 对B进行转置

// v21 临时版本，回退i和j的展开，方便理清位置
void UKernelV20(const int mstart, const int mend, 
                const int nstart, const int nend, 
                const int kstart, const int kend, 
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; i ++) {
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0i0 = vld1q_f32(C + i * ldc + j);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i*lda + k);

                float32x4_t vb0k0 = vld1q_f32(B + k * ldb + j);
                float32x4_t vb1k0 = vld1q_f32(B + (k+1) * ldb + j);
                float32x4_t vb2k0 = vld1q_f32(B + (k+2) * ldb + j);
                float32x4_t vb3k0 = vld1q_f32(B + (k+3) * ldb + j);

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0k0, va0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb1k0, va0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb2k0, va0, 2);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb3k0, va0, 3);
            }
            for (; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }

            vst1q_f32(C + i * ldc + j, vc0i0);
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// v22 基于v21对B进行 转置 操作。即调换j和k
void UKernelPTV21(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend; i ++) {
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0i0 = vld1q_f32(C + i * ldc + j);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i*lda + k);
                
                // // 转置B前
                // float32x4_t vb0k0 = vld1q_f32(B + k * ldb + j);
                // float32x4_t vb1k0 = vld1q_f32(B + (k+1) * ldb + j);
                // float32x4_t vb2k0 = vld1q_f32(B + (k+2) * ldb + j);
                // float32x4_t vb3k0 = vld1q_f32(B + (k+3) * ldb + j);
                // 转置B后
                float32x4_t vb0j0 = vld1q_f32(B + j * ldb + k);
                float32x4_t vb1j0 = vld1q_f32(B + (j+1) * ldb + k);
                float32x4_t vb2j0 = vld1q_f32(B + (j+2) * ldb + k);
                float32x4_t vb3j0 = vld1q_f32(B + (j+3) * ldb + k);

                vc0i0 = vfmaq_laneq_f32(vc0i0, vb0j0, va0, 0);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb1j0, va0, 1);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb2j0, va0, 2);
                vc0i0 = vfmaq_laneq_f32(vc0i0, vb3j0, va0, 3);
            }
            for (; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }

            vst1q_f32(C + i * ldc + j, vc0i0);
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// v23 基于v22，将i展开，使B的加载得到复用;
void UKernelPTV22(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend - 3; i += 4) {
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
            float32x4_t vc1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc3 = vld1q_f32(C + (i+3) * ldc + j);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i * lda + k);
                float32x4_t va1 = vld1q_f32(A + (i+1) * lda + k);
                float32x4_t va2 = vld1q_f32(A + (i+2) * lda + k);
                float32x4_t va3 = vld1q_f32(A + (i+3) * lda + k);
                
                float32x4_t vb0j0 = vld1q_f32(B + j * ldb + k);
                float32x4_t vb1j0 = vld1q_f32(B + (j+1) * ldb + k);
                float32x4_t vb2j0 = vld1q_f32(B + (j+2) * ldb + k);
                float32x4_t vb3j0 = vld1q_f32(B + (j+3) * ldb + k);

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
            }
            for (; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }

            vst1q_f32(C + i * ldc + j, vc0);
            vst1q_f32(C + (i+1) * ldc + j, vc1);
            vst1q_f32(C + (i+2) * ldc + j, vc2);
            vst1q_f32(C + (i+3) * ldc + j, vc3);
        }
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
    for (; i < mend; ++i) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// v24 基于v23，继续将i展开，使B的加载进一步得到复用，但收益较小。
//     思考：B 已经做了一次转置，但B的访问，内层的j会有+1+2+3，属跳跃式访问，不在一个cache line（64B，而vld1q_f32访问4*4=16B）内。
//          是否可以将PackV1中的转置过程中，把B排成内层循环连续的情况。
void UKernelPTV23(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc) {

    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
            float32x4_t vc1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc3 = vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc4 = vld1q_f32(C + (i+4) * ldc + j);
            float32x4_t vc5 = vld1q_f32(C + (i+5) * ldc + j);
            float32x4_t vc6 = vld1q_f32(C + (i+6) * ldc + j);
            float32x4_t vc7 = vld1q_f32(C + (i+7) * ldc + j);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i * lda + k);
                float32x4_t va1 = vld1q_f32(A + (i+1) * lda + k);
                float32x4_t va2 = vld1q_f32(A + (i+2) * lda + k);
                float32x4_t va3 = vld1q_f32(A + (i+3) * lda + k);
                float32x4_t va4 = vld1q_f32(A + (i+4) * lda + k);
                float32x4_t va5 = vld1q_f32(A + (i+5) * lda + k);
                float32x4_t va6 = vld1q_f32(A + (i+6) * lda + k);
                float32x4_t va7 = vld1q_f32(A + (i+7) * lda + k);
                
                float32x4_t vb0j0 = vld1q_f32(B + j * ldb + k);
                float32x4_t vb1j0 = vld1q_f32(B + (j+1) * ldb + k);
                float32x4_t vb2j0 = vld1q_f32(B + (j+2) * ldb + k);
                float32x4_t vb3j0 = vld1q_f32(B + (j+3) * ldb + k);

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
            for (; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
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
        for (; j < nend; ++j) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
    for (; i < mend; ++i) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
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

void GemmNeon(const int M, const int N, const int K,
              const float *A, const int lda,
              const float *B, const int ldb,
              float *C, const int ldc, UKernelFunc ukernel) {
    int i, ii, j, jj, k, kk;
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

// 对B矩阵进行专置操作
void PackV1(const int N, const int K, const float *B, const int ldb, float *nB, int *nldb) {
    *nldb = K;
    for (size_t j = 0; j < N; j ++) {
        for (size_t k = 0; k < K; k ++) {
            nB[j*(*nldb)+k] = B[k*ldb+j]; // *(B++) = src[k*ldb+j];
        }
    }
}

void GemmNeonPackT(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {
    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 80;
     
    int nldb;   
    float *nB = (float *)malloc(N * K * sizeof(float));
    PackV1(N, K, B, ldb, nB, &nldb);

    for (i = 0; i < M; i += T) {
        for (k = 0; k < K; k += T) {
            for (j = 0; j < N; j += T) {
                ukernel(i, std::min(i + T, M),
                        j, std::min(j + T, N),
                        k, std::min(k + T, K),
                        A, lda, nB, nldb, C, ldc);
            }
        }
    }

    free(nB);
}

void GemmNeonPackTL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {
    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
     
    int nldb;   
    float *nB = (float *)malloc(N * K * sizeof(float));
    PackV1(N, K, B, ldb, nB, &nldb);

    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    A, lda, nB, nldb, C, ldc);
        }
    }

    free(nB);
}

/// V24 对B进行Pack 使内层连续, pack和转置代价相近，pack收益更高
void UKernelPBV24(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc) {
                    
    int i, j, k;
    for (i = mstart; i < mend - 7; i += 8) {
        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
        for (j = nstart; j < nend - 3; j += 4) {
            float32x4_t vc0 = vld1q_f32(C + i * ldc + j);
            float32x4_t vc1 = vld1q_f32(C + (i+1) * ldc + j);
            float32x4_t vc2 = vld1q_f32(C + (i+2) * ldc + j);
            float32x4_t vc3 = vld1q_f32(C + (i+3) * ldc + j);
            float32x4_t vc4 = vld1q_f32(C + (i+4) * ldc + j);
            float32x4_t vc5 = vld1q_f32(C + (i+5) * ldc + j);
            float32x4_t vc6 = vld1q_f32(C + (i+6) * ldc + j);
            float32x4_t vc7 = vld1q_f32(C + (i+7) * ldc + j);

            for (k = kstart; k < kend - 3; k += 4) {
                float32x4_t va0 = vld1q_f32(A + i * lda + k);
                float32x4_t va1 = vld1q_f32(A + (i+1) * lda + k);
                float32x4_t va2 = vld1q_f32(A + (i+2) * lda + k);
                float32x4_t va3 = vld1q_f32(A + (i+3) * lda + k);
                float32x4_t va4 = vld1q_f32(A + (i+4) * lda + k);
                float32x4_t va5 = vld1q_f32(A + (i+5) * lda + k);
                float32x4_t va6 = vld1q_f32(A + (i+6) * lda + k);
                float32x4_t va7 = vld1q_f32(A + (i+7) * lda + k);
                
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
            //     C[i * ldc + j] += A[i * lda + k] * B[lbid++];
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
        //     for (k = kstart; k < kend; k++) {
        //         C[i * ldc + j] += A[i * lda + k] * B[lbid++];
        //     }
        // }
    }
    // for (; i < mend; ++i) {
    //     int lbid = bid;
    //     for (j = nstart; j < nend; j++) {
    //         for (k = kstart; k < kend; ++k) {
    //             C[i * ldc + j] += A[i * lda + k] * B[lbid++];
    //         }
    //     }
    // }
}

// 基于V23和PackV1，带入T，目的使V23/V24中内层B的访问连续，直接删掉i循环，套入其访问过程。
// v2Temp: 套入访问过程。 V2: v2Temp基础上，使B内层访问连续。
void PackB4x4V2Temp(const int T, const int N, const int K, const float *B, const int ldb, float *nB, int &nldb) {
    nldb = K;
    int j, jj, k;

    for (j = 0; j < N; j += T) {
        for (jj = j; j < std::min(j + T, N) - 3; j += 4) {
            for (k = 0; k < K - 3; k += 4) {
                // float32x4_t vb0j0 = vld1q_f32(B + j * ldb + k);
                // float32x4_t vb1j0 = vld1q_f32(B + (j + 1) * ldb + k);
                // float32x4_t vb2j0 = vld1q_f32(B + (j + 2) * ldb + k);
                // float32x4_t vb3j0 = vld1q_f32(B + (j + 3) * ldb + k);

                nB[j * nldb + k] = B[k * ldb + j];
                nB[j * nldb + (k+1)] = B[(k+1) * ldb + j];
                nB[j * nldb + (k+2)] = B[(k+2) * ldb + j];
                nB[j * nldb + (k+3)] = B[(k+3) * ldb + j];

                nB[(j+1) * nldb + k] = B[k * ldb + (j+1)];
                nB[(j+1) * nldb + (k+1)] = B[(k+1) * ldb + (j+1)];
                nB[(j+1) * nldb + (k+2)] = B[(k+2) * ldb + (j+1)];
                nB[(j+1) * nldb + (k+3)] = B[(k+3) * ldb + (j+1)];

                nB[(j+2) * nldb + k] = B[k * ldb + (j+2)];
                nB[(j+2) * nldb + (k+1)] = B[(k+1) * ldb + (j+2)];
                nB[(j+2) * nldb + (k+2)] = B[(k+2) * ldb + (j+2)];
                nB[(j+2) * nldb + (k+3)] = B[(k+3) * ldb + (j+2)];

                nB[(j+3) * nldb + k] = B[k * ldb + (j+3)];
                nB[(j+3) * nldb + (k+1)] = B[(k+1) * ldb + (j+3)];
                nB[(j+3) * nldb + (k+2)] = B[(k+2) * ldb + (j+3)];
                nB[(j+3) * nldb + (k+3)] = B[(k+3) * ldb + (j+3)];
            }
            // for (; k < K; ++k) {
            //     nB[j * nldb + k] = B[k * ldb + j];
            // }
        }
        // for (; j < std::min(j + T, N); ++j) {
        //     for (k = 0; k < K; k++) {
        //         nB[j * nldb + k] = B[k * ldb + j];
        //     }
        // }
    }
}

void PackB4x4V2(const int T, const int N, const int K, const float *B, const int ldb, float *nB) {
    int j, jj, k;

    int bid = 0;
    for (j = 0; j < N; j += T) {
        for (jj = j; jj < std::min(j + T, N) - 3; jj += 4) {
            for (k = 0; k < K - 3; k += 4) {
                nB[bid++] = B[k * ldb + jj];
                nB[bid++] = B[(k+1) * ldb + jj];
                nB[bid++] = B[(k+2) * ldb + jj];
                nB[bid++] = B[(k+3) * ldb + jj];

                nB[bid++] = B[k * ldb + (jj+1)];
                nB[bid++] = B[(k+1) * ldb + (jj+1)];
                nB[bid++] = B[(k+2) * ldb + (jj+1)];
                nB[bid++] = B[(k+3) * ldb + (jj+1)];

                nB[bid++] = B[k * ldb + (jj+2)];
                nB[bid++] = B[(k+1) * ldb + (jj+2)];
                nB[bid++] = B[(k+2) * ldb + (jj+2)];
                nB[bid++] = B[(k+3) * ldb + (jj+2)];

                nB[bid++] = B[k * ldb + (jj+3)];
                nB[bid++] = B[(k+1) * ldb + (jj+3)];
                nB[bid++] = B[(k+2) * ldb + (jj+3)];
                nB[bid++] = B[(k+3) * ldb + (jj+3)];
            }
            // for (; k < K; ++k) {
            //     nB[bid++] = B[k * ldb + jj];
            // }
        }
        // for (; jj < std::min(j + T, N); ++jj) {
        //     for (k = 0; k < K; k++) {
        //         nB[bid++] = B[k * ldb + jj];
        //     }
        // }
    }
}

void GemmNeonPackBL2(const int M, const int N, const int K,
                  const float *A, const int lda,
                  const float *B, const int ldb,
                  float *C, const int ldc, UKernelFunc ukernel) {
    int i, ii, j, jj, k, kk;
    memset(C, 0, sizeof(float) * ldc * M);
    
    int T = 64;
     
    int nldb;
    float *nB = (float *)malloc(N * K * sizeof(float));
    // PackB4x4V2Temp(T, N, K, B, ldb, nB, nldb); // 转置
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

    free(nB);
}


/// 基于V24，观察到C不需要load，只需基于0去累加。所以可以把C的load省去
//          同时加B的预取
void UKernelPBV25(const int mstart, const int mend,
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

        int lbid = bid; // bid 只跟j/jj/k三层循环有关，bid由外面的j指定，内部两层循环则这里指定。
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
                float32x4_t va0 = vld1q_f32(a0 + k);
                float32x4_t va1 = vld1q_f32(a1 + k);
                float32x4_t va2 = vld1q_f32(a2 + k);
                float32x4_t va3 = vld1q_f32(a3 + k);
                float32x4_t va4 = vld1q_f32(a4 + k);
                float32x4_t va5 = vld1q_f32(a5 + k);
                float32x4_t va6 = vld1q_f32(a6 + k);
                float32x4_t va7 = vld1q_f32(a7 + k);

                __builtin_prefetch(B + lbid + 160);                
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

// 基于 UKernelPBV25，通过gcc -O3 -S -c gemm.cpp得到gemm.s
// 搜索 UKernelPBV25，得到该函数的汇编代码：
// 从
// 	.globl	_Z12UKernelPBV25iiiiiiPKfiS0_iPfi // -- Begin function _Z12UKernelPBV25iiiiiiPKfiS0_iPfi
// 	.p2align	2
// 	.type	_Z12UKernelPBV25iiiiiiPKfiS0_iPfi,@function
// _Z12UKernelPBV25iiiiiiPKfiS0_iPfi:      // @_Z12UKernelPBV25iiiiiiPKfiS0_iPfi
// 到
// .Lfunc_end23:
// 	.size	_Z12UKernelPBV25iiiiiiPKfiS0_iPfi, .Lfunc_end23-_Z12UKernelPBV25iiiiiiPKfiS0_iPfi
//                                         // -- End function
// 修改_Z12UKernelPBV25iiiiiiPKfiS0_iPfi函数名为新函数名UKernelPBV25Asm，函数参数即原V25的格式，
// cpp 调用的地方用 extern "C" 包含住。

// 基于V25加数据预取
extern "C" void UKernelPBV25Asm(const int mstart, const int mend,
                 const int nstart, const int nend,
                 const int kstart, const int kend,
                 const float *A, const int lda,
                 const float *B, const int bid,
                 float *C, const int ldc);

// 基于V25Asm 排流水 - 难以知道哪些通用寄存器可用。不容易排
// extern "C" void UKernelPBV25AsmOpt(const int mstart, const int mend,
//                  const int nstart, const int nend,
//                  const int kstart, const int kend,
//                  const float *A, const int lda,
//                  const float *B, const int bid,
//                  float *C, const int ldc);

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


// 基于GemmNeonPackBL2，将nB矩阵从堆放到栈中，能得到轻微加速(0.087100s -> 0.086598s)
void GemmNeonPackBL2V2(const int M, const int N, const int K,
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


void PackA8x4V2(const int T, const int M, const int K, const float *A, const int lda, float *nA) {
    int i, ii, k;

    int aid = 0;
    for (i = 0; i < M; i += T) {
        for (ii = i; ii < std::min(i + T, M) - 7; ii += 8) {
            for (k = 0; k < K - 3; k += 4) {
                // float32x4_t va0 = vld1q_f32(A + i * lda + k);
                // float32x4_t va1 = vld1q_f32(A + (i+1) * lda + k);
                // float32x4_t va2 = vld1q_f32(A + (i+2) * lda + k);
                // float32x4_t va3 = vld1q_f32(A + (i+3) * lda + k);
                // float32x4_t va4 = vld1q_f32(A + (i+4) * lda + k);
                // float32x4_t va5 = vld1q_f32(A + (i+5) * lda + k);
                // float32x4_t va6 = vld1q_f32(A + (i+6) * lda + k);
                // float32x4_t va7 = vld1q_f32(A + (i+7) * lda + k);
                nA[aid++] = A[ii*lda + k];
                nA[aid++] = A[ii*lda + k+1];
                nA[aid++] = A[ii*lda + k+2];
                nA[aid++] = A[ii*lda + k+3];

                nA[aid++] = A[(ii+1)*lda + k];
                nA[aid++] = A[(ii+1)*lda + k+1];
                nA[aid++] = A[(ii+1)*lda + k+2];
                nA[aid++] = A[(ii+1)*lda + k+3];

                nA[aid++] = A[(ii+2)*lda + k];
                nA[aid++] = A[(ii+2)*lda + k+1];
                nA[aid++] = A[(ii+2)*lda + k+2];
                nA[aid++] = A[(ii+2)*lda + k+3];

                nA[aid++] = A[(ii+3)*lda + k];
                nA[aid++] = A[(ii+3)*lda + k+1];
                nA[aid++] = A[(ii+3)*lda + k+2];
                nA[aid++] = A[(ii+3)*lda + k+3];

                nA[aid++] = A[(ii+4)*lda + k];
                nA[aid++] = A[(ii+4)*lda + k+1];
                nA[aid++] = A[(ii+4)*lda + k+2];
                nA[aid++] = A[(ii+4)*lda + k+3];

                nA[aid++] = A[(ii+5)*lda + k];
                nA[aid++] = A[(ii+5)*lda + k+1];
                nA[aid++] = A[(ii+5)*lda + k+2];
                nA[aid++] = A[(ii+5)*lda + k+3];

                nA[aid++] = A[(ii+6)*lda + k];
                nA[aid++] = A[(ii+6)*lda + k+1];
                nA[aid++] = A[(ii+6)*lda + k+2];
                nA[aid++] = A[(ii+6)*lda + k+3];

                nA[aid++] = A[(ii+7)*lda + k];
                nA[aid++] = A[(ii+7)*lda + k+1];
                nA[aid++] = A[(ii+7)*lda + k+2];
                nA[aid++] = A[(ii+7)*lda + k+3];
            }
            // for (; k < K; ++k) {
            //     nA[aid++] = A[ii*lda + k];
            //     nA[aid++] = A[(ii+1)*lda + k];
            //     nA[aid++] = A[(ii+2)*lda + k];
            //     nA[aid++] = A[(ii+3)*lda + k];
            //     nA[aid++] = A[(ii+4)*lda + k];
            //     nA[aid++] = A[(ii+5)*lda + k];
            //     nA[aid++] = A[(ii+6)*lda + k];
            //     nA[aid++] = A[(ii+7)*lda + k];
            // }
        }
        // for (; ii < std::min(i + T, M); ++ii) {
        //     for (k = 0; k < K; k++) {
        //         nA[aid++] = A[ii*lda + k];
        //     }
        // }
    }
}


// v26，基于v25。考虑到pack都是在循环前统一进行的，即pack的时候数据加载进cache，在计算时仍然需要从内存加载进cache。
//      两次进cache的过程可以省略掉一次. (矩阵较大时不适用？因为会超过L1的范围（64B）)
void GemmNeonL2PackABV31(const int M, const int N, const int K,
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

void GemmNeonPackABL2(const int M, const int N, const int K,
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

void GemmNeonPackABL2V2(const int M, const int N, const int K,
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

void MatrixMulSIMDv3(const int M, const int N, const int K,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float32x4_t apart = vdupq_n_f32(A[i*lda + k]);
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
                C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
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
        printf("%s -> time: %f s, mean value: %f\n",                                               \
               #func, double(clock() - stime)/CLOCKS_PER_SEC, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

#define TEST_MODULE_UKERNEL(func, kernel)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(float)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C, kernel); \
        }                                                                                          \
        printf("%s -> time: %f s, mean value: %f\n",                                               \
               #func#kernel, double(clock() - stime)/CLOCKS_PER_SEC, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

int main() {

    float *mat_a = (float *)malloc(HEIGHT_A * WIDTH_A * sizeof(float));
    float *mat_b = (float *)malloc(HEIGHT_B * WIDTH_B * sizeof(float));
    float *mat_c = (float *)malloc(HEIGHT_C * WIDTH_C * sizeof(float));

    //float *mat_ret_simd_v4 = (float *)_aligned_malloc(HEIGHT_C * WIDTH_C * sizeof(float), 32);
    GenMatrix(HEIGHT_A, WIDTH_A, mat_a);
    GenMatrix(HEIGHT_B, WIDTH_B, mat_b);

    // TEST_MODULE(GemmV1);
    TEST_MODULE(GemmV2);
    TEST_MODULE(GemmV3);
    TEST_MODULE(GemmV4);
    TEST_MODULE(GemmV5);

    TEST_MODULE_UKERNEL(GemmNeon, UKernelV6);
    TEST_MODULE_UKERNEL(GemmNeon, UKernelV7);
    TEST_MODULE_UKERNEL(GemmNeon, UKernelV8);
    TEST_MODULE_UKERNEL(GemmNeon, UKernelV9);
    TEST_MODULE_UKERNEL(GemmNeon, UKernelV10);

    TEST_MODULE_UKERNEL(GemmNeon, UKernelV20);
    TEST_MODULE_UKERNEL(GemmNeonPackT, UKernelPTV21);
    TEST_MODULE_UKERNEL(GemmNeonPackT, UKernelPTV22);
    TEST_MODULE_UKERNEL(GemmNeonPackTL2, UKernelPTV22);
    TEST_MODULE_UKERNEL(GemmNeonPackTL2, UKernelPTV23);

    TEST_MODULE_UKERNEL(GemmNeonPackBL2, UKernelPBV24);
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV24);   // 基于v24进行，不packA 
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25);
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25Asm); // 基于V25，加预取

    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25MixAsm);
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25MixAsmOpt);
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25MixAsmOptV2);

    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV26);    // 没啥用
    TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV27);

    TEST_MODULE_UKERNEL(GemmNeonPackABL2, UKernelPABV30); // PackA和B
    TEST_MODULE_UKERNEL(GemmNeonPackABL2V2, UKernelPABV30);  // AB用栈，与UKernelPBV25对标
    TEST_MODULE(GemmNeonL2PackABV31); // 将pack藏入到kernel中
    TEST_MODULE_UKERNEL(GemmNeonPackABL2V2, UKernelPABV30Asm);
    TEST_MODULE_UKERNEL(GemmNeonPackABL2V2, UKernelPABV30MixASM);
    // TEST_MODULE_UKERNEL(GemmNeonPackBL2V2, UKernelPBV25MixAsm);

    // TEST_MODULE_UKERNEL(GemmNeonPackBL2V3, UKernelPBV25Asm);
    /*
    //  测试设备: MT6765V/CB, 4核A53
        A53
        L1 cache 大小为 16~64KB，一般为16k，i7多为32K
        L1 cache line 大小为 64B (64字节)
        L2 cache 大小为 128KiB~2MiB
        L2 cache line 大小为 64B

        64B == 16个4(float)
        16KB == 4096个4(float) == 64 * 64个float

        GemmV1 -> time: 7.522371 s, mean value: 92430664.000000
        GemmV2 -> time: 0.847263 s, mean value: 92430664.000000
        GemmV3 -> time: 0.404122 s, mean value: 92430664.000000
        GemmV4 -> time: 0.329029 s, mean value: 92430664.000000
        GemmV5 -> time: 0.274101 s, mean value: 92430664.000000
        GemmNeonUKernelV6 -> time: 0.275483 s, mean value: 92430664.000000
        GemmNeonUKernelV7 -> time: 0.515028 s, mean value: 92430664.000000
        GemmNeonUKernelV8 -> time: 0.272625 s, mean value: 92430664.000000
        GemmNeonUKernelV9 -> time: 0.141661 s, mean value: 92430664.000000
        GemmNeonUKernelV10 -> time: 0.204835 s, mean value: 92430664.000000
        GemmNeonUKernelV20 -> time: 0.470878 s, mean value: 92430664.000000
        GemmNeonPackTUKernelPTV21 -> time: 0.326319 s, mean value: 92430664.000000
        GemmNeonPackTUKernelPTV22 -> time: 0.154177 s, mean value: 92430664.000000
        GemmNeonPackTL2UKernelPTV22 -> time: 0.129545 s, mean value: 92430664.000000
        GemmNeonPackTL2UKernelPTV23 -> time: 0.124593 s, mean value: 92430664.000000
        GemmNeonPackBL2UKernelPBV24 -> time: 0.089169 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV24 -> time: 0.087821 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV25 -> time: 0.076751 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV25Asm -> time: 0.076478 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV25MixAsm -> time: 0.141486 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV25MixAsmOpt -> time: 0.085762 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV25MixAsmOptV2 -> time: 0.083344 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV26 -> time: 0.080930 s, mean value: 92430664.000000
        GemmNeonPackBL2V2UKernelPBV27 -> time: 0.081412 s, mean value: 92430664.000000
        GemmNeonPackABL2UKernelPABV30 -> time: 0.088672 s, mean value: 92430664.000000
        GemmNeonPackABL2V2UKernelPABV30 -> time: 0.085463 s, mean value: 92430664.000000
        GemmNeonL2PackABV31 -> time: 0.084739 s, mean value: 92430664.000000
        GemmNeonPackABL2V2UKernelPABV30Asm -> time: 0.079307 s, mean value: 92430664.000000
        GemmNeonPackABL2V2UKernelPABV30MixASM -> time: 0.077714 s, mean value: 92430664.000000
    */
      
    free(mat_a);
    free(mat_b);
    free(mat_c);

    //_aligned_free(mat_ret_simd_v4);
    return 0;
}