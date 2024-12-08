/*!
* \brief gemm_int8: int32_t *C = int8_t *A x int8_t *B.
* https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=dot
*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <memory.h>
#include <time.h>

#include "pocket-ai/prof/cpu_selector.hpp"
#include "pocket-ai/prof/peak_perf_detector.hpp"

#include <arm_neon.h>

// 生成随机数 -127 到 128 的随机整数
void GenMatrix(const int height, const int width, int8_t *mat) {
    std::srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = rand() % 256 - 127; 
        }
    }
}

// Just for checking the result.
int32_t GetMean(const int32_t* mat, const int height, const int width) {
    int num = height * width;
    long total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += mat[i*width + j];
        }
    }
    return total / num;
}

void PrintMatrix(const int32_t* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d,", mat[i*width + j]);
        }
        printf("\n");
    }
}

// 对B矩阵进行专置操作
void NormalTranspose(const int N, const int K, const int8_t *B, const int ldb, int8_t *nB, int *nldb) {
    *nldb = K;
    for (size_t j = 0; j < N; j ++) {
        for (size_t k = 0; k < K; k ++) {
            nB[j*(*nldb)+k] = B[k*ldb+j]; // *(B++) = src[k*ldb+j];
        }
    }
}

//////////////////////////////////////////////
// 对标gemm_fp32的v2
void GemmV1(const int M, const int N, const int K,
                const int8_t *A, const int lda,
                const int8_t *B, const int ldb,
                int32_t *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(int32_t) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
            }
        }
    }
}

typedef void (*UKernelFunc)(const int mstart, const int mend, 
                            const int nstart, const int nend, 
                            const int kstart, const int kend, 
                            const int8_t *A, const int lda,
                            const int8_t *B, const int ldb,
                            int32_t *C, const int ldc);

void GemmTile(const int M, const int N, const int K,
              const int8_t *A, const int lda,
              const int8_t *B, const int ldb,
              int32_t *C, const int ldc, UKernelFunc ukernel) {
    int i, j, k;
    memset(C, 0, sizeof(int32_t) * ldc * M);

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

//////////////////////////////////////////////
// 对标gemm_fp32的v6
void UKernelV2(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (k = kstart; k < kend; ++k) {
            for (j = nstart; j < nend; j++) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}


// 尝试使ukernel内的数据在L2中得到覆盖
// cpu越弱，分块可能小一点好，使cache能覆盖住，但不是绝对，需要尝试
// 如果K太大了，需要对K分块。
void GemmTileL2(const int M, const int N, const int K,
                const int8_t *A, const int lda,
                const int8_t *B, const int ldb,
                int32_t *C, const int ldc, UKernelFunc ukernel) {
    int i, j, k;
    memset(C, 0, sizeof(int32_t) * ldc * M);
    
    int T = 32;
    for (i = 0; i < M; i += T) {
        for (j = 0; j < N; j += T) {
            ukernel(i, std::min(i + T, M),
                    j, std::min(j + T, N),
                    0, K,
                    A, lda, B, ldb, C, ldc);
        }
    }
}

// 对应gemm_fp32的V10, 改写neon
// 在内层读取int8时使用vld1_s8，将会读取8个元素，所以先将k和j都改成8个一组。
void UKernelV3(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (j = nstart; j < nend-7; j += 8) {
            int32x4_t vc0il = vld1q_s32(C + (i+0) * ldc + j);
            int32x4_t vc0ih = vld1q_s32(C + (i+0) * ldc + j + 4);
            for (k = kstart; k < kend-7; k += 8) {
                int8x8_t vai0s = vld1_s8(A + (i+0)*lda + k);  // 从地址中加载8个int8, 共64位，不满128位则不带q
                int16x8_t vai0 = vmovl_s8(vai0s);             // l长指令，将8x8转为16x8
                int16x4_t ai0l = vget_low_s16(vai0);          // 拆分成高低两份16x4
                int16x4_t ai0h = vget_high_s16(vai0);

                int8x8_t vb0ks = vld1_s8(B + (k+0) * ldb + j);
                int8x8_t vb1ks = vld1_s8(B + (k+1) * ldb + j);
                int8x8_t vb2ks = vld1_s8(B + (k+2) * ldb + j);
                int8x8_t vb3ks = vld1_s8(B + (k+3) * ldb + j);
                int8x8_t vb4ks = vld1_s8(B + (k+4) * ldb + j);
                int8x8_t vb5ks = vld1_s8(B + (k+5) * ldb + j);
                int8x8_t vb6ks = vld1_s8(B + (k+6) * ldb + j);
                int8x8_t vb7ks = vld1_s8(B + (k+7) * ldb + j);

                int16x8_t vb0k = vmovl_s8(vb0ks);
                int16x8_t vb1k = vmovl_s8(vb1ks);
                int16x8_t vb2k = vmovl_s8(vb2ks);
                int16x8_t vb3k = vmovl_s8(vb3ks);
                int16x8_t vb4k = vmovl_s8(vb4ks);
                int16x8_t vb5k = vmovl_s8(vb5ks);
                int16x8_t vb6k = vmovl_s8(vb6ks);
                int16x8_t vb7k = vmovl_s8(vb7ks);

                int16x4_t vb0kl = vget_low_s16(vb0k); 
                int16x4_t vb0kh = vget_high_s16(vb0k);
                int16x4_t vb1kl = vget_low_s16(vb1k); 
                int16x4_t vb1kh = vget_high_s16(vb1k);
                int16x4_t vb2kl = vget_low_s16(vb2k); 
                int16x4_t vb2kh = vget_high_s16(vb2k);
                int16x4_t vb3kl = vget_low_s16(vb3k); 
                int16x4_t vb3kh = vget_high_s16(vb3k);
                int16x4_t vb4kl = vget_low_s16(vb4k); 
                int16x4_t vb4kh = vget_high_s16(vb4k);
                int16x4_t vb5kl = vget_low_s16(vb5k); 
                int16x4_t vb5kh = vget_high_s16(vb5k);
                int16x4_t vb6kl = vget_low_s16(vb6k); 
                int16x4_t vb6kh = vget_high_s16(vb6k);
                int16x4_t vb7kl = vget_low_s16(vb7k); 
                int16x4_t vb7kh = vget_high_s16(vb7k);


                vc0il = vmlal_lane_s16(vc0il, vb0kl, ai0l, 0); // vc0il[i] += vb0kl[i] * ai0l[0] => int32x4_t += int16x4_t x int16x4_t, l长指令
                vc0ih = vmlal_lane_s16(vc0ih, vb0kh, ai0l, 0);
                // C[i * ldc + (j+0)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+0)]; 
                // C[i * ldc + (j+1)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+1)];
                // C[i * ldc + (j+2)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+2)];
                // C[i * ldc + (j+3)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+3)]; //vc0il = vmlal_lane_s16(vc0il, vb0kl, ai0l, 0); (ai0l, 0)对应A的k0, (ai0l, 1) 对应A的k0, (ai0h, 0) 对应A的k4, 
                // C[i * ldc + (j+4)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+4)];
                // C[i * ldc + (j+5)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+5)];
                // C[i * ldc + (j+6)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+6)];
                // C[i * ldc + (j+7)] += A[i * lda + (k+0)] * B[(k+0) * ldb + (j+7)]; // vc0ih = vmlal_lane_s16(vc0ih, vb0kh, ai0l, 0);
                vc0il = vmlal_lane_s16(vc0il, vb1kl, ai0l, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb1kh, ai0l, 1);
                vc0il = vmlal_lane_s16(vc0il, vb2kl, ai0l, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb2kh, ai0l, 2);
                vc0il = vmlal_lane_s16(vc0il, vb3kl, ai0l, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb3kh, ai0l, 3);

                vc0il = vmlal_lane_s16(vc0il, vb4kl, ai0h, 0);
                vc0ih = vmlal_lane_s16(vc0ih, vb4kh, ai0h, 0);
                vc0il = vmlal_lane_s16(vc0il, vb5kl, ai0h, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb5kh, ai0h, 1);
                vc0il = vmlal_lane_s16(vc0il, vb6kl, ai0h, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb6kh, ai0h, 2);
                vc0il = vmlal_lane_s16(vc0il, vb7kl, ai0h, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb7kh, ai0h, 3);
                // C[i * ldc + (j+0)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+0)];
                // C[i * ldc + (j+1)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+1)];
                // C[i * ldc + (j+2)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+2)];
                // C[i * ldc + (j+3)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+3)]; // vc0il = vmlal_lane_s16(vc0il, vb7kl, ai0h, 3);
                // C[i * ldc + (j+4)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+4)];
                // C[i * ldc + (j+5)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+5)];
                // C[i * ldc + (j+6)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+6)];
                // C[i * ldc + (j+7)] += A[i * lda + (k+7)] * B[(k+7) * ldb + (j+7)]; // vc0ih = vmlal_lane_s16(vc0ih, vb7kh, ai0h, 3);
            }
            vst1q_s32(C + (i+0) * ldc + j, vc0il);
            vst1q_s32(C + (i+0) * ldc + j + 4, vc0ih);
            for (; k < kend; ++k) {
                for (int jj = 0; jj < 8; jj++) {
                    C[i * ldc + j+jj] += A[i * lda + k] * B[k * ldb + j+jj];
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}

// 基于v3展开i循环，提高内层循环计算访存比，对A77仅轻微快一丝, 对A55会快很多。
void UKernelV4(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        for (j = nstart; j < nend-7; j += 8) {
            int32x4_t vc0il = vld1q_s32(C + (i+0) * ldc + j);
            int32x4_t vc0ih = vld1q_s32(C + (i+0) * ldc + j + 4);
            int32x4_t vc1il = vld1q_s32(C + (i+1) * ldc + j);
            int32x4_t vc1ih = vld1q_s32(C + (i+1) * ldc + j + 4);
            int32x4_t vc2il = vld1q_s32(C + (i+2) * ldc + j);
            int32x4_t vc2ih = vld1q_s32(C + (i+2) * ldc + j + 4);
            int32x4_t vc3il = vld1q_s32(C + (i+3) * ldc + j);
            int32x4_t vc3ih = vld1q_s32(C + (i+3) * ldc + j + 4);
            for (k = kstart; k < kend-7; k += 8) {
                int8x8_t vai0s = vld1_s8(A + (i+0)*lda + k);
                int8x8_t vai1s = vld1_s8(A + (i+1)*lda + k);
                int8x8_t vai2s = vld1_s8(A + (i+2)*lda + k);
                int8x8_t vai3s = vld1_s8(A + (i+3)*lda + k);
                int16x8_t vai0 = vmovl_s8(vai0s);     // l长指令，将8x8转为16x8
                int16x8_t vai1 = vmovl_s8(vai1s);
                int16x8_t vai2 = vmovl_s8(vai2s);
                int16x8_t vai3 = vmovl_s8(vai3s);

                int16x4_t ai0l = vget_low_s16(vai0);  // 拆分成高低两份16x4
                int16x4_t ai0h = vget_high_s16(vai0);
                int16x4_t ai1l = vget_low_s16(vai1);
                int16x4_t ai1h = vget_high_s16(vai1);
                int16x4_t ai2l = vget_low_s16(vai2);
                int16x4_t ai2h = vget_high_s16(vai2);
                int16x4_t ai3l = vget_low_s16(vai3);
                int16x4_t ai3h = vget_high_s16(vai3);

                int8x8_t vb0ks = vld1_s8(B + (k+0) * ldb + j);
                int8x8_t vb1ks = vld1_s8(B + (k+1) * ldb + j);
                int8x8_t vb2ks = vld1_s8(B + (k+2) * ldb + j);
                int8x8_t vb3ks = vld1_s8(B + (k+3) * ldb + j);
                int8x8_t vb4ks = vld1_s8(B + (k+4) * ldb + j);
                int8x8_t vb5ks = vld1_s8(B + (k+5) * ldb + j);
                int8x8_t vb6ks = vld1_s8(B + (k+6) * ldb + j);
                int8x8_t vb7ks = vld1_s8(B + (k+7) * ldb + j);

                int16x8_t vb0k = vmovl_s8(vb0ks);
                int16x8_t vb1k = vmovl_s8(vb1ks);
                int16x8_t vb2k = vmovl_s8(vb2ks);
                int16x8_t vb3k = vmovl_s8(vb3ks);
                int16x8_t vb4k = vmovl_s8(vb4ks);
                int16x8_t vb5k = vmovl_s8(vb5ks);
                int16x8_t vb6k = vmovl_s8(vb6ks);
                int16x8_t vb7k = vmovl_s8(vb7ks);

                int16x4_t vb0kl = vget_low_s16(vb0k); 
                int16x4_t vb0kh = vget_high_s16(vb0k);
                int16x4_t vb1kl = vget_low_s16(vb1k); 
                int16x4_t vb1kh = vget_high_s16(vb1k);
                int16x4_t vb2kl = vget_low_s16(vb2k); 
                int16x4_t vb2kh = vget_high_s16(vb2k);
                int16x4_t vb3kl = vget_low_s16(vb3k); 
                int16x4_t vb3kh = vget_high_s16(vb3k);
                int16x4_t vb4kl = vget_low_s16(vb4k); 
                int16x4_t vb4kh = vget_high_s16(vb4k);
                int16x4_t vb5kl = vget_low_s16(vb5k); 
                int16x4_t vb5kh = vget_high_s16(vb5k);
                int16x4_t vb6kl = vget_low_s16(vb6k); 
                int16x4_t vb6kh = vget_high_s16(vb6k);
                int16x4_t vb7kl = vget_low_s16(vb7k); 
                int16x4_t vb7kh = vget_high_s16(vb7k);

                // i0
                vc0il = vmlal_lane_s16(vc0il, vb0kl, ai0l, 0);
                vc0ih = vmlal_lane_s16(vc0ih, vb0kh, ai0l, 0);
                vc0il = vmlal_lane_s16(vc0il, vb1kl, ai0l, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb1kh, ai0l, 1);
                vc0il = vmlal_lane_s16(vc0il, vb2kl, ai0l, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb2kh, ai0l, 2);
                vc0il = vmlal_lane_s16(vc0il, vb3kl, ai0l, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb3kh, ai0l, 3);

                vc0il = vmlal_lane_s16(vc0il, vb4kl, ai0h, 0);
                vc0ih = vmlal_lane_s16(vc0ih, vb4kh, ai0h, 0);
                vc0il = vmlal_lane_s16(vc0il, vb5kl, ai0h, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb5kh, ai0h, 1);
                vc0il = vmlal_lane_s16(vc0il, vb6kl, ai0h, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb6kh, ai0h, 2);
                vc0il = vmlal_lane_s16(vc0il, vb7kl, ai0h, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb7kh, ai0h, 3);

                // i1
                vc1il = vmlal_lane_s16(vc1il, vb0kl, ai1l, 0);
                vc1ih = vmlal_lane_s16(vc1ih, vb0kh, ai1l, 0);
                vc1il = vmlal_lane_s16(vc1il, vb1kl, ai1l, 1);
                vc1ih = vmlal_lane_s16(vc1ih, vb1kh, ai1l, 1);
                vc1il = vmlal_lane_s16(vc1il, vb2kl, ai1l, 2);
                vc1ih = vmlal_lane_s16(vc1ih, vb2kh, ai1l, 2);
                vc1il = vmlal_lane_s16(vc1il, vb3kl, ai1l, 3);
                vc1ih = vmlal_lane_s16(vc1ih, vb3kh, ai1l, 3);

                vc1il = vmlal_lane_s16(vc1il, vb4kl, ai1h, 0);
                vc1ih = vmlal_lane_s16(vc1ih, vb4kh, ai1h, 0);
                vc1il = vmlal_lane_s16(vc1il, vb5kl, ai1h, 1);
                vc1ih = vmlal_lane_s16(vc1ih, vb5kh, ai1h, 1);
                vc1il = vmlal_lane_s16(vc1il, vb6kl, ai1h, 2);
                vc1ih = vmlal_lane_s16(vc1ih, vb6kh, ai1h, 2);
                vc1il = vmlal_lane_s16(vc1il, vb7kl, ai1h, 3);
                vc1ih = vmlal_lane_s16(vc1ih, vb7kh, ai1h, 3);

                // i2
                vc2il = vmlal_lane_s16(vc2il, vb0kl, ai2l, 0);
                vc2ih = vmlal_lane_s16(vc2ih, vb0kh, ai2l, 0);
                vc2il = vmlal_lane_s16(vc2il, vb1kl, ai2l, 1);
                vc2ih = vmlal_lane_s16(vc2ih, vb1kh, ai2l, 1);
                vc2il = vmlal_lane_s16(vc2il, vb2kl, ai2l, 2);
                vc2ih = vmlal_lane_s16(vc2ih, vb2kh, ai2l, 2);
                vc2il = vmlal_lane_s16(vc2il, vb3kl, ai2l, 3);
                vc2ih = vmlal_lane_s16(vc2ih, vb3kh, ai2l, 3);

                vc2il = vmlal_lane_s16(vc2il, vb4kl, ai2h, 0);
                vc2ih = vmlal_lane_s16(vc2ih, vb4kh, ai2h, 0);
                vc2il = vmlal_lane_s16(vc2il, vb5kl, ai2h, 1);
                vc2ih = vmlal_lane_s16(vc2ih, vb5kh, ai2h, 1);
                vc2il = vmlal_lane_s16(vc2il, vb6kl, ai2h, 2);
                vc2ih = vmlal_lane_s16(vc2ih, vb6kh, ai2h, 2);
                vc2il = vmlal_lane_s16(vc2il, vb7kl, ai2h, 3);
                vc2ih = vmlal_lane_s16(vc2ih, vb7kh, ai2h, 3);

                // i3
                vc3il = vmlal_lane_s16(vc3il, vb0kl, ai3l, 0);
                vc3ih = vmlal_lane_s16(vc3ih, vb0kh, ai3l, 0);
                vc3il = vmlal_lane_s16(vc3il, vb1kl, ai3l, 1);
                vc3ih = vmlal_lane_s16(vc3ih, vb1kh, ai3l, 1);
                vc3il = vmlal_lane_s16(vc3il, vb2kl, ai3l, 2);
                vc3ih = vmlal_lane_s16(vc3ih, vb2kh, ai3l, 2);
                vc3il = vmlal_lane_s16(vc3il, vb3kl, ai3l, 3);
                vc3ih = vmlal_lane_s16(vc3ih, vb3kh, ai3l, 3);

                vc3il = vmlal_lane_s16(vc3il, vb4kl, ai3h, 0);
                vc3ih = vmlal_lane_s16(vc3ih, vb4kh, ai3h, 0);
                vc3il = vmlal_lane_s16(vc3il, vb5kl, ai3h, 1);
                vc3ih = vmlal_lane_s16(vc3ih, vb5kh, ai3h, 1);
                vc3il = vmlal_lane_s16(vc3il, vb6kl, ai3h, 2);
                vc3ih = vmlal_lane_s16(vc3ih, vb6kh, ai3h, 2);
                vc3il = vmlal_lane_s16(vc3il, vb7kl, ai3h, 3);
                vc3ih = vmlal_lane_s16(vc3ih, vb7kh, ai3h, 3);
            }
            vst1q_s32(C + (i+0) * ldc + j, vc0il);
            vst1q_s32(C + (i+0) * ldc + j + 4, vc0ih);
            vst1q_s32(C + (i+1) * ldc + j, vc1il);
            vst1q_s32(C + (i+1) * ldc + j + 4, vc1ih);
            vst1q_s32(C + (i+2) * ldc + j, vc2il);
            vst1q_s32(C + (i+2) * ldc + j + 4, vc2ih);
            vst1q_s32(C + (i+3) * ldc + j, vc3il);
            vst1q_s32(C + (i+3) * ldc + j + 4, vc3ih);
            for (; k < kend; ++k) {
                for (int ii = 0; ii < 4; ii++) {
                    for (int jj = 0; jj < 8; jj++) {
                        C[(i+ii) * ldc + j+jj] += A[(i+ii) * lda + k] * B[k * ldb + j+jj];                        
                    }
                }
            }
        }
        for (; j < nend; j++) {
            for (int ii = 0; ii < 4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[k * ldb + j];
                }
            }
        }
    }
    for (; i < mend; i++) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
            }
        }        
    }
}

void UKernelV5(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int bid,
               int32_t *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend-3; i += 4) {
        int lbid = bid;
        for (j = nstart; j < nend-7; j += 8) {
            int32x4_t vc0il = vld1q_s32(C + (i+0) * ldc + j);
            int32x4_t vc0ih = vld1q_s32(C + (i+0) * ldc + j + 4);
            int32x4_t vc1il = vld1q_s32(C + (i+1) * ldc + j);
            int32x4_t vc1ih = vld1q_s32(C + (i+1) * ldc + j + 4);
            int32x4_t vc2il = vld1q_s32(C + (i+2) * ldc + j);
            int32x4_t vc2ih = vld1q_s32(C + (i+2) * ldc + j + 4);
            int32x4_t vc3il = vld1q_s32(C + (i+3) * ldc + j);
            int32x4_t vc3ih = vld1q_s32(C + (i+3) * ldc + j + 4);
            for (k = kstart; k < kend-7; k += 8) {
                int8x8_t vai0s = vld1_s8(A + (i+0)*lda + k);
                int8x8_t vai1s = vld1_s8(A + (i+1)*lda + k);
                int8x8_t vai2s = vld1_s8(A + (i+2)*lda + k);
                int8x8_t vai3s = vld1_s8(A + (i+3)*lda + k);
                int16x8_t vai0 = vmovl_s8(vai0s);     // l长指令，将8x8转为16x8
                int16x8_t vai1 = vmovl_s8(vai1s);
                int16x8_t vai2 = vmovl_s8(vai2s);
                int16x8_t vai3 = vmovl_s8(vai3s);

                int16x4_t ai0l = vget_low_s16(vai0);  // 拆分成高低两份16x4
                int16x4_t ai0h = vget_high_s16(vai0);
                int16x4_t ai1l = vget_low_s16(vai1);
                int16x4_t ai1h = vget_high_s16(vai1);
                int16x4_t ai2l = vget_low_s16(vai2);
                int16x4_t ai2h = vget_high_s16(vai2);
                int16x4_t ai3l = vget_low_s16(vai3);
                int16x4_t ai3h = vget_high_s16(vai3);

                int8x8_t vb0ks = vld1_s8(B + lbid + 0);
                int8x8_t vb1ks = vld1_s8(B + lbid + 8);
                int8x8_t vb2ks = vld1_s8(B + lbid + 16);
                int8x8_t vb3ks = vld1_s8(B + lbid + 24);
                int8x8_t vb4ks = vld1_s8(B + lbid + 32);
                int8x8_t vb5ks = vld1_s8(B + lbid + 40);
                int8x8_t vb6ks = vld1_s8(B + lbid + 48);
                int8x8_t vb7ks = vld1_s8(B + lbid + 56);
                lbid += 64;

                int16x8_t vb0k = vmovl_s8(vb0ks);
                int16x8_t vb1k = vmovl_s8(vb1ks);
                int16x8_t vb2k = vmovl_s8(vb2ks);
                int16x8_t vb3k = vmovl_s8(vb3ks);
                int16x8_t vb4k = vmovl_s8(vb4ks);
                int16x8_t vb5k = vmovl_s8(vb5ks);
                int16x8_t vb6k = vmovl_s8(vb6ks);
                int16x8_t vb7k = vmovl_s8(vb7ks);

                int16x4_t vb0kl = vget_low_s16(vb0k); 
                int16x4_t vb0kh = vget_high_s16(vb0k);
                int16x4_t vb1kl = vget_low_s16(vb1k); 
                int16x4_t vb1kh = vget_high_s16(vb1k);
                int16x4_t vb2kl = vget_low_s16(vb2k); 
                int16x4_t vb2kh = vget_high_s16(vb2k);
                int16x4_t vb3kl = vget_low_s16(vb3k); 
                int16x4_t vb3kh = vget_high_s16(vb3k);
                int16x4_t vb4kl = vget_low_s16(vb4k); 
                int16x4_t vb4kh = vget_high_s16(vb4k);
                int16x4_t vb5kl = vget_low_s16(vb5k); 
                int16x4_t vb5kh = vget_high_s16(vb5k);
                int16x4_t vb6kl = vget_low_s16(vb6k); 
                int16x4_t vb6kh = vget_high_s16(vb6k);
                int16x4_t vb7kl = vget_low_s16(vb7k); 
                int16x4_t vb7kh = vget_high_s16(vb7k);

                // i0
                vc0il = vmlal_lane_s16(vc0il, vb0kl, ai0l, 0);
                vc0ih = vmlal_lane_s16(vc0ih, vb0kh, ai0l, 0);
                vc0il = vmlal_lane_s16(vc0il, vb1kl, ai0l, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb1kh, ai0l, 1);
                vc0il = vmlal_lane_s16(vc0il, vb2kl, ai0l, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb2kh, ai0l, 2);
                vc0il = vmlal_lane_s16(vc0il, vb3kl, ai0l, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb3kh, ai0l, 3);

                vc0il = vmlal_lane_s16(vc0il, vb4kl, ai0h, 0);
                vc0ih = vmlal_lane_s16(vc0ih, vb4kh, ai0h, 0);
                vc0il = vmlal_lane_s16(vc0il, vb5kl, ai0h, 1);
                vc0ih = vmlal_lane_s16(vc0ih, vb5kh, ai0h, 1);
                vc0il = vmlal_lane_s16(vc0il, vb6kl, ai0h, 2);
                vc0ih = vmlal_lane_s16(vc0ih, vb6kh, ai0h, 2);
                vc0il = vmlal_lane_s16(vc0il, vb7kl, ai0h, 3);
                vc0ih = vmlal_lane_s16(vc0ih, vb7kh, ai0h, 3);

                // i1
                vc1il = vmlal_lane_s16(vc1il, vb0kl, ai1l, 0);
                vc1ih = vmlal_lane_s16(vc1ih, vb0kh, ai1l, 0);
                vc1il = vmlal_lane_s16(vc1il, vb1kl, ai1l, 1);
                vc1ih = vmlal_lane_s16(vc1ih, vb1kh, ai1l, 1);
                vc1il = vmlal_lane_s16(vc1il, vb2kl, ai1l, 2);
                vc1ih = vmlal_lane_s16(vc1ih, vb2kh, ai1l, 2);
                vc1il = vmlal_lane_s16(vc1il, vb3kl, ai1l, 3);
                vc1ih = vmlal_lane_s16(vc1ih, vb3kh, ai1l, 3);

                vc1il = vmlal_lane_s16(vc1il, vb4kl, ai1h, 0);
                vc1ih = vmlal_lane_s16(vc1ih, vb4kh, ai1h, 0);
                vc1il = vmlal_lane_s16(vc1il, vb5kl, ai1h, 1);
                vc1ih = vmlal_lane_s16(vc1ih, vb5kh, ai1h, 1);
                vc1il = vmlal_lane_s16(vc1il, vb6kl, ai1h, 2);
                vc1ih = vmlal_lane_s16(vc1ih, vb6kh, ai1h, 2);
                vc1il = vmlal_lane_s16(vc1il, vb7kl, ai1h, 3);
                vc1ih = vmlal_lane_s16(vc1ih, vb7kh, ai1h, 3);

                // i2
                vc2il = vmlal_lane_s16(vc2il, vb0kl, ai2l, 0);
                vc2ih = vmlal_lane_s16(vc2ih, vb0kh, ai2l, 0);
                vc2il = vmlal_lane_s16(vc2il, vb1kl, ai2l, 1);
                vc2ih = vmlal_lane_s16(vc2ih, vb1kh, ai2l, 1);
                vc2il = vmlal_lane_s16(vc2il, vb2kl, ai2l, 2);
                vc2ih = vmlal_lane_s16(vc2ih, vb2kh, ai2l, 2);
                vc2il = vmlal_lane_s16(vc2il, vb3kl, ai2l, 3);
                vc2ih = vmlal_lane_s16(vc2ih, vb3kh, ai2l, 3);

                vc2il = vmlal_lane_s16(vc2il, vb4kl, ai2h, 0);
                vc2ih = vmlal_lane_s16(vc2ih, vb4kh, ai2h, 0);
                vc2il = vmlal_lane_s16(vc2il, vb5kl, ai2h, 1);
                vc2ih = vmlal_lane_s16(vc2ih, vb5kh, ai2h, 1);
                vc2il = vmlal_lane_s16(vc2il, vb6kl, ai2h, 2);
                vc2ih = vmlal_lane_s16(vc2ih, vb6kh, ai2h, 2);
                vc2il = vmlal_lane_s16(vc2il, vb7kl, ai2h, 3);
                vc2ih = vmlal_lane_s16(vc2ih, vb7kh, ai2h, 3);

                // i3
                vc3il = vmlal_lane_s16(vc3il, vb0kl, ai3l, 0);
                vc3ih = vmlal_lane_s16(vc3ih, vb0kh, ai3l, 0);
                vc3il = vmlal_lane_s16(vc3il, vb1kl, ai3l, 1);
                vc3ih = vmlal_lane_s16(vc3ih, vb1kh, ai3l, 1);
                vc3il = vmlal_lane_s16(vc3il, vb2kl, ai3l, 2);
                vc3ih = vmlal_lane_s16(vc3ih, vb2kh, ai3l, 2);
                vc3il = vmlal_lane_s16(vc3il, vb3kl, ai3l, 3);
                vc3ih = vmlal_lane_s16(vc3ih, vb3kh, ai3l, 3);

                vc3il = vmlal_lane_s16(vc3il, vb4kl, ai3h, 0);
                vc3ih = vmlal_lane_s16(vc3ih, vb4kh, ai3h, 0);
                vc3il = vmlal_lane_s16(vc3il, vb5kl, ai3h, 1);
                vc3ih = vmlal_lane_s16(vc3ih, vb5kh, ai3h, 1);
                vc3il = vmlal_lane_s16(vc3il, vb6kl, ai3h, 2);
                vc3ih = vmlal_lane_s16(vc3ih, vb6kh, ai3h, 2);
                vc3il = vmlal_lane_s16(vc3il, vb7kl, ai3h, 3);
                vc3ih = vmlal_lane_s16(vc3ih, vb7kh, ai3h, 3);
            }
            vst1q_s32(C + (i+0) * ldc + j, vc0il);
            vst1q_s32(C + (i+0) * ldc + j + 4, vc0ih);
            vst1q_s32(C + (i+1) * ldc + j, vc1il);
            vst1q_s32(C + (i+1) * ldc + j + 4, vc1ih);
            vst1q_s32(C + (i+2) * ldc + j, vc2il);
            vst1q_s32(C + (i+2) * ldc + j + 4, vc2ih);
            vst1q_s32(C + (i+3) * ldc + j, vc3il);
            vst1q_s32(C + (i+3) * ldc + j + 4, vc3ih);
            for (; k < kend; ++k) {
                for (int ii = 0; ii < 4; ii++) {
                    for (int jj = 0; jj < 8; jj++) {
                        C[(i+ii) * ldc + j+jj] += A[(i+ii) * lda + k] * B[lbid + jj];                        
                    }
                }
                lbid += 8;
            }
        }
        for (; j < nend; j++) {
            for (int ii = 0; ii < 4; ii++) {
                for (k = kstart; k < kend; ++k) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[lbid+k];
                }
            }
            lbid += kend-kstart;
        }
    }
    for (; i < mend; i++) {
        int lbid = bid;
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[lbid++];
            }
        }        
    }
}

// 改写自gemm_fp32中的PackTB2BC.
// 按普通矩阵乘的方式去计算，转置了的B会将其恢复成未转置的状态来处理。
void PackTB2BC(const bool is_transposed_b, const int T, const int N, const int K, const int8_t *B, const int ldb, int8_t *nB, int *nldb) {
    int8_t *tB = nullptr;
    *nldb = N;
    // 如果是B本身转置过的, 将其转置回去, 变为正常gemm
    if (is_transposed_b) {
        tB = new int8_t[N*K];
        for (size_t j = 0; j < N; j ++) {
            for (size_t k = 0; k < K; k ++) {
                tB[k*(*nldb)+j] = B[j*ldb+k]; // *(B++) = src[k*ldb+j];
            }
        }
    }
    else {
        tB = (int8_t *)B;
    }

    int bid = 0;
    for (int oj = 0; oj < N; oj += T) {
        int nstart = oj;
        int nend = std::min(oj + T, N);
        int kstart = 0;
        int kend = K;

        int j, k;
        for (j = nstart; j < nend-7; j += 8) {
            for (k = kstart; k < kend-7; k += 8) {
                // int8x8_t vb0ks = vld1_s8(B + (k+0) * ldb + j);
                // int8x8_t vb1ks = vld1_s8(B + (k+1) * ldb + j);
                // int8x8_t vb2ks = vld1_s8(B + (k+2) * ldb + j);
                // int8x8_t vb3ks = vld1_s8(B + (k+3) * ldb + j);
                // int8x8_t vb4ks = vld1_s8(B + (k+4) * ldb + j);
                // int8x8_t vb5ks = vld1_s8(B + (k+5) * ldb + j);
                // int8x8_t vb6ks = vld1_s8(B + (k+6) * ldb + j);
                // int8x8_t vb7ks = vld1_s8(B + (k+7) * ldb + j);
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+0) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+1) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+2) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+3) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+4) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+5) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+6) * (*nldb) + j+r];
                for (int r=0; r<8; r++)  nB[bid++] = tB[(k+7) * (*nldb) + j+r];
            }
            for (; k < kend; ++k) {
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

// 按转置B矩阵乘的方式去计算，未转置的B会将其转置后处理。
void PackB2BTC(const bool is_transposed_b, const int T, const int N, const int K, const int8_t *B, const int ldb, int8_t *nB, int *nldb) {
    int8_t *tB = nullptr;
    *nldb = K;
    // 如果是B本身转置过的, 将其转置回去, 变为正常gemm
    if (is_transposed_b) {
        tB = (int8_t *)B;
    }
    else {
        tB = new int8_t[N*K];
        for (size_t j = 0; j < N; j ++) {
            for (size_t k = 0; k < K; k ++) {
                tB[k*(*nldb)+j] = B[j*ldb+k]; // *(B++) = src[k*ldb+j];
            }
        }
    }

    int bid = 0;
    for (int oj = 0; oj < N; oj += T) {
        int nstart = oj;
        int nend = std::min(oj + T, N);
        int kstart = 0;
        int kend = K;

        int j, k;
        for (j = nstart; j < nend-3; j += 4) {
            for (k = kstart; k < kend-15; k += 16) {
                for (int r=0; r<16; r++)  nB[bid++] = tB[(j+0) * (*nldb) + k+r];
                for (int r=0; r<16; r++)  nB[bid++] = tB[(j+1) * (*nldb) + k+r];
                for (int r=0; r<16; r++)  nB[bid++] = tB[(j+2) * (*nldb) + k+r];
                for (int r=0; r<16; r++)  nB[bid++] = tB[(j+3) * (*nldb) + k+r];
            }
            for (; k < kend; ++k) {
                for (int jj = 0; jj < 4; jj++) {
                    nB[bid++] = tB[(j + jj) * (*nldb) + k];
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                nB[bid++] = tB[j * (*nldb) + k];
            }
        }
    }
    if (!is_transposed_b) {
        delete[] tB;
    }
}

// is_transposed_b 表示B矩阵是否已经被转置了
typedef void (*UPackBFunc)(const bool is_transposed_b, const int T, const int N, const int K, 
                           const int8_t *B, const int ldb, int8_t *nB, int *nldb);

void GemmTilePackTBL2(const int M, const int N, const int K,
                  const int8_t *A, const int lda,
                  const int8_t *B, const int ldb,
                  int32_t *C, const int ldc, 
                  UPackBFunc upack, bool is_transposed_b, UKernelFunc ukernel, bool is_b_continuous) {
    int i, j, k;
    memset(C, 0, sizeof(int32_t) * ldc * M);
    
    int T = 64;
    int nldb;
    int8_t *nB = (int8_t *)malloc(N * K * sizeof(int8_t)); // float nB[N*K];
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


// 已转置B的实现方式，对应上面的UKernelV2
void UKernelTBV1(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    for (i = mstart; i < mend; ++i) {
        for (j = nstart; j < nend; j++) {
            for (k = kstart; k < kend; ++k) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// neon实现，针对armv8.2，使用dot指令, 提升非常显著
void UKernelTBV2(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    int32x4_t zero = vdupq_n_s32(0);
    for (i = mstart; i < mend; ++i) {
        for (j = nstart; j < nend-3; j += 4) {
            int32x4_t vcj0 = zero;
            int32x4_t vcj1 = zero;
            int32x4_t vcj2 = zero;
            int32x4_t vcj3 = zero;
            for (k = kstart; k < kend-15; k += 16) {
                int8x16_t va = vld1q_s8(A + i*lda + k);
                int8x16_t vbj0 = vld1q_s8(B + (j+0)*ldb + k);
                int8x16_t vbj1 = vld1q_s8(B + (j+1)*ldb + k);
                int8x16_t vbj2 = vld1q_s8(B + (j+2)*ldb + k);
                int8x16_t vbj3 = vld1q_s8(B + (j+3)*ldb + k);

                // 16个8，每4个为一组，计算点积，得到4个32位的点积结果
                // c[0] += (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3])
                // c[1] += (a[4]*b[4]+a[5]*b[5]+a[6]*b[6]+a[7]*b[7])
                // ...
                // c[3] += (a[12]*b[12]+a[13]*b[13]+a[14]*b[14]+a[15]*b[15])
                vcj0 = vdotq_s32(vcj0, va, vbj0);
                vcj1 = vdotq_s32(vcj1, va, vbj1);
                vcj2 = vdotq_s32(vcj2, va, vbj2);
                vcj3 = vdotq_s32(vcj3, va, vbj3);
            }
            C[i * ldc + (j+0)] += vgetq_lane_s32(vcj0, 0) + vgetq_lane_s32(vcj0, 1) + vgetq_lane_s32(vcj0, 2) + vgetq_lane_s32(vcj0, 3);
            C[i * ldc + (j+1)] += vgetq_lane_s32(vcj1, 0) + vgetq_lane_s32(vcj1, 1) + vgetq_lane_s32(vcj1, 2) + vgetq_lane_s32(vcj1, 3);
            C[i * ldc + (j+2)] += vgetq_lane_s32(vcj2, 0) + vgetq_lane_s32(vcj2, 1) + vgetq_lane_s32(vcj2, 2) + vgetq_lane_s32(vcj2, 3);
            C[i * ldc + (j+3)] += vgetq_lane_s32(vcj3, 0) + vgetq_lane_s32(vcj3, 1) + vgetq_lane_s32(vcj3, 2) + vgetq_lane_s32(vcj3, 3);

            for (; k < kend; k ++) {
                for (int jj=0; jj<4; jj++) {
                    C[i * ldc + (j+jj)] += A[i * lda + k] * B[(j+jj) * ldb + k];                    
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }

    // for (i = mstart; i < mend; ++i) {
    //     for (j = nstart; j < nend-3; j += 4) {
    //         int vc0[4] = {0};
    //         int vc1[4] = {0};
    //         int vc2[4] = {0};
    //         int vc3[4] = {0};
    //         for (k = kstart; k < kend-15; k += 16) {
    //             vc0[0] += A[i * lda + k+0] * B[j * ldb + k+0];
    //             vc0[0] += A[i * lda + k+1] * B[j * ldb + k+1];
    //             vc0[0] += A[i * lda + k+2] * B[j * ldb + k+2];
    //             vc0[0] += A[i * lda + k+3] * B[j * ldb + k+3];
    //             vc0[1] += A[i * lda + k+4] * B[j * ldb + k+4];
    //             vc0[1] += A[i * lda + k+5] * B[j * ldb + k+5];
    //             vc0[1] += A[i * lda + k+6] * B[j * ldb + k+6];
    //             vc0[1] += A[i * lda + k+7] * B[j * ldb + k+7];
    //             vc0[2] += A[i * lda + k+8] * B[j * ldb + k+8];
    //             vc0[2] += A[i * lda + k+9] * B[j * ldb + k+9];
    //             vc0[2] += A[i * lda + k+10] * B[j * ldb + k+10];
    //             vc0[2] += A[i * lda + k+11] * B[j * ldb + k+11];
    //             vc0[3] += A[i * lda + k+12] * B[j * ldb + k+12];
    //             vc0[3] += A[i * lda + k+13] * B[j * ldb + k+13];
    //             vc0[3] += A[i * lda + k+14] * B[j * ldb + k+14];
    //             vc0[3] += A[i * lda + k+15] * B[j * ldb + k+15];
    //             //
    //             vc1[0] += A[i * lda + k+0] * B[(j+1) * ldb + k+0];
    //             vc1[0] += A[i * lda + k+1] * B[(j+1) * ldb + k+1];
    //             vc1[0] += A[i * lda + k+2] * B[(j+1) * ldb + k+2];
    //             vc1[0] += A[i * lda + k+3] * B[(j+1) * ldb + k+3];
    //             vc1[1] += A[i * lda + k+4] * B[(j+1) * ldb + k+4];
    //             vc1[1] += A[i * lda + k+5] * B[(j+1) * ldb + k+5];
    //             vc1[1] += A[i * lda + k+6] * B[(j+1) * ldb + k+6];
    //             vc1[1] += A[i * lda + k+7] * B[(j+1) * ldb + k+7];
    //             vc1[2] += A[i * lda + k+8] * B[(j+1) * ldb + k+8];
    //             vc1[2] += A[i * lda + k+9] * B[(j+1) * ldb + k+9];
    //             vc1[2] += A[i * lda + k+10] * B[(j+1) * ldb + k+10];
    //             vc1[2] += A[i * lda + k+11] * B[(j+1) * ldb + k+11];
    //             vc1[3] += A[i * lda + k+12] * B[(j+1) * ldb + k+12];
    //             vc1[3] += A[i * lda + k+13] * B[(j+1) * ldb + k+13];
    //             vc1[3] += A[i * lda + k+14] * B[(j+1) * ldb + k+14];
    //             vc1[3] += A[i * lda + k+15] * B[(j+1) * ldb + k+15];
    //                             //
    //             vc2[0] += A[i * lda + k+0] * B[(j+2) * ldb + k+0];
    //             vc2[0] += A[i * lda + k+1] * B[(j+2) * ldb + k+1];
    //             vc2[0] += A[i * lda + k+2] * B[(j+2) * ldb + k+2];
    //             vc2[0] += A[i * lda + k+3] * B[(j+2) * ldb + k+3];
    //             vc2[1] += A[i * lda + k+4] * B[(j+2) * ldb + k+4];
    //             vc2[1] += A[i * lda + k+5] * B[(j+2) * ldb + k+5];
    //             vc2[1] += A[i * lda + k+6] * B[(j+2) * ldb + k+6];
    //             vc2[1] += A[i * lda + k+7] * B[(j+2) * ldb + k+7];
    //             vc2[2] += A[i * lda + k+8] * B[(j+2) * ldb + k+8];
    //             vc2[2] += A[i * lda + k+9] * B[(j+2) * ldb + k+9];
    //             vc2[2] += A[i * lda + k+10] * B[(j+2) * ldb + k+10];
    //             vc2[2] += A[i * lda + k+11] * B[(j+2) * ldb + k+11];
    //             vc2[3] += A[i * lda + k+12] * B[(j+2) * ldb + k+12];
    //             vc2[3] += A[i * lda + k+13] * B[(j+2) * ldb + k+13];
    //             vc2[3] += A[i * lda + k+14] * B[(j+2) * ldb + k+14];
    //             vc2[3] += A[i * lda + k+15] * B[(j+2) * ldb + k+15];
    //                             //
    //             vc3[0] += A[i * lda + k+0] * B[(j+3) * ldb + k+0];
    //             vc3[0] += A[i * lda + k+1] * B[(j+3) * ldb + k+1];
    //             vc3[0] += A[i * lda + k+2] * B[(j+3) * ldb + k+2];
    //             vc3[0] += A[i * lda + k+3] * B[(j+3) * ldb + k+3];
    //             vc3[1] += A[i * lda + k+4] * B[(j+3) * ldb + k+4];
    //             vc3[1] += A[i * lda + k+5] * B[(j+3) * ldb + k+5];
    //             vc3[1] += A[i * lda + k+6] * B[(j+3) * ldb + k+6];
    //             vc3[1] += A[i * lda + k+7] * B[(j+3) * ldb + k+7];
    //             vc3[2] += A[i * lda + k+8] * B[(j+3) * ldb + k+8];
    //             vc3[2] += A[i * lda + k+9] * B[(j+3) * ldb + k+9];
    //             vc3[2] += A[i * lda + k+10] * B[(j+3) * ldb + k+10];
    //             vc3[2] += A[i * lda + k+11] * B[(j+3) * ldb + k+11];
    //             vc3[3] += A[i * lda + k+12] * B[(j+3) * ldb + k+12];
    //             vc3[3] += A[i * lda + k+13] * B[(j+3) * ldb + k+13];
    //             vc3[3] += A[i * lda + k+14] * B[(j+3) * ldb + k+14];
    //             vc3[3] += A[i * lda + k+15] * B[(j+3) * ldb + k+15];
    //         }
    //         C[i * ldc + (j+0)] += vc0[0] + vc0[1] + vc0[2] + vc0[3];
    //         C[i * ldc + (j+1)] += vc1[0] + vc1[1] + vc1[2] + vc1[3];
    //         C[i * ldc + (j+2)] += vc2[0] + vc2[1] + vc2[2] + vc2[3];
    //         C[i * ldc + (j+3)] += vc3[0] + vc3[1] + vc3[2] + vc3[3];

    //         for (; k < kend; k ++) {
    //             for (int jj=0; jj<4; jj++) {
    //                 C[i * ldc + (j+jj)] += A[i * lda + k] * B[(j+jj) * ldb + k];                    
    //             }
    //         }
    //     }
    //     for (; j < nend; j++) {
    //         for (k = kstart; k < kend; k++) {
    //             C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
    //         }
    //     }
    // }
}

// 基于v2，将i展开，提高内层计算访存比, 性能不升反降。
void UKernelTBV3(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    int32x4_t zero = vdupq_n_s32(0);
    for (i = mstart; i < mend-3; i += 4) {
        for (j = nstart; j < nend-3; j += 4) {
            int32x4_t vci0j0 = zero, vci0j1 = zero, vci0j2 = zero, vci0j3 = zero;
            int32x4_t vci1j0 = zero, vci1j1 = zero, vci1j2 = zero, vci1j3 = zero;
            int32x4_t vci2j0 = zero, vci2j1 = zero, vci2j2 = zero, vci2j3 = zero;
            int32x4_t vci3j0 = zero, vci3j1 = zero, vci3j2 = zero, vci3j3 = zero;
            for (k = kstart; k < kend-15; k += 16) {
                int8x16_t vai0 = vld1q_s8(A + (i+0)*lda + k);
                int8x16_t vai1 = vld1q_s8(A + (i+1)*lda + k);
                int8x16_t vai2 = vld1q_s8(A + (i+2)*lda + k);
                int8x16_t vai3 = vld1q_s8(A + (i+3)*lda + k);

                int8x16_t vbj0 = vld1q_s8(B + (j+0)*ldb + k);
                int8x16_t vbj1 = vld1q_s8(B + (j+1)*ldb + k);
                int8x16_t vbj2 = vld1q_s8(B + (j+2)*ldb + k);
                int8x16_t vbj3 = vld1q_s8(B + (j+3)*ldb + k);

                vci0j0 = vdotq_s32(vci0j0, vai0, vbj0);
                vci0j1 = vdotq_s32(vci0j1, vai0, vbj1);
                vci0j2 = vdotq_s32(vci0j2, vai0, vbj2);
                vci0j3 = vdotq_s32(vci0j3, vai0, vbj3);

                vci1j0 = vdotq_s32(vci1j0, vai1, vbj0);
                vci1j1 = vdotq_s32(vci1j1, vai1, vbj1);
                vci1j2 = vdotq_s32(vci1j2, vai1, vbj2);
                vci1j3 = vdotq_s32(vci1j3, vai1, vbj3);

                vci2j0 = vdotq_s32(vci2j0, vai2, vbj0);
                vci2j1 = vdotq_s32(vci2j1, vai2, vbj1);
                vci2j2 = vdotq_s32(vci2j2, vai2, vbj2);
                vci2j3 = vdotq_s32(vci2j3, vai2, vbj3);

                vci3j0 = vdotq_s32(vci3j0, vai3, vbj0);
                vci3j1 = vdotq_s32(vci3j1, vai3, vbj1);
                vci3j2 = vdotq_s32(vci3j2, vai3, vbj2);
                vci3j3 = vdotq_s32(vci3j3, vai3, vbj3);
            }
            C[(i+0) * ldc + (j+0)] += vgetq_lane_s32(vci0j0, 0) + vgetq_lane_s32(vci0j0, 1) + vgetq_lane_s32(vci0j0, 2) + vgetq_lane_s32(vci0j0, 3);
            C[(i+0) * ldc + (j+1)] += vgetq_lane_s32(vci0j1, 0) + vgetq_lane_s32(vci0j1, 1) + vgetq_lane_s32(vci0j1, 2) + vgetq_lane_s32(vci0j1, 3);
            C[(i+0) * ldc + (j+2)] += vgetq_lane_s32(vci0j2, 0) + vgetq_lane_s32(vci0j2, 1) + vgetq_lane_s32(vci0j2, 2) + vgetq_lane_s32(vci0j2, 3);
            C[(i+0) * ldc + (j+3)] += vgetq_lane_s32(vci0j3, 0) + vgetq_lane_s32(vci0j3, 1) + vgetq_lane_s32(vci0j3, 2) + vgetq_lane_s32(vci0j3, 3);

            C[(i+1) * ldc + (j+0)] += vgetq_lane_s32(vci1j0, 0) + vgetq_lane_s32(vci1j0, 1) + vgetq_lane_s32(vci1j0, 2) + vgetq_lane_s32(vci1j0, 3);
            C[(i+1) * ldc + (j+1)] += vgetq_lane_s32(vci1j1, 0) + vgetq_lane_s32(vci1j1, 1) + vgetq_lane_s32(vci1j1, 2) + vgetq_lane_s32(vci1j1, 3);
            C[(i+1) * ldc + (j+2)] += vgetq_lane_s32(vci1j2, 0) + vgetq_lane_s32(vci1j2, 1) + vgetq_lane_s32(vci1j2, 2) + vgetq_lane_s32(vci1j2, 3);
            C[(i+1) * ldc + (j+3)] += vgetq_lane_s32(vci1j3, 0) + vgetq_lane_s32(vci1j3, 1) + vgetq_lane_s32(vci1j3, 2) + vgetq_lane_s32(vci1j3, 3);

            C[(i+2) * ldc + (j+0)] += vgetq_lane_s32(vci2j0, 0) + vgetq_lane_s32(vci2j0, 1) + vgetq_lane_s32(vci2j0, 2) + vgetq_lane_s32(vci2j0, 3);
            C[(i+2) * ldc + (j+1)] += vgetq_lane_s32(vci2j1, 0) + vgetq_lane_s32(vci2j1, 1) + vgetq_lane_s32(vci2j1, 2) + vgetq_lane_s32(vci2j1, 3);
            C[(i+2) * ldc + (j+2)] += vgetq_lane_s32(vci2j2, 0) + vgetq_lane_s32(vci2j2, 1) + vgetq_lane_s32(vci2j2, 2) + vgetq_lane_s32(vci2j2, 3);
            C[(i+2) * ldc + (j+3)] += vgetq_lane_s32(vci2j3, 0) + vgetq_lane_s32(vci2j3, 1) + vgetq_lane_s32(vci2j3, 2) + vgetq_lane_s32(vci2j3, 3);

            C[(i+3) * ldc + (j+0)] += vgetq_lane_s32(vci3j0, 0) + vgetq_lane_s32(vci3j0, 1) + vgetq_lane_s32(vci3j0, 2) + vgetq_lane_s32(vci3j0, 3);
            C[(i+3) * ldc + (j+1)] += vgetq_lane_s32(vci3j1, 0) + vgetq_lane_s32(vci3j1, 1) + vgetq_lane_s32(vci3j1, 2) + vgetq_lane_s32(vci3j1, 3);
            C[(i+3) * ldc + (j+2)] += vgetq_lane_s32(vci3j2, 0) + vgetq_lane_s32(vci3j2, 1) + vgetq_lane_s32(vci3j2, 2) + vgetq_lane_s32(vci3j2, 3);
            C[(i+3) * ldc + (j+3)] += vgetq_lane_s32(vci3j3, 0) + vgetq_lane_s32(vci3j3, 1) + vgetq_lane_s32(vci3j3, 2) + vgetq_lane_s32(vci3j3, 3);
            for (; k < kend; k ++) {
                for (int ii=0; ii<4; ii++) {
                    for (int jj=0; jj<4; jj++) {
                        C[(i+ii) * ldc + (j+jj)] += A[(i+ii) * lda + k] * B[(j+jj) * ldb + k];                    
                    }                    
                }
            }
        }
        for (; j < nend; j++) {
            for (int ii=0; ii<4; ii++) {
                for (k = kstart; k < kend; k++) {
                    C[(i+ii) * ldc + j] += A[(i+ii) * lda + k] * B[j * ldb + k];
                }
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

// 基于v2，将点积结果汇总部分 改为 转置后叠加。无提升。
void UKernelTBV4(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int ldb,
               int32_t *C, const int ldc) {
    int i, j, k;
    int32x4_t zero = vdupq_n_s32(0);
    for (i = mstart; i < mend; ++i) {
        for (j = nstart; j < nend-3; j += 4) {
            int32x4_t vcj0 = zero;
            int32x4_t vcj1 = zero;
            int32x4_t vcj2 = zero;
            int32x4_t vcj3 = zero;

            int32x4_t vc = vld1q_s32(C + i * ldc + j);
            for (k = kstart; k < kend-15; k += 16) {
                int8x16_t va = vld1q_s8(A + i*lda + k);
                int8x16_t vbj0 = vld1q_s8(B + (j+0)*ldb + k);
                int8x16_t vbj1 = vld1q_s8(B + (j+1)*ldb + k);
                int8x16_t vbj2 = vld1q_s8(B + (j+2)*ldb + k);
                int8x16_t vbj3 = vld1q_s8(B + (j+3)*ldb + k);

                vcj0 = vdotq_s32(vcj0, va, vbj0);
                vcj1 = vdotq_s32(vcj1, va, vbj1);
                vcj2 = vdotq_s32(vcj2, va, vbj2);
                vcj3 = vdotq_s32(vcj3, va, vbj3);
            }

            // 参考matrix_transpose.cpp 中的 TransposeFp32x4x4_uzp
            int32x4x2_t t0 = vuzpq_s32(vcj0, vcj1);
            int32x4x2_t t1 = vuzpq_s32(vcj2, vcj3);
            int32x4x2_t s0 = vuzpq_s32(t0.val[0], t1.val[0]);
            int32x4x2_t s1 = vuzpq_s32(t0.val[1], t1.val[1]);

            int32x4_t sum0 = vaddq_s32(s0.val[0], s1.val[0]);
            int32x4_t sum1 = vaddq_s32(s0.val[1], s1.val[1]);
            int32x4_t sum2 = vaddq_s32(sum0, sum1);

            vc = vaddq_s32(vc, sum2);
            vst1q_s32(C + i * ldc + j, vc);

            for (; k < kend; k ++) {
                for (int jj=0; jj<4; jj++) {
                    C[i * ldc + (j+jj)] += A[i * lda + k] * B[(j+jj) * ldb + k];                    
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[j * ldb + k];
            }
        }
    }
}

// 使B连续
void UKernelTBV5(const int mstart, const int mend, 
               const int nstart, const int nend, 
               const int kstart, const int kend, 
               const int8_t *A, const int lda,
               const int8_t *B, const int bid,
               int32_t *C, const int ldc) {
    int i, j, k;
    int32x4_t zero = vdupq_n_s32(0);
    for (i = mstart; i < mend; ++i) {
        int lbid = bid;
        for (j = nstart; j < nend-3; j += 4) {
            int32x4_t vcj0 = zero;
            int32x4_t vcj1 = zero;
            int32x4_t vcj2 = zero;
            int32x4_t vcj3 = zero;

            int32x4_t vc = vld1q_s32(C + i * ldc + j);
            for (k = kstart; k < kend-15; k += 16) {
                int8x16_t va = vld1q_s8(A + i*lda + k);
                int8x16_t vbj0 = vld1q_s8(B + lbid);
                int8x16_t vbj1 = vld1q_s8(B + lbid + 16);
                int8x16_t vbj2 = vld1q_s8(B + lbid + 32);
                int8x16_t vbj3 = vld1q_s8(B + lbid + 48);
                lbid += 64;

                vcj0 = vdotq_s32(vcj0, va, vbj0);
                vcj1 = vdotq_s32(vcj1, va, vbj1);
                vcj2 = vdotq_s32(vcj2, va, vbj2);
                vcj3 = vdotq_s32(vcj3, va, vbj3);
            }

            // 参考matrix_transpose.cpp 中的 TransposeFp32x4x4_uzp
            int32x4x2_t t0 = vuzpq_s32(vcj0, vcj1);
            int32x4x2_t t1 = vuzpq_s32(vcj2, vcj3);
            int32x4x2_t s0 = vuzpq_s32(t0.val[0], t1.val[0]);
            int32x4x2_t s1 = vuzpq_s32(t0.val[1], t1.val[1]);

            int32x4_t sum0 = vaddq_s32(s0.val[0], s1.val[0]);
            int32x4_t sum1 = vaddq_s32(s0.val[1], s1.val[1]);
            int32x4_t sum2 = vaddq_s32(sum0, sum1);

            vc = vaddq_s32(vc, sum2);
            vst1q_s32(C + i * ldc + j, vc);
            
            for (; k < kend; k ++) {
                for (int jj=0; jj<4; jj++) {
                    C[i * ldc + (j+jj)] += A[i * lda + k] * B[lbid++];                    
                }
            }
        }
        for (; j < nend; j++) {
            for (k = kstart; k < kend; k++) {
                C[i * ldc + j] += A[i * lda + k] * B[lbid++];
            }
        }
    }
}

#define TEST_MODULE(func)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(int32_t)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %d\n",                                               \
               #func, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

#define TEST_MODULE_UKERNEL(func, kernel)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(int32_t)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C, kernel); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %d\n",                                               \
               #func#kernel, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

// is_transposed_b: 当前输入的B矩阵是否已经被转置
// is_continuous_b: kernel是否按B连续访问的方式进行访问。
#define TEST_MODULE_PACK_UKERNEL(func, pack, is_transposed_b, kernel, is_continuous_b)                                     \
    do {                                                      \
        memset(mat_c, 0, HEIGHT_C * WIDTH_C * sizeof(int32_t)); \
        time_t stime = clock();                               \
        for (int i = 0; i < 2; i++) {                       \
            func(HEIGHT_C, WIDTH_C, WIDTH_A, mat_a, WIDTH_A, mat_b, WIDTH_B, mat_c, WIDTH_C, pack, is_transposed_b, kernel, is_continuous_b); \
        }                                                                                          \
        double time_used =  double(clock() - stime)/CLOCKS_PER_SEC; \
        printf("%s -> time: ( %f ) s ( %f ) gflops, mean value: %d\n",                                               \
               #func#kernel, time_used, GFLOP/time_used, GetMean(mat_c, HEIGHT_C, WIDTH_C));  \
    } while (0)

int main() {
    pai::prof::CpuSelector cpu_selector;
    cpu_selector.FetchCpuInfo(true);
    // cpu_selector.BindCoreWithId(2);
    cpu_selector.BindCoreWithFreq(true);

    pai::prof::PeakPerfDetector ppf;
    ppf.RunFmla(8);
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
    
    int8_t *mat_a = (int8_t *)malloc(HEIGHT_A * WIDTH_A * sizeof(int8_t));
    int8_t *mat_b = (int8_t *)malloc(HEIGHT_B * WIDTH_B * sizeof(int8_t));
    int32_t *mat_c = (int32_t *)malloc(HEIGHT_C * WIDTH_C * sizeof(int32_t));

    GenMatrix(HEIGHT_A, WIDTH_A, mat_a);
    GenMatrix(HEIGHT_B, WIDTH_B, mat_b);

    // 普通矩阵乘，C=AxB
    TEST_MODULE(GemmV1);                           // 普通实现，对应gemm_fp32的V2
    TEST_MODULE_UKERNEL(GemmTile, UKernelV2);      // 对应gemm_fp32的V6，三层分块
    TEST_MODULE_UKERNEL(GemmTileL2, UKernelV2);    // 二层分块
    TEST_MODULE_UKERNEL(GemmTile, UKernelV3);      // neon首版实现，展开j和k循环。
    TEST_MODULE_UKERNEL(GemmTile, UKernelV4);      // 进一步展开i循环，提高内层循环的计算访存比。对A77无提升, 对于A55有较大提升
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackTB2BC, false, UKernelV5, true); // 将B矩阵排布成访存连续，对A77无收益, 对于A55有较大提升

    // 转置B的gemm
    int8_t *old_mat_b = mat_b;
    int new_ldb;   
    int8_t *BT = (int8_t *)malloc(HEIGHT_B * WIDTH_B * sizeof(int8_t));
    NormalTranspose(WIDTH_B, HEIGHT_B, mat_b, WIDTH_B, BT, &new_ldb);
    mat_b = BT;
    WIDTH_B = new_ldb;

    TEST_MODULE_UKERNEL(GemmTile, UKernelTBV1);
    TEST_MODULE_UKERNEL(GemmTile, UKernelTBV2);
    TEST_MODULE_UKERNEL(GemmTile, UKernelTBV3);   // 不升反降，寄存器用太多？
    TEST_MODULE_UKERNEL(GemmTileL2, UKernelTBV2); // k不拆分，使vdotq_s32可以一直把k算完再汇总
    TEST_MODULE_UKERNEL(GemmTileL2, UKernelTBV4); // 将点积汇总改为转置相加的方式时实现，无收益
    TEST_MODULE_PACK_UKERNEL(GemmTilePackTBL2, PackB2BTC, true, UKernelTBV5, true); // 将B矩阵排布成访存连续, 无收益

    /*
        测试设备: Cortex A77 2.6GHz armv8.2 https://blog.csdn.net/qq_45683435/article/details/103218558
        L1 cache 每个核心配备 64KB 的 L1 指令缓存和 64KB 的 L1 数据缓存，即总共 128KB 的 L1 缓存。
        L1 cache line 大小为 64B (64字节)
        L2 cache 256KB 或 512KB 
        L2 cache line 大小为 64B
        L3 cache 多核共享4M

        64B == 16个4(float)
        16KB == 4096个4(float) == 64 * 64个float

        cortex-A77:
        bind cpus: cpu_4:2600000, cpu_5:2600000, cpu_6:2600000, cpu_7:2600000, 
        GemmV1 -> time: ( 0.205483 ) s ( 3.270343 ) gflops, mean value: 228
        GemmTileUKernelV2 -> time: ( 0.137391 ) s ( 4.891150 ) gflops, mean value: 228
        GemmTileL2UKernelV2 -> time: ( 0.159068 ) s ( 4.224608 ) gflops, mean value: 228
        GemmTileUKernelV3 -> time: ( 0.071577 ) s ( 9.388491 ) gflops, mean value: 228
        GemmTileUKernelV4 -> time: ( 0.071951 ) s ( 9.339689 ) gflops, mean value: 228
        GemmTilePackTBL2UKernelV5 -> time: ( 0.070408 ) s ( 9.544370 ) gflops, mean value: 228
        GemmTileUKernelTBV1 -> time: ( 0.120310 ) s ( 5.585571 ) gflops, mean value: 228
        GemmTileUKernelTBV2 -> time: ( 0.032088 ) s ( 20.942408 ) gflops, mean value: 228
        GemmTileUKernelTBV3 -> time: ( 0.039327 ) s ( 17.087497 ) gflops, mean value: 228
        GemmTileL2UKernelTBV2 -> time: ( 0.021512 ) s ( 31.238378 ) gflops, mean value: 228
        GemmTileL2UKernelTBV4 -> time: ( 0.021469 ) s ( 31.300945 ) gflops, mean value: 228
        GemmTilePackTBL2UKernelTBV5 -> time: ( 0.021867 ) s ( 30.731238 ) gflops, mean value: 228

        cortex-A55:
        cpu numbers 8
        bind cpus: cpu_0:2000000, cpu_1:2000000, cpu_2:2000000, cpu_3:2000000, 
        GemmV1 -> time: ( 0.685754 ) s ( 0.979943 ) gflops, mean value: 241
        GemmTileUKernelV2 -> time: ( 0.308892 ) s ( 2.175518 ) gflops, mean value: 241
        GemmTileL2UKernelV2 -> time: ( 0.777406 ) s ( 0.864413 ) gflops, mean value: 241
        GemmTileUKernelV3 -> time: ( 0.272656 ) s ( 2.464644 ) gflops, mean value: 241
        GemmTileUKernelV4 -> time: ( 0.183999 ) s ( 3.652194 ) gflops, mean value: 241
        GemmTilePackTBL2UKernelV5 -> time: ( 0.152501 ) s ( 4.406528 ) gflops, mean value: 241
        GemmTileUKernelTBV1 -> time: ( 0.498860 ) s ( 1.347071 ) gflops, mean value: 241
        GemmTileUKernelTBV2 -> time: ( 0.161518 ) s ( 4.160527 ) gflops, mean value: 241
        GemmTileUKernelTBV3 -> time: ( 0.152467 ) s ( 4.407511 ) gflops, mean value: 241
        GemmTileL2UKernelTBV2 -> time: ( 0.119827 ) s ( 5.608085 ) gflops, mean value: 241
        GemmTileL2UKernelTBV4 -> time: ( 0.122126 ) s ( 5.502514 ) gflops, mean value: 241
        GemmTilePackTBL2UKernelTBV5 -> time: ( 0.131820 ) s ( 5.097861 ) gflops, mean value: 241
    */
      
    free(mat_a);
    free(mat_b);
    free(mat_c);

    return 0;
}