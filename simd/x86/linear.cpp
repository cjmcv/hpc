/*!
* \brief linear op = gemm(A x Bt).
*/
#include <iostream>
#include <cstring>
#include "time.h"

#include "pocket-ai/util/type.hpp"
#include "immintrin.h"

using namespace pai::util;

float RandomFloat(float min, float max) {
    float random = ((float)rand()) / RAND_MAX; // [0, 1]
    return min + random * (max - min);         // [min, max]
}

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = i%9 + j%9;//RandomFloat(-10, 10);
        }
    }
}

// Just for checking the result.
template <typename T>
float GetMean(const T* mat, const int height, const int width) {
    int num = height * width;
    float total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (sizeof(T) == 4)
                total += mat[i*width + j];
            else if (sizeof(T) == 2)
                total += ConvertBf16ToFp32(mat[i*width + j]);
            else
                printf("The data type T does not match.\n");
        }
    }
    return total / num;
}

void MatrixPrint(const float* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f, ", mat[i * width + j]);
        }
        printf("\n");
    }
}

// Normal method FP32
void LinearNormalFp32(const int M, const int N, const int K,
                    const float *A, const float *B, float *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[j*K + k];
            }
            C[i*N + j] = sum;
        }
    }
}

// -msse m128
void LinearSseFp32(const int M, const int N, const int K,
                const float *A, const float *B, float *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m128 vsum = _mm_setzero_ps();
            for (k = 0; k < K-3; k += 4) {
                __m128 a = _mm_loadu_ps(A + i*K + k);
                __m128 b = _mm_loadu_ps(B + j*K + k);
                vsum = _mm_add_ps(vsum, _mm_mul_ps(a, b));
            }
            float sum = vsum[0] + vsum[1] + vsum[2] + vsum[3];
            for (k; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            C[i*N + j] = sum;
        }
    }
}

// -mavx2
void LinearAvxFp32(const int M, const int N, const int K,
                const float *A, const float *B, float *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m256 vsum = _mm256_setzero_ps();
            for (k = 0; k < K-7; k += 8) {
                __m256 a = _mm256_loadu_ps(A + i*K + k);
                __m256 b = _mm256_loadu_ps(B + j*K + k);
                vsum = _mm256_fmadd_ps(a, b, vsum);
            }
            float sum = vsum[0] + vsum[1] + vsum[2] + vsum[3] 
                      + vsum[4] + vsum[5] + vsum[6] + vsum[7];
            for (k; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            C[i*N + j] = sum;
        }
    }
}

// -mavx512f
void LinearAvx512Fp32(const int M, const int N, const int K,
                    const float *A, const float *B, float *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m512 vsum = _mm512_setzero_ps();
            for (k = 0; k < K-15; k += 16) {
                __m512 a = _mm512_loadu_ps(A + i*K + k);
                __m512 b = _mm512_loadu_ps(B + j*K + k);
                vsum = _mm512_fmadd_ps(a, b, vsum);
            }
            float sum = 0;
            for (k; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            for (int t=0; t<16; t++) {
                sum += vsum[t];                
            }
            C[i*N + j] = sum;
        }
    }
}

// Normal method BF16
void LinearNormalBf16(const int M, const int N, const int K,
                      const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = ConvertBf16ToFp32(A[i * K + k]);
                float b = ConvertBf16ToFp32(B[j * K + k]);
                sum += a * b;
            }
            C[i * N + j] = ConvertFp32ToBf16(sum);
        }
    }
}

// ����֪ʶ��bf16��intel����ģ���bf16��fp32��ָ��������ͬ��ֻ��β�����������򻯣�Ӳ��ʵ���ϸ��򵥡�
//          ��fp16��fp32���ϴ�֧���Ѷȸ��ߣ����Ի����intel cpu֧���µ�bf16��ȴ��֧���ϵ�fp16�������һ��ת����fp32����㡣
// -mavx512bf16
void LinearAvx512Bf16(const int M, const int N, const int K,
                    const bfloat16_t *A, const bfloat16_t *B, bfloat16_t *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m512 vsum = _mm512_setzero_ps();
            for (k = 0; k < K-31; k+=32) {     // һ��32��16λ=512λ
                __m512i ai = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(A + i * K + k));  // ���ڴ��ȡ����������
                __m512bh a_bh = reinterpret_cast<__m512bh>(ai);                                    // ����ǿת
                __m512i bi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + j * K + k));
                __m512bh b_bh = reinterpret_cast<__m512bh>(bi);

                vsum = _mm512_dpbf16_ps(vsum, a_bh, b_bh);  // 32��bf16�ĵ�����㣬a_bh��b_bh������õ�16��fp32����16��fp32��sum��ӡ�
            }
            // float *sum_ptr = (float*)&vsum;
            // float sum2 = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3] + sum_ptr[4] + sum_ptr[5] + sum_ptr[6] + sum_ptr[7] + 
            //              sum_ptr[8] + sum_ptr[9] + sum_ptr[10] + sum_ptr[11] + sum_ptr[12] + sum_ptr[13] + sum_ptr[14] + sum_ptr[15];
            float sum = _mm512_reduce_add_ps(vsum);
            for (k; k < K; k++) {
                float a = ConvertBf16ToFp32(A[i * K + k]);
                float b = ConvertBf16ToFp32(B[j * K + k]);
                sum += a * b;
            }
            C[i * N + j] = ConvertFp32ToBf16(sum);
        }
    }
}

// ģ�⣺���Բ�Ȩ��B��ʼ��ʱ��תΪfp32������A����ԭΪfp16��
//      ����ʱת��Ϊfp32���������C�ڷ���ʱת����fp16.
void LinearAvx512AhBfCh(const int M, const int N, const int K, 
                        HalfToFp32Table *half_table, float *At, float *Ct,
                        const half_t *A, const float *B, half_t *C) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            At[i*K + k] = half_table->LoopupTable(A[i*K + k]);
    LinearAvx512Fp32(M, N, K, At, B, Ct);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = ConvertFp32ToHalf(Ct[i*N+j]);   
}

// Normal method Int8
void LinearNormalInt8(const int M, const int N, const int K,
                    const int8_t *A, const int8_t *B, int32_t *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];    
            }
            C[i * N + j] = sum;
        }
    }
}

// -mavx512vnni -mavx512bw
void LinearAvx512Int8(const int M, const int N, const int K,
                     const int8_t *A, const int8_t *B, int32_t *C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m512i vsum = _mm512_setzero_si512();
            for (k = 0; k < K-63; k += 64) {        // һ��64��8λ=512λ
                __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(A + i * K + k));
                __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + j * K + k));
                
                // _mm512 ������512λ��dp �����bus Byte Unsigned/Signed ������8λ��
                // d��ʾ�ۼ�����double word��32λ��byte 8��word 16���ģ���s�Ǳ��͵���˼����֮��Ӧ��_mm512_dpbusd_epi32�ǲ�����ָ�
                // ע�⣺word��arm����32λ����x86��16λ������
                // epi32 �����32λ���ͣ���vsum��16��int32
                vsum = _mm512_dpbusd_epi32(vsum, a, b);
            }
            int sum = _mm512_reduce_add_epi32(vsum);
            for (k; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}


#define LOOP(cnt, func, A, B, C) {  \
    func(M, N, K, A, B, C);         \
    float mean = GetMean(C, M, N);  \
    time_t stime;                   \
    stime = clock();                \
    for (int i = 0; i < cnt; i++) { \
        func(M, N, K, A, B, C);     \
    }                               \
    double duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC; \
    printf("%s -> time: %f s, result: %f\n", #func, duration, mean); \
} while (0);

int main() {

    int M = 200, N = 300, K = 400;
    float *A = new float[M * K];
    float *B = new float[N * K];
    float *C = new float[M * N];

    srand((unsigned int)time(NULL));
    GenMatrix(M, K, A);
    GenMatrix(N, K, B);

    LOOP(50, LinearNormalFp32, A, B, C);
    LOOP(50, LinearSseFp32, A, B, C);
    LOOP(50, LinearAvxFp32, A, B, C);
    LOOP(50, LinearAvx512Fp32, A, B, C);

    // bf16
    bfloat16_t *A16 = new bfloat16_t[M * K];
    bfloat16_t *B16 = new bfloat16_t[N * K];
    bfloat16_t *C16 = new bfloat16_t[M * N];
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A16[i*K + k] = ConvertFp32ToBf16(A[i*K + k]);
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            B16[j*K + k] = ConvertFp32ToBf16(B[j*K + k]);
    LOOP(50, LinearNormalBf16, A16, B16, C16);
    LOOP(50, LinearAvx512Bf16, A16, B16, C16);

    // half
    float *At = new float[M * K];
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A16[i*K + k] = ConvertFp32ToHalf(A[i*K + k]);
    HalfToFp32Table half_table;
    half_table.Create();
    {
        LinearAvx512AhBfCh(M, N, K, &half_table, A, C,A16, B, C16);
        float mean = GetMean(C, M, N);  
        time_t stime;                   
        stime = clock();                
        for (int i = 0; i < 50; i++) { 
            LinearAvx512AhBfCh(M, N, K, &half_table, A, C,A16, B, C16);  
        }                               
        double duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC; 
        printf("LinearAvx512AhBfCh -> time: %f s, result: %f\n", duration, mean); 
    }

    // int8
    // printf("The input data is truncated and converted from FP32 to INT8, so the result of int8 is different from that of float.\n");
    int8_t *A8 = new int8_t[M * K];
    int8_t *B8 = new int8_t[N * K];
    int32_t *C32 = new int32_t[M * N];
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A8[i*K + k] = A[i*K + k];
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            B8[j*K + k] = B[j*K + k];

    LOOP(50, LinearNormalInt8, A8, B8, C32);
    LOOP(50, LinearAvx512Int8, A8, B8, C32);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A16;
    delete[] B16;
    delete[] C16;
    delete[] A8;
    delete[] B8;
    delete[] C32;

    printf("\nDone!");
    return 0;
}