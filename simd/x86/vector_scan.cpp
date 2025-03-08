/*!
* \brief Scan. Prefix Sum.
* \example: input: 1,2,3,4
*           operation: Add
*           ouput: 1,3,6,10 (out[i]=sum(in[0:i]))
*/

#include <iostream>
#include <string.h>
#include "time.h"

#include <immintrin.h>

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

inline __m256 ScanM256(__m256 x) {
    __m256 t0, t1;
    //shift1_SIMD + add
    t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
    //shift2_SIMD + add
    t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
    //shift3_SIMD + add
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 41));
    return x;
}

void VectorScanSIMDv2(const float *vec_in, const int len, float *vec_out) {
    __m256 offset = _mm256_setzero_ps();
    for (int i = 0; i < len; i += 8) {
        __m256 x = _mm256_loadu_ps(vec_in + i);
        __m256 y = ScanM256(x);
        y = _mm256_add_ps(y, offset);
        _mm256_storeu_ps(vec_out + i, y);
        // broadcast last element
        __m256 t0 = _mm256_permute2f128_ps(y, y, 0x11);
        offset = _mm256_permute_ps(t0, 0xff);
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
    double duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC;
    printf("Normal -> time: %f s, result: %f.\n", duration, vec_out[len - 1]);

    memset(vec_out, 0, sizeof(float) * len);

    stime = clock();
    for (int i = 0; i < loops; i++)
        VectorScanSIMDv1(vec_in, len, vec_out);
    duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC;
    printf("SIMDv1 -> time: %f s, result: %f.\n", duration, vec_out[len - 1]);

    stime = clock();
    for (int i = 0; i < loops; i++)
        VectorScanSIMDv2(vec_in, len, vec_out);
    duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC;
    printf("SIMDv2 -> time: %f s, result: %f.\n", duration, vec_out[len - 1]);

    delete[] vec_in;
    delete[] vec_out;

    printf("Done!");
    return 0;
}