
#include <iostream>
#include "time.h"

#include "immintrin.h"

void TransposeNormalInt32(int *src, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j <= i; j++) {
            int temp = *(src + i * width + j);
            *(src + i * width + j) = *(src + j * width + i);
            *(src + j * width + i) = temp;
        }
    }
}

void PrintMatrixInt32(int *src, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d, ", src[i * width + j]);
        }
        printf("\n");
    }
}

void PrintMatrixFloat(float *src, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f, ", src[i * width + j]);
        }
        printf("\n");
    }
}
// _mm_unpacklo_epi32
// 浮点可用：_mm_unpackhi_ps / _mm_unpacklo_ps
void Transpose4x4Int32(int *src, int width) {
    // 4个32位组成1个128位
    __m128i S0 = _mm_loadu_si128((__m128i *)(src + 0 * width)); // S0: 0 1 2 3
    __m128i S1 = _mm_loadu_si128((__m128i *)(src + 1 * width)); // S1: 4 5 6 7  
    __m128i S2 = _mm_loadu_si128((__m128i *)(src + 2 * width)); // S2: 8 9 10 11
    __m128i S3 = _mm_loadu_si128((__m128i *)(src + 3 * width)); // S3: 12 13 14 15

                                                                // 以32位为处理单元
    __m128i S01L = _mm_unpacklo_epi32(S0, S1); // S01L: 0 4 1 5
    __m128i S01H = _mm_unpackhi_epi32(S0, S1); // S01H: 2 6 3 7
    __m128i S23L = _mm_unpacklo_epi32(S2, S3); // S23L: 8 12 9 13
    __m128i S23H = _mm_unpackhi_epi32(S2, S3); // S23H: 10 14 11 15

                                              // 以64位为处理单元，即以两个32位作为一个操作数
    _mm_storeu_si128((__m128i *)(src + 0 * width), _mm_unpacklo_epi64(S01L, S23L)); // D0: (0 4) (8 12)
    _mm_storeu_si128((__m128i *)(src + 1 * width), _mm_unpackhi_epi64(S01L, S23L)); // D1: (1 5) (9 13)
    _mm_storeu_si128((__m128i *)(src + 2 * width), _mm_unpacklo_epi64(S01H, S23H)); // D2: (2 6) (10 14)
    _mm_storeu_si128((__m128i *)(src + 3 * width), _mm_unpackhi_epi64(S01H, S23H)); // D3: (3 7) (11 15)
}

void Transpose4x4_F_Kernel(__m128 &row0, __m128 &row1, __m128 &row2, __m128 &row3) {
    // 注：_MM_SHUFFLE中参数排序从高到低排，为3,2,1,0。
    // 掩码正序为(0,1,0,1) => (m1的0位，m1的1位，m2的0位，m2的1位)，则掩码为_MM_SHUFFLE(1,0,1,0)=0x44
    // 掩码正序为(2,3,2,3) => (m1[2],m1[3],m2[2],m2[3]) = _MM_SHUFFLE(3,2,3,2) = 0xEE
    __m128 _Tmp0 = _mm_shuffle_ps((row0), (row1), 0x44); // 0, 1, 4, 5
    __m128 _Tmp2 = _mm_shuffle_ps((row0), (row1), 0xEE); // 2, 3, 6, 7
    __m128 _Tmp1 = _mm_shuffle_ps((row2), (row3), 0x44); // 8, 9,12,13
    __m128 _Tmp3 = _mm_shuffle_ps((row2), (row3), 0xEE); // 10,11,14,15
                                                        // 输入0行应为(0,4,8,12) => (T0[0],T0[2],T1[0],T1[2]) => 掩码正序为(0,2,0,2) => _MM_SHUFFLE(2,0,2,0) = 0x88
                                                        // 输入1行应为(1,5,9,13) => (T0[1],T0[3],T1[1],T1[3]) => 掩码正序为(1,3,1,3) => _MM_SHUFFLE(3,1,3,1) = 0xDD
    row0 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88); // 0, 4, 8,12
    row1 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD); // 1, 5, 9,13
    row2 = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88); // 2, 6,10,14
    row3 = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD); // 3, 7,11,15
}

// 取自xmmintrin.h的_MM_TRANSPOSE4_PS
void Transpose4x4Fp32(float *src, int width) {
    __m128 row0 = _mm_loadu_ps(src + 0 * width); // 0, 1, 2, 3
    __m128 row1 = _mm_loadu_ps(src + 1 * width); // 4, 5, 6, 7
    __m128 row2 = _mm_loadu_ps(src + 2 * width); // 8, 9,10,11
    __m128 row3 = _mm_loadu_ps(src + 3 * width); // 12,13,14,15

    Transpose4x4_F_Kernel(row0, row1, row2, row3);

    _mm_storeu_ps(src + 0 * width, row0); // 0, 4, 8,12
    _mm_storeu_ps(src + 1 * width, row1); // 1, 5, 9,13
    _mm_storeu_ps(src + 2 * width, row2); // 2, 6,10,14
    _mm_storeu_ps(src + 3 * width, row3); // 3, 7,11,15
}

// 沿用Transpose4x4_F_Kernel的掩码组合，最后还需要对4x4矩阵位置进行组合
void Transpose8x8Fp32V1(float *src, int width) {
    __m256 row0 = _mm256_loadu_ps(src + 0 * width); //  0, 1, 2, 3, 4, 5, 6, 7
    __m256 row1 = _mm256_loadu_ps(src + 1 * width); //  8, 9,10,11,12,13,14,15
    __m256 row2 = _mm256_loadu_ps(src + 2 * width); // 16,17,18,19,20,21,22,23
    __m256 row3 = _mm256_loadu_ps(src + 3 * width); // 24,25,26,27,28,29,30,31,
    __m256 row4 = _mm256_loadu_ps(src + 4 * width); // 32,33,34,35,36,37,38,39,
    __m256 row5 = _mm256_loadu_ps(src + 5 * width); // 40,41,42,43,44,45,46,47,
    __m256 row6 = _mm256_loadu_ps(src + 6 * width); // 48,49,50,51,52,53,54,55,
    __m256 row7 = _mm256_loadu_ps(src + 7 * width); // 56,57,58,59,60,61,62,63

    //
    // _MM_SHUFFLE(1, 0, 1, 0)则正序为0101，而处理的是__m256，则实际分为两个__m128,掩码为0101 / 0101
    // 注：重组指令掩码在__m256的浮点重组指令上均分为两段处理
    __m256 _Tmp0 = _mm256_shuffle_ps((row0), (row1), _MM_SHUFFLE(1, 0, 1, 0)); // 0, 1, 8, 9 /  4, 5,12,13
    __m256 _Tmp2 = _mm256_shuffle_ps((row0), (row1), _MM_SHUFFLE(3, 2, 3, 2)); // 2, 3,10,11 /  6, 7,14,15
    __m256 _Tmp1 = _mm256_shuffle_ps((row2), (row3), _MM_SHUFFLE(1, 0, 1, 0)); //16,17,24,25 / 20,21,28,29
    __m256 _Tmp3 = _mm256_shuffle_ps((row2), (row3), _MM_SHUFFLE(3, 2, 3, 2)); //18,19,26,27 / 22,23,30,31

    __m256 _Tmp4 = _mm256_shuffle_ps((row4), (row5), _MM_SHUFFLE(1, 0, 1, 0)); //32,33,40,41 / 36,37,44,45
    __m256 _Tmp6 = _mm256_shuffle_ps((row4), (row5), _MM_SHUFFLE(3, 2, 3, 2)); //34,35,42,43 / 38,39,46,47
    __m256 _Tmp5 = _mm256_shuffle_ps((row6), (row7), _MM_SHUFFLE(1, 0, 1, 0)); //48,49,56,57 / 52,53,60,61
    __m256 _Tmp7 = _mm256_shuffle_ps((row6), (row7), _MM_SHUFFLE(3, 2, 3, 2)); //50,51,58,59 / 54,55,62,63

                                                                              //

    __m256 Tmp0 = _mm256_shuffle_ps(_Tmp0, _Tmp1, _MM_SHUFFLE(2, 0, 2, 0)); // 0, 8,16,24 / 4,12,20,28
    __m256 Tmp1 = _mm256_shuffle_ps(_Tmp0, _Tmp1, _MM_SHUFFLE(3, 1, 3, 1)); // 1, 9,17,25 / 5,13,21,29
    __m256 Tmp2 = _mm256_shuffle_ps(_Tmp2, _Tmp3, _MM_SHUFFLE(2, 0, 2, 0)); // 2,10,18,26 / 6,14,22,30
    __m256 Tmp3 = _mm256_shuffle_ps(_Tmp2, _Tmp3, _MM_SHUFFLE(3, 1, 3, 1)); // 3,11,19,27 / 7,15,23,31

    __m256 Tmp4 = _mm256_shuffle_ps(_Tmp4, _Tmp5, _MM_SHUFFLE(2, 0, 2, 0)); //32,40,48,56 / 36,44,52,60
    __m256 Tmp5 = _mm256_shuffle_ps(_Tmp4, _Tmp5, _MM_SHUFFLE(3, 1, 3, 1)); //33,41,49,57 / 37,45,53,61
    __m256 Tmp6 = _mm256_shuffle_ps(_Tmp6, _Tmp7, _MM_SHUFFLE(2, 0, 2, 0)); //34,42,50,58 / 38,46,54,62
    __m256 Tmp7 = _mm256_shuffle_ps(_Tmp6, _Tmp7, _MM_SHUFFLE(3, 1, 3, 1)); //35,43,51,59 / 39,47,55,63

    //// 用_mm256_extractf128_ps重排4X4的子矩阵位置，耗时过大！
    //// _mm256_extractf128_ps(Tmp4, 1) = (32,40,48,56), _mm256_extractf128_ps(Tmp0, 0) = (0, 8,16,24)
    //// (src + 0 * width) => _mm256_set_m128(hi, lo) = (0, 8,16,24),(32,40,48,56)
    //_mm256_storeu_ps(src + 0 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp4, 1), _mm256_extractf128_ps(Tmp0, 0)));
    //_mm256_storeu_ps(src + 1 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp5, 1), _mm256_extractf128_ps(Tmp1, 0)));
    //_mm256_storeu_ps(src + 2 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp6, 1), _mm256_extractf128_ps(Tmp2, 0)));
    //_mm256_storeu_ps(src + 3 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp7, 1), _mm256_extractf128_ps(Tmp3, 0)));
    //_mm256_storeu_ps(src + 4 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp4, 0), _mm256_extractf128_ps(Tmp0, 1)));
    //_mm256_storeu_ps(src + 5 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp5, 0), _mm256_extractf128_ps(Tmp1, 1)));
    //_mm256_storeu_ps(src + 6 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp6, 0), _mm256_extractf128_ps(Tmp2, 1)));
    //_mm256_storeu_ps(src + 7 * width, _mm256_set_m128(_mm256_extractf128_ps(Tmp7, 0), _mm256_extractf128_ps(Tmp3, 1)));

    // 用_mm256_permute2f128_ps重排4X4的子矩阵位置。
    // _mm256_permute2f128_ps把两个256位数据按128为单位，以m0低位、m0高位、m1低位、m1高位分别编号为0，1，2，3
    // 控制位以16进制(控制位均以二进制或十六进制等表示，所以低位是在右边)，
    // 控制位如0x20,则输出为(m0低位对应0，m1低位对应2);0x31,则输出为（m0高位对应1,m1高位对应3）
    _mm256_storeu_ps(src + 0 * width, _mm256_permute2f128_ps(Tmp0, Tmp4, 0x20));
    _mm256_storeu_ps(src + 1 * width, _mm256_permute2f128_ps(Tmp1, Tmp5, 0x20));
    _mm256_storeu_ps(src + 2 * width, _mm256_permute2f128_ps(Tmp2, Tmp6, 0x20));
    _mm256_storeu_ps(src + 3 * width, _mm256_permute2f128_ps(Tmp3, Tmp7, 0x20));
    _mm256_storeu_ps(src + 4 * width, _mm256_permute2f128_ps(Tmp0, Tmp4, 0x31));
    _mm256_storeu_ps(src + 5 * width, _mm256_permute2f128_ps(Tmp1, Tmp5, 0x31));
    _mm256_storeu_ps(src + 6 * width, _mm256_permute2f128_ps(Tmp2, Tmp6, 0x31));
    _mm256_storeu_ps(src + 7 * width, _mm256_permute2f128_ps(Tmp3, Tmp7, 0x31));
}

// 基于Transpose4x4_F_Kernel扩展为8x8。
void Transpose8x8Fp32V2(float *src, int width) {
    __m128 row00 = _mm_loadu_ps(src + 0 * width); // 0, 1, 2, 3，4, 5, 6, 7
    __m128 row01 = _mm_loadu_ps(src + 0 * width + 4);
    __m128 row10 = _mm_loadu_ps(src + 1 * width); // 8, 9,10,11,12,13,14,15
    __m128 row11 = _mm_loadu_ps(src + 1 * width + 4);
    __m128 row20 = _mm_loadu_ps(src + 2 * width);
    __m128 row21 = _mm_loadu_ps(src + 2 * width + 4);
    __m128 row30 = _mm_loadu_ps(src + 3 * width);
    __m128 row31 = _mm_loadu_ps(src + 3 * width + 4);
    __m128 row40 = _mm_loadu_ps(src + 4 * width);
    __m128 row41 = _mm_loadu_ps(src + 4 * width + 4);
    __m128 row50 = _mm_loadu_ps(src + 5 * width); //
    __m128 row51 = _mm_loadu_ps(src + 5 * width + 4);
    __m128 row60 = _mm_loadu_ps(src + 6 * width); //
    __m128 row61 = _mm_loadu_ps(src + 6 * width + 4);
    __m128 row70 = _mm_loadu_ps(src + 7 * width); //
    __m128 row71 = _mm_loadu_ps(src + 7 * width + 4);

    // 4个4x4子块分别做转置
    Transpose4x4_F_Kernel(row00, row10, row20, row30);
    Transpose4x4_F_Kernel(row01, row11, row21, row31);
    Transpose4x4_F_Kernel(row40, row50, row60, row70);
    Transpose4x4_F_Kernel(row41, row51, row61, row71);

    // 以4x4子块为单元做2x2转置
    // 0号子块不动
    _mm_storeu_ps(src + 0 * width, row00);
    _mm_storeu_ps(src + 1 * width, row10);
    _mm_storeu_ps(src + 2 * width, row20);
    _mm_storeu_ps(src + 3 * width, row30);
    // 2号子块挪到1号子块位置
    _mm_storeu_ps(src + 0 * width + 4, row40);
    _mm_storeu_ps(src + 1 * width + 4, row50);
    _mm_storeu_ps(src + 2 * width + 4, row60);
    _mm_storeu_ps(src + 3 * width + 4, row70);
    // 1号子块挪到2号子块位置
    _mm_storeu_ps(src + 4 * width, row01);
    _mm_storeu_ps(src + 5 * width, row11);
    _mm_storeu_ps(src + 6 * width, row21);
    _mm_storeu_ps(src + 7 * width, row31);
    // 3号子块不动
    _mm_storeu_ps(src + 4 * width + 4, row41);
    _mm_storeu_ps(src + 5 * width + 4, row51);
    _mm_storeu_ps(src + 6 * width + 4, row61);
    _mm_storeu_ps(src + 7 * width + 4, row71);
}

#define LOOP(cnt, type_flag, func, data, dim) {      \
    printf("\n%s transported result: \n", #func);   \
    func(data, dim);                \
    if (type_flag == 0)             \
        PrintMatrixInt32((int*)data, dim);  \
    else                            \
        PrintMatrixFloat((float*)data, dim);\
    printf("\n");                   \
    time_t stime;                   \
    stime = clock();                \
    for (int i = 0; i < cnt; i++) { \
        func(data, dim);            \
    }                               \
    double duration = static_cast<double>(clock() - stime) / CLOCKS_PER_SEC; \
    printf("%s -> time: %f s\n", #func, duration);               \
} while (0);

void TransposeTest4x4() {

    int height = 4;
    int width = 4;

    int src_int[16];
    int src_int_2[16];
    float src_float[16];
    for (int i = 0; i < 16; i++) {
        src_int[i] = i;
        src_int_2[i] = i;
        src_float[i] = i;
    }

    printf("Before:\n");
    PrintMatrixInt32(src_int, width);

    int iter = 100000000;
    LOOP(iter, 0, TransposeNormalInt32, src_int, width);
    LOOP(iter, 0, Transpose4x4Int32, src_int, width);
    LOOP(iter, 1, Transpose4x4Fp32, src_float, width);
}

void TransposeTest8x8() {

    int height = 8;
    int width = 8;

    int src_int[64];
    float src_float_v1[64];
    float src_float_v2[64];
    for (int i = 0; i < 64; i++) {
        src_int[i] = i;
        src_float_v1[i] = i;
        src_float_v2[i] = i;
    }

    printf("Before.\n");
    PrintMatrixInt32(src_int, width);

    int iter = 100000000;
    LOOP(iter, 0, TransposeNormalInt32, src_int, width);
    LOOP(iter, 1, Transpose8x8Fp32V1, src_float_v1, width);
    LOOP(iter, 1, Transpose8x8Fp32V2, src_float_v2, width);
}

int main() {

    //Transpose4x4_I_Normal->time: 218
    //Transpose4x4Int32->time : 54
    //Transpose4x4Fp32->time : 53
    TransposeTest4x4();

    printf("\n\n/////////////////////////////////////\n\n");

    //TransposeNormalInt32 -> time: 56
    //Transpose8x8Fp32V1->time: 14
    //Transpose8x8Fp32V2->time : 23
    TransposeTest8x8();

    printf("Done!");
    return 0;
}