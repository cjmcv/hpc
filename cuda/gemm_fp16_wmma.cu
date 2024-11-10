// %%cuda
/*!
* \brief gemm: C = A * B.
*/
#include <iostream>
#include "time.h"

#include "pocket-ai/engine/cu/common.hpp"
#include "pocket-ai/engine/cu/common_ptx.hpp"

using namespace pai::cu;

////////////////////////////////////////////////////////////////////////////////

// Initialize the input data.
void GenHalfMatrix(const int height, const int width, half *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = __float2half((float)(rand() % 20 - 10)); // int: -10 ~ 10
            // mat[i*width + j] = __float2half(1); // int: -10 ~ 10
        }
    }
}

// Just for checking the result.
float GetHalfMean(const half* mat, const int height, const int width) {
    int num = height * width;
    float total = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            total += __half2float(mat[i*width + j]);
        }
    }
    return total / num;
}

// Just for checking the result too.
void HalfMatrixPrint(const half* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << __half2float(mat[i*width + j]) << ",";
        }
        std::cout << std::endl;
    }
}

// CPU普通实现版本，主要用于核对后续优化版本结果的正确性
void GemmHostV1(const int M, const int N, const int K,
    const half *A, const int lda,
    const half *B, const int ldb,
    half *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(half) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float acc = 0;
            for (k = 0; k < K; ++k) {
                acc += __half2float(A[i*lda + k])*__half2float(B[k*ldb + j]);
            }
            C[i*ldc + j] = __float2half(acc);
        }
    }
}

// CUDA version 1: 72 ms、
// 基于GemmHostV2直接一一对应改写而成,
// 其中的 bi,bj 使用 blockIdx.x,blockIdx.y 代替
// 其中的 i,j 使用 threadIdx.x,threadIdx.y 代替
// (注意：如GemmHostV2中block应为正方形)
// 所以去掉块内线程i/j和块的bi/bj，只需留下 k 循环.
//
// \ C[ty, tx] = A[ty, k] * B[k, tx]
// for k -> K
//     C[bi*bs + ty, bj*bs + tx] += A[bi*bs + ty, k] * B[k, bj*bs + tx]
__global__ void GemmKernelv1(const int M, const int N, const int K,
                             const half* __restrict__ A, const int lda,
                             const half* __restrict__ B, const int ldb,
                             half* __restrict__ C, const int ldc) {

    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;

    half c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
        c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
    }
    C[gid_y * ldc + gid_x] = c_sub_acc;
}

// wmma v1.
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void GemmWmmaKernelv1(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {
    const int WARP_SIZE = 32;

    const int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int wid_x = gid_x / WARP_SIZE;  // warp维度为(32, 1), 所以x方向要除，y方向不用
    const int wid_y = gid_y;

    // 得到warp id后直接乘以wmma的块大小，就是对应的偏移量，一个warp的线程处理完一个wmma块的计算
    const int fid_x = wid_x * WMMA_N;
    const int fid_y = wid_y * WMMA_M;

    if (fid_y >= M || fid_x >= N) {
        printf("fid_y = %d, fid_x = %d.\n", fid_y, fid_x);
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    for (int fid_K = 0; fid_K < K; fid_K += WMMA_K) {
        wmma::load_matrix_sync(A_frag, A + fid_y * lda + fid_K, lda);
        wmma::load_matrix_sync(B_frag, B + fid_K * ldb + fid_x, ldb);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + fid_y * N + fid_x, C_frag, N, wmma::mem_row_major);
}


// v2 
//   上面v1中wmma::load_matrix_sync是直接从全局内存中搬运数据到寄存器中使用，
// AB矩阵在全局内存中重复访问，常规优化思路是先搬到共享内存中，重复访问操作放在共享内存中进行。
//   这里为了简化索引, 定义block为一维256个线程, 一行的32个线程为一个warp, 则有8个warp。
// 数据分块不围绕block大小进行，直接定义分块大小为BM x BN x BK = 128 x 256 x 32, 
// 即一个block里8个warp需要完成128 x 256 x 32的计算。
// 
// loop k:
//     TA[BM][BK] * TB[BK][BN] => TC[BM][BN]
//   在K循环的每次迭代中，TA有128*32个元素，对应256个线程，则一个线程需要读出16个half元素, 对应2个float4。
// 同理TB有32*256个元素，则一个线程对应32个元素，对应4个float4，以完成从全局内存到共享内存的数据写入。
// 
// TA[128,32] 线程排布：
//     0行: 0, 1, 2, 3 => 4个float4 = 32个half，为一行
//     1行: 0, 1, 2, 3
//     2行：4, 5, 6, 7
//     ...
//   126行: 252, 253, 254, 255, 
//   127行: 252, 253, 254, 255
// 
// TB[256,32] 线程排布：
//     0行: 0, 1, 2, 3 => 4个float4 = 32个half，为一行
//     1行: 0, 1, 2, 3
//     2行：0, 1, 2, 3
//     3行：0, 1, 2, 3
//     4行：4, 5, 6, 7
//     ...
//   252行: 252, 253, 254, 255, 
//   253行: 252, 253, 254, 255,
//   254行: 252, 253, 254, 255, 
//   255行: 252, 253, 254, 255
//
// 计算时fragment布局：
//   处理单元为16x16，则TC[128][256] = 8x16个wmma，即一个warp需要处理16个[16,16] = 4x4个[16,16]
//   w0, w1, w2, w3,  => 一个warp处理临近4x4个16x16
//   w4，w5, w6, w7
// 则4x4个16x16 warp子块计算，则需要4x4的frag_acc，分别累加。
// 
// 因为block子块k方向为32，可凑够2个16，可全部取出一次算完。
// 则可以使用frag_a[4][2]和frag_b[2][4]来计算得到frag_c[4][4].
//
// 《《《 bank conflict 》》》
// https://zhuanlan.zhihu.com/p/603016056
// 共享内存中的bank冲突问题（定义一个warp的不同线程访问同一个bank的不同位置）：
// 如TA[128][32]中，则一行有16个bank，如一个warp 32个线程读取左上角的第一个16x16子块, 则一个线程会读取8个元素
//   0:   b0-b3       b4-b7      b8-b11     b12-b15                  
//   1:  b16-b19     b20-b23    b24-b27     b28-b31
//   2:   b0-b3       b4-b7      b8-b11     b12-b15
//   3:  b16-b19     b20-b23    b24-b27     b28-b31
// 从bank的分布看，一次读取的16x16子块只涉及到了b0-b7和b16-b23，即32个bank只涉及到了16个bank，只有一半，冲突肯定很严重
////////////////
// 为了处理bank冲突，对TA加pad为[128][40], 多了8个fp16，即4个bank，则有
//   0:   b0-b3      b4-b7      b8-b11    b12-b15    b16-b19         
//   1:  b20-b23    b24-b27    b28-b31     b0-b3      b4-b7 
//   2:   b8-b11    b12-b15    b16-b19    b20-b23    b24-b27
//   3:  b28-b31     b0-b3      b4-b7      b8-b11    b12-b15
//   4:  b16-b19    b20-b23    b24-b27    b28-b31     b0-b3
//       ........
//  16:  ........
//  此时的16x16子块涉及到了从b0-b31的所有bank，理论上可以解决bank冲突。
//  需要进一步分析各个线程如何取数据才能知道具体如何使其conflict free，参考mma的ldmatrix(8x8,一行占4个bank)
//  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-load-instruction-ldmatrix
//  猜测可以是：
//   0:   b0-b3      b4-b7      b8-b11    b12-b15    b16-b19
//        t0-t3
//   1:  b20-b23    b24-b27    b28-b31     b0-b3      b4-b7 
//        t4-t7
//   2:   b8-b11    b12-b15    b16-b19    b20-b23    b24-b27
//        t8-t11
//   3:  b28-b31     b0-b3      b4-b7      b8-b11    b12-b15
//       t12-t15
//   4:  b16-b19    b20-b23    b24-b27    b28-b31     b0-b3
//       t16-t19
//   5:   b4-b7      b8-b11    b12-b15    b16-b19    b20-b23
//       t20-t23
//   6:  b24-b27    b28-b31     b0-b3      b4-b7      b8-b11
//       t24-t27
//   7:  b12-b15    b16-b19    b20-b23    b24-b27    b28-b31
//       t28-t31
//  16:  ........
//  则所有线程都不会调用到同一个bank，达到conflict free的效果。

// 一个warp，从smem_a中加载frag_a[4][2], 从smem_b中加载frag_b[2][4]
#define LD_SMEM2FRAG_42x24(frag_a, smem_a, comp_c_frag_m, a_sw, frag_b, smem_b, comp_c_frag_n, b_sw) \
    wmma::load_matrix_sync(frag_a[0][0], &smem_a[(comp_c_frag_m * 64     )*a_sw +  0], a_sw); \
    wmma::load_matrix_sync(frag_a[1][0], &smem_a[(comp_c_frag_m * 64 + 16)*a_sw +  0], a_sw); \
    wmma::load_matrix_sync(frag_a[2][0], &smem_a[(comp_c_frag_m * 64 + 32)*a_sw +  0], a_sw); \
    wmma::load_matrix_sync(frag_a[3][0], &smem_a[(comp_c_frag_m * 64 + 48)*a_sw +  0], a_sw); \
    wmma::load_matrix_sync(frag_a[0][1], &smem_a[(comp_c_frag_m * 64     )*a_sw + 16], a_sw); \
    wmma::load_matrix_sync(frag_a[1][1], &smem_a[(comp_c_frag_m * 64 + 16)*a_sw + 16], a_sw); \
    wmma::load_matrix_sync(frag_a[2][1], &smem_a[(comp_c_frag_m * 64 + 32)*a_sw + 16], a_sw); \
    wmma::load_matrix_sync(frag_a[3][1], &smem_a[(comp_c_frag_m * 64 + 48)*a_sw + 16], a_sw); \
    \
    wmma::load_matrix_sync(frag_b[0][0], &smem_b[          comp_c_frag_n * 64     ], b_sw); \
    wmma::load_matrix_sync(frag_b[0][1], &smem_b[          comp_c_frag_n * 64 + 16], b_sw); \
    wmma::load_matrix_sync(frag_b[0][2], &smem_b[          comp_c_frag_n * 64 + 32], b_sw); \
    wmma::load_matrix_sync(frag_b[0][3], &smem_b[          comp_c_frag_n * 64 + 48], b_sw); \
    wmma::load_matrix_sync(frag_b[1][0], &smem_b[16*b_sw + comp_c_frag_n * 64     ], b_sw); \
    wmma::load_matrix_sync(frag_b[1][1], &smem_b[16*b_sw + comp_c_frag_n * 64 + 16], b_sw); \
    wmma::load_matrix_sync(frag_b[1][2], &smem_b[16*b_sw + comp_c_frag_n * 64 + 32], b_sw); \
    wmma::load_matrix_sync(frag_b[1][3], &smem_b[16*b_sw + comp_c_frag_n * 64 + 48], b_sw); \

// 一个warp，计算[4][2] x [2][4]个frag
#define COMP_FRAG_42x24(frag_c, frag_a, frag_b) \
    for (int i = 0; i < 4; i++) { \
        for (int j = 0; j < 4; j++) { \
            for (int k = 0; k < 2; k++) { \
                wmma::mma_sync(frag_c[i][j], frag_a[i][k], frag_b[k][j], frag_c[i][j]); \
            } \
        } \
    }

// 一个block处理4x4个frag 
#define ST_FRAG_4X4(C, comp_c_frag_m, comp_c_frag_n, frag_c) \
    int store_c_gmem_m = blockIdx.y * BM + comp_c_frag_m * 64; \
    int store_c_gmem_n = blockIdx.x * BN + comp_c_frag_n * 64; \
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N); \
    for (int i = 0; i < 4; i++) { \
        for (int j = 0; j < 4; j++) { \
            wmma::store_matrix_sync(&C[store_c_gmem_addr + i * 16 * N + j * 16], \
                                    frag_c[i][j], N, wmma::mem_row_major); \
        } \
    }

__global__ void GemmWmmaKernelv2(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {
    const int WARP_SIZE = 32;

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half smem_a[BM * (BK + APAD)];
    __shared__ half smem_b[BK * (BN + BPAD)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    // a分块128*32对应256个线程，一个线程读16个数据，为使用向量化读取。
    // 将共享内存按float4来处理，为128*8个half=128*4个float4, 即一个线程读取16/4/2 = 2个float4。
    // 一个float4为8个half，按half取下标时, tid(m,k): 
    //   0(0,0-7), 1(0,8-15), 2(0,16-23), 3(0,24-31), ...
    //   0(1,0-7), 1(1,8-15), 2(1,16-23), 3(1,24-31), ...
    //   4(2,0-7), 5(2,8-15), 6(2,16-23), 7(2,24-31), ...
    //   4(3,0-7), 5(3,8-15), 6(3,16-23), 7(3,24-31), ...
    // 以 (线程号)->m下标 为格式：
    // load_a_smem_m = (0,1,2,3)->0, (4,5,6,7)->2, (8,9,10,11)->4, 
    //                    索引只需以奇数行计算，偶数行加1即可，则乘以2，
    //                    (0,1,2,3)->0, (4,5,6,7)->1*2, (8,9,10,11)->2*2, 
    // 所以 load_a_smem_m = tid/4*2;
    // load_a_smem_k = 0->0, 1->8, 2->16, 3->24, 4->0, 5->8, 6->16, 7->24,...
    //                    即以乘以8递增即可，而最大为24=3*8, %4 即可循环取到0-3.
    // 所以 load_a_smem_k = tid%4*8;
    //
    // b分块32*256对应256个线程，一个线程读32个数据，为使用向量化读取。
    // 将共享内存按float4来处理，为32*64个half4=32*32个float4, 即一个线程读取32/4/2 = 4个float4。
    // 一个float4为8个half，按half取下标时, 线程号(k,n): 
    //   0(0,0-7),  1(0,8-15),  2(0,16-23),  3(0,24-31),  4(0,32-39),  5(0,40-47), ... , 31(0,248-255)
    //   0(1,0-7),  1(1,8-15),  2(1,16-23),  3(1,24-31),  4(1,32-39),  5(1,40-47), ... , 31(1,248-255)
    //   0(2,0-7),  1(2,8-15),  2(2,16-23),  3(2,24-31),  4(2,32-39),  5(2,40-47), ... , 31(2,248-255)
    //   0(3,0-7),  1(3,8-15),  2(3,16-23),  3(3,24-31),  4(3,32-39),  5(3,40-47), ... , 31(3,248-255)
    //  32(4,0-7), 33(4,8-15), 34(4,16-23), 35(4,24-31), ...
    // load_b_smem_k = (0,1,2,3,...31)->0, (32,...,63)->4, (64,...,95)->8, 
    //                    索引只需以4行取1行计算，其他行分别+1+2+3即可，则转为乘以4，
    //                    (0,1,2,3,...31)->0*4, (32,...,63)->1*4, (64,...,95)->2*4, 
    // 所以 load_b_smem_k = tid/32*4;
    // load_b_smem_n = 0->0, 1->8, 2->16, 3->24, ... , 31->248, 32->0, 33->8, ...
    //                    即以乘以8递增即可，而最大为248=31*8, %31 即可循环取到0-31, 也可以用 &31 来得到。
    // 所以 load_b_smem_n = tid%31*8;
    int load_a_smem_m = (threadIdx.x >> 2) << 1;
    int load_a_smem_k = (threadIdx.x &  3) << 3;
    int load_b_smem_k = (threadIdx.x >> 5) << 2;
    int load_b_smem_n = (threadIdx.x & 31) << 3;

    // 全局内存线程索引，其中k方向只需给出线程在block内负责的数据的起始点即可，在k循环时跨block递增访问。
    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, lda);  
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, ldb);

    // warp索引计算fragment
    int wid = threadIdx.x / WARP_SIZE;
    int comp_c_frag_m = wid &  1; // wid => 0-7 -> 01010101
    int comp_c_frag_n = wid >> 1; // wid => 0-7 -> 00112233

    for (int bk = 0; bk < K / BK; bk++) {
        // 一个线程，从gmem内存A中加载2X8个元素到smem_a, 从global内存B中加载4X8个元素到smem_b，FLOAT4为8个half，等于16个字节。
        int a_sw = BK + APAD;
        int b_sw = BN + BPAD;
        FLOAT4(smem_a[(load_a_smem_m    ) * a_sw + load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr          ]);
        FLOAT4(smem_a[(load_a_smem_m + 1) * a_sw + load_a_smem_k]) = FLOAT4(A[load_a_gmem_addr +     lda]);
        FLOAT4(smem_b[(load_b_smem_k    ) * b_sw + load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr          ]);
        FLOAT4(smem_b[(load_b_smem_k + 1) * b_sw + load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr +     ldb]);
        FLOAT4(smem_b[(load_b_smem_k + 2) * b_sw + load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr + 2 * ldb]);
        FLOAT4(smem_b[(load_b_smem_k + 3) * b_sw + load_b_smem_n]) = FLOAT4(B[load_b_gmem_addr + 3 * ldb]);

        load_a_gmem_addr += BK;        // a 矩阵往x方向偏移
        load_b_gmem_addr += BK * ldb;  // b 矩阵往y方向偏移
        __syncthreads();

        // [128,32] * [32, 256] = [128, 256] 按[16*16]划分warp计算区域，则有 [8,2] * [2,16] = [8,16]个子块。
        // 而这里一个block有8个warp，则每个warp需要处理对应C矩阵的16个块，即16次wmma。
        // w0[0-3, 0-3], w2[0-3, 4-7], w4[0-3, 8-11], w6[0-3, 12-15]
        // w1[4-7, 0-3], w3[4-7, 4-7], w5[4-7, 8-11], w7[4-7, 12-15]
        // 每个warp处理临近的4*4个子块。换算为下标
        // w0[0-63, 0-63],   w2[0-63, 64-127],   w4[0-63, 128-191],   w6[0-63, 192-255]
        // w1[64-127, 0-63], w3[64-127, 64-127], w5[64-127, 128-191], w7[64-127, 192-255]
        // =》
        // w0[0-63, 0-63]     = sa[0-63][32] * sb[32][0-63]     (m=0, n=0)
        // w1[64-127, 0-63]   = sa[64-127][32] * sb[32][0-63]   (m=1, n=0)
        // w2[0-63, 64-127]   = sa[0-63][32] * sb[32][64-127]   (m=0, n=1)
        // w3[64-127, 64-127] = sa[64-127][32] * sb[32][64-127] (m=1, n=1)
        // w3[0-63, 128-191]  = sa[0-63][32] * sb[32][128-191]  (m=0, n=2)
        // ...
        LD_SMEM2FRAG_42x24(frag_a, smem_a, comp_c_frag_m, a_sw, 
                           frag_b, smem_b, comp_c_frag_n, b_sw);

        COMP_FRAG_42x24(frag_c, frag_a, frag_b);
        __syncthreads();
    }

    // 一个block处理4x4个frag
    ST_FRAG_4X4(C, comp_c_frag_m, comp_c_frag_n, frag_c);
}

// 初步使用__cvta_generic_to_shared和cp.async指令替换FLOAT4赋值，从gmem拷贝数据到smem
// 以往从gmem拷贝到smem是使用寄存器从gmem读取数据，然后赋值到smem的，中间经过了寄存器。
// 而cp.async是异步拷贝指令，可将数据从gmem越过寄存器直接拷贝到smem (Ampere架构之后)
__global__ void GemmWmmaKernelv3(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {

    const int WARP_SIZE = 32;

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half smem_a[BM * (BK + APAD)];
    __shared__ half smem_b[BK * (BN + BPAD)];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    // https://forums.developer.nvidia.com/t/problem-about-ptx-instruction-cp-async-ca-shared-global/224219/2
    // 需要进行地址空间转换后才能使用cp.async.ca.shared.global
    // 
    int s_a_base_addr = __cvta_generic_to_shared(smem_a); 
    int s_b_base_addr = __cvta_generic_to_shared(smem_b);
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        // 将上面v2中的FLOAT4赋值一一替换成CP_ASYNC_CA
        CP_ASYNC_CA(load_a_smem_addr_0, &A[load_a_gmem_addr        ], 16);
        CP_ASYNC_CA(load_a_smem_addr_1, &A[load_a_gmem_addr +     K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0, &B[load_b_gmem_addr        ], 16);
        CP_ASYNC_CA(load_b_smem_addr_1, &B[load_b_gmem_addr +     N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2, &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3, &B[load_b_gmem_addr + 3 * N], 16);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();

        LD_SMEM2FRAG_42x24(frag_a, smem_a, comp_c_frag_m, (BK + APAD), 
                           frag_b, smem_b, comp_c_frag_n, (BN + BPAD));

        COMP_FRAG_42x24(frag_c, frag_a, frag_b);

        __syncthreads();
    }
    ST_FRAG_4X4(C, comp_c_frag_m, comp_c_frag_n, frag_c);
}

// Double buffer实现，双倍smem进行pingpong，减少gmem->smem和comp smem的相互等待
// gmem0 -> smem0
//    gmem1 -> smem1
//    comp smem0
//    gmem0 -> smem0
//    comp seme1
//    。。。
// comp smem 1
__global__ void GemmWmmaKernelv4(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *smem_a = smem;
    half *smem_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(smem_a);
    int s_b_base_addr = __cvta_generic_to_shared(smem_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    {
        CP_ASYNC_CA(load_a_smem_addr_0, &A[load_a_gmem_addr        ], 16);
        CP_ASYNC_CA(load_a_smem_addr_1, &A[load_a_gmem_addr +     K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0, &B[load_b_gmem_addr        ], 16);
        CP_ASYNC_CA(load_b_smem_addr_1, &B[load_b_gmem_addr +     N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2, &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3, &B[load_b_gmem_addr + 3 * N], 16);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        int step_next_a_byte = smem_sel_next * s_a_db_offset * sizeof(half);
        int step_next_b_byte = smem_sel_next * s_b_db_offset * sizeof(half);
        CP_ASYNC_CA(load_a_smem_addr_0 + step_next_a_byte, &A[load_a_gmem_addr        ], 16);
        CP_ASYNC_CA(load_a_smem_addr_1 + step_next_a_byte, &A[load_a_gmem_addr +     K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0 + step_next_b_byte, &B[load_b_gmem_addr        ], 16);
        CP_ASYNC_CA(load_b_smem_addr_1 + step_next_b_byte, &B[load_b_gmem_addr +     N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2 + step_next_b_byte, &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3 + step_next_b_byte, &B[load_b_gmem_addr + 3 * N], 16);

        half *cur_smem_a = &smem_a[smem_sel * s_a_db_offset];
        half *cur_smem_b = &smem_b[smem_sel * s_b_db_offset];
        LD_SMEM2FRAG_42x24(frag_a, cur_smem_a, comp_c_frag_m, (BK + APAD), 
                           frag_b, cur_smem_b, comp_c_frag_n, (BN + BPAD));

        COMP_FRAG_42x24(frag_c, frag_a, frag_b);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    half *cur_smem_a = &smem_a[smem_sel * s_a_db_offset];
    half *cur_smem_b = &smem_b[smem_sel * s_b_db_offset];
    LD_SMEM2FRAG_42x24(frag_a, cur_smem_a, comp_c_frag_m, (BK + APAD), 
                       frag_b, cur_smem_b, comp_c_frag_n, (BN + BPAD));
   
    COMP_FRAG_42x24(frag_c, frag_a, frag_b);

    ST_FRAG_4X4(C, comp_c_frag_m, comp_c_frag_n, frag_c);
}

__global__ void GemmWmmaKernelv5(const int M, const int N, const int K,
                                const half* __restrict__ A, const int lda,
                                const half* __restrict__ B, const int ldb,
                                half* __restrict__ C, const int ldc) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    if (bx >= N / BN || by >= M / BM)
        return;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half *smem_a = smem;
    half *smem_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(smem_a);
    int s_b_base_addr = __cvta_generic_to_shared(smem_b);

    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 +     (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    {
        CP_ASYNC_CA(load_a_smem_addr_0, &A[load_a_gmem_addr        ], 16);
        CP_ASYNC_CA(load_a_smem_addr_1, &A[load_a_gmem_addr +     K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0, &B[load_b_gmem_addr        ], 16);
        CP_ASYNC_CA(load_b_smem_addr_1, &B[load_b_gmem_addr +     N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2, &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3, &B[load_b_gmem_addr + 3 * N], 16);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        int step_next_a_byte = smem_sel_next * s_a_db_offset * sizeof(half);
        int step_next_b_byte = smem_sel_next * s_b_db_offset * sizeof(half);
        CP_ASYNC_CA(load_a_smem_addr_0 + step_next_a_byte, &A[load_a_gmem_addr        ], 16);
        CP_ASYNC_CA(load_a_smem_addr_1 + step_next_a_byte, &A[load_a_gmem_addr +     K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0 + step_next_b_byte, &B[load_b_gmem_addr        ], 16);
        CP_ASYNC_CA(load_b_smem_addr_1 + step_next_b_byte, &B[load_b_gmem_addr +     N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2 + step_next_b_byte, &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3 + step_next_b_byte, &B[load_b_gmem_addr + 3 * N], 16);

        half *cur_smem_a = &smem_a[smem_sel * s_a_db_offset];
        half *cur_smem_b = &smem_b[smem_sel * s_b_db_offset];
        LD_SMEM2FRAG_42x24(frag_a, cur_smem_a, comp_c_frag_m, (BK + APAD), 
                           frag_b, cur_smem_b, comp_c_frag_n, (BN + BPAD));

        COMP_FRAG_42x24(frag_c, frag_a, frag_b);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    half *cur_smem_a = &smem_a[smem_sel * s_a_db_offset];
    half *cur_smem_b = &smem_b[smem_sel * s_b_db_offset];
    LD_SMEM2FRAG_42x24(frag_a, cur_smem_a, comp_c_frag_m, (BK + APAD), 
                       frag_b, cur_smem_b, comp_c_frag_n, (BN + BPAD));

    COMP_FRAG_42x24(frag_c, frag_a, frag_b);
    ST_FRAG_4X4(C, comp_c_frag_m, comp_c_frag_n, frag_c);
}

float MatrixMulCUDA(int version_id, int step,
                    const int M, const int N, const int K,
                    const half *A, const int lda,
                    const half *B, const int ldb,
                    half *C, const int ldc) {
    GpuTimer gpu_timer;

    const int block_side_size = 32;
    dim3 threads_per_block(block_side_size, block_side_size);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);
    
    // Warm up.
    for (int i=0; i<10; i++) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    cudaMemset(C, 0, sizeof(half) * M * N);

    // Record the start event
    gpu_timer.Start();

    if (version_id == 0) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    else if (version_id == 1) {
        // warp维度可认为是 (32，1)
        //   x方向一个block有32个线程合计一个warp (threads_per_block.x / WARP_SIZE)，共处理1*16个元素，
        // 则为一个block处理16个元素需要多少个block，直接用N/16即可。
        //   y方向一个block有32个线程，合计32个warp (threads_per_block.y / 1)，共处理32*16个元素，
        // 则直接用M/(32*16)即可,32*16数值较大，注意向上取整。
        const int WARP_SIZE = 32;
        const int WMMA_M = 16;
        const int WMMA_N = 16;
        const int WMMA_K = 16;
        const int warps_pre_block_x = threads_per_block.x / WARP_SIZE;
        const int warps_pre_block_y = threads_per_block.y / 1;
        dim3 blocks_per_grid_r( DivCeil(N, warps_pre_block_x*WMMA_N), DivCeil(M, warps_pre_block_y*WMMA_M) );
        GemmWmmaKernelv1<WMMA_M, WMMA_N, WMMA_K> << <blocks_per_grid_r, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (version_id == 2) {
        const int block_size = 256;
        const int BM = 128, BN = 256;
        dim3 blocks_per_grid_r( DivCeil(N, BN), DivCeil(M, BM) );
        GemmWmmaKernelv2<< <blocks_per_grid_r, block_size >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (version_id == 3) {
        const int block_size = 256;
        const int BM = 128, BN = 256;
        dim3 blocks_per_grid_r( DivCeil(N, BN), DivCeil(M, BM) );
        GemmWmmaKernelv3<< <blocks_per_grid_r, block_size >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (version_id == 4) {
        const int block_size = 256;
        const int BM = 128, BN = 256, BK = 32;
        dim3 blocks_per_grid_r( DivCeil(N, BN), DivCeil(M, BM) );
        // 单个线程块处理共享内存的全部容量：Volta为96KB，Turing为64KB.
        // 当内核的共享内存使用量超过48KB时，意味着只有在部分架构下才能使用，
        // 需要使用动态共享内存（而不是静态大小的数组），并使用cudaFuncSetAttribute来显式指定最大共享内存使用量
        cudaFuncSetAttribute(GemmWmmaKernelv4,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304); // 96KB，volta架构的最大值
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half); // 53KB 已超过 48KB
        GemmWmmaKernelv4<< <blocks_per_grid_r, block_size, dsmem >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    else if (version_id == 5) {
        const int block_size = 256;
        const int BM = 128, BN = 256, BK = 32;
        const int NSPLIT = 4096;
        int split_num = (N + NSPLIT - 1) / NSPLIT;
        dim3 blocks_per_grid_r(DivCeil( DivCeil(N, BN), split_num), DivCeil(M, BM), split_num);
        cudaFuncSetAttribute(GemmWmmaKernelv5,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
        GemmWmmaKernelv5<< <blocks_per_grid_r, block_size, dsmem >> >
            (M, N, K, A, lda, B, ldb, C, ldc);
    }
    // Record the stop event
    gpu_timer.Stop();

    return gpu_timer.ElapsedMillis();
}

#define TEST_CUDA_MODULE_UKERNEL(version_id, step)                            \
    do {                                                                      \
        CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice)); \
        CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice)); \
        msec_total = MatrixMulCUDA(version_id, step, height_a, width_b, width_a, d_a, width_a, d_b, width_b, d_c, width_b); \
        CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost)); \
        printf("gpu version %d step %2d -> time: %f s, mean value = %f\n", version_id, step, msec_total/1000.f, GetHalfMean(h_c, height_a, width_b)); \
    } while (0)

int main() {
    int ret = InitEnvironment(0);
    if (ret != 0) {
        printf("Failed to initialize the environment for cuda.");
        return -1;
    }

    // Normal test
    int height_a = 4096, width_a = 4096;
    int height_b = 4096, width_b = 4096;
    // // Test split-k
    // int height_a = 64, width_a = 4096;
    // int height_b = 4096, width_b = 64;
    // // Debug
    // int height_a = 32, width_a = 16;
    // int height_b = 16, width_b = 32;
    if (width_a != height_b) {
        printf("width_a should be equal to height_b.\n");
        return 1;
    }

    const int mem_size_a = sizeof(half) * height_a * width_a;
    const int mem_size_b = sizeof(half) * height_b * width_b;
    const int mem_size_c = sizeof(half) * height_a * width_b;

    half *h_a = (half *)malloc(mem_size_a);
    half *h_b = (half *)malloc(mem_size_b);
    half *h_c = (half *)malloc(mem_size_c);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        printf("Fail to malloc.\n");
        return 1;
    }

    // Initialize 
    srand(time(NULL));
    GenHalfMatrix(height_a, width_a, h_a);
    GenHalfMatrix(height_b, width_b, h_b);

    // CPU
    // time_t t = clock();
    // GemmHostV1(height_a, width_b, width_a, h_a, width_a,h_b, width_b, h_c, width_b);
    // printf("cpu version 1 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetHalfMean(h_c, height_a, width_b));
    // HalfMatrixPrint(h_c, height_a, width_b);

    // GPU
    // Allocate memory in host. 
    float msec_total;
    half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
    CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
    CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

    TEST_CUDA_MODULE_UKERNEL(0, 1);
    TEST_CUDA_MODULE_UKERNEL(1, 1);
    TEST_CUDA_MODULE_UKERNEL(2, 1);
    TEST_CUDA_MODULE_UKERNEL(3, 1);
    TEST_CUDA_MODULE_UKERNEL(4, 1);
    TEST_CUDA_MODULE_UKERNEL(5, 1);
    // printf("Print output C:\n");
    // for (int i=0; i<height_a; i++) {
    //     for (int j=0; j<width_b; j++) {
    //         printf("%f, ", h_c[i*width_b+j]);
    //     }
    //     printf("\n");
    // }

    // Normal test.
    // GPU Device 0: "Tesla T4" with compute capability 7.5 with 40 multi-processors.
    // cpu version 1 -> time: 168.878989 s, mean value = 1023.253418
    // gpu version 1 step  1 -> time: 0.034384 s, mean value = 1023.097839
    // gpu version 2 step  1 -> time: 0.000495 s, mean value = 1023.098633

    // GPU Device 0: "NVIDIA GeForce RTX 3080" with compute capability 8.6 with 68 multi-processors.

    // gpu version 0 step  1 -> time: 0.014940 s, mean value = 1018.071594
    // gpu version 1 step  1 -> time: 0.001727 s, mean value = 1018.270447
    // gpu version 2 step  1 -> time: 0.000460 s, mean value = 1018.270447

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    CleanUpEnvironment();

    return 0;
}
