// %%cuda
/*!
* \brief gemm: C = A * B.
*/
#include <iostream>
#include "time.h"

#include "pocket-ai/engine/cu/common.hpp"
using namespace pai::cu;

////////////////////////////////////////////////////////////////////////////////

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // mat[i*width + j] = ((float)rand() / RAND_MAX) * 200 - 100; // float: -100 ~ 100
            mat[i*width + j] = (float)(rand() % 200 - 100); // int: -100 ~ 100
            // printf("%f, ", mat[i*width + j]);
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

// Just for checking the result too.
void MatrixPrint(const float* mat, const int height, const int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << mat[i*width + j] << ",";
        }
        std::cout << std::endl;
    }
}

// CPU version 1: 1583 ms
// 普通实现版本
void GemmHostV1(const int M, const int N, const int K,
    const float *A, const int lda,
    const float *B, const int ldb,
    float *C, const int ldc) {
    int i, j, k;
    memset(C, 0, sizeof(float) * ldc * M);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < K; ++k) {
                C[i*ldc + j] += A[i*lda + k]*B[k*ldb + j];
            }
        }
    }
}

// CPU version 2: 3389 ms
// 按i和j方向分块的矩阵乘法，便于改写成cuda
// （暂时省略边界处理）
void GemmHostV2(const int M, const int N, const int K,
                const float *A, const int lda,
                const float *B, const int ldb,
                float *C, const int ldc) {
    int bi, bj;
    int i, j, k;
    const int block_size = 32;
    int block_num_M = M / block_size;
    int block_num_N = N / block_size;
    memset(C, 0, sizeof(float) * ldc * M);

    // Loop over all of the blocks.
    for (bi = 0; bi < block_num_M; ++bi) {
        for (bj = 0; bj < block_num_N; ++bj) {
            // Loop over all of the elements in a block.
            for (i = bi*block_size; i < (bi + 1)*block_size; ++i) {
                for (j = bj*block_size; j < (bj + 1)*block_size; ++j) { 
                    for (k = 0; k < K; ++k) {
                        C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
                    }
                }
            }
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
                             const float* __restrict__ A, const int lda,
                             const float* __restrict__ B, const int ldb,
                             float* __restrict__ C, const int ldc) {

    const int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (gid_x >= N || gid_y >= M) {
        return;
    }

    float c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
        c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
    }
    C[gid_y * ldc + gid_x] = c_sub_acc;
}

// CUDA version 2.
// 使用共享内存优化：先将数据从全局内存拷贝到共享内存，在共享内存中进行乘加运算，最后写回全局内存
//    因为共享内存以block划分，所以需要将逐个block的数据填充到shared[threadIdx.y][threadIdx.x]中，
// 则A和B矩阵均往各自K方向取block的数据进行填充。所以k方向多拆一个循环来索引块。
// 最终从多次读取全局内存计算 变成 一次读取全局内存到共享内存，多次读取共享内存计算
// 参考host端三层循环，对于最内层循环，A读取会重复 j 次，B读取会重复 i 次
// ps: 用template <int BLOCK_SIZE>的原因是kernel内以固定大小的方式开辟共享内存空间，无法使用变量blockDim
template <int BLOCK_SIZE>
__global__ void GemmKernelv2(const int M, const int N, const int K,
                             const float* __restrict__ A, const int lda,
                             const float* __restrict__ B, const int ldb,
                             float* __restrict__ C, const int ldc) {

    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x >= N || gid_y >= M) {
        return;
    }

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    float c_sub_acc = 0;
    // 按 K 方向分块读入共享内存，一次读一个block
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        a_shared[threadIdx.y][threadIdx.x] = A[gid_y * lda + (bk + threadIdx.x)];
        b_shared[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * ldb + gid_x];
        // 等待块内线程同步
        __syncthreads();

        // 计算块内元素
        for (int k = 0; k < BLOCK_SIZE; k++) {
            c_sub_acc += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
        __syncthreads();
    }

    C[gid_y * ldc + gid_x] += c_sub_acc;
}

// CUDA version 3.
//   基于v2的split-k实现，主要针对m和n很小，但k很大的情况。这种情况常见于conv的im2col得到的矩阵中（height/width小channel大的conv）。 
// 假设m和n均为64，而k为3200，此时观察v3会发现，只有一个block的线程在执行，而block里每个线程的计算任务量随k的增大而增大，
// 则会造成使用的线程数少，硬件资源无法充分调用，且每个线程使用周期长的问题。
//   具体做法：
//   把gemm的k方向split成多个k_n更小的k_size，从而得到了k_n个 [m, k_tile] x [k_tile, n]矩阵乘，
// 每个矩阵乘的k loop大小缩短，从而每个线程的计算时间缩短，并且可以创建更多的线程数量来执行计算。
//   为简化方案，令split-k为2，并基于v2修改（出于简化思路考虑，与其他版本的优化方式不冲突）：
// 线程数x方向多一倍，按M*2N的大小分配，即原线程(i,j)对应C矩阵的(i,j)元素的计算，
// 现在是(i,j)[0~M,0~N],(i, j-N)[0~M,N~2N]两个线程同时对(i,j)元素负责。
// 而(i,j)负责k从0到N/2的部分，(i, j-N)负责从N/2到N的部分。最后两个线程reduce到一起输出给C(i,j)
//// CPU 分析步骤（按k方向两段拆分）：
// 1）将内层循环拆分
// for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//         c_sub_acc1 = 0, c_sub_acc2 = 0;
//         for (k = 0; k < K/2; k++) {
//             c_sub_acc1 += A[i][k] * B[k][j];
//         }
//         for (k = K/2; k < K; k++) {
//             c_sub_acc2 += A[i][k] * B[k][j];
//         }
//         C[i * ldc + j] += c_sub_acc1 + c_sub_acc2;
//     }
// }
// 2）因要划分不同线程处理不同的k的部分，所以要从i和j两层大循环做拆分，
// 因只split为2段，线程数多一倍，这里选取x方向多一倍线程，则需拆分j循环。
// for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//         c_sub_acc1 = 0;
//         for (k = 0; k < K/2; k++) {
//             c_sub_acc1 += A[i][k] * B[k][j];
//         }
//         C[i * ldc + j] += c_sub_acc1;
//     }
// }
// for (i = 0; i < M; ++i) {
//     for (j = N; j < 2*N; ++j) {
//         nj = j-N;
//         c_sub_acc2 = 0;
//         for (k = K/2; k < K; k++) {
//             c_sub_acc2 += A[i][k] * B[k][nj];
//         }
//         C[i * ldc + nj] += c_sub_acc2;
//     }
// }
template <int BLOCK_SIZE>
__global__ void GemmKernelv3(const int M, const int N, const int K,
                             const float* __restrict__ A, const int lda,
                             const float* __restrict__ B, const int ldb,
                             float* __restrict__ C, const int ldc) {

    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid_x >= 2*N || gid_y >= M) {
        return;
    }

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    float c_sub_acc_1 = 0;
    float c_sub_acc_2 = 0;
    // 按 K 方向分块读入共享内存，一次读一个block
    if (gid_x < N) {
        for (int bk = 0; bk < K/2; bk += BLOCK_SIZE) {
            a_shared[threadIdx.y][threadIdx.x] = A[gid_y * lda + (bk + threadIdx.x)];
            b_shared[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * ldb + gid_x];
            // 等待块内线程同步
            __syncthreads();

            // 计算块内元素
            for (int k = 0; k < BLOCK_SIZE; k++) {
                c_sub_acc_1 += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
            }
            // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
            __syncthreads();
        }
        atomicAdd(&C[gid_y * ldc + gid_x], c_sub_acc_1);
    }
    else {
        for (int bk = K/2; bk < K; bk += BLOCK_SIZE) {
            a_shared[threadIdx.y][threadIdx.x] = A[gid_y * lda + (bk + threadIdx.x)];
            b_shared[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * ldb + gid_x-N];
            // 等待块内线程同步
            __syncthreads();

            // 计算块内元素
            for (int k = 0; k < BLOCK_SIZE; k++) {
                c_sub_acc_2 += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
            }
            // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
            __syncthreads();
        }
        atomicAdd(&C[gid_y * ldc + gid_x-N], c_sub_acc_2);
    }
}

// CUDA version 4.
//   分析v2，计算的过程实质为全局内存->共享内存->寄存器内存，
// 而v2的最内层k循环中需重复访问的数据存在于共享内存中，就会有重复的从共享内存读取数据到寄存器的操作。
// v3的优化思路是使重复读取数据进行计算的操作放到更快的寄存器中完成，提高共享内存的计算访存比。
// 
// 如无法理解这个重复读取的过程，可以考虑把M和N的循环放回，转为串行代码，观察a和b的读取次数：
// for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//         for (k = 0; k < BLOCK_SIZE; k++) {
//             c_sub_acc += a_shared[i][k] * b_shared[k][j];
//         }
//     }
// }
// 可以看到对于a_shared[i][k]来说，每个元素重复读取了j次，b_shared[k][j]则对应重复了i次。
// 则v3的思路是将这个重复从共享内存读取数据做计算的操作，转变为从共享内存中读取一次，在寄存器重复读取数据计算。
// 可考虑子在一次读取到共享内存后，再进行分块一次读取到寄存器中，则对内存k循环做一次分块操作。提高与共享内存的计算访存比
// for (i = 0; i < M; i+=STEP) {
//     for (j = 0; j < N; j+=STEP) {
//         for (k = 0; k < BLOCK_SIZE; k++) {
//             for (si = 0; si < STEP; si++) {
//                 for (sj = 0; sj < STEP; sj++) {
//                     // c要对应i和j进行累加，i和j放到内层，则需要用STEP*STEP个变量去做累加
//                     c_sub_acc[i+si][j+sj] += a_shared[i+si][k] * b_shared[k][j+sj];
//                 }
//             }
//         }
//     }
// }
// 看最里面两层循环，a和b涉及的元素各有STEP个，计算次数则是STEP*STEP，所以可以先读到寄存器中，再计算
// for (i = 0; i < M; i+=STEP) {
//     for (j = 0; j < N; j+=STEP) {
//         for (k = 0; k < BLOCK_SIZE; k++) {
//             for (s = 0; s < STEP; s++) {
//                 a_reg[s] = a_shared[i+s][k];
//                 b_reg[s] = b_shared[k][j+s];
//             }
//             for (si = 0; si < STEP; si++) {
//                 for (sj = 0; sj < STEP; sj++) {
//                     // c要对应i和j进行累加，i和j放到内层，则需要用STEP*STEP个变量去做累加
//                     c_sub_acc[i+si][j+sj] += a_reg[i+si] * b_reg[k][j+sj];
//                 }
//             }
//         }
//     }
// }
// 观察最外两层i和j循环，在cuda改写时，i和j会被线程的x,y调度隐藏掉。而i和j需要按STEP递增，
// 同步到cuda则意味着线程(x,y)要处理STEP*STEP个元素，而非原来的1个元素。
// 则线程布局时，可以是block线程数不变，block数量在x和y方向上减少STEP倍；或减少block内线程数，维持block数量不变。
template <int BLOCK_SIZE, int STEP>
__global__ void GemmKernelv4(const int M, const int N, const int K,
                             const float* __restrict__ A, const int lda,
                             const float* __restrict__ B, const int ldb,
                             float* __restrict__ C, const int ldc) {
    
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    float a_reg[STEP] = {0};
    float b_reg[STEP] = {0};
    float c_reg[STEP][STEP] = {{0}};
    __shared__ float a_shared[BLOCK_SIZE*STEP][BLOCK_SIZE*STEP];
    __shared__ float b_shared[BLOCK_SIZE*STEP][BLOCK_SIZE*STEP];

    int gid_sx = gid_x * STEP;
    int gid_sy = gid_y * STEP;
    int tid_sx = threadIdx.x * STEP;
    int tid_sy = threadIdx.y * STEP;
    if (gid_sx >= N || gid_sy >= M) {
        return;
    }

    // 按 K 方向分块读入共享内存，一次读取临近的四个block, 一个线程处理四个元素
    for (int bk = 0; bk < K; bk += BLOCK_SIZE*STEP) {
        for (int si=0; si<STEP; si++) {
            for (int sj=0; sj<STEP; sj++) {
                a_shared[tid_sy+si][tid_sx+sj] = A[(gid_sy+si) * lda + (bk + tid_sx+sj)];
                b_shared[tid_sy+si][tid_sx+sj] = B[(bk + tid_sy+si) * ldb + gid_sx+sj];
            }
        }
        
        // 等待块内线程同步
        __syncthreads();

        // 计算块内元素, 每个线程处理临近四个元素（如STEP==2）
        // for (int k = 0; k < BLOCK_SIZE*STEP; k++) {
        //     for (int si=0; si<STEP; si++) {
        //         for (int sj=0; sj<STEP; sj++) {
        //             c_reg[si][sj] += a_shared[tid_sy+si][k] * b_shared[k][tid_sx+sj];
        //         }
        //     }
        // }
        for (int k = 0; k < BLOCK_SIZE*STEP; k++) {
            // 对于a，读取共享内存到寄存器 STEP 次，计算时读取寄存器 STEP*STEP 次; 对于b亦然。
            for (int s = 0; s < STEP; s++) {
                a_reg[s] = a_shared[tid_sy+s][k];
                b_reg[s] = b_shared[k][tid_sx+s];
            }
            for (int si=0; si<STEP; si++) {
                for (int sj=0; sj<STEP; sj++) {
                    c_reg[si][sj] += a_reg[si] * b_reg[sj];
                }
            }
        }

        // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
        __syncthreads();
    }

    for (int si=0; si<STEP; si++) {
        for (int sj=0; sj<STEP; sj++) {
            C[(gid_sy+si) * ldc + gid_sx+sj] += c_reg[si][sj];
            // printf("%f(%d, %d) \n", C[(gid_sy+si) * ldc + gid_sx+sj], (gid_sy+si), gid_sx+sj);
        }
    }
}

template <int BLOCK_SIZE, int STEP>
__global__ void GemmKernelv5(const int M, const int N, const int K,
                             const float* __restrict__ A, const int lda,
                             const float* __restrict__ B, const int ldb,
                             float* __restrict__ C, const int ldc) {
    
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (gid_x >= N || gid_y >= M) {
        return;
    }

    // float4 a_reg[STEP] = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    // float4 b_reg[STEP] = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    float4 c_reg[STEP] = {make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    __shared__ float4 a_shared[BLOCK_SIZE*STEP][BLOCK_SIZE];
    __shared__ float4 b_shared[BLOCK_SIZE*STEP][BLOCK_SIZE];

    int gid_sx = gid_x * STEP;
    int gid_sy = gid_y * STEP;    
    int tid_sx = threadIdx.x * STEP;
    int tid_sy = threadIdx.y * STEP;
    if (gid_sx >= N || gid_sy >= M) {
        return;
    }

    // 按 K 方向分块读入共享内存，一次读取临近的四个block, 一个线程处理四个元素
    for (int bk = 0; bk < K; bk += BLOCK_SIZE*STEP) {
        for (int si=0; si<STEP; si++) {
            a_shared[tid_sy+si][tid_sx/STEP] = FLOAT4(A[(gid_sy+si) * lda + (bk + tid_sx)]);
            b_shared[tid_sy+si][tid_sx/STEP] = FLOAT4(B[(bk + tid_sy+si) * ldb + gid_sx]);
        }
        
        // 等待块内线程同步
        __syncthreads();

        // 计算块内元素, 每个线程处理临近四个元素
        // for (int k = 0; k < BLOCK_SIZE*STEP; k++) {
        //     for (int si=0; si<STEP; si++) {
        //         for (int sj=0; sj<STEP; sj++) {
        //             c_reg[si][sj] += a_shared[tid_sy+si][k] * b_shared[k][tid_sx+sj];
        //         }
        //     }
        // }

        for (int k = 0; k < BLOCK_SIZE*STEP; k+=STEP) {
            // // 从共享内存读到寄存器的次数一致，下面写法是没有效果的
            // a_reg[0] = a_shared[tid_sy + 0][k/STEP];
            // a_reg[1] = a_shared[tid_sy + 1][k/STEP];
            // a_reg[2] = a_shared[tid_sy + 2][k/STEP];
            // a_reg[3] = a_shared[tid_sy + 3][k/STEP];

            // b_reg[0] = b_shared[k+0][tid_sx / STEP];
            // b_reg[1] = b_shared[k+1][tid_sx / STEP];
            // b_reg[2] = b_shared[k+2][tid_sx / STEP];
            // b_reg[3] = b_shared[k+3][tid_sx / STEP];

            // K0
            c_reg[0].x += a_shared[tid_sy + 0][k/STEP].x * b_shared[k][tid_sx / STEP].x;
            c_reg[0].y += a_shared[tid_sy + 0][k/STEP].x * b_shared[k][tid_sx / STEP].y;
            c_reg[0].z += a_shared[tid_sy + 0][k/STEP].x * b_shared[k][tid_sx / STEP].z;
            c_reg[0].w += a_shared[tid_sy + 0][k/STEP].x * b_shared[k][tid_sx / STEP].w;

            c_reg[1].x += a_shared[tid_sy + 1][k/STEP].x * b_shared[k][tid_sx / STEP].x;
            c_reg[1].y += a_shared[tid_sy + 1][k/STEP].x * b_shared[k][tid_sx / STEP].y;
            c_reg[1].z += a_shared[tid_sy + 1][k/STEP].x * b_shared[k][tid_sx / STEP].z;
            c_reg[1].w += a_shared[tid_sy + 1][k/STEP].x * b_shared[k][tid_sx / STEP].w;

            c_reg[2].x += a_shared[tid_sy + 2][k/STEP].x * b_shared[k][tid_sx / STEP].x;
            c_reg[2].y += a_shared[tid_sy + 2][k/STEP].x * b_shared[k][tid_sx / STEP].y;
            c_reg[2].z += a_shared[tid_sy + 2][k/STEP].x * b_shared[k][tid_sx / STEP].z;
            c_reg[2].w += a_shared[tid_sy + 2][k/STEP].x * b_shared[k][tid_sx / STEP].w;

            c_reg[3].x += a_shared[tid_sy + 3][k/STEP].x * b_shared[k][tid_sx / STEP].x;
            c_reg[3].y += a_shared[tid_sy + 3][k/STEP].x * b_shared[k][tid_sx / STEP].y;
            c_reg[3].z += a_shared[tid_sy + 3][k/STEP].x * b_shared[k][tid_sx / STEP].z;
            c_reg[3].w += a_shared[tid_sy + 3][k/STEP].x * b_shared[k][tid_sx / STEP].w;

            // K1
            c_reg[0].x += a_shared[tid_sy + 0][k/STEP].y * b_shared[k+1][tid_sx / STEP].x;
            c_reg[0].y += a_shared[tid_sy + 0][k/STEP].y * b_shared[k+1][tid_sx / STEP].y;
            c_reg[0].z += a_shared[tid_sy + 0][k/STEP].y * b_shared[k+1][tid_sx / STEP].z;
            c_reg[0].w += a_shared[tid_sy + 0][k/STEP].y * b_shared[k+1][tid_sx / STEP].w;

            c_reg[1].x += a_shared[tid_sy + 1][k/STEP].y * b_shared[k+1][tid_sx / STEP].x;
            c_reg[1].y += a_shared[tid_sy + 1][k/STEP].y * b_shared[k+1][tid_sx / STEP].y;
            c_reg[1].z += a_shared[tid_sy + 1][k/STEP].y * b_shared[k+1][tid_sx / STEP].z;
            c_reg[1].w += a_shared[tid_sy + 1][k/STEP].y * b_shared[k+1][tid_sx / STEP].w;

            c_reg[2].x += a_shared[tid_sy + 2][k/STEP].y * b_shared[k+1][tid_sx / STEP].x;
            c_reg[2].y += a_shared[tid_sy + 2][k/STEP].y * b_shared[k+1][tid_sx / STEP].y;
            c_reg[2].z += a_shared[tid_sy + 2][k/STEP].y * b_shared[k+1][tid_sx / STEP].z;
            c_reg[2].w += a_shared[tid_sy + 2][k/STEP].y * b_shared[k+1][tid_sx / STEP].w;

            c_reg[3].x += a_shared[tid_sy + 3][k/STEP].y * b_shared[k+1][tid_sx / STEP].x;
            c_reg[3].y += a_shared[tid_sy + 3][k/STEP].y * b_shared[k+1][tid_sx / STEP].y;
            c_reg[3].z += a_shared[tid_sy + 3][k/STEP].y * b_shared[k+1][tid_sx / STEP].z;
            c_reg[3].w += a_shared[tid_sy + 3][k/STEP].y * b_shared[k+1][tid_sx / STEP].w;

            // K2
            c_reg[0].x += a_shared[tid_sy + 0][k/STEP].z * b_shared[k+2][tid_sx / STEP].x;
            c_reg[0].y += a_shared[tid_sy + 0][k/STEP].z * b_shared[k+2][tid_sx / STEP].y;
            c_reg[0].z += a_shared[tid_sy + 0][k/STEP].z * b_shared[k+2][tid_sx / STEP].z;
            c_reg[0].w += a_shared[tid_sy + 0][k/STEP].z * b_shared[k+2][tid_sx / STEP].w;

            c_reg[1].x += a_shared[tid_sy + 1][k/STEP].z * b_shared[k+2][tid_sx / STEP].x;
            c_reg[1].y += a_shared[tid_sy + 1][k/STEP].z * b_shared[k+2][tid_sx / STEP].y;
            c_reg[1].z += a_shared[tid_sy + 1][k/STEP].z * b_shared[k+2][tid_sx / STEP].z;
            c_reg[1].w += a_shared[tid_sy + 1][k/STEP].z * b_shared[k+2][tid_sx / STEP].w;

            c_reg[2].x += a_shared[tid_sy + 2][k/STEP].z * b_shared[k+2][tid_sx / STEP].x;
            c_reg[2].y += a_shared[tid_sy + 2][k/STEP].z * b_shared[k+2][tid_sx / STEP].y;
            c_reg[2].z += a_shared[tid_sy + 2][k/STEP].z * b_shared[k+2][tid_sx / STEP].z;
            c_reg[2].w += a_shared[tid_sy + 2][k/STEP].z * b_shared[k+2][tid_sx / STEP].w;

            c_reg[3].x += a_shared[tid_sy + 3][k/STEP].z * b_shared[k+2][tid_sx / STEP].x;
            c_reg[3].y += a_shared[tid_sy + 3][k/STEP].z * b_shared[k+2][tid_sx / STEP].y;
            c_reg[3].z += a_shared[tid_sy + 3][k/STEP].z * b_shared[k+2][tid_sx / STEP].z;
            c_reg[3].w += a_shared[tid_sy + 3][k/STEP].z * b_shared[k+2][tid_sx / STEP].w;

            // K3
            c_reg[0].x += a_shared[tid_sy + 0][k/STEP].w * b_shared[k+3][tid_sx / STEP].x;
            c_reg[0].y += a_shared[tid_sy + 0][k/STEP].w * b_shared[k+3][tid_sx / STEP].y;
            c_reg[0].z += a_shared[tid_sy + 0][k/STEP].w * b_shared[k+3][tid_sx / STEP].z;
            c_reg[0].w += a_shared[tid_sy + 0][k/STEP].w * b_shared[k+3][tid_sx / STEP].w;

            c_reg[1].x += a_shared[tid_sy + 1][k/STEP].w * b_shared[k+3][tid_sx / STEP].x;
            c_reg[1].y += a_shared[tid_sy + 1][k/STEP].w * b_shared[k+3][tid_sx / STEP].y;
            c_reg[1].z += a_shared[tid_sy + 1][k/STEP].w * b_shared[k+3][tid_sx / STEP].z;
            c_reg[1].w += a_shared[tid_sy + 1][k/STEP].w * b_shared[k+3][tid_sx / STEP].w;

            c_reg[2].x += a_shared[tid_sy + 2][k/STEP].w * b_shared[k+3][tid_sx / STEP].x;
            c_reg[2].y += a_shared[tid_sy + 2][k/STEP].w * b_shared[k+3][tid_sx / STEP].y;
            c_reg[2].z += a_shared[tid_sy + 2][k/STEP].w * b_shared[k+3][tid_sx / STEP].z;
            c_reg[2].w += a_shared[tid_sy + 2][k/STEP].w * b_shared[k+3][tid_sx / STEP].w;

            c_reg[3].x += a_shared[tid_sy + 3][k/STEP].w * b_shared[k+3][tid_sx / STEP].x;
            c_reg[3].y += a_shared[tid_sy + 3][k/STEP].w * b_shared[k+3][tid_sx / STEP].y;
            c_reg[3].z += a_shared[tid_sy + 3][k/STEP].w * b_shared[k+3][tid_sx / STEP].z;
            c_reg[3].w += a_shared[tid_sy + 3][k/STEP].w * b_shared[k+3][tid_sx / STEP].w;
        }

        // 再次同步，避免该块内个别线程已经计算完进入下一次循环中，往共享内存写数据，与正在共享内存正在计算中的数据相冲突
        __syncthreads();
    }

    for (int si=0; si<STEP; si++) {
        C[(gid_sy+si) * ldc + gid_sx+0] += c_reg[si].x;
        C[(gid_sy+si) * ldc + gid_sx+1] += c_reg[si].y;
        C[(gid_sy+si) * ldc + gid_sx+2] += c_reg[si].z;
        C[(gid_sy+si) * ldc + gid_sx+3] += c_reg[si].w;
    }
}

float MatrixMulCUDA(int version_id, int step,
                    const int M, const int N, const int K,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    float *C, const int ldc) {
    GpuTimer gpu_timer;

    const int block_side_size = 32;
    dim3 threads_per_block(block_side_size, block_side_size);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (M + threads_per_block.y - 1) / threads_per_block.y);
    
    // Warm up.
    for (int i=0; i<10; i++) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    cudaMemset(C, 0, sizeof(float) * M * N);

    // Record the start event
    gpu_timer.Start();

    if (version_id == 1) {
        GemmKernelv1<< <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);        
    }
    else if (version_id == 2) {
        GemmKernelv2<block_side_size> << <blocks_per_grid, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);    
    }
    else if (version_id == 3) {
        int partition_k = 2;
        dim3 blocks_per_grid_k(blocks_per_grid.x * partition_k, blocks_per_grid.y); // x方向多开一倍线程
        GemmKernelv3<block_side_size> << <blocks_per_grid_k, threads_per_block >> >
            (M, N, K, A, lda, B, ldb, C, ldc);   
    }
    else if (version_id == 4) {
        if (step == 2) {
            // 一个线程处理2*2个数据，block数量xy方向都减半，然后一个block的线程数量不变。
            const int STEP = 2;
            dim3 blocks_per_grid_r(blocks_per_grid.x/STEP, blocks_per_grid.y/STEP);
            GemmKernelv4<block_side_size, STEP> << <blocks_per_grid_r, threads_per_block >> >
                (M, N, K, A, lda, B, ldb, C, ldc);            
        }
        else if (step == 4) {
            // 一个线程处理4*4个数据，一个block的线程数量xy方向都减4倍，block数量不变。不然共享内存占用太多。
            // 虽计算访存比增加，但因一个block线程数太少，无法提高使用率，起不到加速作用。
            const int STEP = 4;
            dim3 threads_per_block_r(block_side_size/STEP, block_side_size/STEP);
            GemmKernelv4<block_side_size/STEP, STEP> << <blocks_per_grid, threads_per_block_r >> >
                (M, N, K, A, lda, B, ldb, C, ldc);            
        }
        else if (step == 22) {
            // 一个线程处理4*4个数据，一个block的线程数量xy方向都减半，block数量xy方向也减半。
            const int STEP = 4;
            dim3 threads_per_block_r(block_side_size/2, block_side_size/2);
            dim3 blocks_per_grid_r(blocks_per_grid.x/2, blocks_per_grid.y/2);
            GemmKernelv4<block_side_size/2, STEP> << <blocks_per_grid_r, threads_per_block_r >> >
                (M, N, K, A, lda, B, ldb, C, ldc);            
        }
        else if (step == 8) {
            // 一个线程处理4*4个数据，一个block的线程数量xy方向都减半，block数量xy方向也减半。
            const int STEP = 8;
            dim3 threads_per_block_r(block_side_size/4, block_side_size/4);
            dim3 blocks_per_grid_r(blocks_per_grid.x/2, blocks_per_grid.y/2);
            GemmKernelv4<block_side_size/4, STEP> << <blocks_per_grid_r, threads_per_block_r >> >
                (M, N, K, A, lda, B, ldb, C, ldc);            
        }
    }
    else if (version_id == 5) {
        if (step == 4) {
            // 一个线程处理4*4个数据，一个block的线程数量xy方向都减半，block数量xy方向也减半。
            const int STEP = 4;
            dim3 threads_per_block_r(block_side_size/STEP, block_side_size/STEP);
            GemmKernelv5<block_side_size/STEP, STEP> << <blocks_per_grid, threads_per_block_r >> >
                (M, N, K, A, lda, B, ldb, C, ldc);            

            // const int STEP = 4;
            // dim3 threads_per_block_r(block_side_size/2, block_side_size/2);
            // dim3 blocks_per_grid_r(blocks_per_grid.x/2, blocks_per_grid.y/2);
            // GemmKernelv4<block_side_size/2, STEP> << <blocks_per_grid_r, threads_per_block_r >> >
            //     (M, N, K, A, lda, B, ldb, C, ldc);    
        }
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
        printf("gpu version %d step %2d -> time: %f s, mean value = %f\n", version_id, step, msec_total/1000.f, GetMean(h_c, height_a, width_b)); \
    } while (0)

int main() {
    int ret = InitEnvironment(0);
    if (ret != 0) {
        printf("Failed to initialize the environment for cuda.");
        return -1;
    }

    // // Normal test
    int height_a = 1280, width_a = 4096;
    int height_b = 4096, width_b = 2048;
    // // Test split-k
    // int height_a = 64, width_a = 4096;
    // int height_b = 4096, width_b = 64;
    // // Debug
    // int height_a = 32, width_a = 64;
    // int height_b = 64, width_b = 32;
    if (width_a != height_b) {
        printf("width_a should be equal to height_b.\n");
        return 1;
    }

    const int mem_size_a = sizeof(float) * height_a * width_a;
    const int mem_size_b = sizeof(float) * height_b * width_b;
    const int mem_size_c = sizeof(float) * height_a * width_b;

    float *h_a = (float *)malloc(mem_size_a);
    float *h_b = (float *)malloc(mem_size_b);
    float *h_c = (float *)malloc(mem_size_c);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        printf("Fail to malloc.\n");
        return 1;
    }

    // Initialize 
    srand(time(NULL));
    GenMatrix(height_a, width_a, h_a);
    GenMatrix(height_b, width_b, h_b);

    // // CPU
    // time_t t = clock();
    // GemmHostV1(height_a, width_b, width_a, h_a, width_a,h_b, width_b, h_c, width_b);
    // printf("cpu version 1 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetMean(h_c, height_a, width_b));
    // //MatrixPrint(h_c, height_a, width_b);

    // t = clock();
    // GemmHostV2(height_a, width_b, width_a, h_a, width_a, h_b, width_b, h_c, width_b);
    // printf("cpu version 2 -> time: %f s, mean value = %f\n", double(clock() - t)/CLOCKS_PER_SEC, GetMean(h_c, height_a, width_b));
    // //MatrixPrint(h_c, height_a, width_b);

    // GPU
    // Allocate memory in host. 
    float msec_total;
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
    CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
    CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));

    TEST_CUDA_MODULE_UKERNEL(1, 1);
    TEST_CUDA_MODULE_UKERNEL(2, 1);
    TEST_CUDA_MODULE_UKERNEL(3, 1);
    TEST_CUDA_MODULE_UKERNEL(4, 2);
    TEST_CUDA_MODULE_UKERNEL(4, 4);
    TEST_CUDA_MODULE_UKERNEL(4, 22);
    TEST_CUDA_MODULE_UKERNEL(4, 8);
    TEST_CUDA_MODULE_UKERNEL(5, 4);

    // printf("Print output C:\n");
    // for (int i=0; i<height_a; i++) {
    //     for (int j=0; j<width_b; j++) {
    //         printf("%f, ", h_c[i*width_b+j]);
    //     }
    //     printf("\n");
    // }

    // Normal test.
    // GPU Device 0: "Tesla T4" with compute capability 7.5 with 40 multi-processors.
    // gpu version 1 step  1 -> time: 0.031672 s, mean value = 917.531372
    // gpu version 2 step  1 -> time: 0.024245 s, mean value = 917.531372
    // gpu version 3 step  1 -> time: 0.025030 s, mean value = 917.531372
    // gpu version 4 step  2 -> time: 0.012044 s, mean value = 917.531372
    // gpu version 4 step  4 -> time: 0.013400 s, mean value = 917.531372
    // gpu version 4 step 22 -> time: 0.008451 s, mean value = 917.531372
    // gpu version 4 step  8 -> time: 0.016665 s, mean value = 917.531372
    // gpu version 5 step  4 -> time: 0.008395 s, mean value = 917.531372

    // Test split-k
    // gpu version 1 -> time: 0.000979 s, mean value = 2628.805664
    // gpu version 2 -> time: 0.000694 s, mean value = 2628.805664
    // gpu version 3 -> time: 0.000352 s, mean value = 2628.805664 // split-2
    // gpu version 4 -> time: 0.000965 s, mean value = 2628.805664

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    CleanUpEnvironment();

    return 0;
}
