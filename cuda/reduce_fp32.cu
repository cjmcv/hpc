#include <iostream>
#include "time.h"

#include "pocket-ai/engine/cu/common.hpp"
using namespace pai::cu;

// Initialize the input data.
void GenMatrix(const int height, const int width, float *mat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i*width + j] = (float)(rand() % 200 - 100); // int: -100 ~ 100
        }
    }
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

void ReduceHost(const float *x, float *y, int N) {
    *y = 0;
    for (int i = 0; i < N; ++i) {
        *y += x[i];
    }
}

template <int BLOCK_SIZE>
__global__ void ReduceKernelFp32V1(float *src, float *dst, int N) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 将数据从全局内存加载到共享内存
    if (idx < N) {
        smem[tid] = src[idx];
    } else {
        smem[tid] = 0;
    }
    __syncthreads();

    // 块内规约操作
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    // 将每个块的部分和写入全局内存
    if (tid == 0) {
        atomicAdd(dst, smem[0]);
    }
}

// Warp Reduce Sum
// __shfl_xor_sync为warp级别异或函数，使用异或操作来获取对应目标的值。
// 如lane_mask为16，表示lane0可以拿到lane16的值作为函数返回值，lane1对应lane17，lane2对应lane18等。
// 如lane0和lane16，初始时都分别有个自己对应的sum值，当这两个线程进入到函数内时，里面会根据lane_mask==16进行处理，
// 把lane16的sum值传到lane0中，以函数返回值传出。所以直接用lane0用自己的sum值去累加__shfl_xor_sync的返回值，就能得到lane0=lane0+lane16。
// 同一时刻，lane1=lane1+lane17；lane2=lane2+lane18...；lane15=lane15+land31；完成for循环里的一次处理；
// 下一次，mask除以2等于8，则lane0对应lane8；直到循环结束，最后一次是lane0对应lane1的相加；
// 循环结束后，lane0可拿到整个warp的32个线程对应各自的sum值的累加值，完成warp内的规约。
template<const int kWarpSize = 32>
__device__ __forceinline__ float WarpReduceFp32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// v2, 基于warp reduce，先做warp规约，
//     block内将warp规约结果聚集到smem，用第一个warp对smem做warp规约，完成block的规约。
//     最后将每个block的规约结果用原子加累加在一起，完成计算。
template <int BLOCK_SIZE>
__global__ void ReduceKernelFp32V2(float *src, float *dst, int N)
{
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid; // BLOCK_SIZE = blockDim.x
    if (gid >= N) {
        return;
    }
    // Warp规约.
    float sum = WarpReduceFp32<WARP_SIZE>(src[gid]);

    // 各个block将自己负责的每个warp的结果拷贝到该block的smem中。
    // 则smem里每个元素就是该block内每个warp的结果。
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;    
    __shared__ float smem[NUM_WARPS];
    int wid = tid >> 5;
    int lane = tid % WARP_SIZE;
    // WarpReduceFp32计算后，每个warp的lane0会拿到warp的最终结果，即需要用lane0所持有的sum值。
    if (lane == 0)
        smem[wid] = sum;
    __syncthreads();
    
    // 需要对smem里的数据做规约
    // lane的范围是0-31, 则block大小不应超过1024。
    // 假设block里有8个warp，用第一个warp取前8个线程，
    // 同样按warp_size=8的方式做一个warp reduce，完成block内的smem的规约。
    if (wid == 0) {
        if (lane >= NUM_WARPS)
            return;
        sum = WarpReduceFp32<NUM_WARPS>(smem[lane]);
    }
    // block内第一个warp的lane0，也就是tid0，
    // 最后一个将所有block的tid0进行累加，完成最后的规约。
    if (tid == 0)
        atomicAdd(dst, sum);
}

int main() {
    int ret = InitEnvironment(0);
    if (ret != 0) {
        printf("Failed to initialize the environment for cuda.");
        return -1;
    }

    // Normal test
    int N = 1024000;
    const int size = sizeof(float) * N;
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(sizeof(float));
    if (h_x == NULL) {
        printf("Fail to malloc.\n");
        return 1;
    }

    // Initialize 
    srand(time(NULL));
    GenMatrix(1, N, h_x);

    ReduceHost(h_x, h_y, N);
    printf("cpu version -> time: %f s, value = %f\n", 0.0f, *h_y);

    /////////////////////////

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_x, size));
    CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(float) * 1));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    GpuTimer *gpu_timer = new GpuTimer;
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x);
    for (int i=0; i<10; i++) {
        ReduceKernelFp32V2<256><< <blocks_per_grid, threads_per_block >> >(d_x, d_y, N);
    }

    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
    gpu_timer->Start();
    ReduceKernelFp32V1<256><< <blocks_per_grid, threads_per_block >> >(d_x, d_y, N);
    gpu_timer->Stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    printf("gpu version v1-> time: %f s, value = %f\n", gpu_timer->ElapsedMillis(), *h_y);  

    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)));
    gpu_timer->Start();
    ReduceKernelFp32V2<256><< <blocks_per_grid, threads_per_block >> >(d_x, d_y, N);
    gpu_timer->Stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    printf("gpu version v2-> time: %f s, value = %f\n", gpu_timer->ElapsedMillis(), *h_y);        

    free(h_x);
    cudaFree(d_x);
    cudaFree(d_y);
    delete gpu_timer;
    CleanUpEnvironment();
    return 0;
}