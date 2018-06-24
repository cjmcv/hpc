/*!
* \brief Vector dot product: h_result = SUM(A * B).
*/

#include "cuda_util.h"

// Initialize the input data.
void GenArray(const int len, float *arr) {
  for (int i = 0; i < len; i++) {
    arr[i] = 1;//(float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
}
// CPU version: 965ms
// Normal version in cpu as a reference
float VectorDotProductCPU(const float *vec_a, const float *vec_b, const int len) {
  float h_result = 0;
  for (int i = 0; i<len; i++) {
    h_result += vec_a[i] * vec_b[i];
  }
  return h_result;
}

// CUDA kernel v1 : 283ms
// Multiply and save to shared memory.
// Accumulate data from all of the shared memory to fewer blocks.
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv1(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i];
    __syncthreads();

    int count = BLOCK_SIZE / 2;
    while (count >= 1) {
      if(threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      // Synchronize the threads within the block,
      // then go to next round together.
      __syncthreads();
      count /= 2; 
    }
    
    if(threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel v2 : 201ms
// Compute two blocks' data to the shared memory of one block.
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv2(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];  // Mainly in here.
    __syncthreads();

    int count = BLOCK_SIZE >> 1;
    while (count >= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      // Synchronize the threads within the block,
      // then go to next round together.
      __syncthreads();
      count >>= 1;
    }

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

// CUDA kernel v3 : 179ms
// Condition: The block size should be bigger than 32
// Unroll the last warp
template <int BLOCK_SIZE>
__global__ void VectorDotProductKernelv3(const float *vec_a, const float *vec_b, const int len, float &res) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len / 2;
    i += blockDim.x * gridDim.x) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = vec_a[i] * vec_b[i] + vec_a[i + gridDim.x / 2] * vec_b[i + gridDim.x / 2];
    __syncthreads();

    for (int count = BLOCK_SIZE >> 1; count > 32; count >>= 1) {
      if (threadIdx.x < count) {
        smem[threadIdx.x] += smem[count + threadIdx.x];
      }
      __syncthreads();
    }

    // Mainly in here. Unroll the last warp. (It still need __syncthreads() in a warp ?)
    if (threadIdx.x < 32) {
      smem[threadIdx.x] += smem[threadIdx.x + 32]; __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 16]; __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 8];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 4];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 2];  __syncthreads();
      smem[threadIdx.x] += smem[threadIdx.x + 1];  __syncthreads();
    }

    if (threadIdx.x == 0)
      atomicAdd(&res, smem[0]);
  }
}

float VectorDotProductCUDA(const int loops, const float *vec_a, const float *vec_b, const int len, float &result) {
  // Time recorder.
  cjmcv_cuda_util::GpuTimer gpu_timer;

  const int threads_per_block = 1024; // data_len % threads_per_block == 0
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Warm up.
  VectorDotProductKernelv3<threads_per_block> << <blocks_per_grid, threads_per_block >> >
    (vec_a, vec_b, len, result);
  
  gpu_timer.Start();

  for (int i = 0; i < loops; i++) {
    cudaMemset(&result, 0, sizeof(float));
    VectorDotProductKernelv3<threads_per_block> << <blocks_per_grid, threads_per_block >> >
      (vec_a, vec_b, len, result);
  }

  gpu_timer.Stop();

  return gpu_timer.ElapsedMillis();
}

int main() {
  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  const int loops = 100;
  const int data_len = 10240000; // data_len % threads_per_block == 0
  const int data_mem_size = sizeof(float) * data_len;
  float *h_vector_a = (float *)malloc(data_mem_size);
  float *h_vector_b = (float *)malloc(data_mem_size);
  if (h_vector_a == NULL || h_vector_b == NULL) {
    printf("Fail to malloc.\n");
    return 1;
  }
  
  // Initialize 
  srand(0);
  GenArray(data_len, h_vector_a);
  GenArray(data_len, h_vector_b);

  // CPU
  time_t t = clock();
  float h_result = 0;
  for (int i = 0; i < loops; i++)
    h_result = VectorDotProductCPU(h_vector_a, h_vector_b, data_len);
  printf("\nIn cpu version 1, msec_total = %lld, h_result = %f\n", clock() - t, h_result);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  float *d_vector_a = NULL, *d_vector_b = NULL;
  float *d_result = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_vector_a, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_vector_b, data_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(float)));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_vector_a, h_vector_a, data_mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vector_b, h_vector_b, data_mem_size, cudaMemcpyHostToDevice));

  msec_total = VectorDotProductCUDA(loops, d_vector_a, d_vector_b, data_len, *d_result);
  
  CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
  printf("\nIn gpu version 1, msec_total = %f, h_result = %f\n", msec_total, h_result);

  free(h_vector_a);
  free(h_vector_b);

  cudaFree(d_vector_a);
  cudaFree(d_vector_b);
  cudaFree(d_result);
  cjmcv_cuda_util::CleanUpEnvironment();

  system("pause");
  return 0;
}
