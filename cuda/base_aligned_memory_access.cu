/*!
* \brief An experiment on aligned memory access.
* \reference https://www.cnblogs.com/1024incn/p/4573566.html
*/

#include "cuda_util.h"

// Initialize the input data.
void GenArray(float *arr, const int len) {
  for (int i = 0; i < len; i++) {
    arr[i] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
}

__global__ void TestMisalignedReadKernel(float *A, float *B, float *C, const int len, int offset) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = i + offset;
  if (k < len)
    C[i] = A[k] + B[k];
}

__global__ void TestMisalignedWriteKernel(float *A, float *B, float *C, const int len, int offset) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = i + offset;
  if (k < len)
    C[k] = A[i] + B[i];  // Just switch the place between i and k.
}

int main() {
  int dev_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(dev_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  // set up array size
  int len = 1 << 20; // total number of elements to reduce
  printf("len = %d\n", len);
  size_t mem_size = len * sizeof(float);

  // execution configuration
  const int threads_per_block = 512;
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Allocate host memory
  float *h_a = (float *)malloc(mem_size);
  float *h_b = (float *)malloc(mem_size);
  float *h_c = (float *)malloc(mem_size);
  // Initialize host array
  GenArray(h_a, len);
  GenArray(h_b, len);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((float**)&d_a, mem_size));
  CUDA_CHECK(cudaMalloc((float**)&d_b, mem_size));
  CUDA_CHECK(cudaMalloc((float**)&d_c, mem_size));
  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice));

  // Warm up
  TestMisalignedReadKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, 256);
  
  // Test.
  cjmcv_cuda_util::GpuTimer *timer = new cjmcv_cuda_util::GpuTimer;

  const int offset_size = 12;
  int offset[12] = {1, 16, 129, 254, 513, 1020, 0, 32, 128, 160, 512, 1024};
  for (int i = 0; i < offset_size; i++) {
    timer->Start();
    for(int tc = 0; tc < 10; tc++)
      TestMisalignedReadKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, offset[i]);
    timer->Stop();
    printf("Read<<< %4d, %4d >>>, offset %4d, elapsed %f sec, offset%%32 = %d\n",
      blocks_per_grid, threads_per_block,
      offset[i], timer->ElapsedMillis(), offset[i]%32);
    if (i == (offset_size-1) / 2)
      printf("\n");
  }

  printf("-------------------------------------------------------------------------------\n");

  for (int i = 0; i < offset_size; i++) {
    timer->Start();
    for (int tc = 0; tc < 10; tc++)
      TestMisalignedWriteKernel << < blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, len, offset[i]);
    timer->Stop();
    printf("Write<<< %4d, %4d >>>, offset %4d, elapsed %f sec, offset%%32 = %d\n",
      blocks_per_grid, threads_per_block,
      offset[i], timer->ElapsedMillis(), offset[i] % 32);
    if (i == (offset_size-1) / 2)
      printf("\n");
  }
 
  // Copy kernel result back to host side and check device results
  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost));

  // Free host and device memory
  delete timer;
  if (d_a) CUDA_CHECK(cudaFree(d_a));
  if (d_b) CUDA_CHECK(cudaFree(d_b));
  if (d_c) CUDA_CHECK(cudaFree(d_c));

  if (h_a) free(h_a);
  if (h_b) free(h_b);
  if (h_c) free(h_c);

  // Reset device
  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}