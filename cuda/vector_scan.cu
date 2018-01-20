/*!
* \brief Scan. Prefix Sum.
* \example: input: 1,2,3,4
*           operation: Add
*           ouput: 1,3,6,10 (out[i]=sum(in[0:i]))
*/
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "time.h"

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s \
              : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

// Initialize the input data.
void GenArray(const int len, float *arr) {
  for (int i = 0; i < len; i++) {
    arr[i] = 1;//(float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
  }
} 

// CPU version
// Normal version in cpu as a reference
void VectorScanCPU(const float *vec_in, const int len, float *vec_out) {
  vec_out[0] = vec_in[0];
  for (int i = 1; i<len; i++) {
    vec_out[i] = vec_in[i] + vec_out[i - 1];
  }
}

// CUDA kernel v1
// Hillis Steele Scan
// Limiting conditions : The size of vector should be smaller than block size.
// s1    1      2       3         4         5         6 
// s2    1   3(1+2)   5(2+3)    7(3+4)    9(4+5)   11(5+6)
// s4    1      3     6(1+5)   10(3+7)   14(5+9)   18(7+11)
// s8    1      3       6        10     15(1+14)   21(3+18)
// s16   1      3       6        10        15        21
template <int BLOCK_SIZE>
__global__ void VectorScanKernelv1(const float *vec_in, const int len, float *vec_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float smem[BLOCK_SIZE];
  smem[threadIdx.x] = vec_in[i];
  __syncthreads();

  // Iterative scan.
  for (int step = 1; step < len; step *= 2) {
    if (threadIdx.x >= step) {
      float temp = smem[threadIdx.x - step];
      __syncthreads();
      smem[threadIdx.x] += temp;
    }
    __syncthreads();
  }

  vec_out[i] = smem[threadIdx.x];
}

float VectorScanCUDA(const int loops, const float *vec_in, const int len, float *vec_out) {
  // Time recorder.
  float msec_total = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int threads_per_block = 1024; // data_len % threads_per_block == 0
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Warm up.
  VectorScanKernelv1<threads_per_block> << <blocks_per_grid, threads_per_block >> >
    (vec_in, len, vec_out);
  
  // Record the start event
  CUDA_CHECK(cudaEventRecord(start, NULL));

  for (int i = 0; i < loops; i++) {
    VectorScanKernelv1<threads_per_block> << <blocks_per_grid, threads_per_block >> >
      (vec_in, len, vec_out);
  }

  // Record the stop event
  CUDA_CHECK(cudaEventRecord(stop, NULL));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&msec_total, start, stop));

  return msec_total;
}

int main() {
  const int loops = 10000;
  const int data_len = 1024; // data_len % threads_per_block == 0
  const int data_mem_size = sizeof(float) * data_len;
  float *h_vector_in = (float *)malloc(data_mem_size);
  float *h_vector_out = (float *)malloc(data_mem_size);
  if (h_vector_in == NULL || h_vector_out == NULL) {
    printf("Fail to malloc.\n");
    return 1;
  }
  
  // Initialize 
  srand(0);
  GenArray(data_len, h_vector_in);

  // CPU
  time_t t = clock();
  for (int i = 0; i < loops; i++)
    VectorScanCPU(h_vector_in, data_len, h_vector_out);
  printf("\nIn cpu version 1, msec_total = %lld, h[10000] = %f\n", clock() - t, h_vector_out[1000]);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  float *d_vector_in = NULL, *d_vector_out = NULL;
  CUDA_CHECK(cudaMalloc(&d_vector_in, data_mem_size));
  CUDA_CHECK(cudaMalloc(&d_vector_out, data_mem_size));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_vector_in, h_vector_in, data_mem_size, cudaMemcpyHostToDevice));

  msec_total = VectorScanCUDA(loops, d_vector_in, data_len, d_vector_out);
  
  CUDA_CHECK(cudaMemcpy(h_vector_out, d_vector_out, data_mem_size, cudaMemcpyDeviceToHost));
  printf("\nIn gpu version 1, msec_total = %f, h[10000] = %f\n", msec_total, h_vector_out[1000]);

  free(h_vector_in);
  free(h_vector_out);

  cudaFree(d_vector_in);
  cudaFree(d_vector_out);
  CUDA_CHECK(cudaDeviceReset());

  system("pause");
  return 0;
}
