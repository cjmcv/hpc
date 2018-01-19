/*!
* \brief histogram, mainly introduce atomicAdd.
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
void GenArray(const int up_bound, const int len, unsigned char *arr) {
  for (int i = 0; i < len; i++) {
    arr[i] = (unsigned char)rand() % up_bound;
  }
}

// Just for checking the result.
void PrintHist(const int *hist, const int hist_len) {
  printf("start Printing.\n");
  for (int i = 0; i < hist_len; i++) {
    printf("%d, ", hist[i]);
  }
}

// CPU version: 920ms
// Normal version in cpu as a reference
void HistogramCPU(const unsigned char *data, const int data_len, int *hist, const int hist_len) {
  int i;
  for (i = 0; i < hist_len; i++) {
    hist[i] = 0;
  }
  for (i = 0; i < data_len; i++) {
    hist[data[i]]++;
  }
}

// CUDA version 1 : 212ms
template <int HIST_SIZE>
__global__ void HistogramKernelv1(const unsigned char *data, const int data_len,
  int *hist) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < data_len;
    i += blockDim.x * gridDim.x) {

    atomicAdd(&(hist[data[i]]), 1);
  }
}

// CUDA version 2 : 282ms
// Use shared memory, even slower than v1..
// Attention: blockDim.x should be larger than the size of histogram. 
//            Because using (threadIdx.x < HIST_SIZE) to assign.
template <int HIST_SIZE>
__global__ void HistogramKernelv2(const unsigned char *data, const int data_len,
  int *hist) {
  // Prevents memory access across the border.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < data_len;
    i += blockDim.x * gridDim.x) {
    __shared__ unsigned int smem_hist[HIST_SIZE];
    // Clean up the Shared memory for the block by itself.
    if (threadIdx.x < HIST_SIZE)
      smem_hist[threadIdx.x] = 0;
    __syncthreads();

    // atomicAdd are performed in blocks.
    // And the operation between each block does not conflict.
    atomicAdd(&(smem_hist[data[i]]), 1);
    __syncthreads();

    // Merge the result of each block.
    if (threadIdx.x < HIST_SIZE) {
      atomicAdd(&(hist[threadIdx.x]), smem_hist[threadIdx.x]);
    }
  }
}

float HistogramCUDA(const int loops, const unsigned char *data, const int data_len, 
  int *hist, const int hist_len) {
  // Time recorder.
  float msec_total = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int threads_per_block = 512; // Should be larger than the size of histogram (256)
  const int blocks_per_grid = (data_len + threads_per_block - 1) / threads_per_block;
  cudaMemset(hist, 0, sizeof(int) * hist_len);

  // Warm up.
  HistogramKernelv1<256> << <blocks_per_grid, threads_per_block >> >
    (data, data_len, hist);
  cudaMemset(hist, 0, sizeof(int) * hist_len);

  // Record the start event
  CUDA_CHECK(cudaEventRecord(start, NULL));

  for (int i = 0; i < loops; i++) {
    cudaMemset(hist, 0, sizeof(int) * hist_len);
    HistogramKernelv2<256> << <blocks_per_grid, threads_per_block >> >
      (data, data_len, hist);
  }


  // Record the stop event
  CUDA_CHECK(cudaEventRecord(stop, NULL));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&msec_total, start, stop));

  return msec_total;
}

int InitEnvironment(const int dev_id) {
  CUDA_CHECK(cudaSetDevice(dev_id));
  cudaDeviceProp device_prop;
  cudaError_t error = cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    return 1;
  }
  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
  }
  else {
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", dev_id, device_prop.name, device_prop.major, device_prop.minor);
  }
  return 0;
}

int main() { 
  // 15000000 is very close to the biggest number. 
  // Over this, the cuda program can not get the right answer?
  const int data_len = 15000000;
  const int loops = 100;

  InitEnvironment(0);
  const int hist_len = 256;
  const int hist_mem_size = sizeof(int) * hist_len;
  const int data_mem_size = sizeof(unsigned char) * data_len;
  int *h_hist = (int *)malloc(hist_mem_size);
  unsigned char *h_data = (unsigned char *)malloc(data_mem_size);
  if (h_hist == NULL || h_data == NULL ) {
    printf("Fail to malloc.\n");
    return 1;
  }
  
  // Initialize 
  srand(0);
  GenArray(hist_len, data_len, h_data);
  memset(h_hist, 0, hist_mem_size);

  // CPU
  time_t t = clock();
  for(int i=0; i<loops; i++)
    HistogramCPU(h_data, data_len, h_hist, hist_len);
  printf("\nIn cpu version 1, msec_total = %lld\n", clock() - t);
  PrintHist(h_hist, hist_len);

  // GPU
  // Allocate memory in host. 
  float msec_total;
  int *d_hist;
  unsigned char *d_data;
  CUDA_CHECK(cudaMalloc((void **)&d_hist, hist_mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_data, data_mem_size));

  // Copy host memory to device
  CUDA_CHECK(cudaMemcpy(d_hist, h_hist, hist_mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, data_mem_size, cudaMemcpyHostToDevice));

  msec_total = HistogramCUDA(loops, d_data, data_len, d_hist, hist_len);

  // Copy memory back to host.
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, hist_mem_size, cudaMemcpyDeviceToHost));
  printf("\nIn gpu version 1, msec_total = %f\n", msec_total);
  PrintHist(h_hist, hist_len);

  free(h_hist);
  free(h_data);

  cudaFree(d_hist);
  cudaFree(d_data);
  CUDA_CHECK(cudaDeviceReset());

  system("pause");
  return 0;
}