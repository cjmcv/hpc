/*!
* \brief An experiment on coalesced memory access.
*/

#include "cuda_util.h"
#include "time.h"

// Coalesced Access.
// 89 ms, test <<512,512>> in 1000 times.
//
// example: <<blocks_per_grid, threads_per_block>> 
//         = <<6,6>>
// 
//       t0   t1  t2  t3  t4  t5  t6
//  b0   00   01  02  03  04  05  06
//  b1   10   11  12  13  14  15  16
//   .    .    .   .   .   .   .   .
//  b6   60   61  62  63  64  65  66
//
__global__ void TestCopyKernelOne(int *d_in, int *d_out, const int len) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  d_out[i] = d_in[i];
  //printf("(%d, %d), ", threadIdx.x, i%32);
}

// Non-Coalesced Access.
// 193 ms, test <<512,512>> in 1000 times.
//
// example: <<blocks_per_grid, threads_per_block>> 
//         = <<6,6>>
//
//       b0   b1  b2  b3  b4  b5  b6
//  t0   00   01  02  03  04  05  06
//  t1   10   11  12  13  14  15  16
//   .    .    .   .   .   .   .   .
//  t6   60   61  62  63  64  65  66
//
__global__ void TestCopyKernelTwo(int *d_in, int *d_out, const int len) {
  unsigned int i = threadIdx.x * gridDim.x + blockIdx.x;
  d_out[i] = d_in[i];
  //printf("(%d, %d), ", blockIdx.x, i%32);
}

bool CheckArrayIsEqual(const int *arr1, const int *arr2, const int len) {
  for (int i = 0; i < len; i++) {
    if (arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  int dev_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(dev_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int w = 512;
  int len = w * w;
  // Initialize data in host.  
  int *h_in = new int[len];
  int *h_out1 = new int[len];
  int *h_out2 = new int[len];

  for (int i = 0; i < len; i++) {
    h_in[i] = i;
  }

  int *d_in, *d_out1, *d_out2;
  CUDA_CHECK(cudaMalloc((float**)&d_in, len * sizeof(int)));
  CUDA_CHECK(cudaMalloc((float**)&d_out1, len * sizeof(int)));
  CUDA_CHECK(cudaMalloc((float**)&d_out2, len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, len * sizeof(int), cudaMemcpyHostToDevice));
  
  const int threads_per_block = w;
  const int blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;

  // Warn up.
  TestCopyKernelOne << < blocks_per_grid, threads_per_block >> > (d_in, d_out1, len);
  CUDA_CHECK(cudaMemcpy(h_out1, d_out1, len * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemset(d_out1, 0, len * sizeof(int)));

  cjmcv_cuda_util::GpuTimer *timer = new cjmcv_cuda_util::GpuTimer;
  // The number of iterations.
  int iter_count = 1000;

  ////////// Kernel one ///////////////
  float time_recorder1 = 0;
  for (int i = 0; i < iter_count; i++) {
    timer->Start();
    TestCopyKernelOne << < blocks_per_grid, threads_per_block >> > (d_in, d_out1, len);
    timer->Stop();
    time_recorder1 += timer->ElapsedMillis();
  }
  // The cudaMemcpy function is synchronized to the host, so you need not call cudaDeviceSynchronize() here.
  CUDA_CHECK(cudaMemcpy(h_out1, d_out1, len * sizeof(int), cudaMemcpyDeviceToHost));
  printf("Kernel One - time_elapsed: %f ms.\n", time_recorder1);

  ////////// Kernel two ///////////////
  float time_recorder2 = 0;
  for (int i = 0; i < iter_count; i++) {
    timer->Start();
    TestCopyKernelTwo << < blocks_per_grid, threads_per_block >> > (d_in, d_out2, len);
    timer->Stop();
    time_recorder2 += timer->ElapsedMillis();
  }
  CUDA_CHECK(cudaMemcpy(h_out2, d_out2, len * sizeof(int), cudaMemcpyDeviceToHost)); 
  printf("Kernel Two - time_elapsed: %f ms.\n", time_recorder2);

  // Check results.
  if ((CheckArrayIsEqual(h_in, h_out1, len) == false) || 
      (CheckArrayIsEqual(h_in, h_out2, len) == false)) {
    printf("Error in calculation.\n");
  }
  else {
    printf("PASS.\n");
  }

  cudaFree(d_in);
  cudaFree(d_out1);
  cudaFree(d_out2);

  delete timer;
  delete[] h_in;
  delete[] h_out1;
  delete[] h_out2;

  // Reset device
  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}