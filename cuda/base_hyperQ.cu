/*!
* \brief Demonstrate how HyperQ allows supporting devices to avoid false
*      dependencies between kernels in different streams.
*/

#include "cuda_util.h"

// This subroutine does no real work but runs for at least the specified number
// of clock ticks.
__global__ void KernelA(clock_t *d_o, clock_t clock_count) {
  unsigned int start_clock = (unsigned int)clock();
  clock_t clock_offset = 0;

  while (clock_offset < clock_count) {
    unsigned int end_clock = (unsigned int)clock();

    // The code below should work like 
    // this (thanks to modular arithmetics): 
    // 
    // clock_offset = (clock_t) (end_clock > start_clock ? 
    //                           end_clock - start_clock : 
    //                           end_clock + (0xffffffffu - start_clock));
    //
    // Indeed, let m = 2^32 then
    // end - start = end + m - start (mod m).

    clock_offset = (clock_t)(end_clock - start_clock);
  }
  d_o[0] = clock_offset;
}

void ExperimentA(cudaStream_t *streams, int nstreams) {
  
  clock_t time_clocks = 10000000;
  cjmcv_cuda_util::GpuTimer timer;

  // Allocate device memory for the output (one value for each kernel)
  clock_t *d_a = 0;
  CUDA_CHECK(cudaMalloc((void **)&d_a, nstreams * sizeof(clock_t)));

  // Warn up.
  KernelA << <1, 1 >> > (&d_a[0], time_clocks);

  // Get the running time of KernelA.
  float kernel_time = 0;
  timer.Start();
  KernelA << <1, 1 >> > (&d_a[0], time_clocks);
  timer.Stop();
  kernel_time = timer.ElapsedMillis();

  // Executed in the same stream.
  timer.Start();
  for (int i = 0; i < nstreams; ++i) {
    KernelA << <1, 1 >> > (&d_a[i], time_clocks);
  }
  timer.Stop();
  printf("<In the same stream> Measured time for sample = %.3fs\n", timer.ElapsedMillis() / 1000.0f);

  // Executed in separate streams.
  timer.Start();
  for (int i = 0; i < nstreams; ++i) {
    KernelA << <1, 1, 0, streams[i] >> > (&d_a[i], time_clocks);
  }
  timer.Stop();
  printf("<In separate streams> Measured time for sample = %.3fs\n", timer.ElapsedMillis() / 1000.0f);

  printf("Expected time for serial execution of %d sets of kernels is between approx. %.3fs\n", nstreams, nstreams *kernel_time / 1000.0f);
  printf("Expected time for fully concurrent execution of %d sets of kernels is approx. %.3fs\n", nstreams, kernel_time / 1000.0f);

  CUDA_CHECK(cudaFree(d_a));
}

bool VerifyOutput(int *d_a, int len, int value) {
  int *h_a = (int *)malloc(sizeof(int) * len);
  CUDA_CHECK(cudaMemcpy(h_a, d_a, sizeof(int) * len, cudaMemcpyDeviceToHost));
  bool is_pass = true;
  for (int i = 0; i < len; i++) {
    //printf("%d, ", h_a[i]);
    if (h_a[i] != value)
      is_pass = false;
  }
  if (h_a) free(h_a);
  return is_pass;
}

__global__ void KernelB(int *data, const int len) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < len; i += blockDim.x * gridDim.x) {
    data[i] += 1;
  }
}

void ExperimentB(cudaStream_t *streams, int nstreams, int test_flag) {
  int len = 0;
  int threads_per_block = 0;
  int blocks_per_grid = 0;
  if (test_flag == 0) {
    len = 50000000;
    threads_per_block = 1024;
    blocks_per_grid = (len + threads_per_block - 1) / threads_per_block;  // 4.8w+
  }
  else if (test_flag == 1) {
    len = 50000000;
    threads_per_block = 512;
    blocks_per_grid = 3;
  }
  else {
    len = 50000;
    threads_per_block = 1;
    blocks_per_grid = 1;
  }

  cjmcv_cuda_util::GpuTimer timer;
  int *d_a = 0;
  CUDA_CHECK(cudaMalloc((void **)&d_a, sizeof(int)*len));
  CUDA_CHECK(cudaMemset(d_a, 0, sizeof(float)*len));

  // Warn up.
  KernelB << <blocks_per_grid, threads_per_block >> > (d_a, len);

  // Get the running time of KernelA.
  float kernel_time = 0;
  timer.Start();
  KernelB << <blocks_per_grid, threads_per_block >> > (d_a, len);
  timer.Stop();
  kernel_time = timer.ElapsedMillis();

  // Executed in the same stream. 
  CUDA_CHECK(cudaMemset(d_a, 0, sizeof(float)*len));
  timer.Start();
  for (int i = 0; i < nstreams; ++i) {
    KernelB << <blocks_per_grid, threads_per_block >> > (d_a, len);
  }
  timer.Stop();
  printf("<In the same stream> Measured time for sample = %.3fs, %s\n",
    timer.ElapsedMillis() / 1000.0f, (VerifyOutput(d_a, len, nstreams) ? "PASS" : "NOT")); // NOT, means it is not a serial execution.

  // Executed in separate streams.
  CUDA_CHECK(cudaMemset(d_a, 0, sizeof(float)*len));
  timer.Start();
  for (int i = 0; i < nstreams; ++i) {
    KernelB << <blocks_per_grid, threads_per_block, 0, streams[i] >> > (d_a, len);
  }
  timer.Stop();
  printf("<In separate streams> Measured time for sample = %.3fs, %s\n",
    timer.ElapsedMillis() / 1000.0f, (VerifyOutput(d_a, len, nstreams) ? "PASS" : "NOT")); // NOT, means it is not a serial execution.

  printf("Expected time for serial execution of %d sets of kernels is between approx. %.3fs\n", nstreams, nstreams *kernel_time / 1000.0f);
  printf("Expected time for fully concurrent execution of %d sets of kernels is approx. %.3fs\n", nstreams, kernel_time / 1000.0f);

  if (d_a) CUDA_CHECK(cudaFree(d_a));
}

int main(int argc, char **argv) {
  // HyperQ is available in devices of Compute Capability 3.5 and higher
  int device_id = 0;
  int ret = cjmcv_cuda_util::InitEnvironment(device_id);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  int nstreams = 32;
  // Allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
  for (int i = 0; i < nstreams; i++) {
    CUDA_CHECK(cudaStreamCreate(&(streams[i])));
  }

  printf("Experiment A: \n");
  ExperimentA(streams, nstreams);

  // The acceleration performed by hyperQ is related to the resources invoked when kernel is running. 
  // If all resources are occupied by a single kernel call, hyperQ cannot be used to accelerate in parallel.
  // Refer to ExperimentB_0£¬which takes full use of the resources of device in a kernel call.

  // Close to serial execution.
  printf("\n\nExperiment B_0: \n");
  ExperimentB(streams, nstreams, 0);

  // Only part of it accelerated.
  printf("\n\nExperiment B_1: \n");
  ExperimentB(streams, nstreams, 1);

  // Almost fully concurrent execution.
  printf("\n\nExperiment B_2: \n");
  ExperimentB(streams, nstreams, 2);

  // Release resources
  for (int i = 0; i < nstreams; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }

  if (streams) free(streams);

  cjmcv_cuda_util::CleanUpEnvironment();
}
