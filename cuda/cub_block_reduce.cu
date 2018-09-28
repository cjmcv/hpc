/*!
* \brief Simple demonstration of cub::BlockReduce
*/
#include "cuda_util.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

//Initialize problem.
void Initialize(int *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = i;
}

// Compute solution.
void SolveInCPU(int *h_in, int &h_out, int num_items) {
  h_out = 0;
  for (int i = 0; i < num_items; ++i) {
    h_out += h_in[i];
  }
}

// Kernels
// Simple kernel for performing a block-wide exclusive prefix sum over integers.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockReduceAlgorithm ALGORITHM>
 __global__ void BlockSumKernel(int *d_in, int *d_out, clock_t *d_elapsed) {
  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<int, BLOCK_THREADS, ALGORITHM> BlockReduceT;

  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];
  cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);

  // Start cycle timer
  clock_t start = clock();

  // Compute sum
  int aggregate = BlockReduceT(temp_storage).Sum(data);

  // Stop cycle timer
  clock_t stop = clock();

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0) {
    *d_elapsed = (start > stop) ? start - stop : stop - start;
    *d_out = aggregate;
  }
}

//Test thread block reduction
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockReduceAlgorithm ALGORITHM>
void Test(int grid_size, int timing_iterations) {

  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Allocate host arrays
  int *h_in = new int[TILE_SIZE];

  // Initialize problem and reference output on host
  int h_out;
  Initialize(h_in, TILE_SIZE);
  SolveInCPU(h_in, h_out, TILE_SIZE);

  // Initialize device arrays
  int *d_in = NULL;
  int *d_out = NULL;
  clock_t *d_elapsed = NULL;
  cudaMalloc((void**)&d_in, sizeof(int) * TILE_SIZE);
  cudaMalloc((void**)&d_out, sizeof(int) * 1);
  cudaMalloc((void**)&d_elapsed, sizeof(clock_t));

  // Kernel props
  int max_sm_occupancy;
  CUDA_CHECK(cub::MaxSmOccupancy(max_sm_occupancy, BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>, BLOCK_THREADS));

  // Copy problem to device
  CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice));

  printf("BlockReduce algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
    (ALGORITHM == cub::BLOCK_REDUCE_RAKING) ? "BLOCK_REDUCE_RAKING" : "BLOCK_REDUCE_WARP_REDUCTIONS",
    TILE_SIZE, timing_iterations, grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);

  // Run aggregate/prefix kernel
  BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM> << <grid_size, BLOCK_THREADS >> > (
    d_in, d_out, d_elapsed);

  // Copy the output data from device to cpu.
  int h_out4cub = 0;
  cudaMemcpy(&h_out4cub, d_out, sizeof(int) * 1, cudaMemcpyDeviceToHost);
  printf("Result: (h_out vs h_out4cub) == (%d vs %d)\n", h_out, h_out4cub);

  // Run this several times and average the performance results
  cjmcv_cuda_util::GpuTimer timer;
  float elapsed_millis = 0.0;
  clock_t elapsed_clocks = 0;

  for (int i = 0; i < timing_iterations; ++i) {
    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

    timer.Start();
    // Run aggregate/prefix kernel
    BlockSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM> << <grid_size, BLOCK_THREADS >> > (
      d_in, d_out, d_elapsed);

    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    // Copy clocks from device
    clock_t clocks;
    CUDA_CHECK(cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost));
    elapsed_clocks += clocks;
  }

  // Check for kernel errors and STDIO from the kernel, if any
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Display timing results
  float avg_millis = elapsed_millis / timing_iterations;
  float avg_items_per_sec = float(TILE_SIZE * grid_size) / avg_millis / 1000.0f;
  float avg_clocks = float(elapsed_clocks) / timing_iterations;
  float avg_clocks_per_item = avg_clocks / TILE_SIZE;

  printf("\tAverage BlockReduce::Sum clocks: %.3f\n", avg_clocks);
  printf("\tAverage BlockReduce::Sum clocks per item: %.3f\n", avg_clocks_per_item);
  printf("\tAverage kernel millis: %.4f\n", avg_millis);
  printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

  // Cleanup
  if (h_in) delete[] h_in;
  if (d_in) cudaFree(d_in);
  if (d_out) cudaFree(d_out);
  if (d_elapsed) cudaFree(d_elapsed);
}

int main(int argc, char** argv) {

  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  // Run tests
  int grid_size = 1;
  int timing_iterations = 100;

  //Test<1024, 1, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<512, 2, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<256, 4, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<128, 8, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<64, 16, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<32, 32, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);
  Test<16, 64, cub::BLOCK_REDUCE_RAKING>(grid_size, timing_iterations);

  printf("-------------\n");

  //Test<1024, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<512, 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<256, 4, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<128, 8, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<64, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<32, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);
  Test<16, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>(grid_size, timing_iterations);

  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}

