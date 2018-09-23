/*!
* \brief Simple demonstration of cub::BlockScan
*/
#include "cuda_util.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

void Initialize(int *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = i;
}

// CPU version.
// Solve exclusive-scan problem.
void ExclusiveScanCPU(int *h_in, int *h_out, int num_items) {
  int sum = 0;
  for (int i = 0; i < num_items; ++i) {
    h_out[i] = sum;
    sum += h_in[i];
  }
}

// Kernel.
// Performing a block-wide exclusive prefix sum over integers.
// Process: 1. BlockLoad; 2. BlockScan; 3. BlockStore.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockScanAlgorithm ALGORITHM>
__global__ void BlockPrefixSumKernel(int *d_in, int *d_out, clock_t *d_elapsed) {
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

  // Specialize BlockScan type for our thread block
  typedef cub::BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage    load;
    typename BlockStoreT::TempStorage   store;
    typename BlockScanT::TempStorage    scan;
  } temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, data);
  __syncthreads();

  clock_t start = clock();
  // Compute exclusive prefix sum
  int aggregate;
  BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

  clock_t stop = clock();

  // Barrier for smem reuse
  __syncthreads();

  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0) {
    *d_elapsed = (start > stop) ? start - stop : stop - start;
    d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
  }
}

// Test thread block scan.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, cub::BlockScanAlgorithm ALGORITHM>
void Test(int grid_size, int timing_iterations) {
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Allocate host arrays
  int *h_in = new int[TILE_SIZE];
  int *h_out = new int[TILE_SIZE];
  int *h_out4cub = new int[TILE_SIZE + 1];

  // Initialize problem and reference output on host
  Initialize(h_in, TILE_SIZE);
  ExclusiveScanCPU(h_in, h_out, TILE_SIZE);

  // Initialize device arrays
  int *d_in = NULL;
  int *d_out = NULL;
  clock_t *d_elapsed = NULL;
  cudaMalloc((void**)&d_in, sizeof(int) * TILE_SIZE);
  cudaMalloc((void**)&d_out, sizeof(int) * (TILE_SIZE + 1));
  cudaMalloc((void**)&d_elapsed, sizeof(clock_t));

  // Kernel props
  int max_sm_occupancy;
  CubDebugExit(cub::MaxSmOccupancy(max_sm_occupancy, BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>, BLOCK_THREADS));

  // Copy problem to device
  cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

  printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM occupancy):\n",
    (ALGORITHM == cub::BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING" : (ALGORITHM == cub::BLOCK_SCAN_RAKING_MEMOIZE) ? "BLOCK_SCAN_RAKING_MEMOIZE" : "BLOCK_SCAN_WARP_SCANS",
    TILE_SIZE, timing_iterations, grid_size, BLOCK_THREADS, ITEMS_PER_THREAD, max_sm_occupancy);

  // Run aggregate/prefix kernel
  BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM> << <grid_size, BLOCK_THREADS >> >(
    d_in, d_out, d_elapsed);

  // Copy the output data from device to cpu.
  cudaMemcpy(h_out4cub, d_out, sizeof(int) * (TILE_SIZE + 1), cudaMemcpyDeviceToHost);

  // Simple check results
  bool is_equal = true;
  for (int i = 0; i < TILE_SIZE; i++) {
    if (h_out4cub[i] != h_out[i]) {
      is_equal = false;
      break;
    }
  }
  printf("The result is equal or not: %d -> (%d vs %d)\n", is_equal,
    h_out[TILE_SIZE - 1], h_out4cub[TILE_SIZE - 1]);

  // Run this several times and average the performance results
  cjmcv_cuda_util::GpuTimer timer;
  float elapsed_millis = 0.0;
  clock_t elapsed_clocks = 0;

  for (int i = 0; i < timing_iterations; ++i) {
    // Copy problem to device
    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

    timer.Start();

    // Run aggregate/prefix kernel
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM> << <grid_size, BLOCK_THREADS >> >(
      d_in,
      d_out,
      d_elapsed);

    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    // Copy clocks from device
    clock_t clocks;
    CubDebugExit(cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost));
    elapsed_clocks += clocks;
  }

  // Check for kernel errors and STDIO from the kernel, if any
  CubDebug(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Display timing results
  float avg_millis = elapsed_millis / timing_iterations;
  float avg_items_per_sec = float(TILE_SIZE * grid_size) / avg_millis / 1000.0f;
  float avg_clocks = float(elapsed_clocks) / timing_iterations;
  float avg_clocks_per_item = avg_clocks / TILE_SIZE;

  printf("\tAverage BlockScan::Sum clocks: %.3f\n", avg_clocks);
  printf("\tAverage BlockScan::Sum clocks per item: %.3f\n", avg_clocks_per_item);
  printf("\tAverage kernel millis: %.4f\n", avg_millis);
  printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (h_out4cub) delete[] h_out4cub;
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

  int grid_size = 1;
  int timing_iterations = 100;

  // Run tests. In "GeForce GTX 1070" with compute capability 6.1
  //Test<1024, 1, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // Too many resources requested for launch.
  Test<512, 2, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // 0.3023.(Average kernel millis)
  Test<256, 4, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // 0.2513.
  Test<128, 8, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // 0.2670.
  Test<64, 16, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // 0.3526.
  Test<32, 32, cub::BLOCK_SCAN_RAKING>(grid_size, timing_iterations); // 0.4558.

  printf("-------------\n");

  //Test<1024, 1, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations);
  Test<512, 2, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations); // 0.2396.
  Test<256, 4, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations); // 0.2049.
  Test<128, 8, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations); // 0.2210.
  Test<64, 16, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations); // 0.2766.
  Test<32, 32, cub::BLOCK_SCAN_RAKING_MEMOIZE>(grid_size, timing_iterations); // 0.4179.

  printf("-------------\n");

  //Test<1024, 1, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations);
  Test<512, 2, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations); // 0.1582.
  Test<256, 4, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations); // 0.1516.
  Test<128, 8, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations); // 0.1808.
  Test<64, 16, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations); // 0.2574.
  Test<32, 32, cub::BLOCK_SCAN_WARP_SCANS>(grid_size, timing_iterations); // 0.4188.

  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}

