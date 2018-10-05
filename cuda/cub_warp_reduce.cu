/*!
* \brief Simple demonstration of WarpReduce.
*/

#include <stdio.h>
#include <typeinfo>

#include "cuda_util.h"

#include <cub/warp/warp_reduce.cuh>
#include <cub/util_allocator.cuh>

cub::CachingDeviceAllocator  g_allocator(true);

//Initialize problem.
template <typename T>
void Initialize(T *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = (T) i;
}

// Compute solution.
template <typename T>
void SolveInCPU(T *h_in, T &h_out, int num_items) {
  h_out = (T) 0;
  for (int i = 0; i < num_items; ++i) {
    h_out += h_in[i];
  }
}

/**
 * Full-tile warp reduction kernel
 */
template <int WARPS, int LOGICAL_WARP_THREADS, typename T, typename ReductionOp>
__global__ void FullWarpReduceKernel(T *d_in, T *d_out, ReductionOp reduction_op,
                                     clock_t *d_elapsed) {
  // Cooperative warp-reduce utility type (1 warp)
  typedef cub::WarpReduce<T, LOGICAL_WARP_THREADS> WarpReduce;

  // Allocate temp storage in shared memory
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS];

  // Per-thread tile data
  T input = d_in[threadIdx.x];

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t start = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  // Test warp reduce
  int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;

  //T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
  //  temp_storage[warp_id], input, reduction_op);
  T output = WarpReduce(temp_storage[warp_id]).Reduce(input, reduction_op);

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t stop = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  *d_elapsed = stop - start;

  // Store aggregate
  d_out[threadIdx.x] = (threadIdx.x % LOGICAL_WARP_THREADS == 0) ?
    output :
    input;
}

/**
 * Partially-full warp reduction kernel
 */
template <int WARPS, int LOGICAL_WARP_THREADS, typename T, typename ReductionOp>
__global__ void PartialWarpReduceKernel(T *d_in, T *d_out, ReductionOp reduction_op,
                                        clock_t *d_elapsed, int valid_warp_threads) {
  // Cooperative warp-reduce utility type
  typedef cub::WarpReduce<T, LOGICAL_WARP_THREADS> WarpReduce;

  // Allocate temp storage in shared memory
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS];

  // Per-thread tile data
  T input = d_in[threadIdx.x];

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t start = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  // Test partial-warp reduce
  int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;
  //T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
  //  temp_storage[warp_id], input, reduction_op, valid_warp_threads);
  T output = WarpReduce(temp_storage[warp_id]).Reduce(input, reduction_op, valid_warp_threads);

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t stop = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  *d_elapsed = stop - start;

  // Store aggregate
  d_out[threadIdx.x] = (threadIdx.x % LOGICAL_WARP_THREADS == 0) ?
    output :
    input;
}


/**
 * Test warp reduction
 */
template <int WARPS, int LOGICAL_WARP_THREADS,
          typename T, typename ReductionOp>
void TestReduce(ReductionOp reduction_op,
                int valid_warp_threads = LOGICAL_WARP_THREADS) {

  const int BLOCK_THREADS = LOGICAL_WARP_THREADS * WARPS;

  printf("%d warps, %d warp threads, %d valid lanes, %s (%d bytes) elements:\n",
    WARPS,
    LOGICAL_WARP_THREADS,
    valid_warp_threads,
    typeid(T).name(),
    (int) sizeof(T));

  // Allocate host arrays
  T *h_in = new T[BLOCK_THREADS];
  T h_out = 0;
  // Initialize problem.
  Initialize<T>(h_in, BLOCK_THREADS);
  SolveInCPU(h_in, h_out, BLOCK_THREADS);

  std::cout << "input array: ";
  for (int i = 0; i < BLOCK_THREADS; i++) {
    std::cout << h_in[i] << ",";
  }
  std::cout << std::endl;

  // Initialize/clear device arrays
  T *d_in = NULL;
  T *d_out = NULL;
  clock_t *d_elapsed = NULL;

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * BLOCK_THREADS, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * BLOCK_THREADS));

  // Run kernel
  if (valid_warp_threads == LOGICAL_WARP_THREADS) {
    // Run full-warp kernel
    FullWarpReduceKernel<WARPS, LOGICAL_WARP_THREADS> << <1, BLOCK_THREADS >> > (
      d_in,
      d_out,
      reduction_op,
      d_elapsed);
  }
  else {
    // Run partial-warp kernel
    PartialWarpReduceKernel<WARPS, LOGICAL_WARP_THREADS> << <1, BLOCK_THREADS >> > (
      d_in,
      d_out,
      reduction_op,
      d_elapsed,
      valid_warp_threads);
  }

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  
  clock_t h_elapsed = NULL;
  cudaMemcpy(&h_elapsed, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost);
  // Copy the output data from device to cpu.
  T *h_out4cub_temp_arr = new T[BLOCK_THREADS];
  cudaMemcpy(h_out4cub_temp_arr, d_out, sizeof(T) * BLOCK_THREADS, cudaMemcpyDeviceToHost);

  std::cout << "output array: ";
  for (int i = 0; i < BLOCK_THREADS; i++) {
    std::cout << h_out4cub_temp_arr[i] << ",";
  }
  std::cout << std::endl;

  // Merge the result of each warp.
  T h_out4cub = 0;
  for (int i = 0; i < WARPS; i++) {
    h_out4cub += h_out4cub_temp_arr[i*LOGICAL_WARP_THREADS + 0];
  }
  std::cout << "Result: (h_out vs h_out4cub) == (" << h_out << " vs " \
    << h_out4cub << "), Elapsed clocks:" << h_elapsed << std::endl << std::endl;

  // Cleanup
  if (h_in) delete[] h_in;
  //if (h_out4cub_temp_arr) delete[] h_out4cub_temp_arr;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}

int main(int argc, char** argv) {
  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  // normal type:
  //   (unsigned)char / (unsigned) short / (unsigned) int / (unsigned) long long 
  //   float / double
  //
  // vector type:
  //   uchar1 
  //   uchar2 / ushort2 / uint2 / ulonglong2
  //   uchar4 / ushort4 / uint4 / ulonglong4
  // 
  // functor:
  // Sum / Max / Min / ArgMax / CastOp / ...
  TestReduce<1, 32, int>(cub::Sum());
  TestReduce<2, 16, int>(cub::Sum());
  TestReduce<1, 32, double>(cub::Sum());

  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}