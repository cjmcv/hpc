/*!
* \brief Simple demonstration of WarpReduce.
*/

#include <stdio.h>
#include <typeinfo>

#include "cuda_util.h"

#include <cub/warp/warp_reduce.cuh>
#include <cub/util_allocator.cuh>

cub::CachingDeviceAllocator  g_allocator(true);

/**
 * \brief WrapperFunctor (for precluding test-specialized dispatch to *Sum variants)
 */
template<typename OpT, int LOGICAL_WARP_THREADS>
struct WrapperFunctor {
  OpT op;
  int num_valid;

  inline __host__ __device__ WrapperFunctor(OpT op, int num_valid) 
    : op(op), num_valid(num_valid) {}

  template <typename T>
  inline __host__ __device__ T operator()(const T &a, const T &b) const {
#if CUB_PTX_ARCH != 0
    if ((cub::LaneId() % LOGICAL_WARP_THREADS) >= num_valid)
      cub::ThreadTrap();
#endif

    return op(a, b);
  }

};


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Generic reduction
 */
template <typename T, typename ReductionOp, typename WarpReduce,
          bool PRIMITIVE = cub::Traits<T>::PRIMITIVE>
struct DeviceTest {
  static __device__ __forceinline__ 
    T Reduce(typename WarpReduce::TempStorage &temp_storage, 
             T &data, ReductionOp &reduction_op) {
    return WarpReduce(temp_storage).Reduce(data, reduction_op);
  }

  static __device__ __forceinline__ 
    T Reduce(typename WarpReduce::TempStorage &temp_storage,
             T &data, ReductionOp &reduction_op, const int &valid_warp_threads) {
    return WarpReduce(temp_storage).Reduce(data, reduction_op, valid_warp_threads);
  }

  template <typename FlagT>
  static __device__ __forceinline__ 
    T HeadSegmentedReduce(typename WarpReduce::TempStorage &temp_storage,
                          T &data, FlagT &flag, ReductionOp &reduction_op) {
    return WarpReduce(temp_storage).HeadSegmentedReduce(data, flag, reduction_op);
  }

  template <typename FlagT>
  static __device__ __forceinline__ 
    T TailSegmentedReduce(typename WarpReduce::TempStorage &temp_storage,
                          T &data, FlagT &flag, ReductionOp &reduction_op) {
    return WarpReduce(temp_storage).TailSegmentedReduce(data, flag, reduction_op);
  }
};


/**
 * Summation
 */
template <typename T, typename WarpReduce> 
struct DeviceTest<T, cub::Sum, WarpReduce, true> {
  static __device__ __forceinline__ 
    T Reduce(typename WarpReduce::TempStorage &temp_storage,
              T &data, cub::Sum &reduction_op) {
    return WarpReduce(temp_storage).Sum(data);
  }

  static __device__ __forceinline__ 
    T Reduce(typename WarpReduce::TempStorage &temp_storage, 
             T &data, cub::Sum &reduction_op, 
             const int &valid_warp_threads) {
    return WarpReduce(temp_storage).Sum(data, valid_warp_threads);
  }

  template <typename FlagT>
  static __device__ __forceinline__ 
    T HeadSegmentedReduce(typename WarpReduce::TempStorage &temp_storage,
                          T &data, FlagT &flag, cub::Sum &reduction_op) {
    return WarpReduce(temp_storage).HeadSegmentedSum(data, flag);
  }

  template <typename FlagT>
  static __device__ __forceinline__ 
    T TailSegmentedReduce(typename WarpReduce::TempStorage &temp_storage, 
                          T &data, FlagT &flag, cub::Sum &reduction_op) {
    return WarpReduce(temp_storage).TailSegmentedSum(data, flag);
  }
};


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

  T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
    temp_storage[warp_id], input, reduction_op);

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
  T output = DeviceTest<T, ReductionOp, WarpReduce>::Reduce(
    temp_storage[warp_id], input, reduction_op, valid_warp_threads);

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
 * Head-based segmented warp reduction test kernel
 */
template <int WARPS, int LOGICAL_WARP_THREADS, 
          typename T, typename FlagT, typename ReductionOp>
__global__ void WarpHeadSegmentedReduceKernel(T *d_in, FlagT *d_head_flags,
                                              T *d_out, ReductionOp reduction_op,
                                              clock_t *d_elapsed) {
  // Cooperative warp-reduce utility type
  typedef cub::WarpReduce<T, LOGICAL_WARP_THREADS> WarpReduce;

  // Allocate temp storage in shared memory
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS];

  // Per-thread tile data
  T       input = d_in[threadIdx.x];
  FlagT   head_flag = d_head_flags[threadIdx.x];

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t start = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  // Test segmented warp reduce
  int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;
  T output = DeviceTest<T, ReductionOp, WarpReduce>::HeadSegmentedReduce(
    temp_storage[warp_id], input, head_flag, reduction_op);

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t stop = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  *d_elapsed = stop - start;

  // Store aggregate
  d_out[threadIdx.x] = ((threadIdx.x % LOGICAL_WARP_THREADS == 0) || head_flag) ?
    output :
    input;
}


/**
 * Tail-based segmented warp reduction test kernel
 */
template <int WARPS, int LOGICAL_WARP_THREADS,
  typename T, typename FlagT, typename ReductionOp>
__global__ void WarpTailSegmentedReduceKernel(T *d_in, FlagT *d_tail_flags, 
                                              T *d_out, ReductionOp reduction_op,
                                              clock_t *d_elapsed) {
  // Cooperative warp-reduce utility type
  typedef cub::WarpReduce<T, LOGICAL_WARP_THREADS> WarpReduce;

  // Allocate temp storage in shared memory
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS];

  // Per-thread tile data
  T       input = d_in[threadIdx.x];
  FlagT    tail_flag = d_tail_flags[threadIdx.x];
  FlagT    head_flag = (threadIdx.x == 0) ?
    0 :
    d_tail_flags[threadIdx.x - 1];

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t start = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  // Test segmented warp reduce
  int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;
  T output = DeviceTest<T, ReductionOp, WarpReduce>::TailSegmentedReduce(
    temp_storage[warp_id], input, tail_flag, reduction_op);

  // Record elapsed clocks
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t stop = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  *d_elapsed = stop - start;

  // Store aggregate
  d_out[threadIdx.x] = ((threadIdx.x % LOGICAL_WARP_THREADS == 0) || head_flag) ?
    output :
    input;
}


//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize reduction problem (and solution)
 */
template <typename T, typename ReductionOp>
void Initialize(int gen_mode, int flag_entropy, T *h_in, int *h_flags,
                int warps, int warp_threads, int valid_warp_threads, 
                ReductionOp reduction_op, T *h_head_out, T *h_tail_out) {

  for (int i = 0; i < warps * warp_threads; ++i) {
    // Sample a value for this item
    //InitValue(gen_mode, h_in[i], i);
    h_in[i] = (T)i;
    h_head_out[i] = h_in[i];
    h_tail_out[i] = h_in[i];

    // Sample whether or not this item will be a segment head
    //char bits;
    //RandomBits(bits, flag_entropy);
    h_flags[i] = 1;// bits & 0x1;
  }

  // Accumulate segments (lane 0 of each warp is implicitly a segment head)
  for (int warp = 0; warp < warps; ++warp) {
    int warp_offset = warp * warp_threads;
    int item_offset = warp_offset + valid_warp_threads - 1;

    // Last item in warp
    T head_aggregate = h_in[item_offset];
    T tail_aggregate = h_in[item_offset];

    if (h_flags[item_offset])
      h_head_out[item_offset] = head_aggregate;
    item_offset--;

    // Work backwards
    while (item_offset >= warp_offset) {
      if (h_flags[item_offset + 1]) {
        head_aggregate = h_in[item_offset];
      }
      else {
        head_aggregate = reduction_op(head_aggregate, h_in[item_offset]);
      }

      if (h_flags[item_offset]) {
        h_head_out[item_offset] = head_aggregate;
        h_tail_out[item_offset + 1] = tail_aggregate;
        tail_aggregate = h_in[item_offset];
      }
      else {
        tail_aggregate = reduction_op(tail_aggregate, h_in[item_offset]);
      }

      item_offset--;
    }

    // Record last segment head_aggregate to head offset
    h_head_out[warp_offset] = head_aggregate;
    h_tail_out[warp_offset] = tail_aggregate;
  }
}


/**
 * Test warp reduction
 */
template <int WARPS, int LOGICAL_WARP_THREADS,
          typename T, typename ReductionOp>
void TestReduce(int gen_mode, ReductionOp reduction_op,
                int valid_warp_threads = LOGICAL_WARP_THREADS) {

  const int BLOCK_THREADS = LOGICAL_WARP_THREADS * WARPS;

  // Allocate host arrays
  T   *h_in = new T[BLOCK_THREADS];
  int *h_flags = new int[BLOCK_THREADS];
  T   *h_out = new T[BLOCK_THREADS];
  T   *h_tail_out = new T[BLOCK_THREADS];

  // Initialize problem
  Initialize(gen_mode, -1, h_in, h_flags, WARPS, LOGICAL_WARP_THREADS, valid_warp_threads, reduction_op, h_out, h_tail_out);

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
  printf("\nGen-mode %d, %d warps, %d warp threads, %d valid lanes, %s (%d bytes) elements:\n",
    gen_mode,
    WARPS,
    LOGICAL_WARP_THREADS,
    valid_warp_threads,
    typeid(T).name(),
    (int) sizeof(T));
  fflush(stdout);

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

  // Copy out and display results
  printf("\tReduction results: ...");
  //int compare = CompareDeviceResults(h_out, d_out, BLOCK_THREADS, g_verbose, g_verbose);
  //printf("%s\n", compare ? "FAIL" : "PASS");
  //AssertEquals(0, compare);
  //printf("\tElapsed clocks: ");
  //DisplayDeviceResults(d_elapsed, 1);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_flags) delete[] h_flags;
  if (h_out) delete[] h_out;
  if (h_tail_out) delete[] h_tail_out;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}


/**
 * Test warp segmented reduction
 */
template <int WARPS, int LOGICAL_WARP_THREADS, 
          typename T, typename ReductionOp>
void TestSegmentedReduce(int gen_mode, 
                         int flag_entropy, 
                         ReductionOp reduction_op) {
  const int BLOCK_THREADS = LOGICAL_WARP_THREADS * WARPS;

  // Allocate host arrays
  int compare;
  T   *h_in = new T[BLOCK_THREADS];
  int *h_flags = new int[BLOCK_THREADS];
  T   *h_head_out = new T[BLOCK_THREADS];
  T   *h_tail_out = new T[BLOCK_THREADS];

  // Initialize problem
  Initialize(gen_mode, flag_entropy, h_in, h_flags, WARPS, LOGICAL_WARP_THREADS, LOGICAL_WARP_THREADS, reduction_op, h_head_out, h_tail_out);

  // Initialize/clear device arrays
  T           *d_in = NULL;
  int         *d_flags = NULL;
  T           *d_head_out = NULL;
  T           *d_tail_out = NULL;
  clock_t     *d_elapsed = NULL;

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(int) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_head_out, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_tail_out, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * BLOCK_THREADS, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_flags, h_flags, sizeof(int) * BLOCK_THREADS, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemset(d_head_out, 0, sizeof(T) * BLOCK_THREADS));
  CubDebugExit(cudaMemset(d_tail_out, 0, sizeof(T) * BLOCK_THREADS));

  //if (g_verbose)
  //{
  //    printf("Data:\n");
  //    for (int i = 0; i < WARPS; ++i)
  //        DisplayResults(h_in + (i * LOGICAL_WARP_THREADS), LOGICAL_WARP_THREADS);

  //    printf("\nFlags:\n");
  //    for (int i = 0; i < WARPS; ++i)
  //        DisplayResults(h_flags + (i * LOGICAL_WARP_THREADS), LOGICAL_WARP_THREADS);
  //}

  printf("\nGen-mode %d, head flag entropy reduction %d, %d warps, %d warp threads, %s (%d bytes) elements:\n",
    gen_mode,
    flag_entropy,
    WARPS,
    LOGICAL_WARP_THREADS,
    typeid(T).name(),
    (int) sizeof(T));
  fflush(stdout);

  // Run head-based kernel
  WarpHeadSegmentedReduceKernel<WARPS, LOGICAL_WARP_THREADS> << <1, BLOCK_THREADS >> > (
    d_in,
    d_flags,
    d_head_out,
    reduction_op,
    d_elapsed);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Copy out and display results
  printf("\tHead-based segmented reduction results: ...");
  //compare = CompareDeviceResults(h_head_out, d_head_out, BLOCK_THREADS, g_verbose, g_verbose);
  //printf("%s\n", compare ? "FAIL" : "PASS");
  //AssertEquals(0, compare);
  //printf("\tElapsed clocks: ");
  //DisplayDeviceResults(d_elapsed, 1);

  // Run tail-based kernel
  WarpTailSegmentedReduceKernel<WARPS, LOGICAL_WARP_THREADS> << <1, BLOCK_THREADS >> > (
    d_in,
    d_flags,
    d_tail_out,
    reduction_op,
    d_elapsed);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Copy out and display results
  printf("\tTail-based segmented reduction results: ...");
  //compare = CompareDeviceResults(h_tail_out, d_tail_out, BLOCK_THREADS, g_verbose, g_verbose);
  //printf("%s\n", compare ? "FAIL" : "PASS");
  //AssertEquals(0, compare);
  //printf("\tElapsed clocks: ");
  //DisplayDeviceResults(d_elapsed, 1);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_flags) delete[] h_flags;
  if (h_head_out) delete[] h_head_out;
  if (h_tail_out) delete[] h_tail_out;
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_flags) CubDebugExit(g_allocator.DeviceFree(d_flags));
  if (d_head_out) CubDebugExit(g_allocator.DeviceFree(d_head_out));
  if (d_tail_out) CubDebugExit(g_allocator.DeviceFree(d_tail_out));
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
  TestReduce<1, 32, int>(1, cub::Sum());
  TestReduce<1, 32, double>(1, cub::Sum());
  //TestReduce<2, 16, TestBar>(1, Sum());
  TestSegmentedReduce<1, 32, int>(1, 1, cub::Sum());  //Max()

  cjmcv_cuda_util::CleanUpEnvironment();
  return 0;
}