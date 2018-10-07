/*!
* \brief Simple demonstration of WarpScan.
*/

#include <stdio.h>
#include <typeinfo>

#include "cuda_util.h"

#include <cub/warp/warp_scan.cuh>
#include <cub/util_allocator.cuh>

using namespace cub;

CachingDeviceAllocator  g_allocator(true);

enum ScanMode {
  INCLUSIVE,
  INCLUSIVE_AGGREGATE,
  EXCLUSIVE,
  EXCLUSIVE_AGGREGATE
};

template <typename T>
void Initialize(T *h_in, int num_items) {
  for (int i = 0; i < num_items; ++i)
    h_in[i] = i+1;
}

// CPU version.
// Solve exclusive-scan problem.
template <typename T>
int SolveInCPU(T *h_in, T *h_out, int num_items, ScanMode mode, T &aggregate) {
  T sum = 0;
  if (mode == INCLUSIVE || mode == INCLUSIVE_AGGREGATE) {
    for (int i = 0; i < num_items; ++i) {
      sum += h_in[i];
      h_out[i] = sum;
    }
  }
  else if (mode == EXCLUSIVE || mode == EXCLUSIVE_AGGREGATE) {
    for (int i = 0; i < num_items; ++i) {
      h_out[i] = sum;
      sum += h_in[i];
    }
  }

  if (mode == INCLUSIVE_AGGREGATE || mode == EXCLUSIVE_AGGREGATE)
    aggregate = sum;
  else
    aggregate = 0;

  return 0;
}

/// Scan operation in different mode.
/// Overloading.
template <typename WarpScanT, typename T, typename ScanOpT, typename InitialValueT>
__device__ __forceinline__ void DeviceScan(WarpScanT &warp_scan, T &data,
                                           T &initial_value, ScanOpT &scan_op,
                                           T &aggregate,
                                           Int2Type<INCLUSIVE> test_mode) {
  warp_scan.InclusiveScan(data, data, scan_op);
}
template <typename WarpScanT, typename T, typename ScanOpT, typename InitialValueT>
__device__ __forceinline__ void DeviceScan(WarpScanT &warp_scan, T &data,
                                           T &initial_value, ScanOpT &scan_op,
                                           T &aggregate,
                                           Int2Type<INCLUSIVE_AGGREGATE> test_mode) {
  warp_scan.InclusiveScan(data, data, scan_op, aggregate);
}
template <typename WarpScanT, typename T, typename ScanOpT, typename InitialValueT>
__device__ __forceinline__ void DeviceScan(WarpScanT &warp_scan, T &data,
                                           T &initial_value, ScanOpT &scan_op,
                                           T &aggregate,
                                           Int2Type<EXCLUSIVE> test_mode) {
  warp_scan.ExclusiveScan(data, data, initial_value, scan_op);
}
template <typename WarpScanT, typename T, typename ScanOpT, typename InitialValueT>
__device__ __forceinline__ void DeviceScan(WarpScanT &warp_scan, T &data,
                                           T &initial_value, ScanOpT &scan_op,
                                           T &aggregate,
                                           Int2Type<EXCLUSIVE_AGGREGATE> test_mode) {
  warp_scan.ExclusiveScan(data, data, initial_value, scan_op, aggregate);
}

/**
* WarpScan test kernel
*/
template <int NUM_WARPS, int LOGICAL_WARP_THREADS, ScanMode SCAN_MODE,
          typename T, typename ScanOpT, typename InitialValueT>
__global__ void WarpScanKernel(ScanOpT scan_op, InitialValueT initial_value,
                               T *d_in, T *d_out, T *d_aggregate, 
                               clock_t *d_elapsed) {
  // Cooperative warp-scan utility type (1 warp)
  typedef WarpScan<T, LOGICAL_WARP_THREADS> WarpScanT;

  // Allocate temp storage in shared memory
  __shared__ typename WarpScanT::TempStorage temp_storage[NUM_WARPS];

  // Get warp index
  int warp_id = threadIdx.x / LOGICAL_WARP_THREADS;

  // Per-thread tile data
  T data = d_in[threadIdx.x];

  // Start cycle timer
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t start = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

  T aggregate;

  // Test scan
  WarpScanT warp_scan(temp_storage[warp_id]);
  
  /// The key function. 
  /// You can call them directly like the lines below:
  //  warp_scan.InclusiveSum(data, data, aggregate);
  //  warp_scan.InclusiveSum(data, data);  
  //  warp_scan.ExclusiveSum(data, data, aggregate);
  //  warp_scan.ExclusiveSum(data, data);
  //  warp_scan.InclusiveScan(data, data, scan_op, aggregate);
  //  warp_scan.InclusiveScan(data, data, scan_op);
  //  warp_scan.ExclusiveScan(data, data, initial_value, scan_op, aggregate);
  //  warp_scan.ExclusiveScan(data, data, initial_value, scan_op);
  /// Or you can use Function Overloading to switch them, like this way:
  /// Note: In cub::Int2Type<TEST_MODE>(), the expression must have a constant value.
  ///       So SCAN_MODE can not be one of the input params of this kernel function.
  DeviceScan<WarpScanT, T, ScanOpT, InitialValueT>(
    warp_scan, data, initial_value, scan_op, aggregate, Int2Type<SCAN_MODE>());

  // Stop cycle timer
  __threadfence_block();      // workaround to prevent clock hoisting
  clock_t stop = clock();
  __threadfence_block();      // workaround to prevent clock hoisting

                              // Store data
  d_out[threadIdx.x] = data;
    
  // Store aggregate
  if (SCAN_MODE == INCLUSIVE_AGGREGATE || SCAN_MODE == EXCLUSIVE_AGGREGATE) {
    d_aggregate[threadIdx.x] = aggregate;
  }

  // Store time
  if (threadIdx.x == 0) {
    *d_elapsed = (start > stop) ? start - stop : stop - start;
  }
}

/**
* Test warp scan
*/
template <int NUM_WARPS, int LOGICAL_WARP_THREADS,
          ScanMode SCAN_MODE, typename T,
          typename ScanOpT,typename InitialValueT>        
void Test(ScanOpT scan_op, InitialValueT initial_value) {

  printf("\nTest-mode %d (%s), %s warpscan, %d warp threads, %s (%d bytes) elements:\n",
    SCAN_MODE, typeid(SCAN_MODE).name(),
    (SCAN_MODE == 0 || SCAN_MODE == 1) ? "Inclusive" : "Exclusive",
    LOGICAL_WARP_THREADS,
    typeid(T).name(),
    (int) sizeof(T));

  enum {
    TOTAL_ITEMS = LOGICAL_WARP_THREADS * NUM_WARPS,
  };

  // Allocate host arrays
  T *h_in = new T[TOTAL_ITEMS];
  T *h_out = new T[TOTAL_ITEMS];
  T h_aggregate = 0;

  Initialize(h_in, TOTAL_ITEMS);
  SolveInCPU(h_in, h_out, TOTAL_ITEMS, SCAN_MODE, h_aggregate);

  cjmcv_cuda_util::PrintArray("<CPU> Input array: ", h_in, TOTAL_ITEMS);
  cjmcv_cuda_util::PrintArray("<CPU> Output array: ", h_out, TOTAL_ITEMS);

  // Initialize/clear device arrays
  T *d_in = NULL;
  T *d_out = NULL;
  T *d_aggregate = NULL;
  clock_t *d_elapsed = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * TOTAL_ITEMS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * (TOTAL_ITEMS))); ////////////
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_aggregate, sizeof(T) * TOTAL_ITEMS));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_elapsed, sizeof(clock_t)));
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * TOTAL_ITEMS, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * TOTAL_ITEMS));
  CubDebugExit(cudaMemset(d_aggregate, 0, sizeof(T) * TOTAL_ITEMS));

  // Run aggregate/prefix kernel
  WarpScanKernel<NUM_WARPS, LOGICAL_WARP_THREADS, SCAN_MODE> << <1, TOTAL_ITEMS >> >(
    scan_op,
    initial_value,
    d_in,
    d_out,
    d_aggregate,
    d_elapsed);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  /// Copy out and display results
  T *h_out4cub = new T[TOTAL_ITEMS];
  cudaMemcpy(h_out4cub, d_out, sizeof(T) * TOTAL_ITEMS, cudaMemcpyDeviceToHost);
  cjmcv_cuda_util::PrintArray("<GPU> Output array: ", h_out4cub, TOTAL_ITEMS);

  // Check results.
  bool is_equal = true;
  for (int i = 0; i < TOTAL_ITEMS; i++) {
    if (h_out[i] != h_out4cub[i]) {
      is_equal = false;
      break;
    }
  }
  
  clock_t h_elapsed = NULL;
  cudaMemcpy(&h_elapsed, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost);
    
  T asum = 0;
  if (SCAN_MODE == INCLUSIVE_AGGREGATE || SCAN_MODE == EXCLUSIVE_AGGREGATE) {
    T *h_aggregate4cub = new T[TOTAL_ITEMS];
    cudaMemcpy(h_aggregate4cub, d_aggregate, sizeof(T)*TOTAL_ITEMS, cudaMemcpyDeviceToHost);
    cjmcv_cuda_util::PrintArray("<GPU> Aggregate array: ", h_aggregate4cub, TOTAL_ITEMS);

    //   The computation is based on warp, and when using multiple warps, 
    // The total aggregate should be obtained by adding multiple aggregate in different warps.
    for (int i = 0; i < NUM_WARPS; i++)
      asum += h_aggregate4cub[i*LOGICAL_WARP_THREADS];
    std::cout << "aggregate = (" << h_aggregate << ", " << asum << ")" << std::endl;  
    
    if (h_aggregate4cub) delete[] h_aggregate4cub;
  }

  if(NUM_WARPS == 1) // When NUM_WARPS is 1, the output should be exactly the same as the CPU side
    printf("Test %s, Elapsed clocks: %d\n", (is_equal ? "PASS" : "FAIL"), h_elapsed);
  else
    printf("Test %s, Elapsed clocks: %d\n", ((h_aggregate == asum) ? "PASS" : "FAIL"), h_elapsed);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (h_out4cub) delete[] h_out4cub;
  
  if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
  if (d_aggregate) CubDebugExit(g_allocator.DeviceFree(d_aggregate));
  if (d_elapsed) CubDebugExit(g_allocator.DeviceFree(d_elapsed));
}
          
/**
 * Main.
*/
int main(int argc, char** argv) {

  int ret = cjmcv_cuda_util::InitEnvironment(0);
  if (ret != 0) {
    printf("Failed to initialize the environment for cuda.");
    return -1;
  }

  // Test.
  std::cout << std::endl << "Test module one: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  Test<1, 32, INCLUSIVE, int>(Sum(), (int)0);
  Test<1, 32, INCLUSIVE_AGGREGATE, float>(Sum(), (float)0);
  Test<1, 32, EXCLUSIVE, long long>(Sum(), (long long)0);
  Test<1, 32, EXCLUSIVE_AGGREGATE, double>(Sum(), (double)0);

  std::cout << std::endl << "Test module two: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  Test<1, 32, INCLUSIVE_AGGREGATE, int>(Sum(), (int)0);
  Test<2, 16, INCLUSIVE_AGGREGATE, int>(Sum(), (int)0);
  Test<1, 32, EXCLUSIVE_AGGREGATE, int>(Sum(), (int)0);
  Test<2, 16, EXCLUSIVE_AGGREGATE, int>(Sum(), (int)0);

  std::cout << std::endl << "Test module Three: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
  // The aggregate will be the biggest one in the input array.
  Test<1, 32, INCLUSIVE_AGGREGATE, int>(Max(), (int)0); 
  Test<1, 32, EXCLUSIVE_AGGREGATE, int>(Max(), (int)0);

  cjmcv_cuda_util::CleanUpEnvironment();

  return 0;
}




