# Learning and practice of high performance computing

## Insatll
git clone 


## Application
<details>
  <summary><strong>pocket-ai</strong>  -- A Portable Toolkit for deploying Edge AI and HPC. </summary>
  
  [https://github.com/cjmcv/pocket-ai](https://github.com/cjmcv/pocket-ai)
</details>

## Practice

<details>
  <summary><strong>cux</strong> -- An experimental framework for performance analysis and optimization of CUDA kernel functions. </summary>
  
  [https://github.com/cjmcv/hpc/tree/master/0-frameworks/cux](https://github.com/cjmcv/hpc/tree/master/0-frameworks/cux)
  
  tag: cuda / simd / openmp.
</details>

<details>
  <summary><strong>mrpc</strong> -- Mini-RPC, based on asio.</summary>
  
  [https://github.com/cjmcv/hpc/tree/master/0-frameworks/mrpc](https://github.com/cjmcv/hpc/tree/master/0-frameworks/mrpc)
  
  tag: distributed computing.
</details>

<details>
  <summary><strong>DEPRECATED</strong></summary>
  
  [hcs](https://github.com/cjmcv/hpc/tree/20211017/0-frameworks/hcs) A heterogeneous computing system for multi-task scheduling optimization.

  [vky](https://github.com/cjmcv/hpc/tree/20211017/0-frameworks/vky) A Vulkan-based computing framework

  "hcs" and "vky" have been moved to [pocket-ai](https://github.com/cjmcv/pocket-ai/tree/master/engine) and renamed as graph and vk respectively.
</details>

---

## Learning

### Distributed computing

<details>
  <summary>mpi/mpi4py</summary>
  
* [alg_matrix_multiply](https://github.com/cjmcv/hpc/blob/master/mpi/alg_matrix_multiply.cpp) ： gemm: C = A * B.
* [base_broadcast_scatter_gather](https://github.com/cjmcv/hpc/blob/master/mpi/base_broadcast_scatter_gather.cpp) ： Record the basic usage of Bcast, Scatter, Gather and Allgather.
* [base_group](https://github.com/cjmcv/hpc/blob/master/mpi/base_group.cpp) ： Group communication.
* [base_hello_world](https://github.com/cjmcv/hpc/blob/master/mpi/base_hello_world.cpp) ： Environment Management Routines.
* [base_reduce_alltoall_scan](https://github.com/cjmcv/hpc/blob/master/mpi/base_reduce_alltoall_scan.cpp) ： Record the basic usage of Reduce, Allreduce, Alltoall, Scan and Exscan.
* [base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/base_send_recv.cpp) ： Record the basic usage of MPI_Send/MPI_Recv and MPI_ISend/MPI_IRecv.
* [base_type_contiguous](https://github.com/cjmcv/hpc/blob/master/mpi/base_type_contiguous.cpp) ： Send and receive custom types of data by using MPI_Type_contiguous.
* [base_type_struct](https://github.com/cjmcv/hpc/blob/master/mpi/base_type_struct.cpp) ： Send and receive custom types of data by using MPI_Type_struct.
* [util_bandwidth_test](https://github.com/cjmcv/hpc/blob/master/mpi/util_bandwidth_test.cpp) ： Test bandwidth by point-to-point communications.
* [py_base_broadcast_scatter_gather](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_broadcast_scatter_gather.py) ： Record the basic usage of Bcast, Scatter, Gather and Allgather.
* [py_base_reduce_scan](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_reduce_scan.py) ： Record the basic usage of Reduce and Scan.
* [py_base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_send_recv.py) ： Record the basic usage of Send and Recv.
</details>

### Heterogeneous computing

<details>
  <summary>cuda</summary>

* [cuda_util](https://github.com/cjmcv/hpc/blob/master/cuda/cuda_util.h) ： Utility functions.
* [alg_histogram](https://github.com/cjmcv/hpc/blob/master/cuda/alg_histogram.cu) ： histogram, mainly introduce atomicAdd.
* [alg_matrix_multiply](https://github.com/cjmcv/hpc/blob/master/cuda/alg_matrix_multiply.cu) ： gemm: C = A * B.
* [alg_vector_add](https://github.com/cjmcv/hpc/blob/master/cuda/alg_vector_add.cu) ： Vector addition: C = A + B. 
* [alg_vector_dot_product](https://github.com/cjmcv/hpc/blob/master/cuda/alg_vector_dot_product.cu) ： Vector dot product: h_result = SUM(A * B).
* [alg_vector_scan](https://github.com/cjmcv/hpc/blob/master/cuda/alg_vector_scan.cu) ： Scan. Prefix Sum.
* [base_aligned_memory_access](https://github.com/cjmcv/hpc/blob/master/cuda/base_aligned_memory_access.cu) ： An experiment on aligned memory access.
* [base_bank_conflict](https://github.com/cjmcv/hpc/blob/master/cuda/base_bank_conflict.cu) ： An experiment on Bank Conflict in Shared Memory.
* [base_coalesced_memory_access](https://github.com/cjmcv/hpc/blob/master/cuda/base_coalesced_memory_access.cu) ： An experiment on coalesced memory access.
* [base_float2half](https://github.com/cjmcv/hpc/blob/master/cuda/base_float2half.cu) ： Record the basic usage of float2half.
* [base_graph](https://github.com/cjmcv/hpc/blob/master/cuda/base_graph.cu) ： Record the basic usage of cuda graph.
* [base_hyperQ](https://github.com/cjmcv/hpc/blob/master/cuda/base_hyperQ.cu) ： Demonstrate how HyperQ allows supporting devices to avoid false dependencies between kernels in different streams.
* [base_kernel_layout](https://github.com/cjmcv/hpc/blob/master/cuda/base_kernel_layout.cu) ： Record the basic execution configuration of kernel.
* [base_occupancy](https://github.com/cjmcv/hpc/blob/master/cuda/base_occupancy.cu) ： Record the basic usage of cudaOccupancyMaxPotentialBlockSize.
* [base_texture](https://github.com/cjmcv/hpc/blob/master/cuda/base_texture.cu) ： Record the basic usage of Texture Memory.
* [base_unified_memory](https://github.com/cjmcv/hpc/blob/master/cuda/base_unified_memory.cu) ： A simple task consumer using threads and streams with all data in Unified Memory.
* [base_zero_copy](https://github.com/cjmcv/hpc/blob/master/cuda/base_zero_copy.cu) ： Record the basic usage of Zero Copy.
* [cub_block_reduce](https://github.com/cjmcv/hpc/blob/master/cuda/cub_block_reduce.cu) ： Simple demonstration of cub::BlockReduce.
* [cub_block_scan](https://github.com/cjmcv/hpc/blob/master/cuda/cub_block_scan.cu) ： Simple demonstration of cub::BlockScan.
* [cub_device_reduce](https://github.com/cjmcv/hpc/blob/master/cuda/cub_device_reduce.cu) ： Simple demonstration of DeviceScan::Sum.
* [cub_device_scan](https://github.com/cjmcv/hpc/blob/master/cuda/cub_device_scan.cu) ： Simple demonstration of DeviceScan::ExclusiveSum.
* [cub_warp_reduce](https://github.com/cjmcv/hpc/blob/master/cuda/cub_warp_reduce.cu) ： Simple demonstration of cub::WarpReduce.
* [cub_warp_scan](https://github.com/cjmcv/hpc/blob/master/cuda/cub_warp_scan) ： Simple demonstration of cub::WarpScan.
* [cublas_gemm_float16](https://github.com/cjmcv/hpc/blob/master/cuda/cublas_gemm_float16.cu) ： gemm: C = A * B. Use cublas with half-precision.
* [thrust_iterators](https://github.com/cjmcv/hpc/blob/master/cuda/thrust_iterators.cu) ： Record the basic usage of Iterators in Thrust.
* [thrust_sort](https://github.com/cjmcv/hpc/blob/master/cuda/thrust_sort.cu) ： Sort arrays with Thrust.
* [thrust_transformations](https://github.com/cjmcv/hpc/blob/master/cuda/thrust_transformations.cu) ： Some of the parallel vector operations in Thrust.
* [thrust_vector](https://github.com/cjmcv/hpc/blob/master/cuda/thrust_vector.cu) ： Record the basic usage of Vector in Thrust.
</details>

<details>
  <summary>vulkan</summary>
  
* [vky](https://github.com/cjmcv/hpc/tree/master/vulkan/vky)
</details>

<details>
  <summary>opencl</summary>
  
* [ocl_util](https://github.com/cjmcv/hpc/blob/master/opencl/ocl_util.h) ： Utility functions.
* [alg_dot_product](https://github.com/cjmcv/hpc/blob/master/opencl/alg_dot_product.cpp) ： Vector dot product, h_result = SUM(A * B).
* [alg_vector_add](https://github.com/cjmcv/hpc/blob/master/opencl/alg_vector_add.cpp) ： Vector addition: C = A + B.
* [base_platform_info](https://github.com/cjmcv/hpc/blob/master/opencl/base_platform_info.cpp) ： Query OpenCL platform information.
</details>

### Thread

<details>
  <summary>std</summary>
  
* [alg_quick_sort](https://github.com/cjmcv/hpc/blob/master/std/alg_quick_sort.cpp)： Quick sort using std::thread.
* [alg_vector_dot_product](https://github.com/cjmcv/hpc/tree/master/std/alg_vector_dot_product.cpp)： Vector dot product: h_result = SUM(A * B). Record the basic usage of std::tread and std::sync.
* [base_async](https://github.com/cjmcv/hpc/tree/master/std/base_async.cpp)： Record the basic usage of std::async.
* [util_blocking_queue](https://github.com/cjmcv/hpc/tree/master/std/util_blocking_queue.cpp)： Blocking queue. Mainly implemented by thread, queue and condition_variable.
* [util_internal_thread](https://github.com/cjmcv/hpc/tree/master/std/util_internal_thread.cpp)： Internal Thread. Mainly implemented by std::thread.
* [util_thread_pool](https://github.com/cjmcv/hpc/tree/master/std/util_thread_pool.cpp)： Thread Pool. Mainly implemented by thread, queue, future and condition_variable.
</details>

<details>
  <summary>openmp</summary>
  
* [alg_matrix_multiply](https://github.com/cjmcv/hpc/blob/master/openmp/alg_matrix_multiply.cpp) ： gemm: C = A * B.
* [alg_pi_calculate](https://github.com/cjmcv/hpc/blob/master/openmp/alg_pi_calculate.cpp) ： Calculate PI using parallel, for and reduction.
* [base_flush](https://github.com/cjmcv/hpc/blob/master/openmp/base_flush.cpp) ： Records the basic usage of flush.
* [base_mutex](https://github.com/cjmcv/hpc/blob/master/openmp/base_mutex.cpp) ： Mutex operation in openmp, including critical, atomic, lock.
* [base_parallel_for](https://github.com/cjmcv/hpc/blob/master/openmp/base_parallel_for.cpp) ： Parallel and For.
* [base_schedule](https://github.com/cjmcv/hpc/blob/master/openmp/base_schedule.cpp) ： Records the basic usage of schedule.
* [base_sections_single](https://github.com/cjmcv/hpc/blob/master/openmp/base_sections_single.cpp) ： Records the basic usage of Sections and Single.
* [base_synchronous](https://github.com/cjmcv/hpc/blob/master/openmp/base_synchronous.cpp) ： Synchronous operation in openmp, including barrier, ordered and master.
</details>

<details>
  <summary>tbb</summary>
  
* [base_allocator](https://github.com/cjmcv/hpc/blob/master/tbb/base_allocator.cpp) ： The basic use of allocator.
* [base_atomic](https://github.com/cjmcv/hpc/blob/master/tbb/base_atomic.cpp) ： The basic use of atomic.
* [base_concurrent_hash_map](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_hash_map.cpp) ： The basic use of concurrent_hash_map.
* [base_concurrent_queue](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_queue.cpp) ： The basic use of concurrent queue.
* [base_mutex](https://github.com/cjmcv/hpc/blob/master/tbb/base_mutex.cpp) ： The basic use of mutex in tbb.
* [base_parallel_for](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_for.cpp) ： The basic use of parallel_for.
* [base_parallel_reduce](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_reduce.cpp) ： The basic use of parallel_reduce.
* [base_parallel_scan](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_scan.cpp) ： The basic use of parallel_scan.
* [base_parallel_sort](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_sort.cpp) ： The basic use of base_parallel_sort.
* [base_task_scheduler](https://github.com/cjmcv/hpc/blob/master/tbb/base_task_scheduler.cpp) ： The basic use of base_task_scheduler.
* [count_strings](https://github.com/cjmcv/hpc/blob/master/tbb/count_strings.cpp) ： Count strings. Use the concurrent_hash_map.
</details>

### Coroutines

<details>
  <summary>libco</summary>
  
</details>

<details>
  <summary>asyncio</summary>
  
* [base_future](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_future.py)： Record the basic usage of future.
* [base_gather](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_gather.py)： Use gather to execute tasks in parallel.
* [base_hello_world](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_hello_world.py)： Hello world. Record the basic usage of async, await and loop.
* [base_loop_chain](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_loop_chain.py)： Executes nested coroutines.
</details>

### SIMD

<details>
  <summary>sse/avx</summary>
 
* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/simd/x86/matrix_multiply.cpp) ： Matrix Multiplication. 
* [matrix_transpose](https://github.com/cjmcv/hpc/blob/master/simd/x86/matrix_transpose.cpp) ： Matrix Transpose.
* [vector_dot_product](https://github.com/cjmcv/hpc/blob/master/simd/x86/vector_dot_product.cpp) ： Vector dot product: result = SUM(A * B).
* [vector_scan](https://github.com/cjmcv/hpc/blob/master/simd/x86/vector_scan.cpp) ： Scan. Prefix Sum.
</details>

<details>
  <summary>neon</summary>

* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/simd/arm/gemm.cpp) : Matrix Multiplication. 
* [matrix_transpose](https://github.com/cjmcv/hpc/blob/master/simd/arm/matrix_transpose.cpp) ： Matrix Transpose.
</details>

---