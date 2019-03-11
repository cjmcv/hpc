## A collection of practical code for high performance computing.

---

### CPP11-MT
* [alg_vector_dot_product](https://github.com/cjmcv/hpc/tree/master/cpp11-mt/alg_vector_dot_product.cpp)： Vector dot product: h_result = SUM(A * B). Record the basic usage of std::tread and std::sync.
* [base_async](https://github.com/cjmcv/hpc/tree/master/cpp11-mt/base_async.cpp)： Record the basic usage of std::async.
* [util_blocking_queue](https://github.com/cjmcv/hpc/tree/master/cpp11-mt/util_blocking_queue.cpp)： Blocking queue. Mainly implemented by thread, queue and condition_variable.
* [util_internal_thread](https://github.com/cjmcv/hpc/tree/master/cpp11-mt/util_internal_thread.cpp)： Internal Thread. Mainly implemented by thread.
* [util_thread_pool](https://github.com/cjmcv/hpc/tree/master/cpp11-mt/util_thread_pool.cpp)： Thread Pool. Mainly implemented by thread, queue, future and condition_variable.

### CUDA
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

### Intel-SIMD
* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/intel-simd/matrix_multiply.cpp) ： Matrix Multiplication.
* [vector_dot_product](https://github.com/cjmcv/hpc/blob/master/intel-simd/vector_dot_product.cpp) ： Vector dot product: result = SUM(A * B).
* [vector_scan](https://github.com/cjmcv/hpc/blob/master/intel-simd/vector_scan.cpp) ： Scan. Prefix Sum.

### LLVM
* [fibonacci](https://github.com/cjmcv/hpc/blob/master/llvm/fibonacci.cpp) ： An example of how to build quickly a small module with function Fibonacci and execute it with the JIT.

### MPI
* [base_hello_world](https://github.com/cjmcv/hpc/blob/master/mpi/base_hello_world.cpp) ： Environment Management Routines.
* [base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/base_send_recv.cpp) ： Record the basic usage of MPI_Send/MPI_Recv and MPI_ISend/MPI_IRecv.
* [py_base_broadcast_scatter_gather](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_broadcast_scatter_gather.py) ： Record the basic usage of Bcast, Scatter, Gather and Allgather.
* [py_base_reduce_scan](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_reduce_scan.py) ： Record the basic usage of Reduce and Scan.
* [py_base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_send_recv.py) ： Record the basic usage of Send and Recv.

### OpenCL
* [ocl_util](https://github.com/cjmcv/hpc/blob/master/opencl/ocl_util.h) ： Utility functions.
* [alg_dot_product](https://github.com/cjmcv/hpc/blob/master/opencl/alg_dot_product.cpp) ： Vector dot product, h_result = SUM(A * B).
* [alg_vector_add](https://github.com/cjmcv/hpc/blob/master/opencl/alg_vector_add.cpp) ： Vector addition: C = A + B.
* [base_platform_info](https://github.com/cjmcv/hpc/blob/master/opencl/base_platform_info.cpp) ： Query OpenCL platform information.

### TBB
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

---