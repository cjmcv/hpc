## A collection of practical code for high performance computing.

---

### CUDA
* [aligned_memory_access](https://github.com/cjmcv/hpc/blob/master/cuda/aligned_memory_access.cu) ： An experiment on memory access.
* [base_float2half](https://github.com/cjmcv/hpc/blob/master/cuda/base_float2half.cu) ： Record the basic usage of float2half.
* [base_occupancy](https://github.com/cjmcv/hpc/blob/master/cuda/base_occupancy.cu) ： Record the basic usage of cudaOccupancyMaxPotentialBlockSize.
* [base_texture](https://github.com/cjmcv/hpc/blob/master/cuda/base_texture.cu) ： Record the basic usage of Texture Memory.
* [base_unified_memory](https://github.com/cjmcv/hpc/blob/master/cuda/base_unified_memory.cu) ： A simple task consumer using threads and streams with all data in Unified Memory.
* [base_zero_copy](https://github.com/cjmcv/hpc/blob/master/cuda/base_zero_copy.cu) ： Record the basic usage of Zero Copy.
* [gemm_cublas_float16](https://github.com/cjmcv/hpc/blob/master/cuda/gemm_cublas_float16.cu) ： gemm: C = A * B. Use cublas with half-precision.
* [histogram](https://github.com/cjmcv/hpc/blob/master/cuda/histogram.cu) ： histogram, mainly introduce atomicAdd.
* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/cuda/matrix_multiply.cu) ： gemm: C = A * B.
* [vector_add](https://github.com/cjmcv/hpc/blob/master/cuda/vector_add.cu) ： Vector addition: C = A + B. 
* [vector_dot_product](https://github.com/cjmcv/hpc/blob/master/cuda/vector_dot_product.cu) ： Vector dot product: h_result = SUM(A * B).
* [vector_scan](https://github.com/cjmcv/hpc/blob/master/cuda/vector_scan.cu) ： Scan. Prefix Sum.

### Intel-SIMD
* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/intel-simd/matrix_multiply.cpp) ： Matrix Multiplication.
* [vector_dot_product](https://github.com/cjmcv/hpc/blob/master/intel-simd/vector_dot_product.cpp) ： Vector dot product: result = SUM(A * B).
* [vector_scan](https://github.com/cjmcv/hpc/blob/master/intel-simd/vector_scan.cpp) ： Scan. Prefix Sum.

### LLVM
* [fibonacci](https://github.com/cjmcv/hpc/blob/master/llvm/fibonacci.cpp) ： An example of how to build quickly a small module with function Fibonacci and execute it with the JIT.

### OpenCL
* [base_platform_info](https://github.com/cjmcv/hpc/blob/master/opencl/base_platform_info.cpp) ： Query OpenCL platform information.

### TBB
* [base_allocator](https://github.com/cjmcv/hpc/blob/master/tbb/base_allocator.cpp) ： The basic use of allocator.
* [base_atomic](https://github.com/cjmcv/hpc/blob/master/tbb/base_atomic.cpp) ： The basic use of atomic.
* [base_concurrent_hash_map](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_hash_map.cpp) ： The basic use of concurrent_hash_map.
* [base_concurrent_queue](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_queue.cpp) ： The basic use of concurrent queue.
* [base_mutex](https://github.com/cjmcv/hpc/blob/master/tbb/base_mutex.cpp) ： The basic use of mutex in tbb.
* [base_parallel_for](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_for.cpp) ： The basic use of parallel_for.
* [base_parallel_reduce](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_reduce.cpp) ： The basic use of parallel_reduce.
* [parallel_scan](https://github.com/cjmcv/hpc/blob/master/tbb/parallel_scan.cpp) ： The basic use of parallel_scan.
* [base_parallel_sort](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_sort.cpp) ： The basic use of base_parallel_sort.
* [base_task_scheduler](https://github.com/cjmcv/hpc/blob/master/tbb/base_task_scheduler.cpp) ： The basic use of base_task_scheduler.
* [count_strings](https://github.com/cjmcv/hpc/blob/master/tbb/count_strings.cpp) ： Count strings. Use the concurrent_hash_map.

---