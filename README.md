# Learning and practice of high performance computing and ai infra

## Application
<details>
  <summary><strong>pocket-ai</strong>  -- A Portable Toolkit for building AI Infra. </summary>
  
  [https://github.com/cjmcv/pocket-ai](https://github.com/cjmcv/pocket-ai)

* [engine/cl](https://github.com/cjmcv/pocket-ai/tree/master/engine/cl): A small computing framework based on opencl. This framework is designed to help you quickly call Opencl API to do the calculations you need.

* [engine/vk](https://github.com/cjmcv/pocket-ai/tree/master/engine/vk): A small computing framework based on vulkan. This framework is designed to help you quickly call vulkan's computing API to do the calculations you need.

* [engine/graph](https://github.com/cjmcv/pocket-ai/tree/master/engine/graph): A small multitasking scheduler that can quickly build efficient pipelines for your multiple tasks.

* [engine/infer](https://github.com/cjmcv/pocket-ai/tree/master/engine/infer): A tiny inference engine for microprocessors, with a library size of only 10K+.

* [eval/llm](https://github.com/cjmcv/pocket-ai/tree/master/eval/llm): A small tool is used to quickly verify whether the end-to-end calculation results are correct when accelerating and optimizing the large language model (LLM) inference engine.

* Other small tools.

</details>

## Reading Notes

[ai-infra-notes](https://github.com/cjmcv/ai-infra-notes)

sglang, lighteval, vllm, mlc-llm

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

### Heterogeneous computing

<details>
  <summary>cuda</summary>

* [base_graph](https://github.com/cjmcv/hpc/blob/master/cuda/base_graph.cu) : Record the basic usage of cuda graph.
* [base_unified_memory](https://github.com/cjmcv/hpc/blob/master/cuda/base_unified_memory.cu) : A simple task consumer using threads and streams with all data in Unified Memory.
* [base_zero_copy](https://github.com/cjmcv/hpc/blob/master/cuda/base_zero_copy.cu) : Record the basic usage of Zero Copy.
* [gemm_fp16_wmma](https://github.com/cjmcv/hpc/tree/master/cuda/gemm_fp16_wmma.cu) : Gemm fp16 - wmma
* [gemm_fp32](https://github.com/cjmcv/hpc/tree/master/cuda/gemm_fp32.cu) : Gemm fp32 - cuda core
</details>

<details>
  <summary>vulkan</summary>
  
* [gemm_fp32](https://github.com/cjmcv/hpc/tree/master/vulkan/main_gemm.cpp) : Gemm fp32.

</details>

<details>
  <summary>opencl</summary>
  
* [basic_demo](https://github.com/cjmcv/hpc/blob/master/opencl/basic_demo.cpp) : Introduce the basic calling method and process of OpenCL API (without using pocket-ai).
* [gemm_f32](https://github.com/cjmcv/hpc/blob/master/opencl/gemm_fp32.cl) : Gemm fp32 for Discrete graphics card.
* [gemm_mobile_f32](https://github.com/cjmcv/hpc/blob/master/opencl/gemm_mobile_fp32.cl) : Gemm fp32 for integrated graphics card.
</details>


### SIMD

<details>
  <summary>neon</summary>

* [gemm_fp32](https://github.com/cjmcv/hpc/blob/master/simd/arm/gemm_fp32.cpp) : Gemm fp32.
* [gemm_int8](https://github.com/cjmcv/hpc/blob/master/simd/arm/gemm_int8.cpp) : Gemm int8.
* [matrix_transpose](https://github.com/cjmcv/hpc/blob/master/simd/arm/matrix_transpose.cpp) : Matrix Transpose.
</details>

<details>
  <summary>sse/avx</summary>
 
* [matrix_multiply](https://github.com/cjmcv/hpc/blob/master/simd/x86/matrix_multiply.cpp) : Matrix Multiplication. 
* [matrix_transpose](https://github.com/cjmcv/hpc/blob/master/simd/x86/matrix_transpose.cpp) : Matrix Transpose.
* [vector_dot_product](https://github.com/cjmcv/hpc/blob/master/simd/x86/vector_dot_product.cpp) : Vector dot product: result = SUM(A * B).
* [vector_scan](https://github.com/cjmcv/hpc/blob/master/simd/x86/vector_scan.cpp) : Scan. Prefix Sum.
</details>

### Distributed computing

<details>
  <summary>mpi/mpi4py</summary>
  
* [alg_matrix_multiply](https://github.com/cjmcv/hpc/blob/master/mpi/alg_matrix_multiply.cpp) : gemm: C = A * B.
* [base_broadcast_scatter_gather](https://github.com/cjmcv/hpc/blob/master/mpi/base_broadcast_scatter_gather.cpp) : Record the basic usage of Bcast, Scatter, Gather and Allgather.
* [base_group](https://github.com/cjmcv/hpc/blob/master/mpi/base_group.cpp) : Group communication.
* [base_hello_world](https://github.com/cjmcv/hpc/blob/master/mpi/base_hello_world.cpp) : Environment Management Routines.
* [base_reduce_alltoall_scan](https://github.com/cjmcv/hpc/blob/master/mpi/base_reduce_alltoall_scan.cpp) : Record the basic usage of Reduce, Allreduce, Alltoall, Scan and Exscan.
* [base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/base_send_recv.cpp) : Record the basic usage of MPI_Send/MPI_Recv and MPI_ISend/MPI_IRecv.
* [base_type_contiguous](https://github.com/cjmcv/hpc/blob/master/mpi/base_type_contiguous.cpp) : Send and receive custom types of data by using MPI_Type_contiguous.
* [base_type_struct](https://github.com/cjmcv/hpc/blob/master/mpi/base_type_struct.cpp) : Send and receive custom types of data by using MPI_Type_struct.
* [util_bandwidth_test](https://github.com/cjmcv/hpc/blob/master/mpi/util_bandwidth_test.cpp) : Test bandwidth by point-to-point communications.
* [py_base_broadcast_scatter_gather](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_broadcast_scatter_gather.py) : Record the basic usage of Bcast, Scatter, Gather and Allgather.
* [py_base_reduce_scan](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_reduce_scan.py) : Record the basic usage of Reduce and Scan.
* [py_base_send_recv](https://github.com/cjmcv/hpc/blob/master/mpi/mpi4py/base_send_recv.py) : Record the basic usage of Send and Recv.
</details>

### Thread

<details>
  <summary>std</summary>
  
* [alg_quick_sort](https://github.com/cjmcv/hpc/blob/master/std/alg_quick_sort.cpp): Quick sort using std::thread.
* [alg_vector_dot_product](https://github.com/cjmcv/hpc/tree/master/std/alg_vector_dot_product.cpp): Vector dot product: h_result = SUM(A * B). Record the basic usage of std::tread and std::sync.
* [base_async](https://github.com/cjmcv/hpc/tree/master/std/base_async.cpp): Record the basic usage of std::async.
* [util_blocking_queue](https://github.com/cjmcv/hpc/tree/master/std/util_blocking_queue.cpp): Blocking queue. Mainly implemented by thread, queue and condition_variable.
* [util_internal_thread](https://github.com/cjmcv/hpc/tree/master/std/util_internal_thread.cpp): Internal Thread. Mainly implemented by std::thread.
* [util_thread_pool](https://github.com/cjmcv/hpc/tree/master/std/util_thread_pool.cpp): Thread Pool. Mainly implemented by thread, queue, future and condition_variable.
</details>

<details>
  <summary>openmp</summary>
  
* [alg_matrix_multiply](https://github.com/cjmcv/hpc/blob/master/openmp/alg_matrix_multiply.cpp) : gemm: C = A * B.
* [alg_pi_calculate](https://github.com/cjmcv/hpc/blob/master/openmp/alg_pi_calculate.cpp) : Calculate PI using parallel, for and reduction.
* [base_flush](https://github.com/cjmcv/hpc/blob/master/openmp/base_flush.cpp) : Records the basic usage of flush.
* [base_mutex](https://github.com/cjmcv/hpc/blob/master/openmp/base_mutex.cpp) : Mutex operation in openmp, including critical, atomic, lock.
* [base_parallel_for](https://github.com/cjmcv/hpc/blob/master/openmp/base_parallel_for.cpp) : Parallel and For.
* [base_schedule](https://github.com/cjmcv/hpc/blob/master/openmp/base_schedule.cpp) : Records the basic usage of schedule.
* [base_sections_single](https://github.com/cjmcv/hpc/blob/master/openmp/base_sections_single.cpp) : Records the basic usage of Sections and Single.
* [base_synchronous](https://github.com/cjmcv/hpc/blob/master/openmp/base_synchronous.cpp) : Synchronous operation in openmp, including barrier, ordered and master.
</details>

<details>
  <summary>tbb</summary>
  
* [base_allocator](https://github.com/cjmcv/hpc/blob/master/tbb/base_allocator.cpp) : The basic use of allocator.
* [base_atomic](https://github.com/cjmcv/hpc/blob/master/tbb/base_atomic.cpp) : The basic use of atomic.
* [base_concurrent_hash_map](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_hash_map.cpp) : The basic use of concurrent_hash_map.
* [base_concurrent_queue](https://github.com/cjmcv/hpc/blob/master/tbb/base_concurrent_queue.cpp) : The basic use of concurrent queue.
* [base_mutex](https://github.com/cjmcv/hpc/blob/master/tbb/base_mutex.cpp) : The basic use of mutex in tbb.
* [base_parallel_for](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_for.cpp) : The basic use of parallel_for.
* [base_parallel_reduce](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_reduce.cpp) : The basic use of parallel_reduce.
* [base_parallel_scan](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_scan.cpp) : The basic use of parallel_scan.
* [base_parallel_sort](https://github.com/cjmcv/hpc/blob/master/tbb/base_parallel_sort.cpp) : The basic use of base_parallel_sort.
* [base_task_scheduler](https://github.com/cjmcv/hpc/blob/master/tbb/base_task_scheduler.cpp) : The basic use of base_task_scheduler.
* [count_strings](https://github.com/cjmcv/hpc/blob/master/tbb/count_strings.cpp) : Count strings. Use the concurrent_hash_map.
</details>

### Coroutines

<details>
  <summary>libco</summary>
  
</details>

<details>
  <summary>asyncio</summary>
  
* [base_future](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_future.py): Record the basic usage of future.
* [base_gather](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_gather.py): Use gather to execute tasks in parallel.
* [base_hello_world](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_hello_world.py): Hello world. Record the basic usage of async, await and loop.
* [base_loop_chain](https://github.com/cjmcv/hpc/blob/master/coroutine/asyncio/base_loop_chain.py): Executes nested coroutines.
</details>

---