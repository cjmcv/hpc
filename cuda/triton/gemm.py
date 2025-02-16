# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py

"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.



"""
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = tl.program_id(axis=0)
#    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    pid_m = pid // grid_n
#    pid_n = pid % grid_n
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# 行主序计算9*9的矩阵，计算一行9个元素的输出，需要A取一行9个元素，B取9列9个元素，共需要取90个元素。
# 如果改成列主序，
# Final Result
# ------------

import torch

import triton
import triton.language as tl

DEVICE = 'cuda:0' # triton.runtime.driver.active.get_current_target()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

# triton.Config
# meta：一个字典，包含了内核运行时使用的元参数。这些元参数会作为常量表达式传递给内核函数。例如，{'BLOCK_SIZE': 32} 表示块大小为 32。
#      注意，这里的BLOCK_SIZE_M或BLOCK_SIZE，针对的是数据的布局，指的是数据块，是会显式以入参方式传入kernel函数。而不是实际意义上的线程块，线程块由num_warps决定。
# 其他：
#   num_warps: 每个块使用的线程束（warp）数量，一个warp 32个线程，也决定了一个block里线程的数量。
#   num_stages: 流水线阶段的数量, 核心思想是将一个计算任务分解成多个阶段(如加载/计算/存储)，并且让这些阶段可以重叠执行，从而减少整体的执行时间。
#   num_ctas: 整个grid里block的数量。一般在grid中指定，如无特殊要求，可以不填。
#   maxnreg: 每个线程所能使用的寄存器的最大数目。寄存器总数有限，
#            需要结合线程数量和任务情况(不只是人为创建的部分，还有些代码自动需要较多寄存器存放中间结果)来设定。如无特殊要求，可以不填。
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

# 针对 @triton.jit 装饰的kernel函数，可以使用 @triton.autotune 进行自动调优。
# -- 提供一个 triton.Config 的列表，定义 meta-parameters 的不同定义 和 编译选项。
# -- key中写的MNK是目标kernel的其中三个入参，autotune会根据key指定的参数数值，自动调优。
#    这个例子里，MNK是输入数据的维度，不同输入会有不同的MNK，调优后最佳的配置就会有变化。
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters，可以在autotune中设置，BLOCK_SIZE表示该block负责的数据块维度。
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    # 假设BLOCK_SIZE_M/BLOCK_SIZE_N是128, MNK都是512，则block数量是16=4*4
    # pid取值是0-15.
    # num_pid_m = 4, num_pid_n = 4. 对应M和N维度上各需要多少个block。
    # 令GROUP_SIZE_M=2，即m维度上以2分组。
    #
    # 按行切分矩阵进行分组
    # axis=0是指x维度，pid等同于blockIdx.x, 因为外面调用处的grid是一维的，所以也只能取axis=0。
    # 以pid去指向该线程块负责计算的C矩阵部分。
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # 向上取整，M维度上需要使用多少个block来处理
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # N维度上需要使用多少个block来处理
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # GROUP_SIZE_M 对 num_pid_m 在行方向分组，分好组后，m*n得到该组数据所需的block。
    group_id = pid // num_pid_in_group # block id号除以一组所需的block，划分出 group id号
    first_pid_m = group_id * GROUP_SIZE_M # 一group在m维度上对应的block的起始点
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 当前block对应的group size，m维度最后一组有可能凑不满GROUP_SIZE_M个block，所以需要取小值。
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) # 
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  # 行偏移
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # 列偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.
configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else ['cublas', "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else ['cuBLAS', "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("Triton and Torch match")
    else:
        print("Triton and Torch differ")

    if TORCH_HAS_FP8 and is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency.
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("Triton and Torch match")
        else:
            print("Triton and Torch differ")

    benchmark.run(show_plots=True, print_data=True)

