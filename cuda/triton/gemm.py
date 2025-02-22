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
#    这个例子里，MNK是输入数据的维度，不同输入会有不同的MNK，根据不同的MNK调优出最佳的配置。
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
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters，可以在autotune中设置，BLOCK_SIZE_M/N/K表示该block负责的数据块维度，
        # GROUP_SIZE_M表示M方向分组的维度大小
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

    # 按行切分矩阵进行分组
    # axis=0是指x维度，pid等同于blockIdx.x, 因为外面调用处的grid是一维的，所以也只能取axis=0。
    # 以pid去指向该线程块负责计算的C矩阵部分。 tl.cdiv 是向上取整, // 是向下取整。
    #
    # 假设BLOCK_SIZE_M/BLOCK_SIZE_N是128, MNK都是512，则block数量是16 = 512/128 * 512/128
    # 则pid取值是0-15. num_pid_m = 4, num_pid_n = 4, 对应M和N维度上各需要多少个block。
    # 令GROUP_SIZE_M=2，即m维度上以分组维度是2，即分成了两组，即4*4分成了2 * 2*4。
    # num_pid_in_group=8=2*4，即一组由8个block，则对于group_id 0 = pid 0-7, group_id 1 = pid 8-15
    # first_pid_m = 0 或 2 表示每组M方向的首个下标
    # group_size_m = 2 = min(4-0, 2) / min(4-2, 2), 表示m方向的实际group_size，如m方向不能被GROUP_SIZE_M整除，则最后一组的group_size_m会小于GROUP_SIZE_M。
    # pid_m = 0 + (pid0-7 % 8) % 2  => 0-7 : 0, 1, 0, 1, 0, 1, 0, 1
    #       = 2 + (pid8-15 % 8) % 2 => 8-15: 2, 3, 2, 3, 2, 3, 2, 3
    # pid_n = (pid0-7 % 8) // 2     => 0-7 : 0, 0, 1, 1, 2, 2, 3, 3
    #       = (pid8-15 % 8) // 2    => 8-15: 0, 0, 1, 1, 2, 2, 3, 3
    # 最终排布block的方式为：
    # pid   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    # data 00, 10, 01, 11, 02, 12, 03, 13, 20, 30, 21, 31, 22, 32, 23, 33
    # 即 pid 在4x4块中负责的部分位置，每个区域对应c矩阵的[BLOCK_SIZE_M, BLOCK_SIZE_N]数据块大小
    #   0  2  4  6
    #   1  3  5  7
    #   8 10 12 14
    #   9 11 13 15   (分组后)
    # 
    # 一维id转二维的基本方式是 %和/，如0-15转为4行4列，可以基于行数4，计算y=pid/4=(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3)，x=pid%4=(0,1,2,3,,0,1,2,3,,0,1,2,3,,0,1,2,3)
    # 行用/，列用%，则是行主序，而该例子分组后是行用%，列用/，则为列主序。
    # 即 pid 在4x4块中负责的部分位置：
    #   0  1  2  3
    #   4  5  6  7
    #   8  9 10 11
    #  12 13 14 15   (分组前)
    #
    # 计算c矩阵：
    # 分组前：对于c的第0行，需要a矩阵的第0行(1,2,3,4)和b矩阵的4列全部数据，a的一行加载一次后保持在L2里，则共4+4*4=20。
    #         因L2无法将b全部存入，所以计算c的其他行时，b也需要重复读取，因为老数据已被L2踢出。共需要加载20*4=80块。
    # 分组后：c的0号 = a的0行 + b的0列 = 4+4=8; c的1号=a的1行 + b的0列(L2) = 4，c的2号=a的0行(L2) + b的1列 = 4, c的3号=a的1行(L2) + b的1列(L2) = 0;
    #        c的4号 = a的0行(L2) + b的2列 = 4; c的5号=a的1行 + b的2列(L2) = 4，c的6号=a的0行(L2) + b的3列 = 4, c的7号=a的1行(L2) + b的3列(L2) = 0; 共28块。
    #        假设第二组的数据L2数据全部被换出无法复用，则共加载28x2块=56块即可。分组与分子块思路一致。

    pid = tl.program_id(axis=0)           # block id
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # 对应M和N维度上各需要多少个block
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 一组需要多少个block
    group_id = pid // num_pid_in_group # 以pid换算到group id号
    first_pid_m = group_id * GROUP_SIZE_M # 一组在m维度上对应的block的起始点
    # 当前block对应的group size，m维度最后一组有可能凑不满GROUP_SIZE_M个block，所以需要取小值
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) 
    # 一维block转二维分布，划分每个block负责的数据块位置，分别对应行和列
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) 
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # offs_am: m方向，pid_m负责的块的所有行的索引
    # offs_bn: n方向，pid_n负责的块的所有列的索引
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_am[:, None] 维度从[BLOCK_SIZE_M], 转为[BLOCK_SIZE_M, 1], 
    # stride_am是跨行的步长，广播后逐元素相乘，即从行索引转变成了对应行的首地址偏移量。
    # offs_k[None, :] 维度从[BLOCK_SIZE_K], 转为[1, BLOCK_SIZE_K], stride_ak是a矩阵第二维度的步长，一般就是1.
    # 转变为行首地址偏移的offs_am[BLOCK_SIZE_M, 1]和转变为列偏移的offs_k[1, BLOCK_SIZE_K]相加（广播），再加上a_ptr后，
    # 得到一个偏移量的二维数组a_ptrs[BLOCK_SIZE_M, BLOCK_SIZE_K]，则每个元素对应a矩阵对应行m在k方向的第一个数据块的一个数据的地址索引。
    # b_ptrs[BLOCK_SIZE_K, BLOCK_SIZE_N] 同理， 对应b矩阵对应列n在k方向第一个数据块的地址索引。
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # 循环计算累加数据到一个矩阵块accumulator中暂存，对应c矩阵一个块的结果，
    # accumulator按fp32类型进行累加，尽可能保存精度。计算完了后，在写入c时，再转回fp16，
    # for循环中tl.cdiv(K, BLOCK_SIZE_K)指k维度上需要计算多少个数据块，k指代k维度上的数据块id。
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 沿着a/b矩阵各自k方向取出数据块进行矩阵乘计算。
        # offs_k[None, :]是[1, BLOCK_SIZE_K], 数值是从0到BLOCK_SIZE_K，需要确保数值需要小于K - k * BLOCK_SIZE_K，
        # 如k遍历到最后一个数据块，K - k * BLOCK_SIZE_K 的数据小于BLOCK_SIZE_K，则超出部分不应计算，置为0.
        # a_ptrs[BLOCK_SIZE_M, BLOCK_SIZE_K], mask是[1, BLOCK_SIZE_K]，会将mask广播为[BLOCK_SIZE_M, BLOCK_SIZE_K]在处理。
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # accumulator = accumulator + a*b
        accumulator = tl.dot(a, b, accumulator)
        # 指向k维度的下一个数据块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 应用fp32的激活函数，后转为fp16.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # 写入C矩阵中
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# 可以选择通过meta-parameter，将激活函数的变量传入到matmul_kernel中，从而融合leaky_relu。
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D的线程块布局，由这线程块的数量可知，每个线程块会负责c矩阵的[BLOCK_SIZE_M, BLOCK_SIZE_N]数据块的计算
    # triton kernel编写的优势在于不需要管理线程块内的线程如何计算，最底层只需要关注线程块的调度即可。
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

