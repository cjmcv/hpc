# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py

"""
Vector Addition
"""

import torch

import triton
import triton.language as tl

DEVICE = torch.cuda.current_device() # 'cuda:0' # triton.runtime.driver.active.get_active_torch_device()

# triton.jit 装饰器，用于定义kernel
# BLOCK_SIZE指代1D的blockDim，需要使用tl.constexpr类型
@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    # axis=0是指x维度，pid等同于blockIdx.x，
    # 全程以block (即这里的program) 为单位进行操作，不需要手动管理线程的调度关系。
    pid = tl.program_id(axis=0)
    # 等同于 blockIdx.x*blockDims.x, block_start就是该block要负责的数据的起始点。
    # tl.arange(0, BLOCK_SIZE)取出该block负责的所有offset。
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 因为总共有n_elements个数据，需要把超出范围的部分去掉。
    mask = offsets < n_elements
    # block从DRAM加载数据，读取所负责的数据，同时加mask防止读取越界。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # block把负责的这个数据块结果存到DRAM中
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    # 设定grid的维度, 一维就是(block_per_gird_x, ), 二维是(block_per_gird_y, block_per_gird_x), 注意二维grid是先行后列，cuda的是先列后行
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # BLOCK_SIZE指定为1024，即一维block有1024个线程
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 这里没有调用torch.cuda.synchronize()，所以到这里add_kernel还是异步的。
    return output

# 接受triton.testing.Benchmark为参数的装饰器，用于性能测试。
# x_names是可变参数的名字，对应def benchmark(size, provider)中的size
# x_vals的值对应着x_names，即会把x_vals列表对应的每一个数值，以此填入到benchmark的size参数里。
# line_arg用于区分不同测试线的参数，参数名字是provider，与benchmark(size, provider)对应。
# line_vals=['triton', 'torch']中这两个元素会依次填入到provider入参中。
# line_names别名标志？
# styles=画图风格，y label=y轴标签，plot_name=图表名
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], 
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,  # x axis is logarithmic.
        line_arg='provider', 
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    # 分位数, 假设计算10次，会将10次的计算耗时从小到大排序，然后在50%，20%和80%处取出耗时数据，组成一个列表放在ms里。min_ms和max_ms还是一个数值。
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

    # 可选填 save_path='/path/to/results/', 将结果以csv格式存下来。 
    benchmark.run(print_data=True, show_plots=True)