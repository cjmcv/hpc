/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

// 我现在也没有在看marlin kernel了，上周周报上写的就是我知道的所有内容了，也没有新的了。
// 关于上周法朝哥你提的那个问题，哪些精度的kernel需要进一步优化?

// 我说一下我的看法，就是周报上提的第二句话：
// 一般W4A16的发力点是在batchsize较小的情况，这时的gemm将会处于访存受限状态，w4理论上能够达到4倍fp16的效果。
// 要选择那种精度的kernel，主要就是看batchsize的大小。如果batchsize很小，marlin kernel的w4a16应该是足够了的。
// 就算是换成w4a8提升空间也很有限，因为瓶颈在w4，计算是需要等待访存的，访存耗时没下来，整体耗时就快不了。除非换个w3甚至更小的权重。

// 另外如果batchsize大的时候，gemm就会处于计算受限状态，这时候的瓶颈在计算，所以batchsize大的情况需要关注的就是a16而不是w4了，
// 那这时候用w4a8应该就能起到不少加速效果，随着batchsize越大，加速效果越明显。
// 但是这时候也会有个问题，batchsize大的情况使用w4是否还合适，因为这里会涉及到w4的反量化。
// marlin kernel大家看过应该也知道，它的操作是使用反量化的计算量是换取访存速度的，
// 就是在小矩阵块计算之前做的反量化，这意味着会有大量的重复反量化的操作（不然的话可以先反量化一大块，然后再读fp16）。
// 也就是说，随着batchsize的增大，反量化的计算量会增大，计算的负担会变大，在计算受限下的gemm下，还去增加它的计算量是不太好的。
// 所以这时候把额外的担子压在访存会比较合适，就是也不用w4a8了，直接用w8a8可能会比较合适。

// 前提是kernel要调得比较好，访存和计算能够相互掩盖。


//////////////////////////////////////////////////////////////////////



// marlin kernel实现的是一个fp16 A x int4/int8 B = fp16 C的混合精度的gemm.
// 一般W4A16的发力点是在batchsize很小的情况，这时的gemm将会处于访存受限状态，w4理论上能够达到4倍fp16的效果。
// （计算次数：2MNK，2是乘加运算，访存次数是：MK+KN+MN，即每个矩阵每个元素都要访问一次）
//  (带宽是GB/s，基本单位是按byte来算的，所以如果元素为w4，即1/2个Byte，访存量需要乘以1/2；同理计算量也需要按对应精度算力进行换算）
//  (batchsize越小，计算/访存的比例就会越小, M=1,N=K=100: 2x100x100 / 100+100x100+100=1.96; M=N=K=100: 2000000/30000=66)
// 所以这个kernel设计关键点在于 降低访存时延，同时让计算时延在访存时延下能够隐藏起来。
// 
// 主要围绕着16x8x16的fp16 mma进行计算，即一次mma为A[16,16] x B[16,8] = C[16,8]。
// 子块划分为[16,16]，按n方向两次mma来完成A[16,16] x B[16,16] = C[16,16]

// 以int4为例:
// 1. 在模型加载后，权重矩阵 B 会做一次repack：
//     1) 针对[16, 16]分块, 将维度排布成[k/16, n*16]，使16x16子块排成一行。（一个是合并访问，另外一个int4=128bit=32个4bit，就包含了两行）
//     2) 将子块元素按列混插方式排布，因为gemm里采用的是fp16计算，而B矩阵是int4，
//         在计算之前会涉及到反量化操作，需要对用int32存储的8个int4分别进行拼凑，
//         则需要将相邻的两个int4分别划分到高16位和低16位中，便于在mask后直接进反量化操作。
//     3) 因为B矩阵是int4，涉及到unpack和反量化，无法使用ldmatrix进行读取，
//         所以将子块元素顺序进行按ldmatrix B矩阵的格式进行排布，达到ldmatrix.trans的效果，在反量化后可直接用到mma中。
// 2. gemm 
//    线程分配：block的数量=sm数量，因为GPU调度器把线程块逐个分配到SM上（一个Block只能在一个SM中，且一个SM可负责多个Block的执行），
//           可以让block和sm一一对应，可以在代码层面上让每个sm都有较明确的任务，间接参与了对sm的调度。
//             每个block的线程数量=256/128，一般会选择256，也就是8个warp，因为每个SM都有4个warp调度器，
//           每个调度器有多于1个warp可以让它有更多的调度空间，也不能设置太多，会减少每个warp可用的寄存器数量。
//             为什么留有128的选项：n和k分块维度要能被n和k整除，如不能，则减小分块维度，同时线程数也对应减少。线程数和分块维度的关系？
//    分块层次：》kernel级别分层：m方向对A和C矩阵进行一个切分，循环中会根据当前需要计算的行数launch一个新的kernel进行计算。
//              （即它会更倾向于小batchsize的情况进行优化，也比较符合w4a16的应用场合，因为batchsize上去之后，会转为计算受限。
//                此时瓶颈将会是反量化和fp16的mma，到时这版marlin kernel的收益也不会很好）
//             》block级别分块：k和n方向的切分根据m的总行数来设定，m较小时一般设为128*128，m较大时一般设为64*256。（这个猜测可能是A矩阵访问时延和最后的规约问题有关，这个我没去细看，感兴趣的可以去分析一下）
//               (三维block如何理解？三个维度表示涉及的gemm block所负责的计算范围，而不是线程分布。)
//             》warp级别分块：warp级别分块为16x16x16, 由两条16x8x16的mma组成。
//    优化点归纳：
//             1）全局内存访问基本以8个fp16(int4)为单位进行，与cp.async指令的最大拷贝数量(16Byte)一致，也满足合并访问要求。
//             2）采用的Stream-K方式进行，提高了数据局部性，也避免了矩阵太大资源不够的情况。
//                (sm数量无法排布完整一个块，因为sm的数量几乎都一个比较奇怪的数字如72/84等，所以硬切会造成sm浪费)
//             3) 在L2上做了一些文章，因为w4a16的重心会放在访存的优化上。而且在kernel级别分层时也限制了M的大小，那么正常B矩阵的访存量会远大于A和C矩阵。
//                大权重矩阵B的读取会使瓶颈。所以为了减轻B矩阵的读取负担，会围绕B矩阵来进行布局，则A会涉及重复读取。
//                然后cp.async异步拷贝指令可以同时进行gmem到L2和L2到L1的拷贝，所以他的思路是让A重复从L2中获取数据，使读取A时延在读取B时延下完成隐藏。
//                (B一次读取，A和C多次读取？需要将A矩阵在L2上进行重复读取，A读取的同时B也要读取，B的读取可能会导致A所需的内容被L2丢弃）
//                (block swizzle提高L2缓存命中率的核心点是让同时执行的多个sm对应的block访问的全局数据尽可能相邻：https://github.com/NVIDIA/cutlass/issues/1017)
//                (B矩阵只会被精准访问一次，提高L2缓存命中率主要为了A和C矩阵的重复访问)
//                (kernel级别分层后，m较小，block按列优先时，sm数量足够覆盖多列，可视为成块，C矩阵块的mn都小。
//                 采用stream-k在k方向进行切分，则k也小，所以涉及AB矩阵也成块，维度也较小，都能在L2中进行复用;
//                为什么不做行优先的分块，因为这样需要K维度切分得很小，才能覆盖m方向，且不一定能刚好覆盖mk的数据(因为block数量已经固定)。分块两个维度都不能要求更高，逻辑更复杂)
//             4) 4级流水线 / 异步拷贝 / 双缓存来进行数据预取，隐藏拷贝和计算的等待延迟.
//                为什么是4级？资源和收益的权衡，profile得出？
//             5）ldmatrix读取的是8x8的矩阵，正常访问会存在bank冲突, 通过异或做swizzle, 达到读写的conflict free。
//                (这个操作是以前在cutlass中提出了，现在被广泛使用了。看transform_a)
//                (row=row, col=row^col; 如[1,0]->[1,1=1^0], [1,1]->[1,0=1^1], [1,2]->[1,3=二进制11=01^10]), [1,3]=[1,2=10=01^11])
//             6）内层调用mma的两层循环，外层是B矩阵n方向的4个16x16，内层是m方向所有数据，每次调用两次mma。
//                反量化在较外层，减少反量化次数。
//                每次选取B矩阵4份tile：因为每个线程负责32个fp16的读取(4次，一次2个frag，一个frag有4个fp16),32x32=4x16x16，warp里每个线程都在工作状态。
//             7) 块间规约：mma输出c矩阵为fp32，则顺势在寄存器上进行fp32的规约，最后以rf->smem(fp32->fp16/重排)->gmem的流程写出去。
//                （为什么不直接从rf到gmem，因为数据不连续，无法合并访问，经过smem重排后，可以合并写入到gmem）
//             
//             优化的核心应该是B矩阵重排内容，他把反量化和mma的数据准备都一步到位全部准备好，
//             另外围绕B矩阵进行布局让其他矩阵的重复访问在B矩阵的访问下隐藏延时。
//             另外就是各层级很精细的布局，里面有还有很多细节, 有时间还是很值得去细扣一下的。
//             （B矩阵的访问线程分布情况？为什么要拉成16x16）
// 
#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {}

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {}

}  // namespace marlin

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               vllm::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp) {
  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 
// 表示计算C[16,8]+A[16,16]*B[16,8]=C[16,8], 类型f32.f16.f16.f32，分别对应C,A,B,C。
// 而a_frag里一个元素是4个16x2的Vec，frag_b里一个元素是2个16x2的Vec, frag_c是4个float的Vec。
// 
//    针对A矩阵而言，共16x16个数据，warp有32个线程，则每个线程需要负责写入8个fp16数据，需要4个32位寄存器。
// t0([0,0], [0,1], [8,0], [8,1], [0,8], [0,9], [8,8], [8,9]), t1([0,2], [0,3], [8,2], [8,3], [0,10], [0,11], [8,10], [8,11])
// t2/t3略，t4([1,0], [1,1], [9,0], [9,1], [1,8], [1,9], [9,8], [9,9])
//    针对B矩阵而言，共16x8个数据，每个线程需要负责写入4个fp16数据，需要2个32位寄存器。
// t0([0,0], [1,0], [8,0], [9,0]),  t1([2,0], [3,0], [10,0], [11,0]), t2 略，t3([6,0], [7,0], [14,0], [15,0])
// t4([0,1], [1,1], [8,1], [9,1])...
// B矩阵的写入是按列优先的。上下分两组写入
//    针对C矩阵而言，共16x8个数据，每个线程需要负责写入4个fp32数据，需要4个32位寄存器。
// t0([0,0], [0,1], [8,0], [8,1]),  t1([0,2], [0,3], [8,2], [8,3]), t2/t3 略，t4([1,0], [1,1], [9,0], [9,1])
//
// A和C矩阵的写入是按行优先的，A矩阵按十字切分4份8x8，一个线程先写左上2个元素，左下2个，右上再右下，共8个数据。
// C矩阵写法与A矩阵一致，但C矩阵列方向少了一半数据为上下2份8*8，先写左上2数据，再写左下2数据，共4个。
// B矩阵按列优先写入，同分上下两块，先写上面的2数据，再写下面2数据，先递增列。
//
// 以t0为例，将会提供A矩阵的[0,0], [0,1], [8,0], [8,1], [0,8], [0,9], [8,8], [8,9]
//                  B矩阵的[0,0], [1,0], [8,0], [9,0]
//                  C矩阵的[0,0], [0,1], [8,0], [8,1]
// A和B的数据并不能直接计算得到C的数据。为什么数据这么排布，原因在于该指令会跟ldmatrix进行搭配使用，ldmatrix读取出来后可以直接进入mma指令。
// 
// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename ScalarType<scalar_t>::FragA& a_frag,
                           const typename ScalarType<scalar_t>::FragB& frag_b,
                           typename ScalarType<scalar_t>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t);
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

// ldmatrix.sync.aligned.m8n8.x4.shared.b16 
//   warp级别的加载矩阵指令，有同步指令和内存对齐的特点，加载的是16位8x8的矩阵，x4表示加载4份，即4个16位8x8的矩阵。
//
//   4个出参为4个32位寄存器（FragA是4个16x2的Vec，即有4个fp32的寄存器）
// 对于其中一个出参而言，一个warp有32个线程，共读取8x8=64个元素，则分到这个线程的数据就刚好2个fp16，
// 即刚好是一个fp32的寄存器，所以当前线程提供一个fp32寄存器用作出参即可。x4则提供4个fp32寄存器。
// x1时: 8行32个线程，t0-t3获得t0提供的输入地址的8个fp16数据，1个线程2个数据；t4-t7获得t1输入地址的8个数据。。。
//       (t0将拿到[0,0][0,1]，t1([0,2][0,3])...t3([0,6][0,7]), t4([1,0][1,1])... t31([7,6][7,7])) 
//       与 mma中行优先的A和C矩阵一个线程读取的8x8子块内的顺序是一样的！所以可以用完ldmatrix后直接进入mma指令。
//       对于B矩阵只需要用 ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
//       需要注意bank冲突，t0读取0号32位数据, 属于bank0, 假设共享内存大小为32x32，则t4访问的也是bank0.
// x4时: t0获得t0提供的输入地址的8个数据，t1获得t1输入地址的8个数据。。。
//       (假设4个8x8矩阵被排列成32x8，t0-t31提供每行首地址) 则t0将拿到a0a1..a7, t1拿到a8-a15, ...每个寄存器的数据顺序同x1，
//       (假设4个8x8矩阵被排列成16x16，t0-t16提供每行首地址，t16-t31提供每行右半边首地址) 则t0将拿到a0a1..a7，t16拿到a8-a15，t1拿到a16-a23了。
//
//   1个入参用于设置读取数据的首地址，因为读取的矩阵数据列是连续的，但行不是连续的，所以需要提供每一行的首地址。
// 这里读取的矩阵是4个8x8，即需要提供4x8=32个首地址，一个warp有32个线程，每个线程都会调用到这个指令，
// 指令约定t0-t31来分别提供这32个首地址，通过每个线程的%4输入
//（x1指令：则约定t0-t7来提供8个首地址；x2则约定t0-t15来提供16个首地址）。
// 
// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <typename scalar_t>
__device__ inline void ldsm4(typename ScalarType<scalar_t>::FragA& frag_a,
                             const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// 如实现 res=(a&b)|c, 则lut=(0xf0 & 0xcc) | 0xaa
// 同理如实现 res=(a|b)&c, 则lut=(0xf0 | 0xcc) & 0xaa
// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// prmt.b32{.mode}  d, a, b, c;
// 详见 dequant<half, vllm::kU8.id()>(int q)
// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t, vllm::ScalarTypeId w_type_id>
__device__ inline typename ScalarType<scalar_t>::FragB dequant(int q);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU4B8.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU4B8.id()>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

// uint4转fp16
// 输入参数q为 i4s = {e7,e5,e3,e1,e6,e4,e2,e0}，奇数位放在高位，偶数位放在低位
// 将4个int4 e3,e1,e2,e0 分别转换成 4个fp16
// 对于 e7,e5,e6,e4 可以将i4s右移8位，得到 {0,0,e7,e5,e3,e1,e6,e4}, 使用0x000f000f和0x00f000f0则可以取出e7,e5,e6,e4
template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU4.id()>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400; // (0 11001 0000000000) (0 11001 0000000000)
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  // (i4s & 0x000f000f) | 0x64006400
  // 取出 e1和e0 并分别和0x6400做或运算。随后还要再分别减去1024，即可完成转换。
  // 假设q未预先按交织排布，即为{e7,e6,e5,e4,e3,e2,e1,e0}, 使用 0x000000ff 取出 e1e0，
  // 因e1和e0两个数据紧挨在一起，无法分别和0x6400做或运算转换e1和e0的fp16，所以必要先是交织列格式才行。
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  // (i4s & 0x00f000f0) | 0x64006400 
  // 取出 e3和e2 并分别和0x6400做或运算。
  // 假设e3为 0110 即为6, 但在原数据上表现为0110 0000, 
  // 进行或运算后的fp16格式为 0 11001 0001100000 => 2^(25-15)*(1+(96/2^10))=1024*1.09375=1120
  // 1120 * 0.0625 - 64 = 6
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  const int SUB = 0x64006400; // (0 11001 0000000000) (0 11001 0000000000) 11001 = 25, 2^(25-15)=2^10=1024
  const int MUL = 0x2c002c00; // (0 01011 0000000000) (0 01011 0000000000) 01011 = 11，2^(11-15)=2^-4=1/16=0.0625
  const int ADD = 0xd400d400; // (1 10101 0000000000) (1 10101 0000000000) 10101 = 21, 2^(21-15)=2^6=64, 符号位为1，即-64
  typename ScalarType<half>::FragB frag_b; // 2个fp16x2, 共64位, 4个fp16
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU4.id()>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&lo),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162*>(&hi),
                      *reinterpret_cast<const nv_bfloat162*>(&MUL),
                      *reinterpret_cast<const nv_bfloat162*>(&ADD));
  return frag_b;
}

//
// Fast Int8ToFp16/Int8ToBf16: Efficiently dequantize 8bit int values to fp16 or
// bf16 Reference:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L53-L85
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L125-L175
//
template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU8B128.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU8B128.id()>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388736.f;
  fp32_intermediates[1] -= 8388736.f;
  fp32_intermediates[2] -= 8388736.f;
  fp32_intermediates[3] -= 8388736.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

template <>
__device__ inline typename ScalarType<half>::FragB
dequant<half, vllm::kU8.id()>(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250; // 0000 0000 0000 0000 0101 0010 0101 0000
  static constexpr uint32_t mask_for_elt_23 = 0x5351; // 0000 0000 0000 0000 0101 0011 0101 0001
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464; // 0110 0100 0110 0100 0110 0100 0110 0100

  // 假设q为4个int8，如1,2,3,4, 即为 (0000 0100) (0000 0011) (0000 0010) (0000 0001)
  // 将b和a拼接在一起，即 start_byte_for_fp16和q，得到
  // {q, start_byte_for_fp16}： 0110 0100 0110 0100 0110 0100 0110 0100   0000 0100 0000 0011 0000 0010 0000 0001
  //                               b7        b6        b5        b4          b3         b2        b1       b0
  // 最后通过mask来选择，首先第一个是0x5250, 
  //  第一个byte是0000，选择b0，为 0000 0001; 第二个byte是0101，选择b5，为 0110 0100;
  //  第三个byte是0010，选择b2，为 0000 0011; 第四个byte是0101，选择b5，为 0110 0100;
  //  最后得到 b5 b2 b5 b0: 0110 0100 0000 0011 0110 0100 0000 0001，
  //  按两个fp16划分: 0 11001 0000000011 和 0 11001 0000000001，分别为2^(25-15)x(1+3/1024) = 1027; 2^(25-15)x(1+1/1024) = 1025
  // 同理第二个mask 0x5351，对应b5 b3 b5 b1: 2^(25-15)x(1+4/1024) = 1028; 2^(25-15)x(1+2/1024) = 1026;
  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400; // (0 11001 0000000000) (0 11001 0000000000)

  // 1027和1025分别减去magic num，即1024，得到2和1。
  // 1028和1026分别减去magic num，得到4和3。完成转换。
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  frag_b[1] = __hsub2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&I8s_TO_F16s_MAGIC_NUM));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16, vllm::kU8.id()>(int q) {
  typename ScalarType<nv_bfloat16>::FragB frag_b;

  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(&frag_b);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);

  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <typename scalar_t>
__device__ inline void scale(typename ScalarType<scalar_t>::FragB& frag_b,
                             typename ScalarType<scalar_t>::FragS& frag_s,
                             int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t>
__device__ inline void sub_zp(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::scalar_t2& frag_zp,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 zp =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Same as above, but for act_order (each K is multiplied individually)
template <typename scalar_t>
__device__ inline void scale4(typename ScalarType<scalar_t>::FragB& frag_b,
                              typename ScalarType<scalar_t>::FragS& frag_s_1,
                              typename ScalarType<scalar_t>::FragS& frag_s_2,
                              typename ScalarType<scalar_t>::FragS& frag_s_3,
                              typename ScalarType<scalar_t>::FragS& frag_s_4,
                              int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s_val_1_2;
  s_val_1_2.x = reinterpret_cast<scalar_t*>(&frag_s_1)[i];
  s_val_1_2.y = reinterpret_cast<scalar_t*>(&frag_s_2)[i];

  scalar_t2 s_val_3_4;
  s_val_3_4.x = reinterpret_cast<scalar_t*>(&frag_s_3)[i];
  s_val_3_4.y = reinterpret_cast<scalar_t*>(&frag_s_4)[i];

  frag_b[0] = __hmul2(frag_b[0], s_val_1_2);
  frag_b[1] = __hmul2(frag_b[1], s_val_3_4);
}

// Given 2 floats multiply by 2 scales (halves)
template <typename scalar_t>
__device__ inline void scale_float(float* c,
                                   typename ScalarType<scalar_t>::FragS& s) {
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], ScalarType<scalar_t>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], ScalarType<scalar_t>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(int4 const* __restrict__ a_int4_ptr,
                                    int const* __restrict__ perm_int_ptr,
                                    int4* __restrict__ out_int4_ptr, int size_m,
                                    int size_k, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        int cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor), 因为zero points也是int4，原n表示的是int4的元素个数，需要除以8才是int32的个数
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  using FragA = typename ScalarType<scalar_t>::FragA;   // 4个fp16x2
  using FragB = typename ScalarType<scalar_t>::FragB;   // 2个fp16x2
  using FragC = typename ScalarType<scalar_t>::FragC;   // 4个fp32
  using FragS = typename ScalarType<scalar_t>::FragS;   // 1个fp16x2
  using FragZP = typename ScalarType<scalar_t>::FragZP; // 4个fp16x2

  static constexpr auto w_type = vllm::ScalarType::from_id(w_type_id); // 权重类型，这里是int4

  constexpr int pack_factor = 32 / w_type.size_bits(); // 4bit，pack_factor=8，用于int4和int32的索引下标转换

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  // 重新把并行数算出来，prob_m改回大分块的维度，进行计算
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks); // 这里的并行数，即是m方向切分后，进入该kernel的矩阵的基础上，m方向的大分块的数量
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks; // k方向 大分块的数量 即block tile
  int n_tiles = prob_n / 16 / thread_n_blocks; // n方向 大分块的数量
  // 围绕B矩阵，每个block需要处理大分块的数量：gridDim.x即是block的数量，所有方向大分块block tile的数量乘积 除以 block数量，向上取整
  int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  // 使用 % 和 / 的组合，将一维的blockIdx.x转变为二维索引, %用于行索引，则表示列优先，遍历完一列再下一列
  // * 假设iters为1，围绕k_tiles=5进行排布，blockIdx.x[slice_row, slice_col_par]
  //   则有 0[0,0], 1[1,0], 2[2,0], 3[3,0], 4[4,0], 
  //        5[0,1], 6[1,1], 7[2,1]...
  // * 如果1个block处理两个大块，即iters为2, x[2x%5,2x/5]
  //   则有 0[0,0], 1[2,0], 2[4,0], 3[1,1], 4[3,1]
  // * 如果1个block处理三个大块，即iters为3, x[3x%5,3x/5]
  //   则有 0[0,0], 1[3,0], 2[1,1]
  // 即会按k_tiles为一列的行数，按顺序分配给各个block指定二维首地址，
  // 如iters为3，则0行的[0,0][1,0][2,0]都归0号，[3,0][4,0][0,1]都归1号，
  // 而都归一个block的大块(threadblock tiles)集合算是一个slice，如[0,0][1,0][2,0]
  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;  // number of threadblock tiles in the current slice
  int slice_count =
      0;          // total number of active threadblocks in the current slice
  int slice_idx;  // index of threadblock in current slice; numbered bottom to
                  // top

  int par_id = 0;

  // n_tiles是n方向block tile的数量，slice_col_par是n方向block tile的下标。
  // 如果列数大于等于n_tiles，即表示超出范围了？slice_col_par最大应该是最后一个block负责的第一个block tile，应比n_tiles小？
  // 暂时忽略这里
  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles; // 因为iters是向上取整得到的，所以有可能会大于n_tiles，取余得到剩余的部分。
    par_id = slice_col_par / n_tiles;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
      par_id++;
    }
  };
  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  // a矩阵gmem的下标索引：将fp16改为int4索引。A/C/scale都是fp16，需要除以8来转INT4。B/zeropoint是int4，需要用32来转INT4
  // a_gl_stride 为 A矩阵一行总数
  int a_gl_stride = prob_k / 8;
  // stride of an A matrix tile in shared memory，
  // a矩阵smem的下标索引，16*thread_k_blocks是大块数据量，除以8是将fp16下标转为INT4，
  // 即a_sh_stride 为一个大块的一行INT4总数
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // between shared memory tile reads
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides，prob_n是B矩阵重排之前的n维度的int4总数量，重排后小块的16x16被拉平了，所以需要多乘一个16
  int b_gl_stride = 16 * prob_n / (pack_factor * 4); // 用(pack_factor * 4)来将int4转为INT4，b_gl_stride为一行INT4的总数
  // 等同于marlin kernel中的 32 * thread_n_blocks / 4，32=16*16/8, 
  // 8是pack_factor, 除以8表示将地址从fp16转为INT4(4*32=128bit=8*16)，因为B矩阵是int4，需要再除以4, 即从int4转为INT4，需要除以32.
  // thread_n_blocks是一个block所负责的n方向的小tile数量，
  // thread_n_blocks * 16中的16，表示一个block负责的n方向的元素总个数。因为小分块是16x16
  // 后面一个乘16，表示b矩阵的16x16被拉成了1x(16x16)
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;  // 大分块n方向一行的总个数
  constexpr int b_thread_vecs = w_type.size_bits() == 4 ? 1 : 2;                // int4时，就是等于1.
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;              // int4，即 b_sh_stride_threads 等同于 b_sh_stride

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides without act_order，8是pack_factor，因为scale是fp16，需要除以8来把fp16转为INT4
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8; // n方向一行的总数跨度
  constexpr int s_tb_groups =
      !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;
  int tb_n_warps = thread_n_blocks / 4;
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides，int4转INT4下标，zero point是int4，需要除以8再除以4来转成INT4
  int zp_gl_stride = (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = ((16 * thread_n_blocks) / pack_factor) / 4; // n方向一行的总数跨度
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) +
                (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x * b_thread_vecs;
  int b_sh_rd = threadIdx.x * b_thread_vecs;

  // For act_order
  constexpr int k_iter_size = tb_k / b_sh_wr_iters;
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

  // No act_order
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
    } else {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
    }
  }
  int s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    }
  }
  int zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = threadIdx.x < zp_sh_stride;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) % 4;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    zp_sh_rd = num_ints_per_thread * num_col_threads *
                   ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
               num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
  }

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  // a_sh_wr_trans和a_sh_rd_trans存储的都是共享内存里A矩阵的转换后的偏移量。
  // a_sh_wr_trans用于从gmem到smem
  // a_sh_rd_trans用于从smem到rf
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // 因为B矩阵的访问并不是固定步长的，所以需要在计算过程中计算。
  // 这里通过维护多个指针，来去掉访问一个tile时，多次连续访问之间的依赖关系。
  // 即一个tile的读取，每行其实是分布在多个不同位置的，没必要访问完第一行再计算下一行的位置继续访问。
  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a = sh;                           // [stages * a_sh_stage]
  int4* sh_b = sh_a + (stages * a_sh_stage); // [stages * b_sh_stage] = [4*]
  int4* sh_g_idx = sh_b + (stages * b_sh_stage);
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];                    // No act-order
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;
  constexpr int sh_max_num_groups = 32;

  auto fetch_scales_to_shared = [&](bool is_async, int first_group_id,
                                    int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups < sh_max_num_groups) {
      sh_num_groups = sh_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(&sh_s[(i * s_sh_stride) + threadIdx.x],
                         &scales_ptr[row_offset + (i * s_gl_stride) +
                                     slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset +
                         threadIdx.x];
        }
      }
    }
  };
  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j);
        }

        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (has_act_order) {
        // Fetch g_idx thread-block portion
        int full_pipe = a_off;
        int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
        if (cur_k < prob_k && cur_k < slice_k_finish) {
          int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

          int4 const* cur_g_idx_stage_ptr =
              reinterpret_cast<int4 const*>(&g_idx[cur_k]);

          if (threadIdx.x < g_idx_stage) {
            cp_async4_pred(&sh_g_idx_stage[threadIdx.x],
                           &cur_g_idx_stage_ptr[threadIdx.x]);
          }
        }
      } else {
        if constexpr (group_blocks != -1) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
            // Only fetch scales if this tile starts a new group
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < s_tb_groups; i++) {
              if (s_sh_wr_pred) {
                cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr],
                          &scales_ptr[s_gl_rd]);
              }
              s_gl_rd += s_gl_rd_delta;
            }
          }
        }

        if constexpr (has_zp && group_blocks != -1) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          if constexpr (group_blocks >= thread_k_blocks) {
            // Only fetch zero-points if this tile starts a new group
            if (pipe % (group_blocks / thread_k_blocks) == 0) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          } else {
            for (int i = 0; i < zp_tb_groups; i++) {
              if (zp_sh_wr_pred) {
                cp_async4(&sh_zp_stage[i * zp_sh_stride + zp_sh_wr],
                          &zp_ptr[zp_gl_rd]);
              }
              zp_gl_rd += zp_gl_rd_delta;
            }
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) // A矩阵smem需要读取整个thread_m_blocks维度的内容，即会重复读取。
      ldsm4<scalar_t>(frag_a[k % 2][i],
                      &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) { // B矩阵为int4时，b_thread_vecs就是1。
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      is_same_group[pipe] = false;
      same_group_id[pipe] = 0;
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (!has_act_order) {
      // No act-order case
      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          int4* sh_s_stage =
              sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                                   (pipe / (group_blocks / thread_k_blocks)));
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          int warp_id = threadIdx.x / 32;     // warp下标，256个线程有8个warp，范围是0-7. 
          // n方向warp数量，thread_n_blocks是16x16的块在n方向的数量，256线程下，会有(thread_n_blocks，thread_k_blocks) = (16,4) / (8,8)
          // 如果是(8, 8)， 则 n_warps=2=8/4；
          // 如果是(16, 4)，则 n_warps=4=16/4；
          int n_warps = thread_n_blocks / 4; N方向每个warp处理4个16的tile。n_warps就是n方向需要多少个warp
          // warp_row(warp_id/n_warps), 如(8,8)时，0(0/2),0(1/2),1(2/2),1(3/2),3(7/2). 为[4,2] 布局
          //                           如(16,8)时，0(0/4),0(1/4),0(2/4),0(3/4),1(7/4). 为[2,4] 布局
          int warp_row = warp_id / n_warps; 

          int cur_k = warp_row * 16;          // 块为16x16，乘以16换算为k的实际下标。
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
              sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
        }
      }

      return;
    }

    // Act-order case

    // Determine K of the "current" thread-block
    int cur_k = slice_k_start + tb_k * full_pipe;
    if (cur_k >= prob_k || cur_k >= slice_k_finish) {
      return;
    }

    // Reset (to current thread-block) since we read g_idx portion from the
    // shared memory
    cur_k = 0;

    // Progress to current iteration
    cur_k += k_iter_size * (k % b_sh_wr_iters);

    // Determine "position" inside the thread-block (based on warp and
    // thread-id)
    int warp_id = threadIdx.x / 32;
    int n_warps =
        thread_n_blocks / 4;  // Each warp processes 4 16-size tiles over N

    int warp_row = warp_id / n_warps;
    int warp_col = warp_id % n_warps;

    cur_k += warp_row * 16;

    int th_id = threadIdx.x % 32;
    cur_k += (th_id % 4) * 2;  // Due to tensor-core layout for fp16 B matrix

    int s_col_shift =
        /*slice_n_offset +*/ (act_s_col_warp_stride * warp_col) +
        (th_id / 4) * act_s_col_stride;

    if (is_same_group[pipe]) {
      if (k % 2 == 0) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride +
                 s_col_shift];
      } else {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
      }

      for (int i = 1; i < 4; i++) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
      }
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    constexpr int k_frag_offsets[4] = {0, 1, 8,
                                       9};  // Tensor core offsets per thread

  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int actual_k = cur_k + k_frag_offsets[i];

      int group_id = sh_g_idx_int_ptr[actual_k];
      int rel_group_id = group_id - sh_first_group_id;

      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
          sh_s[rel_group_id * s_sh_stride + s_col_shift];
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    // This code does not handle group_blocks == 0,
    // which signifies act_order.
    // has_zp implies AWQ, which doesn't have act_order,
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
        }

      } else if constexpr (group_blocks >= thread_k_blocks) {
        int4* sh_zp_stage =
            sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) *
                                   (pipe / (group_blocks / thread_k_blocks)));
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      } else {
        int warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks = cur_k / 16;
        int cur_group_id = 0;

        // Suppress bogus and persistent divide-by-zero warning
  #pragma nv_diagnostic push
  #pragma nv_diag_suppress divide_by_zero
        cur_group_id = k_blocks / group_blocks;
  #pragma nv_diagnostic pop

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
    if constexpr (has_zp) {
      FragB frag_zp_0;
      FragB frag_zp_1;
      int zp_quant_0, zp_quant_1;

      if constexpr (w_type.size_bits() == 4) {
        zp_quant_0 = frag_qzp[k % 2][0];
        zp_quant_1 = zp_quant_0 >> 8;
      } else {
        static_assert(w_type.size_bits() == 8);
        zp_quant_0 = frag_qzp[k % 2][0];
        zp_quant_1 = frag_qzp[k % 2][1];
      }

      frag_zp_0 = dequant<scalar_t, w_type_id>(zp_quant_0);
      frag_zp_1 = dequant<scalar_t, w_type_id>(zp_quant_1);

      frag_zp[0] = frag_zp_0[0];
      frag_zp[1] = frag_zp_0[1];
      frag_zp[2] = frag_zp_1[0];
      frag_zp[3] = frag_zp_1[1];
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  // b矩阵一个线程for循环4次，每次读取2份frag，一份frag有4个fp16，即每个线程负责32个fp16，32个线程共1024=32(warp)x32=4x16x16个fp16
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (w_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k % 2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(w_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k % 2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      frag_b0 = dequant<scalar_t, w_type_id>(b_quant_0);
      frag_b1 = dequant<scalar_t, w_type_id>(b_quant_1);

      // Apply zero-point to frag_b0
      if constexpr (has_zp) {
        sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
      }

      // Apply scale to frag_b0
      if constexpr (has_act_order) {
        scale4<scalar_t>(frag_b0, act_frag_s[k % 2][0][j],
                         act_frag_s[k % 2][1][j], act_frag_s[k % 2][2][j],
                         act_frag_s[k % 2][3][j], 0);
      } else {
        if constexpr (group_blocks != -1) {
          scale<scalar_t>(frag_b0, frag_s[k % 2][j], 0);
        }
      }

      // Apply zero-point to frag_b1
      if constexpr (has_zp) {
        sub_zp<scalar_t>(frag_b1, frag_zp[j], 1);
      }

      // Apply scale to frag_b1
      if constexpr (has_act_order) {
        scale4<scalar_t>(frag_b1, act_frag_s[k % 2][0][j],
                         act_frag_s[k % 2][1][j], act_frag_s[k % 2][2][j],
                         act_frag_s[k % 2][3][j], 1);

      } else {
        if constexpr (group_blocks != -1) {
          scale<scalar_t>(frag_b1, frag_s[k % 2][j], 1);
        }
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<scalar_t>(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma<scalar_t>(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    // b_sh_stride_threads 是B矩阵大分块n方向一行的INT4总数，threads是block内线程总数, 
    ///？ b_sh_stride_threads实际对应有B矩阵的 b_sh_stride_threads*32个int4数据，因为重排多乘了16，应对C输出的n方向数据维度是 b_sh_stride_threads*2个int4.
    ///？ 按个数算对应b_sh_stride_threads*2个fp32的规约，再换算成INT4做下标，对应 b_sh_stride_threads*2/4？
    // 由线程布局来看，threads会是大块n长度的一倍或两倍，b_sh_stride_threads是按INT4算，有32倍差距，且多乘了16。
    // 所以threads / b_sh_stride_threads会是2或4. 即表示每个b_sh_stride_threads会使用多少个线程进行规约，一次规约的单位是一个INT4，即4个fp32。
    // red_off是按INT4作为跨度索引，也是每轮轮次各线程取数据的基本偏移量。每个线程一次处理的数据量固定，则线程数越多，跨度越大。其可折半次数也是需要规约的轮次。
    // 除以2，是因为red_off是偏移量的起始点，需要折半。
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      // 线程分组，b_sh_stride_threads个线程为一组。
      // red_idx用于限制需要工作的线程，在多轮规约中，每次数据量减少，需要工作的线程数量也会减半。活跃线程组减半，看下面的i。
      int red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2; // 一次是4x2个INT4，即4x2x4个fp32，再乘以b_sh_stride_threads，得到新的跨度值。
      constexpr int red_sh_delta = b_sh_stride_threads; // 实际一行的数量
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) { // 遍历m维度上的小块总数，即一个线程会访问m方向所有子块。则下面逻辑均对应m维度上的1行block。
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          // 限制活跃线程，随着规约轮次的推进，需要参与归于线程会越来越少。
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            // 一个线程处理4*2*4=32个数据，对应一次 matmul 的数据量（FragC frag_c[thread_m_blocks][4][2]，一次matmul得到4x2个fragC）
            // 则一轮规约计算，一个block内会处理 threads*32个fp32数据，
            for (int j = 0; j < 4 * 2; j++) {

              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd =
                    reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++) // 内层j和k合计 4x2x4个数据，与 matmul 中的4x2x4相对应，一个warp 32个线程负责合计处理32x32个数据，即4份16x16的数据。
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j]; // 一次4个fp32，首次进入无数据，需要提前填充
            }
          }
          // 到这里block内每个warp内每个线程负责的数据量已完成一轮规约。
          // 在下一轮规约时，跨度减半，涉及到当前轮次的跨warp的结果，需要做一次同步。
          __syncthreads();
        }
        // 
        // 省下最后一组，上面最后一次规约的frag_c写到了sh[red_sh_wr]的数据
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd =
                reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                    4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
  // Interestingly, doing direct global accesses here really seems to mess up
  // the compiler and lead to slowdowns, hence we also use async-copies even
  // though these fetches are not actually asynchronous.
  #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
              &sh[c_sh_wr + c_sh_wr_delta * i],
              &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                 c_gl_wr_delta_i * (i % 2)],
              i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
  #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(
                  &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                  Dtype::num2float(reinterpret_cast<scalar_t*>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
  #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<scalar_t*>(&c)[j] =
                  Dtype::float2num(reinterpret_cast<float*>(
                      &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
        }
      }
    }
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    
    // 一个warp处理n方向4份小tile(m方向是单独遍历的)，一个线程则处理连续的4个数据，即一个大int4。
    // 一个warp有32个线程。
    // 之前的thread_block_reduce将block内的结果都汇总到了该block的一个warp内，
    // 先需要将每个block的一个warp的数据使用全局内存进行汇总。
    // 能到这里的线程是block的线程总数256，
    constexpr int active_threads = 32 * thread_n_blocks / 4; 
    bool is_th_active = threadIdx.x < active_threads;

    int par_offset = c_size * n_tiles * par_id;
    int slice_offset = c_size * slice_col;

    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = par_offset + slice_offset;

    if (!is_th_active) {
      return;
    }

    if (!first) { // 如果是第一个slice，则不需要跨slice的规约。
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        sh[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x]; // 一次一个大int4即4个fp32。

        float* sh_c_ptr = reinterpret_cast<float*>(&sh[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {  // 如果是最后一个slice，则不会再写入C_tmp，而是会进入到write_result函数中，直接写到C里。
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k++) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr =
        (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s) {
      scalar_t2 res =
          Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (!has_act_order && group_blocks == -1 &&
                    w_type.size_bits() == 4) {
        res = __hmul2(res, s[0]);
      }

      ((scalar_t2*)sh)[idx] = res;
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if (has_act_order && i == 0) {
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        fetch_scales_to_shared(true, g_idx[slice_k_start], g_idx[last_g_idx]);
      }

      if constexpr (has_zp && group_blocks == -1) {
        if (i == 0) {
          fetch_zp_to_shared();
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    slice_k_start_shared_fetch += tb_k * (stages - 1);
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.
    // 循环展开global fetch和寄存器加载pipeline，以确保所有共享内存访问都是静态的。
    // 请注意，这两个pipeline的长度是相等的，这意味着下一次迭代将始终从索引0开始。
  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
          init_same_group(pipe % stages);
        }
        matmul(k);
      }

      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
      // matmul中fragC不涉及for循环的k，即算到这里fragC已经完成了b_sh_wr_iters的累加规约。
    }

    a_gl_rd += a_gl_rd_delta_o * stages;
    slice_k_start += tb_k * stages;
    slice_k_start_shared_fetch += tb_k * stages;

    if constexpr (has_act_order) {
      int first_group_id = g_idx[slice_k_start];
      int last_g_idx = slice_k_start + stages * tb_k * 2;
      if (last_g_idx >= prob_k) {
        last_g_idx = prob_k - 1;
      }
      int last_group_id = g_idx[last_g_idx];
      if (last_group_id >= sh_first_group_id + sh_num_groups) {
        fetch_scales_to_shared(false, first_group_id, last_group_id);
        __syncthreads();
      }
    }

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (!has_act_order && group_blocks == -1) {
        if constexpr (w_type.size_bits() == 8) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        } else {
          if (last) {
            if (s_sh_wr_pred) {
              cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
            }
            cp_async_fence();
          }
        }
      }

      thread_block_reduce();
      if constexpr (!has_act_order && group_blocks == -1) {
        if constexpr (w_type.size_bits() == 8) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
          }

        } else {
          if (last) {
            cp_async_wait<0>();
            __syncthreads();
            if (threadIdx.x / 32 < thread_n_blocks / 4) {
              reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
              reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (!has_act_order && group_blocks == -1 &&
                    w_type.size_bits() == 8) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + 0]);

              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                  frag_s[j / 2][2 * (j % 2) + 1]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                  frag_s[j / 2][2 * (j % 2) + 1]);
            }
          }
        }
      }

      if (slice_count > 1) {  // only globally reduce if there is more than one
                              // block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[slice_col], last);
      }
      if (last)  // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
  #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
  #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
        }

        // Update slice k/n for scales loading
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;

        } else {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        }

        start_pipes();
      }
    }
  }
}

  #define __CALL_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                    HAS_ACT_ORDER, HAS_ZP, GROUP_BLOCKS, NUM_THREADS)          \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS &&         \
             thread_n_blocks == THREAD_N_BLOCKS &&                             \
             thread_k_blocks == THREAD_K_BLOCKS &&                             \
             has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&             \
             group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {     \
      cudaFuncSetAttribute(                                                    \
          Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS,          \
                 THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER, \
                 HAS_ZP, GROUP_BLOCKS>,                                        \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);        \
      Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS,              \
             THREAD_N_BLOCKS, THREAD_K_BLOCKS, pipe_stages, HAS_ACT_ORDER,     \
             HAS_ZP, GROUP_BLOCKS>                                             \
          <<<blocks, NUM_THREADS, max_shared_mem, stream>>>(                   \
              A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, zp_ptr, g_idx_ptr,        \
              num_groups, prob_m, prob_n, prob_k, locks, use_fp32_reduce);     \
    }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

typedef struct {
  int max_m_blocks;
  thread_config_t tb_cfg;
} exec_config_t;

// 优先采用256线程一个block，因为256个线程有8个warp。
// （费米架构只有2个？）volta/安倍架构，每个SM都有4个warp调度器，每个调度器里有超过1个warp可以隐藏更多的延迟。
// 但warp又不能太多，希望相对较少的warp，使每个warp和小分块上有更多寄存器可用
// m小的情况下选[128,128]; 
// m大时选[64,256]，总数都是16384 = 64*256 = 128*128
thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority
    // For small batchizes, better partioning is slightly more important than better compute utilization

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},

};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * pipe_stages * 2;     // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;

  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

bool is_valid_cache_size(thread_config_t const& th_config, int max_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int scales_cache_size, int max_shared_mem) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;

  int b_size = (tb_k * tb_n / pack_factor) * 4;

  // Get A size
  int m_blocks = div_ceil(prob_m, 16);
  int tb_max_m = 16;

  while (true) {
    if (m_blocks >= max_m_blocks) {
      tb_max_m *= max_m_blocks;
      break;
    }

    max_m_blocks--;
    if (max_m_blocks == 0) {
      TORCH_CHECK(false, "Unexpected m_blocks = ", m_blocks);
    }
  }

  int a_size = (tb_max_m * tb_k) * 2;

  float pipe_size = (a_size + b_size) * pipe_stages;

  TORCH_CHECK(max_shared_mem / 2 > scales_cache_size);  // Sanity

  return pipe_size < 0.95f * (max_shared_mem - scales_cache_size);
}

bool is_valid_config(thread_config_t const& th_config, int max_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  //  Determine cache for scales
  int scales_cache_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full);

  // Check that pipeline fits into cache
  if (!is_valid_cache_size(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                           num_bits, scales_cache_size, max_shared_mem)) {
    return false;
  }

  return true;
}

int determine_reduce_max_m(int prob_m, int max_par) {
  constexpr int tile_m_size = 16;

  if (prob_m <= tile_m_size) {
    return tile_m_size;

  } else if (prob_m <= tile_m_size * 2) {
    return tile_m_size * 2;

  } else if (prob_m <= tile_m_size * 3) {
    return tile_m_size * 3;

  } else if (prob_m <= tile_m_size * 4) {
    return tile_m_size * 4;

  } else {
    int cur_par = min(div_ceil(prob_m, tile_m_size * 4), max_par);
    return tile_m_size * 4 * cur_par;
  }
}

exec_config_t determine_thread_config(int prob_m, int prob_n, int prob_k,
                                      int num_bits, int group_size,
                                      bool has_act_order, bool is_k_full,
                                      int max_shared_mem) {
  int max_m_blocks = 4;
  while (max_m_blocks > 0) {
    // prob_m是A矩阵的行数
    if (prob_m <= 16) {
      for (auto th_config : small_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    } else {
      for (auto th_config : large_batch_thread_configs) {
        if (is_valid_config(th_config, max_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            max_shared_mem)) {
          return exec_config_t{max_m_blocks, th_config};
        }
      }
    }

    max_m_blocks--;  // Process less M blocks per invocation to reduce cache
                     // usage
  }

  return exec_config_t{0, {-1, -1, -1}};
}

  #define GPTQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
                                                                            \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

  #define AWQ_CALL_IF(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
    __CALL_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

template <typename scalar_t>
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* s,
               void* zp, void* g_idx, void* perm, void* a_tmp, int prob_m,
               int prob_n, int prob_k, void* workspace,
               vllm::ScalarType const& q_type, bool has_act_order,
               bool is_k_full, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k, int thread_n,
               int sms, int max_par, bool use_fp32_reduce) {
  if (has_zp) {
    TORCH_CHECK(
        q_type == vllm::kU4 || q_type == vllm::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    TORCH_CHECK(
        q_type == vllm::kU4B8 || q_type == vllm::kU8B128,
        "q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        q_type.str());
  }

  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  // TODO: remove alias when we start supporting other 8bit types
  int num_bits = q_type.size_bits();
  int tot_m = prob_m;
  int tot_m_blocks = div_ceil(tot_m, 16); // 除以16，向上取整，表示为m方向的小tile数量，小tile为16x16.
  int pad = 16 * tot_m_blocks - tot_m;    // 无法凑整的部分补0

  if (sms == -1) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); // 获取sm个数，一个sm分配一个block
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    exec_cfg =
        exec_config_t{4, thread_config_t{thread_k, thread_n, default_threads}};
  } else {
    // Auto config
    exec_cfg =
        determine_thread_config(prob_m, prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, max_shared_mem);
  }

  TORCH_CHECK(exec_cfg.max_m_blocks > 0 &&
                  is_valid_config(exec_cfg.tb_cfg, exec_cfg.max_m_blocks,
                                  prob_m, prob_n, prob_k, num_bits, group_size,
                                  has_act_order, is_k_full, max_shared_mem),
              "Invalid thread config: max_m_blocks = ", exec_cfg.max_m_blocks,
              ", thread_k = ", exec_cfg.tb_cfg.thread_k,
              ", thread_n = ", exec_cfg.tb_cfg.thread_n,
              ", num_threads = ", exec_cfg.tb_cfg.num_threads, " for MKN = [",
              prob_m, ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
              ", group_size = ", group_size,
              ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
              ", max_shared_mem = ", max_shared_mem);

  int num_threads = exec_cfg.tb_cfg.num_threads; // 一个block的线程数量，是256或128，对应8个/4个warp
  thread_k = exec_cfg.tb_cfg.thread_k;
  thread_n = exec_cfg.tb_cfg.thread_n;

  // 线程布局是<<<sm, num_thread>>>, 跟 thread_k 和 thread_n 无关。
  // thread_k和thread_n表示一个大分块的大小，（下面的for循环，每次处理一块？）
  // 小分块是16x16，则thread_k_blocks和thread_n_blocks表示小分块的个数。
  int thread_k_blocks = thread_k / 16; 
  int thread_n_blocks = thread_n / 16; 
 
  // block数量设为与sm数量一致，本质上计算时就是一个sm一次只能调度一个block。
  // 直接使用逻辑，合理的给这些block分配任务，而不必要增加无法并行的block。
  int blocks = sms; 

  TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
              " is not divisible by thread_n = ", thread_n);
  TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
              " is not divisible by thread_k = ", thread_k);

  int group_blocks = 0;
  if (has_act_order) { // awq False
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }

  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  const int4* A_ptr = (const int4*)A; // fp16
  const int4* B_ptr = (const int4*)B; // int4
  int4* C_ptr = (int4*)C;             // fp16
  int4* C_tmp_ptr = (int4*)C_tmp;     // fp32
  const int4* s_ptr = (const int4*)s; // fp16
  const int4* zp_ptr = (const int4*)zp;  // int4
  const int* g_idx_ptr = (const int*)g_idx; // swq 无
  const int* perm_ptr = (const int*)perm;   // awq 无
  int4* a_tmp_ptr = (int4*)a_tmp;        // fp16

  int* locks = (int*)workspace;

  // awq里为False
  if (has_act_order) { 
    // Permute A columns
    int block_rows = div_ceil(prob_m, blocks);
    permute_cols_kernel<<<blocks, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, block_rows);
    A_ptr = a_tmp_ptr;
  }

  // If we have a full K, then we can run the non-act-order version of Marlin
  // (since the weight rows are reordered by increasing group ids, and by having
  // a full K, we have full original groups)
  if (is_k_full) {
    has_act_order = false;
  }

  // Main loop
  for (int i = 0; i < tot_m_blocks; i += exec_cfg.max_m_blocks) {
    int thread_m_blocks = tot_m_blocks - i;  // 剩余的tile数量 = m方向的总tile[16,16]个数 - 当前到第i的tile
    prob_m = tot_m - 16 * i;                 // 剩余行数 = 总行数 - 16x当前tile数
    int par = 1;                             // 先令并行度为1
    if (thread_m_blocks > exec_cfg.max_m_blocks) {   // max_m_blocks 是按资源情况和布局计算过的一次处理最大的分块数量
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / (16 * exec_cfg.max_m_blocks); // 看按最大的max_m_blocks，可以并行执行多少份
      if (par > max_par) par = max_par;     // 限制最大并行数量
      prob_m = (16 * exec_cfg.max_m_blocks) * par; // 按新的并行数量调整当前轮次需要处理的prob_m行
      i += exec_cfg.max_m_blocks * (par - 1);
      thread_m_blocks = exec_cfg.max_m_blocks;
    }

    if (false) {
    }
    GPTQ_CALL_IF(vllm::kU4B8, 16, 4, 256)
    GPTQ_CALL_IF(vllm::kU4B8, 8, 8, 256)
    GPTQ_CALL_IF(vllm::kU4B8, 8, 4, 128)
    GPTQ_CALL_IF(vllm::kU4B8, 4, 8, 128)
    GPTQ_CALL_IF(vllm::kU8B128, 16, 4, 256)
    GPTQ_CALL_IF(vllm::kU8B128, 8, 8, 256)
    GPTQ_CALL_IF(vllm::kU8B128, 8, 4, 128)
    GPTQ_CALL_IF(vllm::kU8B128, 4, 8, 128)

    AWQ_CALL_IF(vllm::kU4, 16, 4, 256)
    AWQ_CALL_IF(vllm::kU4, 8, 8, 256)
    AWQ_CALL_IF(vllm::kU4, 8, 4, 128)
    AWQ_CALL_IF(vllm::kU4, 4, 8, 128)
    AWQ_CALL_IF(vllm::kU8, 16, 4, 256)
    AWQ_CALL_IF(vllm::kU8, 8, 8, 256)
    AWQ_CALL_IF(vllm::kU8, 8, 4, 128)
    AWQ_CALL_IF(vllm::kU8, 4, 8, 128)
    else {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_bits = ", num_bits);
    }

    // 按行跳跃，A_ptr指向某行首地址
    // thread_m_blocks为一个16x16的tile在m方向的个数，16*thread_m_blocks转换为元素个数，par为并行数，如2则直接处理两份；
    // 16 * thread_m_blocks * par得到行号；prob_k为一行元素个数，A_ptr的元素为fp16，prob_k表示有这么多个fp16，
    // 而指针为int4*，注意这里的int4并不是4bit的int，而是4个int，即8个fp16，所以基于int4*去索引，需要除以8.
    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par; 
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par; 
  }
}

}  // namespace marlin

// gptq_marlin_gemm / marlin_mm / Marlin
// 
// A：fp16的激活值[m,k]
// B：int4量化后的权重[k,n]
// C: fp16的输出[m,n]
// max_par: 并行计算64行A矩阵的最大并行数, 
// workspace: model_executor\layers\quantization\gptq_marlin.py#281
//   用来做global_reduce的标志位，shape为[n/GPTQ_MARLIN_MIN_THREAD_N(64) * GPTQ_MARLIN_MAX_PARALLEL(16)]
//   按n维度切分，N上最少线程数是64，则最多切分为N/64，再乘以最大并行数，即最多有N/64*16个slice，每个slice需要一个标志位。
// b_scales[k/groupsize，n]: fp16量化的scale值。b_scales.size(0)是num_groups; b_scales.size(1)等于size_n，group_size=size_k/num_groups;
// b_zeros[(k/groupsize),(n/pack_factor)]，has_zp: 
//                 4bit的zero-points。 在awq上has_zp为True，即有自己的零点；
//                 gptq上has_zp为False，即零点被固定，权重为int4时，零点固定为8，int8时为128？
// pack_factor: 8 = 32 / b_q_type->size_bits()，用于int32和int4之间的下标索引转换。1个int32=8个int4，用int32指针索引时，下标需要除以8.
//
// 以Qwen2-1.5B-Instruct-AWQ为例，查看config.json可看到其group_size为128，继而打印其safetensors，可看到如下维度信息。
// B[8960,192]的int32，其实际类型是int4，所以实际维度是B[8960, 1536=192*8]; 
// 同样qzeros也是int4用int32来存储，其维度为[70, 1536=192*8]，其中行是70*group_size(128)=权重的size_k(8960); scales与qzeros元素个数是一样的。
//
// model.layers.9.mlp.down_proj.qweight torch.Size([8960, 192]) torch.int32
// model.layers.9.mlp.down_proj.qzeros torch.Size([70, 192]) torch.int32
// model.layers.9.mlp.down_proj.scales torch.Size([70, 1536]) torch.float16    size_k = 1536, 1536 / 8 = 192
//
// is_k_full: 在awq中被写死为True，在gptq中为可选。(vllm\model_executor\layers\quantization\utils\marlin_utils.py#300)
//            为True时，表示权重矩阵B的行没有被排序。
// g_idx / perm：Handle sorting for activation reordering if needed. (model_executor\layers\quantization\gptq_marlin.py#286)
//               在awq_marlin的调用中为空，不做考虑 (vllm\model_executor\layers\quantization\awq_marlin.py#266)
// use_fp32_reduce: 默认为True, 表示使用fp32的矩阵对最后完成分块计算时，对各分块结果进行规约。

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& b_zeros,
                               torch::Tensor& g_idx, torch::Tensor& perm,
                               torch::Tensor& workspace,
                               vllm::ScalarTypeTorchPtr const& b_q_type,
                               int64_t size_m, int64_t size_n, int64_t size_k,
                               bool is_k_full, bool has_zp,
                               bool use_fp32_reduce) {
  if (has_zp) {
    TORCH_CHECK(*b_q_type == vllm::kU4 || *b_q_type == vllm::kU8,
                "b_q_type must be u4 or u8 when has_zp = True. Got = ",
                b_q_type->str());
  } else {
    TORCH_CHECK(
        *b_q_type == vllm::kU4B8 || *b_q_type == vllm::kU8B128,
        "b_q_type must be uint4b8 or uint8b128 when has_zp = False. Got = ",
        b_q_type->str());
  }

  int pack_factor = 32 / b_q_type->size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(size_k % marlin::tile_size == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", marlin::tile_size);
  TORCH_CHECK((size_k / marlin::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", marlin::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % marlin::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", marlin::tile_size);
  int actual_size_n = (b_q_weight.size(1) / marlin::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
  TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");

  TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
  TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);
  torch::Tensor a_tmp = torch::empty({size_m, size_k}, options);

  // 用于最后的分块结果规约
  // Alloc C tmp buffer that is going to be used for the global reduce
  int reduce_max_m = marlin::determine_reduce_max_m(size_m, marlin::max_par);
  int reduce_n = size_n;
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (!use_fp32_reduce) {
    reduce_max_m = 0;
    reduce_n = 0;
  }
  torch::Tensor c_tmp = torch::empty({reduce_max_m, reduce_n}, options_fp32);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  int sms = -1;

  // Verify g_idx and perm
  TORCH_CHECK((g_idx.size(0) == 0 && perm.size(0) == 0) ||
                  (g_idx.size(0) == size_k && perm.size(0) == size_k),
              "Unexpected g_idx.size(0) = ", g_idx.size(0),
              " and perm.size(0) = ", perm.size(0),
              ", where size_k = ", size_k);

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(0) != 0; // awq中为False，gptq中可选

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0); // 分组量化，scale和zero的行数即是group的数量

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    TORCH_CHECK(b_zeros.size(0) == num_groups,
                "b_zeros dim 0 = ", b_zeros.size(0),
                " is not num_groups = ", num_groups);
    TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                "b_zeros dim 1 = ", b_scales.size(1),
                " is not size_n / pack_factor = ", size_n / pack_factor);
  }

  // Verify workspace size
  TORCH_CHECK(size_n % marlin::min_thread_n == 0, "size_n = ", size_n,
              ", is not divisible by min_thread_n = ", marlin::min_thread_n);
  int min_workspace_size = (size_n / marlin::min_thread_n) * marlin::max_par;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  if (a.scalar_type() == at::ScalarType::Half) {
    marlin::marlin_mm<half>(
        a.data_ptr<at::Half>(), b_q_weight.data_ptr(), c.data_ptr<at::Half>(),
        c_tmp.data_ptr<float>(), b_scales.data_ptr<at::Half>(),
        b_zeros.data_ptr(), g_idx.data_ptr(), perm.data_ptr(),
        a_tmp.data_ptr<at::Half>(), size_m, size_n, size_k,
        workspace.data_ptr(), *b_q_type, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, marlin::max_par, use_fp32_reduce);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    marlin::marlin_mm<nv_bfloat16>(
        a.data_ptr<at::BFloat16>(), b_q_weight.data_ptr(),
        c.data_ptr<at::BFloat16>(), c_tmp.data_ptr<float>(),
        b_scales.data_ptr<at::BFloat16>(), b_zeros.data_ptr(), g_idx.data_ptr(),
        perm.data_ptr(), a_tmp.data_ptr<at::BFloat16>(), size_m, size_n, size_k,
        workspace.data_ptr(), *b_q_type, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, at::cuda::getCurrentCUDAStream(dev),
        thread_k, thread_n, sms, marlin::max_par, use_fp32_reduce);
  } else {
    TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16");
  }

  return c;
}

#endif
