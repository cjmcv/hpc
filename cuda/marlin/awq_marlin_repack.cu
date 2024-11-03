#include "marlin.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

namespace marlin {

template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {}

}  // namespace marlin

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                int64_t size_k, int64_t size_n,
                                int64_t num_bits) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "marlin_repack_from_gptq(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else

// https://github.com/IST-DASLab/marlin/marlin/__init__.py
// # Precompute permutations for Marlin weight and scale shuffling 
// def _get_perms():
//     perm = []
//     for i in range(32):  # 一个warp有32个线程，这里为每个线程进行分配。
//         perm1 = []
//         col = i // 4     # 0=t0-t3，1=t4-t7, 2=t8-t11 ...
//         for block in [0, 1]:  # 
//             # t0: 0,1,8,9;  t1:2,3,10,11; t2:4,5,12,13; t3:6,7,14,15
//             # t4: 0,1,8,9;  ...
//             for row in [      
//                 2 * (i % 4),
//                 2 * (i % 4) + 1,
//                 2 * (i % 4 + 4),
//                 2 * (i % 4 + 4) + 1
//             ]:
//                 # t0:[0, 16, 128, 144, 8, 24, 136, 152]; t1:[32, 48, 160, 176, 40, 56, 168, 184]
//                 # 即正好是16x16的B矩阵对应warp线程分配情况。
//                 # 对应mma的m16n8k16：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
//                 # t0:([0,0], [1,0], [8,0], [9,0]) 转为16x16的格式 => 0=[0,0], 16=[1,0], 128=[8,0], 144=[9,0]
//                 perm1.append(16 * row + col + 8 * block)  
//         for j in range(4):
//             perm.extend([p + 256 * j for p in perm1])

//     perm = np.array(perm)
//     interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
//     perm = perm.reshape((-1, 8))[:, interleave].ravel()  # 对应后续的faster dequant中的4位混插方式，t0: [0, 128, 8, 136, 16, 144, 24, 152]
//     perm = torch.from_numpy(perm)
//     scale_perm = []
//     for i in range(8):
//         scale_perm.extend([i + 8 * j for j in range(8)])
//     scale_perm_single = []
//     for i in range(4):
//         scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
//     return perm, scale_perm, scale_perm_single

// _perm, _scale_perm, _scale_perm_single = _get_perms()

    // def pack(self, linear, scales):
    //     """Pack a fake-quantized linear layer into this actual Marlin representation.
    //     @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
    //     @scales: corresponding quantization scales of shape `(infeatures, groups)`
    //     """ 
    //     if linear.weight.dtype != torch.half:
    //         raise ValueError('Only `torch.half` weights are supported.')
    //     tile = 16
    //     maxq = 2 ** 4 - 1 # 256-1
    //     s = scales.t() # scale 转置
    //     w = linear.weight.data.t() # 权重也转置
    //     if self.groupsize != self.k:
    //         w = w.reshape((-1, self.groupsize, self.n)) # 
    //         w = w.permute(1, 0, 2)
    //         w = w.reshape((self.groupsize, -1))
    //         s = s.reshape((1, -1))
    //     w = torch.round(w / s).int()
    //     w += (maxq + 1) // 2
    //     w = torch.clamp(w, 0, maxq)
    //     if self.groupsize != self.k:
    //         w = w.reshape((self.groupsize, -1, self.n))
    //         w = w.permute(1, 0, 2)
    //         w = w.reshape((self.k, self.n)).contiguous()
    //         s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    //     else:
    //         s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    //     s = s.reshape((-1, self.n)).contiguous()
    //     w = w.reshape((self.k // tile, tile, self.n // tile, tile)) # 将权重转化成：[(IN/16),16,(OUT/16),16]的矩阵。
    //     w = w.permute((0, 2, 1, 3)) # [(IN/16),(OUT/16),16,16]
    //     w = w.reshape((self.k // tile, self.n * tile)) # [IN/16, OUT*16], 即16*16被展开为一行
    //     res = w
    //     res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape) # 按_get_perms的结果进行perm操作，排成mma和faster dequant所需的顺序
    //     q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    //     res = res.cpu().numpy().astype(np.uint32)
    //     for i in range(8):
    //         q |= res[:, i::8] << 4 * i     # 每8个数据融合成一个int32
    //     q = torch.from_numpy(q.astype(np.int32)).to(w.device)
    //     self.B[:, :] = q.to(self.B.device)
    //     self.s[:, :] = s.to(self.s.device)

namespace marlin {

template <int const num_threads, int const num_bits>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  // [tile_k_size, tile_n_size] = [16, 64]
  int k_tiles = size_k / tile_k_size; // k方向总tile个数
  int n_tiles = size_n / tile_n_size; // n方向总tile个数
  // gridDim.x为block个数，launch时被设为了sm个数，即一个sm一个block
  // block size是一维256个线程。tile的个数 除以 block个数，即每个block（每个sm）负责 block_k_tiles 个tile，
  int block_k_tiles = div_ceil(k_tiles, gridDim.x); 

  // 以block为单位计算，start_k_tile为每个block的第一个tile
  int start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  // 取出每个block负责的最后一个tile
  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  // 除以8，表示int32类型的n方向tile数量，因为b_q_weight_ptr以int32进行索引
  constexpr int tile_n_ints = tile_n_size / pack_factor;  // tile_n_size=64, pack_factor=8

  // 一个warp在tile上的线程分配，n方向用2个线程，k方向用16个，共32个。
  // 即n方向一个线程处理4个int32类型数据(即32个int4)。一个warp处理一个tile[16, 64]
  constexpr int stage_n_threads = tile_n_ints / 4; // 为 2
  constexpr int stage_k_threads = tile_k_size; // 为 16
  constexpr int stage_size = stage_k_threads * stage_n_threads; // 32

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    // n_tile_id 是 n 方向做multi-stage时当前stage的偏移量
    int first_n = n_tile_id * tile_n_size;
    int first_n_packed = first_n / pack_factor; // b_q_weight_ptr类型为int32，其索引也需要由int4改成int32的

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      int k_id = threadIdx.x / stage_n_threads; // 可得到0-15
      int n_id = threadIdx.x % stage_n_threads; // 可得到0-1

      int first_k = k_tile_id * tile_k_size;

      // 异步拷贝16个字节，即4个int，对应一个warp里线程的工作量
      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) +
                                     first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  // 与上面的_get_perms对应 
  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    int warp_id = threadIdx.x / 32;
    int th_id = threadIdx.x % 32;

    // 1个warp 32个线程, 这里launch的一个block有256，即8*32;
    // 这里只使用4个warp，即一半的线程？
    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9}; // 0,1,8,9是mma的16x8x16指令对B矩元素访问的偏移量，如t0取(0,1,8,9), t1取(2,3,10,11)

    int cur_n = warp_id * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    // Undo interleaving
    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];

      int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
      int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) +
                                          sh_stride * cur_elem];

      vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
      vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7}; // 针对后续的faster dequant中的4位混插方式，进一步调整

      uint32_t res = 0;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4); // # 按调整顺序，将每8个数据融合成一个int32
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
  #pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
  #pragma unroll
  // 外层沿着k方向进行，stage沿n方向进行。
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    // 首次进入新的k_tile，将该k对应的第一个n_tile的所有stage加载到共享内存
    start_pipes(k_tile_id, n_tile_id);

    // n_tiles是n方向tile的总数
    while (n_tile_id < n_tiles) {
  #pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        // 如pip为0，则加载第7段到共享内存，(pip=1, load=8),(pip=2, load=9)..(pip=7, load=14)
        // ？为什么pip=0，load=7，是该循环最后一stage，而不是新循环的第一阶？
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages; // repack_stages为8，分8段进行，
    }
  }
}

}  // namespace marlin

  // 一个sm一个block，一个block256个线程，一维布局
  #define CALL_IF(NUM_BITS)                                                   \
    else if (num_bits == NUM_BITS) {                                          \
      cudaFuncSetAttribute(                                                   \
          marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>, \
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);       \
      marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>      \
          <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(       \
              b_q_weight_ptr, out_ptr, size_k, size_n);                       \
    }

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits) {
  // Verify compatibility with marlin tile of 16x64
  TORCH_CHECK(size_k % marlin::tile_k_size == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", marlin::tile_k_size);
  TORCH_CHECK(size_n % marlin::tile_n_size == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", marlin::tile_n_size);

  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int const pack_factor = 32 / num_bits; // 权重是4位(也兼容8位)，即会将int4 pack到int32中，此时pack_factor为8;

  // Verify B
  TORCH_CHECK(b_q_weight.size(0) == size_k,
              "b_q_weight.size(0) = ", b_q_weight.size(0),
              " is not size_k = ", size_k);
  TORCH_CHECK((size_n / pack_factor) == b_q_weight.size(1),  // size_n为元素个数，类型是int32，所以对于int4而言，b_q_weight.size(1)需要等于size_n/8
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_n = ", size_n, ", pack_factor = ", pack_factor);

  // Verify device and strides
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());

  // 输出权重B，维度按tile进行调整
  // 以tile_size为16，则size_k维度除以16，即把16行变成1行；size_n * marlin::tile_size / pack_factor表示矩阵已经按int4进行排布了
  torch::Tensor out = torch::empty(
      {size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor},
      options);

  // Get ptrs
  uint32_t const* b_q_weight_ptr =
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  // Get dev info
  int dev = b_q_weight.get_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);  // 获取到该设备有多少个sm，repack kernel的启动是一个sm一个block。

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  if (false) {
  }
  CALL_IF(4)
  CALL_IF(8)
  else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits);
  }

  return out;
}

#endif
