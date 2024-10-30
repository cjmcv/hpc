// TODO: 分块大小和local_work_size 调整


// GEMM 矩阵乘法例子
// 实现平台：

//    K     N        N
// M     K     =  M

//// CPU版本
// for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//         for (k = 0; k < K; ++k) {
//             C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
//         }
//     }
// }

// V1 初始版本 同 用于独显的 GemmDeviceV1
// B矩阵 gid_x 作为列，相邻线程访问相邻元素，满足全局内存合并访问
__kernel void GemmMobileDeviceV1(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {

    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x < N && gid_y < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        float c_sub_acc = 0;
        for (int k = 0; k < K; k++) {
            c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
        }
        C[gid_y * ldc + gid_x] = c_sub_acc;
    }
}

// V2 一个线程处理16个点.
// V1中的全局内存的计算访存比是1:2，最内层循环线程要读两次全局内存，然后计算一次乘加指令fma
// v2中读8次，计算4*4次fma，计算访存比是2:1
__kernel void GemmMobileDeviceV2(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float c_sub_acc[STEP][STEP] = {0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            for (int si = 0; si < STEP; si++) {
                for (int sj = 0; sj < STEP; sj++) {
                    c_sub_acc[si][sj] += A[(gid_sy+si) * lda + k] * B[k * ldb + gid_sx + sj];
                }
            }

            // // 分析计算访存比, 读8次，计算4*4次fma，计算访存比是2:1
            // float Asi[STEP], Bsj[STEP];
            // Asi[0] = A[(gid_sy+0) * lda + k];
            // Asi[1] = A[(gid_sy+1) * lda + k];
            // Asi[2] = A[(gid_sy+2) * lda + k];
            // Asi[3] = A[(gid_sy+3) * lda + k];
            // Bsj[0] = B[k * ldb + gid_sx + 0];
            // Bsj[1] = B[k * ldb + gid_sx + 1];
            // Bsj[2] = B[k * ldb + gid_sx + 2];
            // Bsj[3] = B[k * ldb + gid_sx + 3];
            // for (int si = 0; si < STEP; si++) {
            //     for (int sj = 0; sj < STEP; sj++) {
            //         c_sub_acc[si][sj] += Asi[si] * Bsj[sj];
            //     }
            // }
        }

        for (int si=0; si<STEP; si++) {
            for (int sj=0; sj<STEP; sj++) {
                C[(gid_sy+si) * ldc + gid_sx+sj] += c_sub_acc[si][sj];
            }
        }
    }
}

// v3_0 向量化过渡版本0, 基于v2 (注释部分), A和B矩阵 使用向量化数据类型改写
__kernel void GemmMobileDeviceV3_0(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float c_sub_acc[STEP][STEP] = {0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi.s0 = A[(gid_sy+0) * lda + k];
            Asi.s1 = A[(gid_sy+1) * lda + k];
            Asi.s2 = A[(gid_sy+2) * lda + k];
            Asi.s3 = A[(gid_sy+3) * lda + k];
            Bsj = vload4(0, B + k * ldb + gid_sx);

            c_sub_acc[0][0] += Asi.s0 * Bsj.s0;
            c_sub_acc[0][1] += Asi.s0 * Bsj.s1;
            c_sub_acc[0][2] += Asi.s0 * Bsj.s2;
            c_sub_acc[0][3] += Asi.s0 * Bsj.s3;

            c_sub_acc[1][0] += Asi.s1 * Bsj.s0;
            c_sub_acc[1][1] += Asi.s1 * Bsj.s1;
            c_sub_acc[1][2] += Asi.s1 * Bsj.s2;
            c_sub_acc[1][3] += Asi.s1 * Bsj.s3;

            c_sub_acc[2][0] += Asi.s2 * Bsj.s0;
            c_sub_acc[2][1] += Asi.s2 * Bsj.s1;
            c_sub_acc[2][2] += Asi.s2 * Bsj.s2;
            c_sub_acc[2][3] += Asi.s2 * Bsj.s3;

            c_sub_acc[3][0] += Asi.s3 * Bsj.s0;
            c_sub_acc[3][1] += Asi.s3 * Bsj.s1;
            c_sub_acc[3][2] += Asi.s3 * Bsj.s2;
            c_sub_acc[3][3] += Asi.s3 * Bsj.s3;
        }

        for (int si=0; si<STEP; si++) {
            for (int sj=0; sj<STEP; sj++) {
                C[(gid_sy+si) * ldc + gid_sx+sj] += c_sub_acc[si][sj];
            }
        }
    }
}

// v3_1 基于v3_0, c_sub_acc也使用向量化数据类型改写
__kernel void GemmMobileDeviceV3_1(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi.s0 = A[(gid_sy+0) * lda + k];
            Asi.s1 = A[(gid_sy+1) * lda + k];
            Asi.s2 = A[(gid_sy+2) * lda + k];
            Asi.s3 = A[(gid_sy+3) * lda + k];
            Bsj = vload4(0, B + k * ldb + gid_sx);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;
        }

        vstore4(acc[0], 0, C + (gid_sy+0) * ldc + gid_sx);
        vstore4(acc[1], 0, C + (gid_sy+1) * ldc + gid_sx);
        vstore4(acc[2], 0, C + (gid_sy+2) * ldc + gid_sx);
        vstore4(acc[3], 0, C + (gid_sy+3) * ldc + gid_sx);
    }
}

// v3_2 基于v3_1, 将A矩阵在外部进行转置, gid_sy充当列，可使访存得到合并。
// 同时也使A的单线程读取满足向量化加载(优化思路同cpu的缓存命中率)。
// note: 全局内存合并访问优化指的一个线程束对全局内存的一次访问请求导致最少数量的数据传输。
//       比如warp内线程访问的地址连续,与线程一一对应, 一个线程访问一个数据.且起始地址是每个线程所存取的大小的16倍.
__kernel void GemmMobileDeviceV3_2(const int M, const int N, const int K,
                           __global const float *A, const int lda,
                           __global const float *B, const int ldb,
                           __global float *C, const int ldc) {
    
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_sx = get_global_id(0)*STEP, gid_sy = get_global_id(1)*STEP;
        gid_sx < N && gid_sy < M; 
        gid_sx += get_global_size(0)*STEP, gid_sy += get_global_size(1)*STEP) {

        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            Asi = vload4(0, A + k * lda + gid_sy);
            Bsj = vload4(0, B + k * ldb + gid_sx);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;
        }

        vstore4(acc[0], 0, C + (gid_sy+0) * ldc + gid_sx);
        vstore4(acc[1], 0, C + (gid_sy+1) * ldc + gid_sx);
        vstore4(acc[2], 0, C + (gid_sy+2) * ldc + gid_sx);
        vstore4(acc[3], 0, C + (gid_sy+3) * ldc + gid_sx);
    }
}

// v4 使用纹理内存
__constant sampler_t default_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void GemmMobileDeviceV4(const int M, const int N, const int K,
                           __read_only image2d_t A, const int lda,
                           __read_only image2d_t B, const int ldb,
                           __write_only image2d_t C, const int ldc) {

    // 用回 gid_x 而非 gid_sx，因为图像按rgba定义，read_imagef按下标读取时会直接跨越4个元素 
    // 需 STEP 也与 CHANNEL 一致为4
    const int STEP = 4;
    float4 acc[STEP] = {(float4)0, (float4)0, (float4)0, (float4)0};
    for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
        gid_x*STEP < N && gid_y*STEP < M; 
        gid_x += get_global_size(0), gid_y += get_global_size(1)) {

        // printf(">>>> (%d, %d)-(%d, %d)", N, M, gid_x*STEP, gid_y*STEP);
        for (int k = 0; k < K; k++) {
            float4 Asi, Bsj;
            // Asi = vload4(0, A + k * lda + gid_y);
            // Bsj = vload4(0, B + k * ldb + gid_x);
            Asi = read_imagef(A, default_sampler, (int2)(gid_y, k));
            Bsj = read_imagef(B, default_sampler, (int2)(gid_x, k));

            // printf("A: (%d, %d)-(%f, %f, %f, %f)\n", gid_y, k, Asi.s0, Asi.s1, Asi.s2, Asi.s3);
            // printf("B: (%d, %d)-(%f, %f, %f, %f)\n", gid_x, k, Bsj.s0, Bsj.s1, Bsj.s2, Bsj.s3);

            acc[0] += Asi.s0 * Bsj;
            acc[1] += Asi.s1 * Bsj;
            acc[2] += Asi.s2 * Bsj;
            acc[3] += Asi.s3 * Bsj;

            // acc[0] = fma(Asi.s0, Bsj, acc[0]);
            // acc[1] = fma(Asi.s1, Bsj, acc[1]);
            // acc[2] = fma(Asi.s2, Bsj, acc[2]);
            // acc[3] = fma(Asi.s3, Bsj, acc[3]);
        }

        // (int2)(列，行)，一次存4列1行，共回存4列4行
        // gid_y是按宽度为4个元素进行索引的，需要乘以STEP，转为按1个元素索引。
        int gid_sy = gid_y * STEP;
        write_imagef(C, (int2)(gid_x, gid_sy+0), acc[0]);
        write_imagef(C, (int2)(gid_x, gid_sy+1), acc[1]);
        write_imagef(C, (int2)(gid_x, gid_sy+2), acc[2]);
        write_imagef(C, (int2)(gid_x, gid_sy+3), acc[3]);
    }
}

// v5, 基于v4, 进一步增大STEP到8，即一个线程将需要处理8*8的数据，
//     读4*4次，计算4*16次fma，计算访存比是4:1, 优化的前提是寄存器充足
__kernel void GemmMobileDeviceV5(const int M, const int N, const int K,
                           __read_only image2d_t A, const int lda,
                           __read_only image2d_t B, const int ldb,
                           __write_only image2d_t C, const int ldc) {

    const int CHANNEL = 4;
    const int STEP = 8;
    float4 acc[STEP][STEP/CHANNEL] = {{0}};
    float4 Asi[STEP/CHANNEL], Bsj[STEP/CHANNEL];

    for (int gid_x = get_global_id(0) * STEP/CHANNEL, gid_y = get_global_id(1) * STEP/CHANNEL;
        gid_x*CHANNEL < N && gid_y*CHANNEL < M; 
        gid_x += get_global_size(0) * STEP/CHANNEL, gid_y += get_global_size(1) * STEP/CHANNEL) {

        for (int k = 0; k < K; k++) {
            Asi[0] = read_imagef(A, default_sampler, (int2)(gid_y, k));
            Asi[1] = read_imagef(A, default_sampler, (int2)(gid_y+1, k));
            Bsj[0] = read_imagef(B, default_sampler, (int2)(gid_x, k));
            Bsj[1] = read_imagef(B, default_sampler, (int2)(gid_x+1, k));

            acc[0][0] += Asi[0].s0 * Bsj[0];
            acc[0][1] += Asi[0].s0 * Bsj[1];
            acc[1][0] += Asi[0].s1 * Bsj[0];
            acc[1][1] += Asi[0].s1 * Bsj[1];
            acc[2][0] += Asi[0].s2 * Bsj[0];
            acc[2][1] += Asi[0].s2 * Bsj[1];
            acc[3][0] += Asi[0].s3 * Bsj[0];
            acc[3][1] += Asi[0].s3 * Bsj[1];

            acc[4][0] += Asi[1].s0 * Bsj[0];
            acc[4][1] += Asi[1].s0 * Bsj[1];
            acc[5][0] += Asi[1].s1 * Bsj[0];
            acc[5][1] += Asi[1].s1 * Bsj[1];
            acc[6][0] += Asi[1].s2 * Bsj[0];
            acc[6][1] += Asi[1].s2 * Bsj[1];
            acc[7][0] += Asi[1].s3 * Bsj[0];
            acc[7][1] += Asi[1].s3 * Bsj[1];

            //// 推算
            // A0
            // Asi[0] = A[k * lda + (gid_sy+0)];
            // Asi[1] = A[k * lda + (gid_sy+1)];
            // Asi[2] = A[k * lda + (gid_sy+2)];
            // Asi[3] = A[k * lda + (gid_sy+3)];
            // A1
            // Asi[4] = A[k * lda + (gid_sy+4)];
            // Asi[5] = A[k * lda + (gid_sy+5)];
            // Asi[6] = A[k * lda + (gid_sy+6)];
            // Asi[7] = A[k * lda + (gid_sy+7)];
            // B0
            // Bsj[0] = B[k * ldb + gid_sx + 0];
            // Bsj[1] = B[k * ldb + gid_sx + 1];
            // Bsj[2] = B[k * ldb + gid_sx + 2];
            // Bsj[3] = B[k * ldb + gid_sx + 3];
            // B1
            // Bsj[4] = B[k * ldb + gid_sx + 4];
            // Bsj[5] = B[k * ldb + gid_sx + 5];
            // Bsj[6] = B[k * ldb + gid_sx + 6];
            // Bsj[7] = B[k * ldb + gid_sx + 7];

            // acc[0][0]
            // c_sub_acc[0][0] += Asi[0] * Bsj[0];
            // c_sub_acc[0][1] += Asi[0] * Bsj[1];
            // c_sub_acc[0][2] += Asi[0] * Bsj[2];
            // c_sub_acc[0][3] += Asi[0] * Bsj[3];
            // acc[0][1]
            // c_sub_acc[0][4] += Asi[0] * Bsj[4];
            // c_sub_acc[0][5] += Asi[0] * Bsj[5];
            // c_sub_acc[0][6] += Asi[0] * Bsj[6];
            // c_sub_acc[0][7] += Asi[0] * Bsj[7];
            // acc[1][0]
            // c_sub_acc[1][0] += Asi[1] * Bsj[0];
            // c_sub_acc[1][1] += Asi[1] * Bsj[1];
            // c_sub_acc[1][2] += Asi[1] * Bsj[2];
            // c_sub_acc[1][3] += Asi[1] * Bsj[3];
            // acc[1][1]
            // c_sub_acc[1][4] += Asi[1] * Bsj[4];
            // c_sub_acc[1][5] += Asi[1] * Bsj[5];
            // c_sub_acc[1][6] += Asi[1] * Bsj[6];
            // c_sub_acc[1][7] += Asi[1] * Bsj[7];
            // ...

            // for (int si = 0; si < 8; si++) {
            //     for (int sj = 0; sj < 8; sj++) {
            //         acc[si][sj] += Asi[si] * Bsj[sj];
            //     }
            // }

        }

        // printf("<<%d, %d>>\n", gid_x, gid_y);
        // printf("0->(%f; %f; %f; %f; %f; %f; %f; %f)\n", acc[0][0], acc[1][0], acc[2][0], acc[3][0], acc[4][0], acc[5][0], acc[6][0], acc[7][0]);
        // printf("1->(%f; %f; %f; %f; %f; %f; %f; %f)\n", acc[0][1], acc[1][1], acc[2][1], acc[3][1], acc[4][1], acc[5][1], acc[6][1], acc[7][1]);
        // 回存 8*8 的数据
        int gid_sy = gid_y * CHANNEL;
        write_imagef(C, (int2)(gid_x, gid_sy+0), acc[0][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+1), acc[1][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+2), acc[2][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+3), acc[3][0]);

        write_imagef(C, (int2)(gid_x, gid_sy+4), acc[4][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+5), acc[5][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+6), acc[6][0]);
        write_imagef(C, (int2)(gid_x, gid_sy+7), acc[7][0]);

        write_imagef(C, (int2)(gid_x+1, gid_sy+0), acc[0][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+1), acc[1][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+2), acc[2][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+3), acc[3][1]);

        write_imagef(C, (int2)(gid_x+1, gid_sy+4), acc[4][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+5), acc[5][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+6), acc[6][1]);
        write_imagef(C, (int2)(gid_x+1, gid_sy+7), acc[7][1]);
        
        //
        // for (int si=0; si<STEP; si++) {
        //     for (int sj=0; sj<STEP; sj++) {
        //         C[(gid_sy+si) * ldc + gid_sx+sj] += c_sub_acc[si][sj];
        //     }
        // }
    }
}
