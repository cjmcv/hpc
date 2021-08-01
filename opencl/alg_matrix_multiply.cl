//#define BLOCK_SIDE_SIZE 8

__kernel void MatrixMulDevice(const int M, const int N, const int K, const float ALPHA,
  __global const float *A, const int lda,
  __global const float *B, const int ldb,
  __global float *C, const int ldc) {

  for (int gid_x = get_global_id(0), gid_y = get_global_id(1);
    gid_x < N && gid_y < M; 
    gid_x += get_global_size(0), gid_y += get_global_size(1)) {

    // ��ά�̲߳��֣�ÿ���̶߳�Ӧ��������е�һ��Ԫ��
    // ÿ���߳�ֻ��Ҫ������k����ı�������
    float c_sub_acc = 0;
    for (int k = 0; k < K; k++) {
      c_sub_acc += A[gid_y * lda + k] * B[k * ldb + gid_x];
    }
    C[gid_y * ldc + gid_x] = c_sub_acc;
  }
}