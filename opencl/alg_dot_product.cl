#define WORKGROUP_SIZE 256

__kernel void DotProduct(__global int *src1, __global int *src2, __global int *dst, int len) {
  int gid = get_global_id(0);
  __local int buffer[WORKGROUP_SIZE];

  // 以原子方式将 dst 地址中内容清零（只执行一次，使用全局工作项标识）
  if (gid == 0)
    atomic_xchg(dst, 0);
  if (gid < WORKGROUP_SIZE)
    buffer[gid] = 0;

  // 获取工作组中的每个工作项
  int lid = get_local_id(0);

  // 填充缓冲区
  buffer[lid] += src1[gid] * src2[gid];

  // 所有的工作项执行到这里。等待对局部缓冲区的访问完成
  barrier(CLK_LOCAL_MEM_FENCE);

  // 只有在第一个工作项执行的时候获取结果，并将其添加到 dst 指向的位置
  if (lid == 0) {
    int sum = 0;
    for (int i = 0; i < WORKGROUP_SIZE; i++)
      sum += buffer[i];
    atomic_add(dst, sum);
  }
}