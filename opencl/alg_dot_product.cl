#define WORKGROUP_SIZE 256

__kernel void DotProductDevice(__global int *src1, __global int *src2, __global int *dst) {
  int gid = get_global_id(0);
  __local int buffer[WORKGROUP_SIZE];

  // The memory of dst is cleared atomically.
  if (gid == 0)
    atomic_xchg(dst, 0);

  // Get the id of each item in a work group.
  int lid = get_local_id(0);

  // Save the intermediate result to buffer.
  buffer[lid] = src1[gid] * src2[gid];

  // Make sure all working groups are done.
  barrier(CLK_LOCAL_MEM_FENCE);

  // The first item of each group is responsible for
  // aggregating the group's results, and add to the dst.
  if (lid == 0) {
    int sum = 0;
    for (int i = 0; i < WORKGROUP_SIZE; i++)
      sum += buffer[i];
    atomic_add(dst, sum);
  }
}