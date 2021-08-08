#define WORKGROUP_SIZE 256

__kernel void DotProductDevice(__global int *src1, __global int *src2, const int len, __global int *dst) {
  
  // Init local memory.
  __local int buffer[WORKGROUP_SIZE];
  int lid = get_local_id(0);
  buffer[lid] = 0;

  for (int gid = get_global_id(0); gid < len; gid += get_global_size(0)) {
    // The memory of dst is cleared atomically.
    if (gid == 0)
      atomic_xchg(dst, 0);

    // Save the intermediate result to buffer. 
    //   Part of the data of the last group may not be involved 
    // in the calculation, so initialization is necessary for local memory
    buffer[lid] = src1[gid] * src2[gid];

    // Make sure all working groups are done.
    barrier(CLK_LOCAL_MEM_FENCE);

    // V1. 1.1ms
    // The first item of each group is responsible for
    // aggregating the group's results, and add to the dst.
    if (lid == 0) {
      int sum = 0;
      for (int i = 0; i < WORKGROUP_SIZE; i++)
        sum += buffer[i];
      atomic_add(dst, sum);
    }

    //// V2. 1.8ms
    //int count = WORKGROUP_SIZE / 2;
    //while (count >= 1) {
    //  if (lid < count) {
    //    buffer[lid] += buffer[count + lid];
    //  }  
    //  barrier(CLK_LOCAL_MEM_FENCE);
    //  count = count / 2;
    //}
    //if(lid == 0)
    //  atomic_add(dst, buffer[lid]);
  }
}
