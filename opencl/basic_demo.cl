// OpenCL Kernel Function for element by element vector addition
__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements) {
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= iNumElements) {   
        return; 
    }
    c[iGID] = a[iGID] + b[iGID];
}
