/*!
* \brief Record the basic usage of float2half.
*/
#include <iostream>
#include <time.h>
#include <assert.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA_CHECK error in line %d of file %s \
              : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

// Taken from:
// https://github.com/dmlc/mshadow/blob/master/mshadow/half.h
union Bits {
  float f;
  int32_t si;
  uint32_t ui;
};
static int const shift = 13;
static int const shiftSign = 16;

static int32_t const infN = 0x7F800000;   // flt32 infinity
static int32_t const maxN = 0x477FE000;   // max flt16 normal as a flt32
static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
static int32_t const signN = 0x80000000;  // flt32 sign bit

static int32_t const infC = infN >> shift;
static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32
static int32_t const maxC = maxN >> shift;
static int32_t const minC = minN >> shift;
static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
static int32_t const norC = 0x00400;  // min flt32 normal down shifted

static int32_t const maxD = infC - maxC - 1;
static int32_t const minD = minC - subC - 1;

// Host version of device function __float2half_rn()
uint16_t float2half(const float& value) {
  Bits v, s;
  v.f = value;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign;  // logical shift
  s.si = mulN;
  s.si = s.f * v.f;  // correct subnormals
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;  // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  return v.ui | sign;
}

// Annotation:
// \function, unsigned short __float2half_rn(float x);  -->  device_functions.h
//   It was already presented in CUDA before CUDA 7.5.
//
// \function, __half __float2half(const float a);       -->  cuda_fp16.h
//   It was introduced in CUDA 7.5 and does the same with __float2half_rn,
//   but return a half.
//
// \function, __half2 __float2half2_rn(const float a);  -->  cuda_fp16.h
//   It returns a half2 which stores two half into an unsigned int. 
//
// \Difference between __float2half_rn and __float2half.
// https://stackoverflow.com/questions/35198856/half-precision-difference-between-float2half-vs-float2half-rn
//   "I found that (for sm_20) the old __float2half_rn() has an additional int16 to int32
//   operation and does a 32bit store. On the other hand, __float2half_() does not have 
//   this conversion and does a 16bit store."
//   __float2half_rn():
//    /*0040*/         I2I.U32.U16 R0, R0;
//    /*0050*/         STL[R2], R0;
//   __float2half():
//    /*0048*/         STL.U16 [R2], R0;

__global__ void ConvertTest() {
  const float flt_in = 1.1234;

  //unsigned short res = __float2half_rn(flt_in);
  half res = __float2half(flt_in);  
  printf("Device version half: %hu\n", res);

  half2 res2 = __float2half2_rn(flt_in);
  printf("Device version half2: %d\n", res2);
}

int main() {
  ConvertTest << <1, 1 >> >();
  cudaDeviceSynchronize();

  unsigned short res = float2half(1.1234);
  printf("Host version half: %hu\n", res);
  return 0;
}