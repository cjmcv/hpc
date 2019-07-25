/*!
* \brief  Float Precision 16.
* \Reference https://github.com/dmlc/mshadow/blob/master/mshadow/half.h
*      
*      The storage of floating point Numbers consists of three parts: 
*      Sign: 0 for Positive / 1 for Negative.
*      Exponent: It is used to store the exponential part of scientific 
*               counting method and adopt shift storage mode.
*      Mantissa: A significant number after the decimal point.
* 
*      Value = (-1)^S * M * 2^E
*            = 1.xxxxxx(2) * 2^n(10)
*
*      fp64: Sign(1), Exponent(11), Mantissa(52).
*      fp32: Sign(1), Exponent(8), Mantissa(23). => 0.000001 * 2^23 = 8.3; 0.0000001 * 2^23 = 0.83
*                                                => 6 significant digits can be retained
*      fp16: Sign(1), Exponent(5), Mantissa(10). => 0.001 => 0.001 * 2^9 = 0.512, 0.001 * 2^10 = 1.024
*                                                => 3 significant digits can be retained
*
*      https://blog.csdn.net/qingtingchen1987/article/details/7719259
*      float 9.125 = 9.125 * 10^0 (decimal)
*      => 9 = 1001, 0.125 = 0.001
*      => 9.125 = 1001.001 => Three to the left => 1.001001 * 2^3 
*              PS: 1.001001(2) == (1).(1/8+1/64) == 1.140625 ===> 1.140625 * 2^3 = 9.125
*      That means E = 3, but the blas is 2^7-1=127, so E(00000001~11111110 == 1~254 ==> -126~127) in memory is actually equal to 127+3 = 130 == 10000010(2)
*                 M = 001001 == 00100100000000000000000
*                 S = 0 for Positive.
*      float 9.125 = 0 10000010 00100100000000000000000
* 
*      In fp16, 2^4-1=15 is the blas of E(00001~11110 == 0~30 ==> -14~15), so E is actually equal to 18 == 10010.
*      fp16  9.125 = 0 10010 0010010000
* 
*      https://blog.csdn.net/tercel_zhang/article/details/52537726
*      Note for E: 1. The minimum exponent (all bits set to 0) is used to define 0 and weak specification Numbers
*                  2. Maximum exponent (all bits full value 1) is used to define ±∞ and NaN (Not a Number)
*                  3. Other indices are used to represent regular Numbers. So the scope of E in float16 is (00001~11110)
*
*      normal, subnormal and non number: https://www.jianshu.com/p/43b1b09f27f4

*/

#ifndef CUX_FP16_H_
#define CUX_FP16_H_

#include <iostream>
#include <cuda_fp16.h>

// This flag dictates rounding for the float2half() routine only (used generally on Windows),
// not the f16c lib or cuda v7.5 (or later) behavior which is fixed at round-to-nearest-even.
#ifndef HALF_ROUND_TO_NEAREST
  #define HALF_ROUND_TO_NEAREST 1
#endif

namespace cux { 

#define HALF_OPERATOR(OP)                                   \
  inline half operator OP (half a, half b) {                \
    return half(float(a) OP float(b));                      \
  }                                                         \
  template<typename T>                                      \
  inline half operator OP (half a, T b) {                   \
    return half(float(a) OP float(b));                      \
  }                                                         \
  template<typename T>                                      \
  inline half operator OP (T a, half b) {                   \
    return half(float(a) OP float(b));                      \
  }

#define HALF_ASSIGNOP(AOP, OP)                              \
  template<typename T>                                      \
  inline half operator AOP (const T& a) {                   \
    return *this = half(float(*this) OP float(a));          \
  }     

#ifdef WIN32
__declspec(align(2)) class half {
#else
class alignas(2) half {
#endif
public:
  union {
    uint16_t half_;
    __half cuhalf_;
  };

  static inline half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  // Constructor and Convertors.
  inline half() {};
  inline half(const float& value) { constructor(value); }
  inline explicit half(const int8_t& value) { constructor(value); }
  inline explicit half(const uint8_t& value) { constructor(value); }
  inline explicit half(const int32_t& value) { constructor(value); }
  inline explicit half(const uint32_t& value) { constructor(value); }

  inline explicit half(const __half& value) { cuhalf_ = value; }
  inline operator float() { return float(half2float(half_));}

  // Operator overloading.
  HALF_ASSIGNOP(+=, +)
  HALF_ASSIGNOP(-=, -)
  HALF_ASSIGNOP(*=, *)
  HALF_ASSIGNOP(/=, /)

  inline half operator+() { return *this; }
  inline half operator-() { return half(-float(*this)); }
  inline half operator=(const half& a) { half_ = a.half_; return a; }
  template<typename T>
  inline half operator=(const T& a) { return *this = half(a); }

private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };
  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits); // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;         // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;       // 13. For Mantissa. float has 23 bits and fp16 has 10 bits, so it need to be shifted 13 bits to the right.
  static int const shiftSign = 16; // For Sign.
  static int32_t const expAdjust = 127 - 15;    // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity -> 0 - 111 1111 1 - 000 0000 0000 0000 0000 0000‬
  static int32_t const maxN = 0x477FE000;   // max flt32 that's a flt16 normal after -> 0 - 100 0111 0(142) - 111 1111 111~0 0000 0000 0000
                                            // ps: max flt16 = 0 - 11110(30) - 111 1111 111 => Exponent=11110, 11111 is the Special value, reserved for special value processing.
                                            //     1 1110 ==cvt2fp32==> 1...1110 = 1 000 1110;
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32           -> 0 - 011 1000 1(113) - 000 0000 000~0 0000 0000 0000
                                            // ps: min flt16 = 0 - 00001(1) - 000 0000 000 -> E=1-15 = -14 ==> -14 + 127 = 113 = 011 1000 1.
  static int32_t const maxZ = 0x33000000;  // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift; // Get the Sign + Exponent of flt32.
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

  // Example:
  //   fp32 9.125 = 0 10000010 00100100000000000000000
  //   fp16 9.125 = 0 10010 0010010000
  inline uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit: 0 10000010 00100100000000000000000 & 1 00000000 00000000000000000000000
                                     //         sign = 0 00000000 00000000000000000000000
    v.si ^= sign;                    // clear sign bit from v: v.si = 0 10000010 00100100000000000000000 = 9.125
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      v.ui = 0; // Handle eventual zeros here to ensure vshift will not exceed 32 below.
    }
    else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      // The only time it's *not* OK to add 0x1000 (i.e. half the flt16 fraction lsb) is
      // when the lsb of the flt16 fraction == 0 (so not rounding up to even) and the additional
      // bits to the right of the lsb are 1000... (including flt32 significand bits
      // that may be lost during the above vshift).  The first term below will always
      // be true for vshift >=12 (since even the 'hidden bit' has been shifted to the
      // right of the '1' bit in 0x1000). And when vshift <= 11, both terms combine to make
      // the proper test of the flt32 significand bits, including those lost during the vshift.
#if HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent to 1, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
#endif
    }
    else if (v.si <= maxN) {
      // Handle norms
#if HALF_ROUND_TO_NEAREST == 1
      // Rounding may increase the exponent, possibly creating an inf, but that's OK.
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
#endif
      v.ui -= expAdjust << fp32FractionBits;  
    }
    else if (v.si <= infN)
      v.si = infN;
    else if (v.si < nanN)
      v.si = nanN;

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // This function corresponds to half2float and is only used to understand the principle
  uint16_t float2half_t(const float& value) {
    Bits v, s;
    v.f = value;
    uint32_t sign = v.si & signN;
                                 
    v.si ^= sign;                
    sign >>= shiftSign;           
    s.si = mulN;                  
    s.si = s.f * v.f;             
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift;
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    return v.ui | sign;
  }

  float half2float(const uint16_t& value) {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  inline void constructor(const T& value) {
#if (defined(__CUDA_ARCH__))
    cuhalf_ = __float2half(float(value));  // NOLINT(*)
#else
    half_ = float2half(float(value));  // NOLINT(*)
#endif
  }
};

// Operator overloading.
// A binocular operator overload should be outside the class definition.
HALF_OPERATOR(+)
HALF_OPERATOR(-)
HALF_OPERATOR(*)
HALF_OPERATOR(/)

}	//namespace cux
#endif //CUX_FP16_H_
