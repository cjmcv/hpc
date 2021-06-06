#include <stdio.h>
#include <time.h>
#include <arm_neon.h>

void TransposeInt32Normal(int *src, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j <= i; j++) {
      int temp = *(src + i * width + j);
      *(src + i * width + j) = *(src + j * width + i);
      *(src + j * width + i) = temp;
    }
  }
}

void TransposeFp32x4x4_trn(float *src) {
  float32x4_t q0 = vld1q_f32(src);
  float32x4_t q1 = vld1q_f32(src + 1 * 4);
  float32x4_t q2 = vld1q_f32(src + 2 * 4);
  float32x4_t q3 = vld1q_f32(src + 3 * 4);
  //【 0,  1， 2， 3】
  //【 4,  5， 6， 7】
  //【 8,  9，10，11】
  //【12, 13，14，15】
  float32x4x2_t q01 = vtrnq_f32(q0, q1);
  float32x4x2_t q23 = vtrnq_f32(q2, q3);
  //【 0， 4， 2， 6】
  //【 1， 5， 3， 7】
  //【 8，12，10，14】
  //【 9，13，11，15】
  float32x4_t qq0 = q01.val[0];
  float32x2_t d00 = vget_low_f32(qq0);
  float32x2_t d01 = vget_high_f32(qq0);
  float32x4_t qq1 = q01.val[1];
  float32x2_t d10 = vget_low_f32(qq1);
  float32x2_t d11 = vget_high_f32(qq1);
  float32x4_t qq2 = q23.val[0];
  float32x2_t d20 = vget_low_f32(qq2);
  float32x2_t d21 = vget_high_f32(qq2);
  float32x4_t qq3 = q23.val[1];
  float32x2_t d30 = vget_low_f32(qq3);
  float32x2_t d31 = vget_high_f32(qq3);
  //【 0， 4， 8，12】
  //【 1， 5， 9，13】
  //【 2， 6，10，14】
  //【 3， 7，11，15】
  vst1q_f32(src, vcombine_f32(d00, d20));
  vst1q_f32(src + 1 * 4, vcombine_f32(d10, d30));
  vst1q_f32(src + 2 * 4, vcombine_f32(d01, d21));
  vst1q_f32(src + 3 * 4, vcombine_f32(d11, d31));
}

void TransposeFp32x4x4_uzp(float *src) {
  float32x4_t q0 = vld1q_f32(src);
  float32x4_t q1 = vld1q_f32(src + 1 * 4);
  float32x4_t q2 = vld1q_f32(src + 2 * 4);
  float32x4_t q3 = vld1q_f32(src + 3 * 4);
  //【 0,  1， 2， 3】
  //【 4,  5， 6， 7】
  //【 8,  9，10，11】
  //【12, 13，14，15】
  float32x4x2_t t0 = vuzpq_f32(q0, q1);
  float32x4x2_t t1 = vuzpq_f32(q2, q3);
  //【 0,  2， 4， 6】=> t0.val[0]
  //【 1,  3， 5， 7】=> t0.val[1]
  //【 8, 10，12，14】=> t1.val[0]
  //【 9, 11，13，15】=> t1.val[1]
  float32x4x2_t s0 = vuzpq_f32(t0.val[0], t1.val[0]);
  float32x4x2_t s1 = vuzpq_f32(t0.val[1], t1.val[1]);
  //【 0， 4， 8，12】=> s0.val[0]
  //【 2， 6，10，14】=> s0.val[1]
  //【 1， 5， 9，13】=> s1.val[0]
  //【 3， 7，11，15】=> s1.val[1]
  vst1q_f32(src, s0.val[0]);
  vst1q_f32(src + 1 * 4, s1.val[0]);
  vst1q_f32(src + 2 * 4, s0.val[1]);
  vst1q_f32(src + 3 * 4, s1.val[1]);
}

void TransposeInt16x4x4_trn(int16_t *src) {
  int width = 4;
  // 指令后缀均不带q，使用的是64位寄存器，所以读一次是16x4=64位
  int16x4_t a0 = vld1_s16(src);
  int16x4_t a1 = vld1_s16(src + width);
  int16x4_t a2 = vld1_s16(src + 2 * width);
  int16x4_t a3 = vld1_s16(src + 3 * width);
  //【 0,  1， 2， 3】
  //【 4,  5， 6， 7】
  //【 8,  9，10，11】
  //【12, 13，14，15】
  int16x4x2_t b01 = vtrn_s16(a0, a1);
  int16x4x2_t b23 = vtrn_s16(a2, a3);
  //【 0， 4， 2， 6】
  //【 1， 5， 3， 7】
  //【 8，12，10，14】
  //【 9，13，11，15】
  // b01.val[0] = [0， 4， 2， 6],每个数字占16位共64位，
  // 用vreinterpret_s32_s16重新定义为s32数据，即0和4算一个数字，2和6算一个数字。
  int32x2x2_t c02 = vtrn_s32(vreinterpret_s32_s16(b01.val[0]), vreinterpret_s32_s16(b23.val[0]));
  int32x2x2_t c13 = vtrn_s32(vreinterpret_s32_s16(b01.val[1]), vreinterpret_s32_s16(b23.val[1]));
  //【 0， 4， 8，12】
  //【 1， 5， 9，13】
  //【 2， 6，10，14】
  //【 3， 7，11，15】
  // 将类型从s32x2重新解释为s16x4，并保存到目标地址上
  vst1_s16(src, vreinterpret_s16_s32(c02.val[0]));
  vst1_s16(src + 1 * 4, vreinterpret_s16_s32(c13.val[0]));
  vst1_s16(src + 2 * 4, vreinterpret_s16_s32(c02.val[1]));
  vst1_s16(src + 3 * 4, vreinterpret_s16_s32(c13.val[1]));
}

/////////////////////////////////////
void main() {
    
  int height = 4;
  int width = 4;
  int src_int[16];
  float src_float_trn[16];
  float src_float_trn_v2[16];
  float src_float_uzp[16];
  short src_s16[16];
  for (int i = 0; i < 16; i++) {
    src_int[i] = i;
    src_float_trn[i] = i;
    src_float_trn_v2[i] = i;
    src_float_uzp[i] = i;
    src_s16[i] = i;
  }
  
  printf("Before.\n");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d, ", src_int[i * width + j]);
    }
    printf("\n");
  }
  
  printf("After transport (int32) normal:\n");
  TransposeInt32Normal(src_int, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d, ", src_int[i * width + j]);
    }
    printf("\n");
  }
  
  printf("After transport (float) - trn:\n");
  TransposeFp32x4x4_trn(src_float_trn);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%f, ", src_float_trn[i * width + j]);
    }
    printf("\n");
  }
  
  printf("After transport (float) - uzp:\n");
  TransposeFp32x4x4_uzp(src_float_uzp);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%f, ", src_float_uzp[i * width + j]);
    }
    printf("\n");
  }
  
  printf("After transport (s16) - trn:\n");
  TransposeInt16x4x4_trn(src_s16);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d, ", src_s16[i * width + j]);
    }
    printf("\n");
  }
  
  ////////////////////////////////
  // Performance
  ////////////////////////////////
  printf("\n");
  int iter = 1000;
  time_t stime;
  stime = clock();
  for (int i = 0; i < iter; i++) {
    TransposeInt32Normal(src_int, width);
  }
  printf("TransposeInt32Normal -> time: %d.\n", clock() - stime);
  //////////////////////////////////////////////////////////////////////////
  stime = clock();
  for (int i = 0; i < iter; i++) {
    TransposeFp32x4x4_trn(src_float_trn);
  }
  printf("TransposeFp32x4x4_trn v1 -> time: %d.\n", clock() - stime);
  //////////////////////////////////////////////////////////////////////////
  stime = clock();
  for (int i = 0; i < iter; i++) {
    TransposeFp32x4x4_uzp(src_float_uzp);
  }
  printf("TransposeFp32x4x4_uzp -> time: %d.\n", clock() - stime);
  //////////////////////////////////////////////////////////////////////////
  stime = clock();
  for (int i = 0; i < iter; i++) {
    TransposeInt16x4x4_trn(src_s16);
  }
  printf("Transposeint16x4x4_trn -> time: %d.\n", clock() - stime);
  //////////////////////////////////////////////////////////////////////////
  return;
}