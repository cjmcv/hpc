
// 对应 原版 UKernelPBV24
// extern "C" void UKernelPBV30Asm(const int mstart, const int mend,
//                  const int nstart, const int nend,
//                  const int kstart, const int kend,
//                  const float *A, const int lda,
//                  const float *B, const int bid,
//                  float *C, const int ldc);

// armv7: 参数的前4个字（32*4 bits）通过寄存器r0~r3来传递，多出的内容从栈上传递

.globl	UKernelPBV30Asm // -- Begin function UKernelPBV30Asm
.p2align	2
.type	UKernelPBV30Asm,@function

UKernelPBV30Asm:      // @UKernelPBV30Asm
// %bb.0:
	sub	sp, sp, #224            // =224
	sub	w8, w1, #7              // =7   => w8 = w1 - 7 => mend - 7
                                        // kill: def $w0 killed $w0 def $x0
	cmp	w8, w0                  // w0 原始放参数0，即mstart，i=mstart，则for循环中即用于i => i 和 mend - 7 比较
	stp	x28, x27, [sp, #128]    // 16-byte Folded Spill
	stp	x26, x25, [sp, #144]    // 16-byte Folded Spill
	stp	x24, x23, [sp, #160]    // 16-byte Folded Spill
	stp	x22, x21, [sp, #176]    // 16-byte Folded Spill
	stp	x20, x19, [sp, #192]    // 16-byte Folded Spill
	stp	x29, x30, [sp, #208]    // 16-byte Folded Spill
                                        // kill: def $w7 killed $w7 def $x7
	str	x6, [sp, #40]           // 8-byte Folded Spill
	str	x4, [sp, #120]          // 8-byte Folded Spill
	str	x2, [sp, #64]           // 8-byte Folded Spill
	b.le	.LBB23_10           // 如果 w8 小于 w0，即 i < mend - 7，则跳转
// %bb.1:
	ldr	x13, [sp, #120]         // 8-byte Folded Reload
	ldr	x10, [sp, #240]
	sub	w12, w5, #3             // =3
	sub	w11, w3, #3             // =3
	sxtw	x15, w13
	sub	w13, w5, w13
	sxtw	x16, w12
	lsl	w12, w13, #2
	str	x10, [sp, #32]          // 8-byte Folded Spill
	ldr	w10, [sp, #232]
	sxtw	x17, w11
	sub	w11, w12, #16           // =16
	ldr	x12, [sp, #40]          // 8-byte Folded Reload
	ldr	x14, [sp, #64]          // 8-byte Folded Reload
	sxtw	x8, w8
	ldr	w9, [sp, #248]
	str	x8, [sp, #56]           // 8-byte Folded Spill
	nop
	smaddl	x8, w7, w0, x15
	str	w10, [sp, #28]          // 4-byte Folded Spill
	ldr	x10, [sp, #224]
	add	x8, x12, x8, lsl #2
	sxtw	x14, w14
	str	x8, [sp, #112]          // 8-byte Folded Spill
	and	w8, w11, #0xfffffff0
	str	x14, [sp, #16]          // 8-byte Folded Spill
	sxtw	x14, w0
	add	w8, w8, #16             // =16
	str	x14, [sp, #72]          // 8-byte Folded Spill
	sxtw	x14, w7
	str	w8, [sp, #92]           // 4-byte Folded Spill
	sxtw	x8, w9
	stp	x8, x14, [sp]           // 16-byte Folded Spill
	sbfiz	x14, x7, #5, #32
	add	x8, x10, #32            // =32
	str	x14, [sp, #48]          // 8-byte Folded Spill
	str	x8, [sp, #80]           // 8-byte Folded Spill
	stp	x15, x17, [sp, #96]     // 16-byte Folded Spill
	ldr	x8, [sp, #64]           // 8-byte Folded Reload
	cmp	w17, w8
	b.gt	.LBB23_4
.LBB23_2:                               // =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #72]           // 8-byte Folded Reload
	ldr	x9, [sp, #56]           // 8-byte Folded Reload
	add	x8, x8, #8              // =8
	str	x8, [sp, #72]           // 8-byte Folded Spill
	cmp	x8, x9
	ldr	x8, [sp, #48]           // 8-byte Folded Reload
	ldr	x9, [sp, #112]          // 8-byte Folded Reload
	add	x9, x9, x8
	str	x9, [sp, #112]          // 8-byte Folded Spill
	b.ge	.LBB23_10
// %bb.3:                               //   in Loop: Header=BB23_2 Depth=1
	ldr	x8, [sp, #64]           // 8-byte Folded Reload
	cmp	w17, w8
	b.le	.LBB23_2
.LBB23_4:
	ldr	x15, [sp, #72]          // 8-byte Folded Reload
	ldp	x4, x6, [sp]            // 16-byte Folded Reload
	ldr	x5, [sp, #32]           // 8-byte Folded Reload
	add	x9, x15, #1             // =1
	mul	x8, x15, x4
	add	x7, x5, x8, lsl #2
	mul	x8, x9, x4
	add	x19, x5, x8, lsl #2
	ldr	x8, [sp, #40]           // 8-byte Folded Reload
	add	x10, x15, #2            // =2
	add	x13, x15, #5            // =5
	add	x14, x15, #6            // =6
	mul	x17, x10, x4
	mul	x2, x13, x4
	mul	x3, x14, x4
	mul	x9, x9, x6
	mul	x14, x14, x6
	add	x20, x5, x17, lsl #2
	ldr	x17, [sp, #104]         // 8-byte Folded Reload
	add	x23, x5, x2, lsl #2
	add	x26, x8, x9, lsl #2
	add	x9, x8, x14, lsl #2
	ldr	x14, [sp, #16]          // 8-byte Folded Reload
	ldr	w2, [sp, #28]           // 4-byte Folded Reload
	add	x11, x15, #3            // =3
	add	x12, x15, #4            // =4
	add	x15, x15, #7            // =7
	mul	x0, x11, x4
	mul	x1, x12, x4
	mul	x4, x15, x4
	mul	x10, x10, x6
	mul	x11, x11, x6
	mul	x12, x12, x6
	mul	x13, x13, x6
	mul	x15, x15, x6
	add	x21, x5, x0, lsl #2
	add	x22, x5, x1, lsl #2
	add	x24, x5, x3, lsl #2
	add	x25, x5, x4, lsl #2
	add	x27, x8, x10, lsl #2
	add	x28, x8, x11, lsl #2
	add	x29, x8, x12, lsl #2
	add	x30, x8, x13, lsl #2
	add	x11, x8, x15, lsl #2
	b	.LBB23_6
.LBB23_5:                               //   in Loop: Header=BB23_6 Depth=1
	add	x14, x14, #4            // =4
	cmp	x14, x17
	str	q7, [x15]
	str	q6, [x13]
	str	q5, [x6]
	str	q4, [x8]
	str	q3, [x3]
	str	q2, [x10]
	str	q1, [x5]
	str	q0, [x1]
	b.ge	.LBB23_2
.LBB23_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_8 Depth 2
	lsl	x12, x14, #2
	add	x15, x7, x12
	add	x13, x19, x12
	add	x6, x20, x12
	add	x8, x21, x12
	add	x3, x22, x12
	add	x10, x23, x12
	add	x5, x24, x12
	add	x1, x25, x12
	ldr	x12, [sp, #120]         // 8-byte Folded Reload
	ldr	q7, [x15]
	ldr	q6, [x13]
	ldr	q5, [x6]
	ldr	q4, [x8]
	ldr	q3, [x3]
	ldr	q2, [x10]
	ldr	q1, [x5]
	ldr	q0, [x1]
	cmp	w16, w12
	b.le	.LBB23_5
// %bb.7:                               //   in Loop: Header=BB23_6 Depth=1
	ldr	x12, [sp, #80]          // 8-byte Folded Reload
	ldr	x0, [sp, #112]          // 8-byte Folded Reload
	ldr	x4, [sp, #96]           // 8-byte Folded Reload
	add	x12, x12, w2, sxtw #2
.LBB23_8:                               //   Parent Loop BB23_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x17, x4, #2
	ldr	q16, [x0], #16
	ldp	q20, q19, [x12, #-32]
	ldr	q21, [x26, x17]
	ldr	q22, [x27, x17]
	ldr	q23, [x28, x17]
	ldr	q24, [x29, x17]
	ldr	q25, [x30, x17]
	ldr	q26, [x9, x17]
	ldr	q27, [x11, x17]
	ldp	q18, q17, [x12], #64
	fmla	v7.4s, v20.4s, v16.s[0]
	fmla	v6.4s, v20.4s, v21.s[0]
	fmla	v5.4s, v20.4s, v22.s[0]
	fmla	v4.4s, v20.4s, v23.s[0]
	fmla	v3.4s, v20.4s, v24.s[0]
	fmla	v2.4s, v20.4s, v25.s[0]
	fmla	v1.4s, v20.4s, v26.s[0]
	fmla	v0.4s, v20.4s, v27.s[0]
	fmla	v7.4s, v19.4s, v16.s[1]
	fmla	v6.4s, v19.4s, v21.s[1]
	fmla	v5.4s, v19.4s, v22.s[1]
	fmla	v4.4s, v19.4s, v23.s[1]
	fmla	v3.4s, v19.4s, v24.s[1]
	fmla	v2.4s, v19.4s, v25.s[1]
	fmla	v1.4s, v19.4s, v26.s[1]
	fmla	v0.4s, v19.4s, v27.s[1]
	add	x4, x4, #4              // =4
	fmla	v7.4s, v18.4s, v16.s[2]
	fmla	v6.4s, v18.4s, v21.s[2]
	fmla	v5.4s, v18.4s, v22.s[2]
	fmla	v4.4s, v18.4s, v23.s[2]
	fmla	v3.4s, v18.4s, v24.s[2]
	fmla	v2.4s, v18.4s, v25.s[2]
	fmla	v1.4s, v18.4s, v26.s[2]
	fmla	v0.4s, v18.4s, v27.s[2]
	cmp	x4, x16
	fmla	v7.4s, v17.4s, v16.s[3]
	fmla	v6.4s, v17.4s, v21.s[3]
	fmla	v5.4s, v17.4s, v22.s[3]
	fmla	v4.4s, v17.4s, v23.s[3]
	fmla	v3.4s, v17.4s, v24.s[3]
	fmla	v2.4s, v17.4s, v25.s[3]
	fmla	v1.4s, v17.4s, v26.s[3]
	fmla	v0.4s, v17.4s, v27.s[3]
	b.lt	.LBB23_8
// %bb.9:                               //   in Loop: Header=BB23_6 Depth=1
	ldr	w12, [sp, #92]          // 4-byte Folded Reload
	ldr	x17, [sp, #104]         // 8-byte Folded Reload
	add	w2, w12, w2
	b	.LBB23_5
.LBB23_10:
	ldp	x29, x30, [sp, #208]    // 16-byte Folded Reload
	ldp	x20, x19, [sp, #192]    // 16-byte Folded Reload
	ldp	x22, x21, [sp, #176]    // 16-byte Folded Reload
	ldp	x24, x23, [sp, #160]    // 16-byte Folded Reload
	ldp	x26, x25, [sp, #144]    // 16-byte Folded Reload
	ldp	x28, x27, [sp, #128]    // 16-byte Folded Reload
	add	sp, sp, #224            // =224
	ret
.Lfunc_end30:
	.size	UKernelPBV30Asm, .Lfunc_end30-UKernelPBV30Asm
                                        // -- End function


////////////////////////////////////////////////////////////////////

// extern "C" void UKernelPBV31Asm(const int mstart, const int mend,
//                  const int nstart, const int nend,
//                  const int kstart, const int kend,
//                  const float *A, const int lda,
//                  const float *B, const int bid,
//                  float *C, const int ldc);

.globl	UKernelPBV31Asm // -- Begin function UKernelPBV31Asm
.p2align	2
.type	UKernelPBV31Asm,@function

UKernelPBV31Asm:      // @UKernelPBV31Asm
// %bb.0:
	sub	sp, sp, #224            // =224
	sub	w8, w1, #7              // =7
                                        // kill: def $w0 killed $w0 def $x0
	cmp	w8, w0
	stp	x28, x27, [sp, #128]    // 16-byte Folded Spill
	stp	x26, x25, [sp, #144]    // 16-byte Folded Spill
	stp	x24, x23, [sp, #160]    // 16-byte Folded Spill
	stp	x22, x21, [sp, #176]    // 16-byte Folded Spill
	stp	x20, x19, [sp, #192]    // 16-byte Folded Spill
	stp	x29, x30, [sp, #208]    // 16-byte Folded Spill
                                        // kill: def $w7 killed $w7 def $x7
	str	x6, [sp, #40]           // 8-byte Folded Spill
	str	x4, [sp, #120]          // 8-byte Folded Spill
	str	x2, [sp, #64]           // 8-byte Folded Spill
	b.le	.LBB31_10
// %bb.1:
	ldr	x13, [sp, #120]         // 8-byte Folded Reload
	ldr	x10, [sp, #240]
	sub	w12, w5, #3             // =3
	sub	w11, w3, #3             // =3
	sxtw	x15, w13
	sub	w13, w5, w13
	sxtw	x16, w12
	lsl	w12, w13, #2
	str	x10, [sp, #32]          // 8-byte Folded Spill
	ldr	w10, [sp, #232]
	sxtw	x17, w11
	sub	w11, w12, #16           // =16
	ldr	x12, [sp, #40]          // 8-byte Folded Reload
	ldr	x14, [sp, #64]          // 8-byte Folded Reload
	sxtw	x8, w8
	ldr	w9, [sp, #248]
	str	x8, [sp, #56]           // 8-byte Folded Spill
	nop
	smaddl	x8, w7, w0, x15
	str	w10, [sp, #28]          // 4-byte Folded Spill
	ldr	x10, [sp, #224]
	add	x8, x12, x8, lsl #2
	sxtw	x14, w14
	str	x8, [sp, #112]          // 8-byte Folded Spill
	and	w8, w11, #0xfffffff0
	str	x14, [sp, #16]          // 8-byte Folded Spill
	sxtw	x14, w0
	add	w8, w8, #16             // =16
	str	x14, [sp, #72]          // 8-byte Folded Spill
	sxtw	x14, w7
	str	w8, [sp, #92]           // 4-byte Folded Spill
	sxtw	x8, w9
	stp	x8, x14, [sp]           // 16-byte Folded Spill
	sbfiz	x14, x7, #5, #32
	add	x8, x10, #32            // =32
	str	x14, [sp, #48]          // 8-byte Folded Spill
	str	x8, [sp, #80]           // 8-byte Folded Spill
	stp	x15, x17, [sp, #96]     // 16-byte Folded Spill
	ldr	x8, [sp, #64]           // 8-byte Folded Reload
	cmp	w17, w8
	b.gt	.LBB31_4
.LBB31_2:                               // =>This Inner Loop Header: Depth=1
	ldr	x8, [sp, #72]           // 8-byte Folded Reload
	ldr	x9, [sp, #56]           // 8-byte Folded Reload
	add	x8, x8, #8              // =8
	str	x8, [sp, #72]           // 8-byte Folded Spill
	cmp	x8, x9
	ldr	x8, [sp, #48]           // 8-byte Folded Reload
	ldr	x9, [sp, #112]          // 8-byte Folded Reload
	add	x9, x9, x8
	str	x9, [sp, #112]          // 8-byte Folded Spill
	b.ge	.LBB31_10
// %bb.3:                               //   in Loop: Header=BB23_2 Depth=1
	ldr	x8, [sp, #64]           // 8-byte Folded Reload
	cmp	w17, w8
	b.le	.LBB31_2
.LBB31_4:
	ldr	x15, [sp, #72]          // 8-byte Folded Reload
	ldp	x4, x6, [sp]            // 16-byte Folded Reload
	ldr	x5, [sp, #32]           // 8-byte Folded Reload
	add	x9, x15, #1             // =1
	mul	x8, x15, x4
	add	x7, x5, x8, lsl #2
	mul	x8, x9, x4
	add	x19, x5, x8, lsl #2
	ldr	x8, [sp, #40]           // 8-byte Folded Reload
	add	x10, x15, #2            // =2
	add	x13, x15, #5            // =5
	add	x14, x15, #6            // =6
	mul	x17, x10, x4
	mul	x2, x13, x4
	mul	x3, x14, x4
	mul	x9, x9, x6
	mul	x14, x14, x6
	add	x20, x5, x17, lsl #2
	ldr	x17, [sp, #104]         // 8-byte Folded Reload
	add	x23, x5, x2, lsl #2
	add	x26, x8, x9, lsl #2
	add	x9, x8, x14, lsl #2
	ldr	x14, [sp, #16]          // 8-byte Folded Reload
	ldr	w2, [sp, #28]           // 4-byte Folded Reload
	add	x11, x15, #3            // =3
	add	x12, x15, #4            // =4
	add	x15, x15, #7            // =7
	mul	x0, x11, x4
	mul	x1, x12, x4
	mul	x4, x15, x4
	mul	x10, x10, x6
	mul	x11, x11, x6
	mul	x12, x12, x6
	mul	x13, x13, x6
	mul	x15, x15, x6
	add	x21, x5, x0, lsl #2
	add	x22, x5, x1, lsl #2
	add	x24, x5, x3, lsl #2
	add	x25, x5, x4, lsl #2
	add	x27, x8, x10, lsl #2
	add	x28, x8, x11, lsl #2
	add	x29, x8, x12, lsl #2
	add	x30, x8, x13, lsl #2
	add	x11, x8, x15, lsl #2
	b	.LBB31_6
.LBB31_5:                               //   in Loop: Header=BB23_6 Depth=1
	add	x14, x14, #4            // =4
	cmp	x14, x17
	str	q7, [x15]
	str	q6, [x13]
	str	q5, [x6]
	str	q4, [x8]
	str	q3, [x3]
	str	q2, [x10]
	str	q1, [x5]
	str	q0, [x1]
	b.ge	.LBB31_2
.LBB31_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_8 Depth 2
	lsl	x12, x14, #2
	add	x15, x7, x12
	add	x13, x19, x12
	add	x6, x20, x12
	add	x8, x21, x12
	add	x3, x22, x12
	add	x10, x23, x12
	add	x5, x24, x12
	add	x1, x25, x12
	ldr	x12, [sp, #120]         // 8-byte Folded Reload
	ldr	q7, [x15]
	ldr	q6, [x13]
	ldr	q5, [x6]
	ldr	q4, [x8]
	ldr	q3, [x3]
	ldr	q2, [x10]
	ldr	q1, [x5]
	ldr	q0, [x1]
	cmp	w16, w12
	b.le	.LBB31_5
// %bb.7:                               //   in Loop: Header=BB23_6 Depth=1
	ldr	x12, [sp, #80]          // 8-byte Folded Reload
	ldr	x0, [sp, #112]          // 8-byte Folded Reload
	ldr	x4, [sp, #96]           // 8-byte Folded Reload
	add	x12, x12, w2, sxtw #2
.LBB31_8:                               //   Parent Loop BB23_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x17, x4, #2              // (i+1) * lda + k
	ldr	q16, [x0], #16           // float32x4_t va0 = vld1q_f32(A + i * lda + k);
	ldp	q20, q19, [x12, #-32]    // float32x4_t vb0j0 = vld1q_f32(B + lbid);  // ldp 一次加载两个
	ldr	q21, [x26, x17]          // float32x4_t va1 = vld1q_f32(A + (i+1) * lda + k); ldr用的是q，加载到q21，neon计算时对应v21
	ldr	q22, [x27, x17]          // float32x4_t va2 = vld1q_f32(A + (i+2) * lda + k);
	ldr	q23, [x28, x17]
	ldr	q24, [x29, x17]
	ldr	q25, [x30, x17]
	ldr	q26, [x9, x17]
	ldr	q27, [x11, x17]
	ldp	q18, q17, [x12], #64     // float32x4_t vb2j0 = vld1q_f32(B + lbid + 8); 
	fmla	v7.4s, v20.4s, v16.s[0]
	fmla	v6.4s, v20.4s, v21.s[0]
	fmla	v5.4s, v20.4s, v22.s[0]
	fmla	v4.4s, v20.4s, v23.s[0]
	fmla	v3.4s, v20.4s, v24.s[0]
	fmla	v2.4s, v20.4s, v25.s[0]
	fmla	v1.4s, v20.4s, v26.s[0]
	fmla	v0.4s, v20.4s, v27.s[0]
	fmla	v7.4s, v19.4s, v16.s[1]
	fmla	v6.4s, v19.4s, v21.s[1]
	fmla	v5.4s, v19.4s, v22.s[1]
	fmla	v4.4s, v19.4s, v23.s[1]
	fmla	v3.4s, v19.4s, v24.s[1]
	fmla	v2.4s, v19.4s, v25.s[1]
	fmla	v1.4s, v19.4s, v26.s[1]
	fmla	v0.4s, v19.4s, v27.s[1]
	add	x4, x4, #4              // =4,  k += 4
	fmla	v7.4s, v18.4s, v16.s[2]
	fmla	v6.4s, v18.4s, v21.s[2]
	fmla	v5.4s, v18.4s, v22.s[2]
	fmla	v4.4s, v18.4s, v23.s[2]
	fmla	v3.4s, v18.4s, v24.s[2]
	fmla	v2.4s, v18.4s, v25.s[2]
	fmla	v1.4s, v18.4s, v26.s[2]
	fmla	v0.4s, v18.4s, v27.s[2]
	cmp	x4, x16
	fmla	v7.4s, v17.4s, v16.s[3]
	fmla	v6.4s, v17.4s, v21.s[3]
	fmla	v5.4s, v17.4s, v22.s[3]
	fmla	v4.4s, v17.4s, v23.s[3]
	fmla	v3.4s, v17.4s, v24.s[3]
	fmla	v2.4s, v17.4s, v25.s[3]
	fmla	v1.4s, v17.4s, v26.s[3]
	fmla	v0.4s, v17.4s, v27.s[3]
	b.lt	.LBB31_8
// %bb.9:                               //   in Loop: Header=BB23_6 Depth=1
	ldr	w12, [sp, #92]          // 4-byte Folded Reload
	ldr	x17, [sp, #104]         // 8-byte Folded Reload
	add	w2, w12, w2
	b	.LBB31_5
.LBB31_10:
	ldp	x29, x30, [sp, #208]    // 16-byte Folded Reload
	ldp	x20, x19, [sp, #192]    // 16-byte Folded Reload
	ldp	x22, x21, [sp, #176]    // 16-byte Folded Reload
	ldp	x24, x23, [sp, #160]    // 16-byte Folded Reload
	ldp	x26, x25, [sp, #144]    // 16-byte Folded Reload
	ldp	x28, x27, [sp, #128]    // 16-byte Folded Reload
	add	sp, sp, #224            // =224
	ret
.Lfunc_end31:
	.size	UKernelPBV31Asm, .Lfunc_end31-UKernelPBV31Asm
                                        // -- End function