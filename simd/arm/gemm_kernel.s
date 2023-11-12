
// 对应 原版 UKernelPBV25
// extern "C" void UKernelPBV25Asm(const int mstart, const int mend,
//                  const int nstart, const int nend,
//                  const int kstart, const int kend,
//                  const float *A, const int lda,
//                  const float *B, const int bid,
//                  float *C, const int ldc);

// armv7: 参数的前4个字（32*4 bits）通过寄存器r0~r3来传递，多出的内容从栈上传递

	.globl	UKernelPBV25Asm // -- Begin function UKernelPBV25Asm
	.p2align	2
	.type	UKernelPBV25Asm,@function
	
UKernelPBV25Asm:      // @UKernelPBV25Asm
// %bb.0:
	sub	sp, sp, #192            // =192
	sub	w8, w1, #7              // =7
                                        // kill: def $w0 killed $w0 def $x0
	cmp	w8, w0
	stp	x28, x27, [sp, #96]     // 16-byte Folded Spill
	stp	x26, x25, [sp, #112]    // 16-byte Folded Spill
	stp	x24, x23, [sp, #128]    // 16-byte Folded Spill
	stp	x22, x21, [sp, #144]    // 16-byte Folded Spill
	stp	x20, x19, [sp, #160]    // 16-byte Folded Spill
	stp	x29, x30, [sp, #176]    // 16-byte Folded Spill
                                        // kill: def $w7 killed $w7 def $x7
	stp	x6, x4, [sp, #24]       // 16-byte Folded Spill
	str	x2, [sp, #88]           // 8-byte Folded Spill
	b.le	.LBB31_10
// %bb.1:
	ldr	w8, [sp, #200]
	ldr	x13, [sp, #32]          // 8-byte Folded Reload
	sub	w14, w5, #3             // =3
	sxtw	x16, w7
	sub	w1, w1, w0
	ldr	w9, [sp, #216]
	str	w8, [sp, #20]           // 4-byte Folded Spill
	sub	w12, w3, #3             // =3
	ldr	x8, [sp, #88]           // 8-byte Folded Reload
	sub	w15, w5, w13
	str	x16, [sp, #8]           // 8-byte Folded Spill
	sbfiz	x16, x7, #5, #32
	sxtw	x17, w14
	sub	w14, w1, #8             // =8
	str	x16, [sp, #80]          // 8-byte Folded Spill
	sxtw	x16, w12
	lsl	w12, w15, #2
	lsr	w14, w14, #3
	ldr	x2, [sp, #208]
	ldr	x11, [sp, #192]
	sub	w12, w12, #16           // =16
	str	w14, [sp, #76]          // 4-byte Folded Spill
	ldr	x14, [sp, #24]          // 8-byte Folded Reload
	and	w12, w12, #0xfffffff0
	sxtw	x3, w8
	sxtw	x8, w13
	add	w5, w12, #16            // =16
	sxtw	x12, w9
	smaddl	x15, w7, w0, x8
	stp	x12, x3, [sp, #40]      // 16-byte Folded Spill
	nop
	smaddl	x12, w9, w0, x3
	mov	w10, wzr
	sxtw	x13, w0
	add	x6, x14, x15, lsl #2
	sbfiz	x9, x9, #5, #32
	add	x19, x11, #32           // =32
	add	x20, x2, x12, lsl #2
	movi	v0.2d, #0000000000000000
	stp	x2, x9, [sp, #56]       // 16-byte Folded Spill
	ldr	x9, [sp, #88]           // 8-byte Folded Reload
	cmp	w16, w9
	b.gt	.LBB31_4
.LBB31_2:                               // =>This Inner Loop Header: Depth=1
	ldr	w9, [sp, #76]           // 4-byte Folded Reload
	add	x13, x13, #8            // =8
	cmp	w10, w9
	ldr	x9, [sp, #64]           // 8-byte Folded Reload
	add	w10, w10, #1            // =1
	add	x20, x20, x9
	ldr	x9, [sp, #80]           // 8-byte Folded Reload
	add	x6, x6, x9
	b.eq	.LBB31_10
// %bb.3:                               //   in Loop: Header=BB31_2 Depth=1
	ldr	x9, [sp, #88]           // 8-byte Folded Reload
	cmp	w16, w9
	b.le	.LBB31_2
.LBB31_4:
	ldp	x2, x21, [sp, #32]      // 16-byte Folded Reload
	ldr	x28, [sp, #56]          // 8-byte Folded Reload
	add	x0, x13, #2             // =2
	add	x1, x13, #1             // =1
	mul	x3, x0, x21
	add	x22, x28, x3, lsl #2
	ldr	x3, [sp, #48]           // 8-byte Folded Reload
	add	x15, x13, #3            // =3
	add	x14, x13, #4            // =4
	add	x12, x13, #5            // =5
	add	x11, x13, #6            // =6
	add	x9, x13, #7             // =7
	cmp	w17, w2
	mul	x2, x1, x21
	mul	x4, x15, x21
	mul	x7, x14, x21
	mul	x25, x12, x21
	mul	x26, x11, x21
	mul	x27, x9, x21
	add	x21, x28, x2, lsl #2
	add	x23, x28, x4, lsl #2
	add	x24, x28, x7, lsl #2
	add	x25, x28, x25, lsl #2
	add	x26, x28, x26, lsl #2
	add	x27, x28, x27, lsl #2
	mov	x2, x20
	b.le	.LBB31_9
// %bb.5:
	ldr	x3, [sp, #8]            // 8-byte Folded Reload
	mul	x4, x1, x3
	ldr	x1, [sp, #24]           // 8-byte Folded Reload
	mul	x0, x0, x3
	mul	x15, x15, x3
	mul	x14, x14, x3
	mul	x12, x12, x3
	mul	x2, x11, x3
	mul	x3, x9, x3
	ldr	x9, [sp, #40]           // 8-byte Folded Reload
	add	x29, x1, x0, lsl #2
	add	x11, x1, x12, lsl #2
	ldp	x0, x12, [sp, #48]      // 16-byte Folded Reload
	mul	x7, x13, x9
	add	x28, x1, x4, lsl #2
	add	x30, x1, x15, lsl #2
	add	x9, x1, x14, lsl #2
	add	x4, x1, x2, lsl #2
	add	x15, x1, x3, lsl #2
	ldr	w1, [sp, #20]           // 4-byte Folded Reload
	add	x2, x12, x7, lsl #2
.LBB31_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB31_7 Depth 2
	add	x14, x19, w1, sxtw #2
	movi	v1.2d, #0000000000000000
	mov	x3, x6
	mov	x12, x8
	movi	v2.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	movi	v16.2d, #0000000000000000

.LBB31_7:                               //   Parent Loop BB31_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x12, #2
	ldr	q17, [x3], #16
	ldp	q21, q20, [x14, #-32]
	ldr	q22, [x28, x7]
	ldr	q23, [x29, x7]
	ldr	q24, [x30, x7]
	ldr	q25, [x9, x7]
	ldr	q26, [x11, x7]
	ldr	q27, [x4, x7]
	ldr	q28, [x15, x7]
	ldp	q19, q18, [x14], #64
	prfm	PLDL1KEEP, [x14, #640]   // 额外手动增加的
	fmla	v1.4s, v21.4s, v17.s[0]
	fmla	v2.4s, v21.4s, v22.s[0]
	fmla	v3.4s, v21.4s, v23.s[0]
	fmla	v4.4s, v21.4s, v24.s[0]
	fmla	v5.4s, v21.4s, v25.s[0]
	fmla	v6.4s, v21.4s, v26.s[0]
	fmla	v7.4s, v21.4s, v27.s[0]
	fmla	v16.4s, v21.4s, v28.s[0]
	fmla	v1.4s, v20.4s, v17.s[1]
	fmla	v2.4s, v20.4s, v22.s[1]
	fmla	v3.4s, v20.4s, v23.s[1]
	fmla	v4.4s, v20.4s, v24.s[1]
	fmla	v5.4s, v20.4s, v25.s[1]
	fmla	v6.4s, v20.4s, v26.s[1]
	fmla	v7.4s, v20.4s, v27.s[1]
	fmla	v16.4s, v20.4s, v28.s[1]
	add	x12, x12, #4            // =4
	fmla	v1.4s, v19.4s, v17.s[2]
	fmla	v2.4s, v19.4s, v22.s[2]
	fmla	v3.4s, v19.4s, v23.s[2]
	fmla	v4.4s, v19.4s, v24.s[2]
	fmla	v5.4s, v19.4s, v25.s[2]
	fmla	v6.4s, v19.4s, v26.s[2]
	fmla	v7.4s, v19.4s, v27.s[2]
	fmla	v16.4s, v19.4s, v28.s[2]
	cmp	x12, x17
	fmla	v1.4s, v18.4s, v17.s[3]
	fmla	v2.4s, v18.4s, v22.s[3]
	fmla	v3.4s, v18.4s, v23.s[3]
	fmla	v4.4s, v18.4s, v24.s[3]
	fmla	v5.4s, v18.4s, v25.s[3]
	fmla	v6.4s, v18.4s, v26.s[3]
	fmla	v7.4s, v18.4s, v27.s[3]
	fmla	v16.4s, v18.4s, v28.s[3]
	b.lt	.LBB31_7
// %bb.8:                               //   in Loop: Header=BB31_6 Depth=1
	lsl	x12, x0, #2
	add	x0, x0, #4              // =4
	cmp	x0, x16
	add	w1, w1, w5
	str	q1, [x2, x12]
	str	q2, [x21, x12]
	str	q3, [x22, x12]
	str	q4, [x23, x12]
	str	q5, [x24, x12]
	str	q6, [x25, x12]
	str	q7, [x26, x12]
	str	q16, [x27, x12]
	b.lt	.LBB31_6
	b	.LBB31_2
.LBB31_9:                               // =>This Inner Loop Header: Depth=1
	lsl	x9, x3, #2
	add	x3, x3, #4              // =4
	str	q0, [x2], #16
	cmp	x3, x16
	str	q0, [x21, x9]
	str	q0, [x22, x9]
	str	q0, [x23, x9]
	str	q0, [x24, x9]
	str	q0, [x25, x9]
	str	q0, [x26, x9]
	str	q0, [x27, x9]
	b.lt	.LBB31_9
	b	.LBB31_2
.LBB31_10:
	ldp	x29, x30, [sp, #176]    // 16-byte Folded Reload
	ldp	x20, x19, [sp, #160]    // 16-byte Folded Reload
	ldp	x22, x21, [sp, #144]    // 16-byte Folded Reload
	ldp	x24, x23, [sp, #128]    // 16-byte Folded Reload
	ldp	x26, x25, [sp, #112]    // 16-byte Folded Reload
	ldp	x28, x27, [sp, #96]     // 16-byte Folded Reload
	add	sp, sp, #192            // =192
	ret
.Lfunc_end31:
	.size	UKernelPBV25Asm, .Lfunc_end31-UKernelPBV25Asm
                                        // -- End function

////////////////////////////////////////////////////////////////

	.globl	UKernelPABV30Asm // -- Begin function UKernelPABV30Asm
	.p2align	2
	.type	UKernelPABV30Asm,@function
UKernelPABV30Asm:     // @UKernelPABV30Asm
// %bb.0:
	sub	sp, sp, #112            // =112
	sub	w8, w1, #7              // =7
                                        // kill: def $w0 killed $w0 def $x0
	cmp	w8, w0
	stp	x28, x27, [sp, #16]     // 16-byte Folded Spill
	stp	x26, x25, [sp, #32]     // 16-byte Folded Spill
	stp	x24, x23, [sp, #48]     // 16-byte Folded Spill
	stp	x22, x21, [sp, #64]     // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]     // 16-byte Folded Spill
	stp	x29, x30, [sp, #96]     // 16-byte Folded Spill
                                        // kill: def $w2 killed $w2 def $x2
	b.le	.LBB29_10
// %bb.1:
	mov	w16, w4
	ldr	w9, [sp, #136]
	ldr	w11, [sp, #120]
	ldr	x8, [sp, #128]
	ldr	x4, [sp, #112]
	sub	w17, w3, #3             // =3
	sub	w3, w5, w16
	lsl	w3, w3, #2
	sxtw	x12, w2
	sub	w1, w1, w0
	sub	w3, w3, #16             // =16
	str	w11, [sp, #12]          // 4-byte Folded Spill
	sub	w11, w5, #3             // =3
	mul	w13, w5, w0
	lsl	w14, w5, #3
	add	x5, x6, #64             // =64
	sub	w1, w1, #8              // =8
	and	w3, w3, #0xfffffff0
	smaddl	x7, w9, w0, x12
	mov	w10, wzr
	sxtw	x15, w0
	str	x5, [sp]                // 8-byte Folded Spill
	sxtw	x17, w17
	lsr	w1, w1, #3
	add	w3, w3, #16             // =16
	sxtw	x5, w9
	sbfiz	x0, x9, #5, #32
	add	x6, x4, #32             // =32
	add	x7, x8, x7, lsl #2
	movi	v0.2d, #0000000000000000
	cmp	w17, w2
	b.gt	.LBB29_4
.LBB29_2:                               // =>This Inner Loop Header: Depth=1
	add	x15, x15, #8            // =8
	cmp	w10, w1
	add	w10, w10, #1            // =1
	add	w13, w13, w14
	add	x7, x7, x0
	b.eq	.LBB29_10
// %bb.3:                               //   in Loop: Header=BB29_2 Depth=1
	cmp	w17, w2
	b.le	.LBB29_2
.LBB29_4:
	add	x9, x15, #1             // =1
	add	x4, x15, #2             // =2
	add	x19, x15, #3            // =3
	add	x20, x15, #4            // =4
	add	x21, x15, #5            // =5
	add	x22, x15, #6            // =6
	add	x23, x15, #7            // =7
	mul	x9, x9, x5
	mul	x4, x4, x5
	mul	x24, x19, x5
	mul	x25, x20, x5
	mul	x26, x21, x5
	mul	x27, x22, x5
	mul	x28, x23, x5
	cmp	w11, w16
	add	x19, x8, x9, lsl #2
	add	x20, x8, x4, lsl #2
	add	x21, x8, x24, lsl #2
	add	x22, x8, x25, lsl #2
	add	x23, x8, x26, lsl #2
	add	x24, x8, x27, lsl #2
	add	x25, x8, x28, lsl #2
	mov	x9, x7
	mov	x4, x12
	b.le	.LBB29_9
// %bb.5:
	ldr	x9, [sp]                // 8-byte Folded Reload
	ldr	w28, [sp, #12]          // 4-byte Folded Reload
	mul	x4, x15, x5
	add	x27, x8, x4, lsl #2
	add	x9, x9, w13, sxtw #2
	mov	x29, x12
.LBB29_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB29_7 Depth 2
	add	x30, x6, w28, sxtw #2
	movi	v1.2d, #0000000000000000
	mov	x26, x9
	mov	w4, w16
	movi	v2.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	movi	v16.2d, #0000000000000000
.LBB29_7:                               //   Parent Loop BB29_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q17, q18, [x26, #-64]
	ldp	q19, q20, [x26, #-32]
	ldp	q21, q22, [x26]
	ldp	q23, q24, [x26, #32]
	ldp	q28, q27, [x30, #-32]
	ldp	q26, q25, [x30], #64
	prfm	PLDL1KEEP, [x26, #256]       // 手动添加
	prfm	PLDL1KEEP, [x30, #640]       // 手动添加
	add	w4, w4, #4              // =4
	cmp	w4, w11
	fmla	v1.4s, v28.4s, v17.s[0]
	fmla	v2.4s, v28.4s, v18.s[0]
	fmla	v3.4s, v28.4s, v19.s[0]
	fmla	v4.4s, v28.4s, v20.s[0]
	fmla	v5.4s, v28.4s, v21.s[0]
	fmla	v6.4s, v28.4s, v22.s[0]
	fmla	v7.4s, v28.4s, v23.s[0]
	fmla	v16.4s, v28.4s, v24.s[0]
	fmla	v1.4s, v27.4s, v17.s[1]
	fmla	v2.4s, v27.4s, v18.s[1]
	fmla	v3.4s, v27.4s, v19.s[1]
	fmla	v4.4s, v27.4s, v20.s[1]
	fmla	v5.4s, v27.4s, v21.s[1]
	fmla	v6.4s, v27.4s, v22.s[1]
	fmla	v7.4s, v27.4s, v23.s[1]
	fmla	v16.4s, v27.4s, v24.s[1]
	fmla	v1.4s, v26.4s, v17.s[2]
	fmla	v2.4s, v26.4s, v18.s[2]
	fmla	v3.4s, v26.4s, v19.s[2]
	fmla	v4.4s, v26.4s, v20.s[2]
	fmla	v5.4s, v26.4s, v21.s[2]
	fmla	v6.4s, v26.4s, v22.s[2]
	fmla	v7.4s, v26.4s, v23.s[2]
	fmla	v16.4s, v26.4s, v24.s[2]
	fmla	v1.4s, v25.4s, v17.s[3]
	fmla	v2.4s, v25.4s, v18.s[3]
	fmla	v3.4s, v25.4s, v19.s[3]
	fmla	v4.4s, v25.4s, v20.s[3]
	fmla	v5.4s, v25.4s, v21.s[3]
	fmla	v6.4s, v25.4s, v22.s[3]
	fmla	v7.4s, v25.4s, v23.s[3]
	fmla	v16.4s, v25.4s, v24.s[3]
	add	x26, x26, #128          // =128
	b.lt	.LBB29_7
// %bb.8:                               //   in Loop: Header=BB29_6 Depth=1
	lsl	x4, x29, #2
	add	x29, x29, #4            // =4
	cmp	x29, x17
	add	w28, w28, w3
	str	q1, [x27, x4]
	str	q2, [x19, x4]
	str	q3, [x20, x4]
	str	q4, [x21, x4]
	str	q5, [x22, x4]
	str	q6, [x23, x4]
	str	q7, [x24, x4]
	str	q16, [x25, x4]
	b.lt	.LBB29_6
	b	.LBB29_2
.LBB29_9:                               // =>This Inner Loop Header: Depth=1
	lsl	x26, x4, #2
	add	x4, x4, #4              // =4
	str	q0, [x9], #16
	cmp	x4, x17
	str	q0, [x19, x26]
	str	q0, [x20, x26]
	str	q0, [x21, x26]
	str	q0, [x22, x26]
	str	q0, [x23, x26]
	str	q0, [x24, x26]
	str	q0, [x25, x26]
	b.lt	.LBB29_9
	b	.LBB29_2
.LBB29_10:
	ldp	x29, x30, [sp, #96]     // 16-byte Folded Reload
	ldp	x20, x19, [sp, #80]     // 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]     // 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]     // 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]     // 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]     // 16-byte Folded Reload
	add	sp, sp, #112            // =112
	ret
.Lfunc_end29:
	.size	UKernelPABV30Asm, .Lfunc_end29-UKernelPABV30Asm
                                        // -- End function