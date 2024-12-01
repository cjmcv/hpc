// 通过 aarch64-linux-android21-clang++ -O3 -S -c gemm.cpp 生成
// 以Begin function为起点，End function为终点, 将原来的函数名_Z10UKernelV13iiiiiiPKfiS0_iPfi 批量替换成UKernelV13Asm
// 在cpp处通过声明，进行调用
// extern "C" void UKernelV13Asm(const int mstart, const int mend,
//                  const int nstart, const int nend,
//                  const int kstart, const int kend,
//                  const float *A, const int lda,
//                  const float *B, const int bid,
//                  float *C, const int ldc);

// armv7: 参数的前4个字（32*4 bits）通过寄存器r0~r3来传递，多出的内容从栈上传递

	.globl	UKernelV13Asm // -- Begin function UKernelV13Asm
	.p2align	2
	.type	UKernelV13Asm,@function
UKernelV13Asm:        // @UKernelV13Asm
// %bb.0:
	sub	sp, sp, #496            // =496
	ldr	w9, [sp, #504]
	ldr	w11, [sp, #520]
	ldr	x8, [sp, #512]
	ldr	x10, [sp, #496]
                                        // kill: def $w1 killed $w1 def $x1
	str	w9, [sp, #112]          // 4-byte Folded Spill
	sub	w9, w1, #7              // =7
                                        // kill: def $w0 killed $w0 def $x0
	cmp	w9, w0
	stp	d15, d14, [sp, #336]    // 16-byte Folded Spill
	stp	d13, d12, [sp, #352]    // 16-byte Folded Spill
	stp	d11, d10, [sp, #368]    // 16-byte Folded Spill
	stp	d9, d8, [sp, #384]      // 16-byte Folded Spill
	stp	x28, x27, [sp, #400]    // 16-byte Folded Spill
	stp	x26, x25, [sp, #416]    // 16-byte Folded Spill
	stp	x24, x23, [sp, #432]    // 16-byte Folded Spill
	stp	x22, x21, [sp, #448]    // 16-byte Folded Spill
	stp	x20, x19, [sp, #464]    // 16-byte Folded Spill
	stp	x29, x30, [sp, #480]    // 16-byte Folded Spill
                                        // kill: def $w7 killed $w7 def $x7
                                        // kill: def $w5 killed $w5 def $x5
                                        // kill: def $w4 killed $w4 def $x4
                                        // kill: def $w3 killed $w3 def $x3
                                        // kill: def $w2 killed $w2 def $x2
	b.le	.LBB19_39
// %bb.1:
	sxtw	x9, w9
	str	x9, [sp, #96]           // 8-byte Folded Spill
	sxtw	x9, w3
	str	x9, [sp, #232]          // 8-byte Folded Spill
	smull	x9, w7, w0
	str	x9, [sp, #208]          // 8-byte Folded Spill
	sbfiz	x9, x7, #5, #32
	str	x9, [sp, #88]           // 8-byte Folded Spill
	add	x9, x10, #64            // =64
	str	x9, [sp, #136]          // 8-byte Folded Spill
	madd	w9, w11, w0, w2
	str	w9, [sp, #116]          // 4-byte Folded Spill
	lsl	w9, w11, #3
	str	w9, [sp, #84]           // 4-byte Folded Spill
	sbfiz	x9, x7, #3, #32
	sub	w13, w5, #3             // =3
	sxtw	x15, w11
	stp	x15, x9, [sp, #64]      // 16-byte Folded Spill
	add	x9, x10, w4, sxtw #2
	sub	w12, w3, #7             // =7
	sub	w14, w5, w4
	sxtw	x15, w2
	stp	x7, x9, [sp, #24]       // 16-byte Folded Spill
	sxtw	x9, w13
	sxtw	x20, w4
	str	x15, [sp, #40]          // 8-byte Folded Spill
	sxtw	x19, w0
	sxtw	x15, w5
	str	x9, [sp, #192]          // 8-byte Folded Spill
	sub	w9, w14, #4             // =4
	sxtw	x12, w12
	sbfiz	x13, x14, #2, #32
	str	x1, [sp, #16]           // 8-byte Folded Spill
	sxtw	x21, w7
	sbfiz	x27, x7, #2, #32
	stp	x12, x15, [sp, #320]    // 16-byte Folded Spill
	nop
	smaddl	x12, w7, w0, x20
	str	x13, [sp, #224]         // 8-byte Folded Spill
	add	x13, x19, #1            // =1
	add	x14, x19, #2            // =2
	add	x16, x19, #3            // =3
	add	x17, x19, #4            // =4
	add	x0, x19, #5             // =5
	add	x1, x19, #6             // =6
	str	x19, [sp, #120]         // 8-byte Folded Spill
	add	x19, x19, #7            // =7
	lsl	w7, w9, #3
	sub	x15, x15, x20
	and	w9, w9, #0xfffffffc
	add	x30, x6, x12, lsl #2
	madd	x12, x13, x21, x20
	madd	x13, x14, x21, x20
	madd	x14, x16, x21, x20
	madd	x16, x17, x21, x20
	madd	x0, x0, x21, x20
	madd	x1, x1, x21, x20
	stp	x20, x4, [sp, #144]     // 16-byte Folded Spill
	stp	x21, x3, [sp, #48]      // 16-byte Folded Spill
	nop
	madd	x21, x19, x21, x20
	and	w20, w7, #0xffffffe0
	add	w19, w9, w4
	add	x7, x6, x12, lsl #2
	add	w12, w20, #32           // =32
	add	x28, x6, x13, lsl #2
	add	x17, x6, x14, lsl #2
	add	x25, x6, x16, lsl #2
	add	x26, x6, x0, lsl #2
	add	x9, x6, x1, lsl #2
	str	w12, [sp, #132]         // 4-byte Folded Spill
	add	w12, w19, #4            // =4
	add	x16, x6, x21, lsl #2
	str	w12, [sp, #128]         // 4-byte Folded Spill
	str	x6, [sp, #200]          // 8-byte Folded Spill
	str	x5, [sp, #160]          // 8-byte Folded Spill
	str	x2, [sp, #104]          // 8-byte Folded Spill
	b	.LBB19_3
.LBB19_2:                               //   in Loop: Header=BB19_3 Depth=1
	ldr	w13, [sp, #116]         // 4-byte Folded Reload
	ldr	w0, [sp, #84]           // 4-byte Folded Reload
	ldr	x12, [sp, #120]         // 8-byte Folded Reload
	ldr	x14, [sp, #88]          // 8-byte Folded Reload
	add	w13, w13, w0
	str	w13, [sp, #116]         // 4-byte Folded Spill
	ldr	x13, [sp, #208]         // 8-byte Folded Reload
	ldr	x0, [sp, #72]           // 8-byte Folded Reload
	add	x12, x12, #8            // =8
	add	x30, x30, x14
	add	x7, x7, x14
	add	x13, x13, x0
	str	x13, [sp, #208]         // 8-byte Folded Spill
	ldp	x13, x2, [sp, #96]      // 16-byte Folded Reload
	add	x28, x28, x14
	add	x17, x17, x14
	add	x25, x25, x14
	add	x26, x26, x14
	add	x9, x9, x14
	cmp	x12, x13
	add	x16, x16, x14
	str	x12, [sp, #120]         // 8-byte Folded Spill
	b.ge	.LBB19_38
.LBB19_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_8 Depth 2
                                        //     Child Loop BB19_13 Depth 2
                                        //       Child Loop BB19_14 Depth 3
                                        //     Child Loop BB19_21 Depth 2
                                        //       Child Loop BB19_22 Depth 3
                                        //       Child Loop BB19_24 Depth 3
                                        //       Child Loop BB19_26 Depth 3
                                        //       Child Loop BB19_28 Depth 3
                                        //       Child Loop BB19_30 Depth 3
                                        //       Child Loop BB19_32 Depth 3
                                        //       Child Loop BB19_34 Depth 3
                                        //       Child Loop BB19_36 Depth 3
	ldr	x12, [sp, #320]         // 8-byte Folded Reload
	ldr	w24, [sp, #112]         // 4-byte Folded Reload
	mov	w19, w2
	cmp	w12, w2
	b.le	.LBB19_18
// %bb.4:                               //   in Loop: Header=BB19_3 Depth=1
	ldr	x19, [sp, #120]         // 8-byte Folded Reload
	ldr	x12, [sp, #64]          // 8-byte Folded Reload
	ldr	x23, [sp, #48]          // 8-byte Folded Reload
	add	x13, x19, #2            // =2
	add	x6, x19, #1             // =1
	add	x14, x19, #3            // =3
	add	x2, x19, #4             // =4
	add	x0, x19, #5             // =5
	add	x1, x19, #6             // =6
	add	x3, x19, #7             // =7
	mul	x29, x19, x12
	mul	x19, x13, x23
	str	x19, [sp, #304]         // 8-byte Folded Spill
	mul	x19, x14, x23
	str	x19, [sp, #296]         // 8-byte Folded Spill
	mul	x19, x2, x23
	str	x19, [sp, #288]         // 8-byte Folded Spill
	mul	x19, x1, x23
	str	x19, [sp, #280]         // 8-byte Folded Spill
	add	x19, x8, x29, lsl #2
	str	x19, [sp, #312]         // 8-byte Folded Spill
	mul	x19, x2, x12
	ldr	x2, [sp, #200]          // 8-byte Folded Reload
	mul	x24, x6, x23
	mul	x21, x0, x23
	mul	x23, x3, x23
	mul	x6, x6, x12
	mul	x22, x13, x12
	mul	x20, x14, x12
	mul	x0, x0, x12
	mul	x1, x1, x12
	mul	x3, x3, x12
	add	x12, x2, x24, lsl #2
	add	x14, x2, x21, lsl #2
	mov	x21, x12
	ldr	x12, [sp, #280]         // 8-byte Folded Reload
	ldr	x13, [sp, #304]         // 8-byte Folded Reload
	str	x6, [sp, #272]          // 8-byte Folded Spill
	str	x14, [sp, #240]         // 8-byte Folded Spill
	add	x6, x2, x12, lsl #2
	ldr	x12, [sp, #272]         // 8-byte Folded Reload
	add	x24, x2, x13, lsl #2
	ldr	x13, [sp, #296]         // 8-byte Folded Reload
	add	x12, x8, x12, lsl #2
	str	x12, [sp, #304]         // 8-byte Folded Spill
	add	x12, x8, x22, lsl #2
	add	x29, x2, x13, lsl #2
	ldr	x13, [sp, #288]         // 8-byte Folded Reload
	str	x12, [sp, #296]         // 8-byte Folded Spill
	add	x12, x8, x20, lsl #2
	str	x12, [sp, #288]         // 8-byte Folded Spill
	add	x12, x8, x19, lsl #2
	str	x12, [sp, #280]         // 8-byte Folded Spill
	add	x12, x8, x0, lsl #2
	str	x12, [sp, #272]         // 8-byte Folded Spill
	add	x12, x8, x1, lsl #2
	add	x13, x2, x13, lsl #2
	add	x2, x2, x23, lsl #2
	mov	x23, x24
	str	x12, [sp, #264]         // 8-byte Folded Spill
	add	x12, x8, x3, lsl #2
	ldp	w24, w20, [sp, #112]    // 8-byte Folded Reload
	ldr	x19, [sp, #40]          // 8-byte Folded Reload
	ldr	x3, [sp, #192]          // 8-byte Folded Reload
	stp	x13, x12, [sp, #248]    // 16-byte Folded Spill
	stp	x23, x21, [sp, #176]    // 16-byte Folded Spill
	str	x29, [sp, #168]         // 8-byte Folded Spill
	cmp	w3, w4
	movi	v0.2d, #0000000000000000
	b.gt	.LBB19_7
	b	.LBB19_10
.LBB19_5:                               //   in Loop: Header=BB19_3 Depth=1
	ldr	x12, [sp, #320]         // 8-byte Folded Reload
	ldp	x14, x13, [sp, #240]    // 16-byte Folded Reload
	add	x19, x19, #8            // =8
	add	w20, w20, #8            // =8
	cmp	x19, x12
	b.ge	.LBB19_17
// %bb.6:                               //   in Loop: Header=BB19_3 Depth=1
	cmp	w3, w4
	movi	v0.2d, #0000000000000000
	b.le	.LBB19_10
.LBB19_7:                               //   in Loop: Header=BB19_3 Depth=1
	ldp	x12, x22, [sp, #136]    // 16-byte Folded Reload
	movi	v16.2d, #0000000000000000
	mov	x1, x30
	movi	v23.2d, #0000000000000000
	add	x0, x12, w24, sxtw #2
	movi	v22.2d, #0000000000000000
	movi	v21.2d, #0000000000000000
	movi	v20.2d, #0000000000000000
	movi	v19.2d, #0000000000000000
	movi	v18.2d, #0000000000000000
	movi	v17.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v2.2d, #0000000000000000
	movi	v1.2d, #0000000000000000
	movi	v0.2d, #0000000000000000
.LBB19_8:                               //   Parent Loop BB19_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x12, x22, #2
	ldr	q24, [x1], #16
	ldp	q31, q8, [x0, #-64]
	ldr	q9, [x21, x12]
	ldr	q10, [x23, x12]
	ldr	q11, [x29, x12]
	ldr	q12, [x13, x12]
	ldr	q13, [x14, x12]
	ldr	q14, [x6, x12]
	ldr	q15, [x2, x12]
	ldp	q29, q30, [x0, #-32]
	ldp	q27, q28, [x0]
	ldp	q25, q26, [x0, #32]
	fmla	v16.4s, v31.4s, v24.s[0]
	fmla	v23.4s, v8.4s, v24.s[0]
	fmla	v22.4s, v31.4s, v9.s[0]
	fmla	v21.4s, v8.4s, v9.s[0]
	fmla	v20.4s, v31.4s, v10.s[0]
	fmla	v19.4s, v8.4s, v10.s[0]
	fmla	v18.4s, v31.4s, v11.s[0]
	fmla	v17.4s, v8.4s, v11.s[0]
	fmla	v7.4s, v31.4s, v12.s[0]
	fmla	v6.4s, v8.4s, v12.s[0]
	fmla	v5.4s, v31.4s, v13.s[0]
	fmla	v4.4s, v8.4s, v13.s[0]
	fmla	v3.4s, v31.4s, v14.s[0]
	fmla	v2.4s, v8.4s, v14.s[0]
	fmla	v1.4s, v31.4s, v15.s[0]
	fmla	v0.4s, v8.4s, v15.s[0]
	fmla	v16.4s, v29.4s, v24.s[1]
	fmla	v23.4s, v30.4s, v24.s[1]
	fmla	v22.4s, v29.4s, v9.s[1]
	fmla	v21.4s, v30.4s, v9.s[1]
	fmla	v20.4s, v29.4s, v10.s[1]
	fmla	v19.4s, v30.4s, v10.s[1]
	fmla	v18.4s, v29.4s, v11.s[1]
	fmla	v17.4s, v30.4s, v11.s[1]
	fmla	v7.4s, v29.4s, v12.s[1]
	fmla	v6.4s, v30.4s, v12.s[1]
	fmla	v5.4s, v29.4s, v13.s[1]
	fmla	v4.4s, v30.4s, v13.s[1]
	fmla	v3.4s, v29.4s, v14.s[1]
	fmla	v2.4s, v30.4s, v14.s[1]
	fmla	v1.4s, v29.4s, v15.s[1]
	fmla	v0.4s, v30.4s, v15.s[1]
	add	x22, x22, #4            // =4
	fmla	v16.4s, v27.4s, v24.s[2]
	fmla	v23.4s, v28.4s, v24.s[2]
	fmla	v22.4s, v27.4s, v9.s[2]
	fmla	v21.4s, v28.4s, v9.s[2]
	fmla	v20.4s, v27.4s, v10.s[2]
	fmla	v19.4s, v28.4s, v10.s[2]
	fmla	v18.4s, v27.4s, v11.s[2]
	fmla	v17.4s, v28.4s, v11.s[2]
	fmla	v7.4s, v27.4s, v12.s[2]
	fmla	v6.4s, v28.4s, v12.s[2]
	fmla	v5.4s, v27.4s, v13.s[2]
	fmla	v4.4s, v28.4s, v13.s[2]
	fmla	v3.4s, v27.4s, v14.s[2]
	fmla	v2.4s, v28.4s, v14.s[2]
	fmla	v1.4s, v27.4s, v15.s[2]
	fmla	v0.4s, v28.4s, v15.s[2]
	cmp	x22, x3
	fmla	v16.4s, v25.4s, v24.s[3]
	fmla	v23.4s, v26.4s, v24.s[3]
	fmla	v22.4s, v25.4s, v9.s[3]
	fmla	v21.4s, v26.4s, v9.s[3]
	fmla	v20.4s, v25.4s, v10.s[3]
	fmla	v19.4s, v26.4s, v10.s[3]
	fmla	v18.4s, v25.4s, v11.s[3]
	fmla	v17.4s, v26.4s, v11.s[3]
	fmla	v7.4s, v25.4s, v12.s[3]
	fmla	v6.4s, v26.4s, v12.s[3]
	fmla	v5.4s, v25.4s, v13.s[3]
	fmla	v4.4s, v26.4s, v13.s[3]
	fmla	v3.4s, v25.4s, v14.s[3]
	fmla	v2.4s, v26.4s, v14.s[3]
	fmla	v1.4s, v25.4s, v15.s[3]
	fmla	v0.4s, v26.4s, v15.s[3]
	add	x0, x0, #128            // =128
	b.lt	.LBB19_8
// %bb.9:                               //   in Loop: Header=BB19_3 Depth=1
	ldr	w12, [sp, #132]         // 4-byte Folded Reload
	add	w24, w12, w24
	ldr	w12, [sp, #128]         // 4-byte Folded Reload
	b	.LBB19_11
.LBB19_10:                              //   in Loop: Header=BB19_3 Depth=1
	movi	v1.2d, #0000000000000000
	movi	v2.2d, #0000000000000000
	movi	v3.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v6.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	movi	v17.2d, #0000000000000000
	movi	v18.2d, #0000000000000000
	movi	v19.2d, #0000000000000000
	movi	v20.2d, #0000000000000000
	movi	v21.2d, #0000000000000000
	movi	v22.2d, #0000000000000000
	movi	v23.2d, #0000000000000000
	movi	v16.2d, #0000000000000000
	mov	w12, w4
.LBB19_11:                              //   in Loop: Header=BB19_3 Depth=1
	ldr	x14, [sp, #312]         // 8-byte Folded Reload
	lsl	x13, x19, #2
	cmp	w12, w5
	add	x14, x14, x13
	stp	q16, q23, [x14]
	ldr	x14, [sp, #304]         // 8-byte Folded Reload
	add	x14, x14, x13
	stp	q22, q21, [x14]
	ldr	x14, [sp, #296]         // 8-byte Folded Reload
	add	x14, x14, x13
	stp	q20, q19, [x14]
	ldr	x14, [sp, #288]         // 8-byte Folded Reload
	add	x14, x14, x13
	stp	q18, q17, [x14]
	ldr	x14, [sp, #280]         // 8-byte Folded Reload
	add	x14, x14, x13
	stp	q7, q6, [x14]
	ldr	x14, [sp, #272]         // 8-byte Folded Reload
	add	x14, x14, x13
	stp	q5, q4, [x14]
	ldp	x0, x14, [sp, #256]     // 16-byte Folded Reload
	add	x14, x14, x13
	add	x13, x0, x13
	stp	q3, q2, [x14]
	stp	q1, q0, [x13]
	b.ge	.LBB19_5
// %bb.12:                              //   in Loop: Header=BB19_3 Depth=1
	sxtw	x0, w12
	ldp	x13, x12, [sp, #200]    // 16-byte Folded Reload
	sxtw	x24, w24
	str	x19, [sp, #216]         // 8-byte Folded Spill
	add	x12, x12, x0
	add	x3, x13, x12, lsl #2
.LBB19_13:                              //   Parent Loop BB19_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB19_14 Depth 3
	add	x4, x24, #1             // =1
	add	x22, x24, #2            // =2
	add	x5, x24, #3             // =3
	add	x14, x24, #4            // =4
	add	x19, x24, #5            // =5
	add	x29, x24, #6            // =6
	add	x23, x24, #7            // =7
	mov	w12, #8
	mov	x1, x3
	mov	w13, w20
.LBB19_14:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	s0, [x1]
	ldr	s1, [x10, x24, lsl #2]
	sbfiz	x21, x13, #2, #32
	ldr	s2, [x8, x21]
	subs	x12, x12, #1            // =1
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #1            // =1
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x4, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #2            // =2
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x22, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #3            // =3
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x5, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #4            // =4
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x14, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #5            // =5
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x19, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #6            // =6
	sbfiz	x21, x21, #2, #32
	ldr	s0, [x1]
	ldr	s1, [x10, x29, lsl #2]
	ldr	s2, [x8, x21]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	add	w21, w13, #7            // =7
	ldr	s0, [x1]
	ldr	s1, [x10, x23, lsl #2]
	sbfiz	x21, x21, #2, #32
	ldr	s2, [x8, x21]
	add	w13, w13, w11
	fmul	s0, s0, s1
	add	x1, x1, x27
	fadd	s0, s2, s0
	str	s0, [x8, x21]
	b.ne	.LBB19_14
// %bb.15:                              //   in Loop: Header=BB19_13 Depth=2
	ldr	x12, [sp, #328]         // 8-byte Folded Reload
	add	x0, x0, #1              // =1
	add	x24, x24, #8            // =8
	add	x3, x3, #4              // =4
	cmp	x0, x12
	b.ne	.LBB19_13
// %bb.16:                              //   in Loop: Header=BB19_3 Depth=1
	ldp	x4, x5, [sp, #152]      // 16-byte Folded Reload
	ldp	x21, x3, [sp, #184]     // 16-byte Folded Reload
	ldr	x19, [sp, #216]         // 8-byte Folded Reload
	ldp	x29, x23, [sp, #168]    // 16-byte Folded Reload
	b	.LBB19_5
.LBB19_17:                              //   in Loop: Header=BB19_3 Depth=1
	ldr	x3, [sp, #56]           // 8-byte Folded Reload
.LBB19_18:                              //   in Loop: Header=BB19_3 Depth=1
	cmp	w19, w3
	b.ge	.LBB19_2
// %bb.19:                              //   in Loop: Header=BB19_3 Depth=1
	cmp	w5, w4
	b.le	.LBB19_2
// %bb.20:                              //   in Loop: Header=BB19_3 Depth=1
	ldr	x21, [sp, #64]          // 8-byte Folded Reload
	ldr	x6, [sp, #120]          // 8-byte Folded Reload
	sxtw	x12, w19
	mul	x13, x6, x21
	add	x14, x6, #1             // =1
	add	x2, x6, #2              // =2
	add	x0, x6, #3              // =3
	add	x1, x6, #4              // =4
	add	x20, x6, #5             // =5
	add	x19, x6, #6             // =6
	add	x6, x6, #7              // =7
	mul	x22, x6, x21
	ldr	x6, [sp, #32]           // 8-byte Folded Reload
	mul	x14, x14, x21
	mul	x2, x2, x21
	mul	x0, x0, x21
	mul	x1, x1, x21
	mul	x29, x20, x21
	mul	x23, x19, x21
	add	x6, x6, w24, sxtw #2
.LBB19_21:                              //   Parent Loop BB19_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB19_22 Depth 3
                                        //       Child Loop BB19_24 Depth 3
                                        //       Child Loop BB19_26 Depth 3
                                        //       Child Loop BB19_28 Depth 3
                                        //       Child Loop BB19_30 Depth 3
                                        //       Child Loop BB19_32 Depth 3
                                        //       Child Loop BB19_34 Depth 3
                                        //       Child Loop BB19_36 Depth 3
	add	x19, x13, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_22:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x30, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_22
// %bb.23:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x14, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_24:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x7, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_24
// %bb.25:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x2, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_26:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x28, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_26
// %bb.27:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x0, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_28:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x17, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_28
// %bb.29:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x1, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_30:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x25, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_30
// %bb.31:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x29, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_32:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x26, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_32
// %bb.33:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x23, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_34:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x9, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_34
// %bb.35:                              //   in Loop: Header=BB19_21 Depth=2
	add	x19, x22, x12
	ldr	s0, [x8, x19, lsl #2]
	mov	x20, xzr
.LBB19_36:                              //   Parent Loop BB19_3 Depth=1
                                        //     Parent Loop BB19_21 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	lsl	x21, x20, #2
	ldr	s1, [x16, x21]
	ldr	s2, [x6, x21]
	add	x20, x20, #1            // =1
	cmp	x15, x20
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x19, lsl #2]
	b.ne	.LBB19_36
// %bb.37:                              //   in Loop: Header=BB19_21 Depth=2
	ldr	x19, [sp, #232]         // 8-byte Folded Reload
	add	x12, x12, #1            // =1
	cmp	x12, x19
	ldr	x19, [sp, #224]         // 8-byte Folded Reload
	add	x6, x6, x19
	b.ne	.LBB19_21
	b	.LBB19_2
.LBB19_38:
	ldp	x1, x7, [sp, #16]       // 16-byte Folded Reload
	ldr	x6, [sp, #200]          // 8-byte Folded Reload
	ldr	x0, [sp, #120]          // 8-byte Folded Reload
                                        // kill: def $w0 killed $w0 killed $x0 def $x0
.LBB19_39:
	cmp	w0, w1
	b.ge	.LBB19_48
// %bb.40:
	cmp	w2, w3
	b.ge	.LBB19_48
// %bb.41:
	sxtw	x17, w4
	sxtw	x13, w1
	sxtw	x1, w5
	smaddl	x16, w0, w7, x17
	sxtw	x9, w2
	sxtw	x12, w0
	sxtw	x11, w11
	sxtw	x14, w3
	sbfiz	x15, x7, #2, #32
	add	x16, x6, x16, lsl #2
	sub	x17, x1, x17
	cmp	w5, w4
	b.gt	.LBB19_44
.LBB19_42:                              // =>This Inner Loop Header: Depth=1
	add	x12, x12, #1            // =1
	cmp	x12, x13
	add	x16, x16, x15
	b.eq	.LBB19_48
// %bb.43:                              //   in Loop: Header=BB19_42 Depth=1
	cmp	w5, w4
	b.le	.LBB19_42
.LBB19_44:
	ldr	w2, [sp, #112]          // 4-byte Folded Reload
	mul	x0, x12, x11
	mov	x1, x9
.LBB19_45:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_46 Depth 2
	add	x3, x1, x0
	ldr	s0, [x8, x3, lsl #2]
	add	x19, x10, w2, sxtw #2
	mov	x7, x17
	mov	x6, x16
.LBB19_46:                              //   Parent Loop BB19_45 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	s1, [x6], #4
	ldr	s2, [x19], #4
	subs	x7, x7, #1              // =1
	add	w2, w2, #1              // =1
	fmul	s1, s1, s2
	fadd	s0, s0, s1
	str	s0, [x8, x3, lsl #2]
	b.ne	.LBB19_46
// %bb.47:                              //   in Loop: Header=BB19_45 Depth=1
	add	x1, x1, #1              // =1
	cmp	x1, x14
	b.ne	.LBB19_45
	b	.LBB19_42
.LBB19_48:
	ldp	x29, x30, [sp, #480]    // 16-byte Folded Reload
	ldp	x20, x19, [sp, #464]    // 16-byte Folded Reload
	ldp	x22, x21, [sp, #448]    // 16-byte Folded Reload
	ldp	x24, x23, [sp, #432]    // 16-byte Folded Reload
	ldp	x26, x25, [sp, #416]    // 16-byte Folded Reload
	ldp	x28, x27, [sp, #400]    // 16-byte Folded Reload
	ldp	d9, d8, [sp, #384]      // 16-byte Folded Reload
	ldp	d11, d10, [sp, #368]    // 16-byte Folded Reload
	ldp	d13, d12, [sp, #352]    // 16-byte Folded Reload
	ldp	d15, d14, [sp, #336]    // 16-byte Folded Reload
	add	sp, sp, #496            // =496
	ret
.Lfunc_end19:
	.size	UKernelV13Asm, .Lfunc_end19-UKernelV13Asm
                                        // -- End function