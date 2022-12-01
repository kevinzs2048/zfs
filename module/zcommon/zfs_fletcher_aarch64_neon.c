/*
 * Implement fast Fletcher4 with NEON instructions. (aarch64)
 *
 * Use the 128-bit NEON SIMD instructions and registers to compute
 * Fletcher4 in two incremental 64-bit parallel accumulator streams,
 * and then combine the streams to form the final four checksum words.
 * This implementation is a derivative of the AVX SIMD implementation by
 * James Guilford and Jinshan Xiong from Intel (see zfs_fletcher_intel.c).
 *
 * Copyright (C) 2016 Romain Dolbeau.
 *
 * Authors:
 *	Romain Dolbeau <romain.dolbeau@atos.net>
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *	- Redistributions of source code must retain the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer.
 *
 *	- Redistributions in binary form must reproduce the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer in the documentation and/or other materials
 *	  provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if defined(__aarch64__)

#include <sys/simd.h>
#include <sys/spa_checksum.h>
#include <sys/string.h>
#include <zfs_fletcher.h>

ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_aarch64_neon_init(fletcher_4_ctx_t *ctx)
{
	memset(ctx->aarch64_neon, 0, 4 * sizeof (zfs_fletcher_aarch64_neon_t));
}

ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_aarch64_neon_fini(fletcher_4_ctx_t *ctx, zio_cksum_t *zcp)
{
	uint64_t A, B, C, D;
	A = ctx->aarch64_neon[0].v[0] + ctx->aarch64_neon[0].v[1];
	B = 2 * ctx->aarch64_neon[1].v[0] + 2 * ctx->aarch64_neon[1].v[1] -
	    ctx->aarch64_neon[0].v[1];
	C = 4 * ctx->aarch64_neon[2].v[0] - ctx->aarch64_neon[1].v[0] +
	    4 * ctx->aarch64_neon[2].v[1] - 3 * ctx->aarch64_neon[1].v[1];
	D = 8 * ctx->aarch64_neon[3].v[0] - 4 * ctx->aarch64_neon[2].v[0] +
	    8 * ctx->aarch64_neon[3].v[1] - 8 * ctx->aarch64_neon[2].v[1] +
	    ctx->aarch64_neon[1].v[1];
	ZIO_SET_CHECKSUM(zcp, A, B, C, D);
}

#define        NEON_INIT_LOOP2()                        \
       asm("eor %[ZERO].16b,%[ZERO].16b,%[ZERO].16b\n" \
       "ld1 { %[ACC0].4s }, %[CTX0]\n"         \
       "ld1 { %[ACC1].4s }, %[CTX1]\n"         \
       "ld1 { %[ACC2].4s }, %[CTX2]\n"         \
       "ld1 { %[ACC3].4s }, %[CTX3]\n"         \
       : [ZERO] "=w" (ZERO),                   \
       [ACC0] "=w" (ACC0), [ACC1] "=w" (ACC1), \
       [ACC2] "=w" (ACC2), [ACC3] "=w" (ACC3)  \
       : [CTX0] "Q" (ctx->aarch64_neon[0]),    \
        [CTX1] "Q" (ctx->aarch64_neon[1]),      \
        [CTX2] "Q" (ctx->aarch64_neon[2]),      \
        [CTX3] "Q" (ctx->aarch64_neon[3]))

#define	NEON_INIT_LOOP()			\
	asm("eor v0.16b,v0.16b,v0.16b\n"	\
	"ld1 { v1.4s }, %[CTX0]\n"		\
	"ld1 { v2.4s }, %[CTX1]\n"		\
	"ld1 { v3.4s }, %[CTX2]\n"		\
	"ld1 { v4.4s }, %[CTX3]\n"		\
	: : [CTX0] "Q" (ctx->aarch64_neon[0]),	\
	[CTX1] "Q" (ctx->aarch64_neon[1]),	\
	[CTX2] "Q" (ctx->aarch64_neon[2]),	\
	[CTX3] "Q" (ctx->aarch64_neon[3]))

#define NEON_SUPERSCALAR8() \
    asm("ldp q21, q20, %[IP]\n"		\
        "uaddw   v19.2d, v19.2d, v21.2s\n"		\
        "uaddw2  v18.2d, v18.2d, v21.4s\n"		\
        "uaddw   v17.2d, v17.2d, v20.2s\n"		\
        "uaddw2  v16.2d, v16.2d, v20.4s\n"		\
        "add     v7.2d, v7.2d, v19.2d\n"		\
        "add     v6.2d, v6.2d, v18.2d\n"		\
        "add     v5.2d, v5.2d, v17.2d\n"		\
        "add     v4.2d, v4.2d, v16.2d\n"		\
        "add     v3.2d, v3.2d, v7.2d\n"		\
        "add     v2.2d, v2.2d, v6.2d\n"		\
        "add     v1.2d, v1.2d, v5.2d\n"		\
        "add     v0.2d, v0.2d, v4.2d\n"		\
        "add     v25.2d, v25.2d, v3.2d\n"		\
        "add     v24.2d, v24.2d, v2.2d\n"		\
        "add     v23.2d, v23.2d, v1.2d\n"		\
        "add     v22.2d, v22.2d, v0.2d\n"::[IP] "Q" (*ip))

#define	NEON_DO_REVERSE "rev32 %[SRC].16b, %[SRC].16b\n"

#define	NEON_DONT_REVERSE ""

#define	NEON_MAIN_LOOP(REVERSE)				\
	asm("ld1 { %[SRC].4s }, %[IP]\n"		\
	REVERSE						\
	"zip1 %[TMP1].4s, %[SRC].4s, %[ZERO].4s\n"	\
	"zip2 %[TMP2].4s, %[SRC].4s, %[ZERO].4s\n"	\
	"add %[ACC0].2d, %[ACC0].2d, %[TMP1].2d\n"	\
	"add %[ACC1].2d, %[ACC1].2d, %[ACC0].2d\n"	\
	"add %[ACC2].2d, %[ACC2].2d, %[ACC1].2d\n"	\
	"add %[ACC3].2d, %[ACC3].2d, %[ACC2].2d\n"	\
	"add %[ACC0].2d, %[ACC0].2d, %[TMP2].2d\n"	\
	"add %[ACC1].2d, %[ACC1].2d, %[ACC0].2d\n"	\
	"add %[ACC2].2d, %[ACC2].2d, %[ACC1].2d\n"	\
	"add %[ACC3].2d, %[ACC3].2d, %[ACC2].2d\n"	\
	: [SRC] "=&w" (SRC),				\
	[TMP1] "=&w" (TMP1), [TMP2] "=&w" (TMP2),	\
	[ACC0] "+w" (ACC0), [ACC1] "+w" (ACC1),		\
	[ACC2] "+w" (ACC2), [ACC3] "+w" (ACC3)		\
	: [ZERO] "w" (ZERO), [IP] "Q" (*ip))

#define	NEON_FINI_LOOP1()			\
	asm("st1 { v1.4s },%[DST0]\n"	\
	"st1 { v2.4s },%[DST1]\n"		\
	"st1 { v3.4s },%[DST2]\n"		\
	"st1 { v4.4s },%[DST3]\n"		\
	: [DST0] "=Q" (ctx->aarch64_neon[0]),	\
	[DST1] "=Q" (ctx->aarch64_neon[1]),	\
	[DST2] "=Q" (ctx->aarch64_neon[2]),	\
	[DST3] "=Q" (ctx->aarch64_neon[3])	\
	: )

#define	NEON_FINI_LOOP2()			\
	asm("st1 { v1.4s },%[DST0]\n"	\
	"st1 { v2.4s },%[DST1]\n"		\
	"st1 { v3.4s },%[DST2]\n"		\
	"st1 { v4.4s },%[DST3]\n"		\
	: [DST0] "=Q" (ctx->aarch64_neon[0]),	\
	[DST1] "=Q" (ctx->aarch64_neon[1]),	\
	[DST2] "=Q" (ctx->aarch64_neon[2]),	\
	[DST3] "=Q" (ctx->aarch64_neon[3])	\
	: )

static void
fletcher_4_aarch64_neon_native(fletcher_4_ctx_t *ctx,
    const void *buf, uint64_t size)
{
	const uint64_t *ip = buf;
	const uint64_t *ipend = (uint64_t *)((uint8_t *)ip + size);

	kfpu_begin();

	NEON_INIT_LOOP();

    for (; ip < ipend; ip += 2) {
		NEON_SUPERSCALAR8();
	}

	NEON_FINI_LOOP1();

	kfpu_end();
}

static void
fletcher_4_aarch64_neon_byteswap(fletcher_4_ctx_t *ctx,
    const void *buf, uint64_t size)
{
	const uint64_t *ip = buf;
	const uint64_t *ipend = (uint64_t *)((uint8_t *)ip + size);
#if defined(_KERNEL)
register unsigned char ZERO asm("v0") __attribute__((vector_size(16)));
register unsigned char ACC0 asm("v1") __attribute__((vector_size(16)));
register unsigned char ACC1 asm("v2") __attribute__((vector_size(16)));
register unsigned char ACC2 asm("v3") __attribute__((vector_size(16)));
register unsigned char ACC3 asm("v4") __attribute__((vector_size(16)));
register unsigned char TMP1 asm("v5") __attribute__((vector_size(16)));
register unsigned char TMP2 asm("v6") __attribute__((vector_size(16)));
register unsigned char SRC asm("v7") __attribute__((vector_size(16)));
#else
unsigned char ZERO __attribute__((vector_size(16)));
unsigned char ACC0 __attribute__((vector_size(16)));
unsigned char ACC1 __attribute__((vector_size(16)));
unsigned char ACC2 __attribute__((vector_size(16)));
unsigned char ACC3 __attribute__((vector_size(16)));
unsigned char TMP1 __attribute__((vector_size(16)));
unsigned char TMP2 __attribute__((vector_size(16)));
unsigned char SRC __attribute__((vector_size(16)));
#endif

	kfpu_begin();

	NEON_INIT_LOOP2();

	for (; ip < ipend; ip += 2) {
		NEON_MAIN_LOOP(NEON_DO_REVERSE);
	}

	NEON_FINI_LOOP2();

	kfpu_end();
}

static boolean_t fletcher_4_aarch64_neon_valid(void)
{
	return (kfpu_allowed());
}

const fletcher_4_ops_t fletcher_4_aarch64_neon_ops = {
	.init_native = fletcher_4_aarch64_neon_init,
	.compute_native = fletcher_4_aarch64_neon_native,
	.fini_native = fletcher_4_aarch64_neon_fini,
	.init_byteswap = fletcher_4_aarch64_neon_init,
	.compute_byteswap = fletcher_4_aarch64_neon_byteswap,
	.fini_byteswap = fletcher_4_aarch64_neon_fini,
	.valid = fletcher_4_aarch64_neon_valid,
	.name = "aarch64_neon"
};

#endif /* defined(__aarch64__) */
