/*
 * Implement fast Fletcher4 using superscalar pipelines.
 *
 * Use regular C code to compute
 * Fletcher4 in four incremental 64-bit parallel accumulator streams,
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

#include <sys/param.h>
#include <sys/byteorder.h>
#include <sys/spa_checksum.h>
#include <sys/string.h>
#include <zfs_fletcher.h>

ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_superscalar4_init(fletcher_4_ctx_t *ctx)
{
	memset(ctx->superscalar, 0, 4 * sizeof (zfs_fletcher_superscalar_t));
}

ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_superscalar4_fini(fletcher_4_ctx_t *ctx, zio_cksum_t *zcp)
{
	uint64_t A, B, C, D;

	A = ctx->superscalar[0].v[0] + ctx->superscalar[0].v[1] +
	    ctx->superscalar[0].v[2] + ctx->superscalar[0].v[3];
	B = 0 - ctx->superscalar[0].v[1] - 2 * ctx->superscalar[0].v[2] -
	    3 * ctx->superscalar[0].v[3] + 4 * ctx->superscalar[1].v[0] +
	    4 * ctx->superscalar[1].v[1] + 4 * ctx->superscalar[1].v[2] +
	    4 * ctx->superscalar[1].v[3];

	C = ctx->superscalar[0].v[2] + 3 * ctx->superscalar[0].v[3] -
	    6 * ctx->superscalar[1].v[0] - 10 * ctx->superscalar[1].v[1] -
	    14 * ctx->superscalar[1].v[2] - 18 * ctx->superscalar[1].v[3] +
	    16 * ctx->superscalar[2].v[0] + 16 * ctx->superscalar[2].v[1] +
	    16 * ctx->superscalar[2].v[2] + 16 * ctx->superscalar[2].v[3];

	D = 0 - ctx->superscalar[0].v[3] + 4 * ctx->superscalar[1].v[0] +
	    10 * ctx->superscalar[1].v[1] + 20 * ctx->superscalar[1].v[2] +
	    34 * ctx->superscalar[1].v[3] - 48 * ctx->superscalar[2].v[0] -
	    64 * ctx->superscalar[2].v[1] - 80 * ctx->superscalar[2].v[2] -
	    96 * ctx->superscalar[2].v[3] + 64 * ctx->superscalar[3].v[0] +
	    64 * ctx->superscalar[3].v[1] + 64 * ctx->superscalar[3].v[2] +
	    64 * ctx->superscalar[3].v[3];

	ZIO_SET_CHECKSUM(zcp, A, B, C, D);
}

#define PREF4X64L1(buffer, PREF_OFFSET, ITR) \
       __asm__("PRFM PLDL1STRM, [%x[v],%[c]]"::[v]"r"(buffer), [c]"I"((PREF_OFFSET) + ((ITR) + 0)*64));

#define PREF1KL1(buffer, PREF_OFFSET)  PREF4X64L1(buffer,(PREF_OFFSET), 0)

#define        SUPERSCALAR4_NEON_INIT() \
    asm("ld1 { %[AA2].4s, %[A3A4].4s }, %[CTX0]\n"             \
       "ld1 { %[BB2].4s, %[B3B4].4s }, %[CTX1]\n"              \
       "ld1 { %[CC2].4s, %[C3C4].4s }, %[CTX2]\n"              \
       "ld1 { %[DD2].2d, %[D3D4].2d }, %[CTX3]\n"              \
       "mov %[vala3a], %[A3A4].2d[0]\n"                        \
       "mov %[vala4a], %[A3A4].2d[1]\n"                        \
       : [AA2] "=w" (AA2), [A3A4] "=w" (A3A4), \
    [BB2] "=w" (BB2), [B3B4] "=w" (B3B4),      \
    [CC2] "=w" (CC2), [C3C4] "=w" (C3C4),      \
       [DD2] "=w" (DD2), [D3D4] "=w" (D3D4), [vala3a] "=r" (vala3a), [vala4a] "=r" (vala4a)    \
       : [CTX0] "Q" (superscalar[0]),     \
       [CTX1] "Q" (superscalar[1]),       \
       [CTX2] "Q" (superscalar[2]),       \
       [CTX3] "Q" (superscalar[3]))

#define        SUPERSCALAR4_NEON_MAIN() \
    asm volatile("mov %[vala3b], %[A3A4].2d[0]\n"                        \
       "mov %[vala4b], %[A3A4].2d[1]\n"                        \
        "add %[AA2].2d, %[AA2].2d, %[TMP1].2d\n"       \
       "add %[A3A4].2d, %[A3A4].2d, %[TMP2].2d\n"      \
       "add %[BB2].2d, %[BB2].2d, %[AA2].2d\n" \
       "add %[B3B4].2d, %[B3B4].2d, %[A3A4].2d\n"      \
       "add %[CC2].2d, %[CC2].2d, %[BB2].2d\n" \
       "add %[C3C4].2d, %[C3C4].2d, %[B3B4].2d\n"                 \
       "add %[DD2].2d, %[D3D4].2d, %[CC2].2d\n"   \
       "add %[D3D4].2d, %[D3D4].2d, %[C3C4].2d\n"      \
       "mov %[vala], %[AA2].2d[0]\n"                        \
       "mov %[vala2], %[AA2].2d[1]\n"                                 \
       "mov %[vala3], %[A3A4].2d[0]\n"                        \
       "mov %[vala4], %[A3A4].2d[1]\n"                        \
       : [AA2] "+w" (AA2), [A3A4] "+w" (A3A4), \
    [BB2] "+w" (BB2), [B3B4] "+w" (B3B4),      \
    [CC2] "+w" (CC2), [C3C4] "+w" (C3C4),      \
       [DD2] "+w" (DD2), [D3D4] "+w" (D3D4), [vala] "=r" (vala),  [vala2] "=r" (vala2),[vala3] "=r" (vala3), [vala4] "=r" (vala4),[vala3b] "=r" (vala3b), [vala4b] "=r" (vala4b)  \
       :[TMP1] "w" (TMP1), [TMP2] "w" (TMP2))

#define        SUPERSCALAR4_NEON_MAIN_LOAD() \
    asm volatile("ld1 { %[SRC1].4s }, %[IP]\n" \
    "eor %[ZERO].16b,%[ZERO].16b,%[ZERO].16b\n"	                                         \
    "zip1 %[TMP1].4s, %[SRC1].4s, %[ZERO].4s\n"	\
	"zip2 %[TMP2].4s, %[SRC1].4s, %[ZERO].4s\n"	   \
    "ld1 { %[SRC2].4s }, %[IP]\n" \
    "mov %[val0], %[TMP1].2d[0]\n"          \
    "mov %[val1], %[TMP1].2d[1]\n"                        \
    "mov %[val2], %[TMP2].2d[0]\n"                        \
    "mov %[val3], %[TMP2].2d[1]\n"                        \
    : [ZERO] "=w" (ZERO), [SRC1] "=&w" (SRC1), [SRC2] "=&w" (SRC2), [TMP1] "=&w" (TMP1), [TMP2] "=&w" (TMP2),  \
      [val0] "=r" (val0), [val1] "=r" (val1), [val2] "=r" (val2), [val3] "=r" (val3) \
    : [IP] "Q" (*ip2))



#define        SUPERSCALAR4_NEON_FINI_LOOP()                   \
       asm volatile("mov %[val00], %[AA2].2d[0]\n"                        \
    "mov %[val01], %[AA2].2d[1]\n"                        \
    "mov %[val02], %[A3A4].2d[0]\n"                        \
    "mov %[val03], %[A3A4].2d[1]\n"                        \
    "mov %[val10], %[BB2].2d[0]\n"                        \
    "mov %[val11], %[BB2].2d[1]\n"                        \
    "mov %[val12], %[B3B4].2d[0]\n"                        \
    "mov %[val13], %[B3B4].2d[1]\n"                        \
    "mov %[val20], %[CC2].2d[0]\n"                        \
    "mov %[val21], %[CC2].2d[1]\n"                        \
    "mov %[val22], %[C3C4].2d[0]\n"                        \
    "mov %[val23], %[C3C4].2d[1]\n"                        \
    "mov %[val30], %[DD2].2d[0]\n"                        \
    "mov %[val31], %[DD2].2d[1]\n"                        \
    "mov %[val32], %[D3D4].2d[0]\n"                        \
    "mov %[val33], %[D3D4].2d[1]\n"                        \
       : [val00] "=r" (val00), [val01] "=r" (val01), [val02] "=r" (val02), [val03] "=r" (val03),       \
       [val10] "=r" (val10), [val11] "=r" (val11), [val12] "=r" (val12), [val13] "=r" (val13),       \
       [val20] "=r" (val20), [val21] "=r" (val21), [val22] "=r" (val22), [val23] "=r" (val23),       \
       [val30] "=r" (val30), [val31] "=r" (val31), [val32] "=r" (val32), [val33] "=r" (val33)       \
       : [AA2] "w" (AA2), [A3A4] "w" (A3A4),   \
    [BB2] "w" (BB2), [B3B4] "w" (B3B4),        \
    [CC2] "w" (CC2), [C3C4] "w" (C3C4),        \
       [DD2] "w" (DD2), [D3D4] "w" (D3D4))


ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_superscalar4_native(fletcher_4_ctx_t *ctx,
    const void *buf, uint64_t size)
{
	const uint32_t *ip = buf;
	const uint32_t *ipend = ip + (size / sizeof (uint32_t));
	uint64_t a, b, c, d;
	uint64_t a2, b2, c2, d2;
	uint64_t a3, b3, c3, d3;
	uint64_t a4, b4, c4, d4;
    uint64_t val0, val1, val2, val3;
    uint64_t val00, val01, val02, val03;
    uint64_t val10, val11, val12, val13;
    uint64_t val20, val21, val22, val23;
    uint64_t val30, val31, val32, val33;
    uint64_t vala, vala2, vala3, vala4, vala3a, vala4a, vala3b, vala4b;
    const uint32_t *ip2 = buf;

#if defined(_KERNEL)
    const uint64_t *ip3 = buf;
    const uint64_t *ipend3 = (uint64_t *)((uint8_t *)ip + size);

    printk("ip2: %pK, %pK, %pK, %pK, %pK", ip2, ip2+1, ip2+2, ip2+3, ip2+4);
    printk("*ip2: %x, %x, %x, %x, %x", *ip2,*(ip2+1),*(ip2+2),*(ip2+3),*(ip2+4));
    printk("ip2: %x, %x, %x, %x, %x", ip2[0], ip2[1], ip2[2], ip2[3], ip2[4]);
    printk("ipend: %pK", ipend);

    printk("ip3+1: %pK, %pK, %pK", ip3, ip3+1, ip3+2);
    printk("*ip3: %llx, %llx, %llx", *ip3,*(ip3+1),*(ip3+2));
    printk("ip3: %llx, %llx, %llx", ip3[0], ip3[1], ip3[2]);
    printk("ip3end: %pK", ipend3);
#endif

    uint64_t superscalar[4][4];
    superscalar[0][0] = ctx->superscalar[0].v[0];
    superscalar[1][0] = ctx->superscalar[1].v[0];
    superscalar[2][0] = ctx->superscalar[2].v[0];
    superscalar[3][0] = ctx->superscalar[3].v[0];
    superscalar[0][1] = ctx->superscalar[0].v[1];
    superscalar[1][1] = ctx->superscalar[1].v[1];
    superscalar[2][1] = ctx->superscalar[2].v[1];
    superscalar[3][1] = ctx->superscalar[3].v[1];
    superscalar[0][2] = ctx->superscalar[0].v[2];
    superscalar[1][2] = ctx->superscalar[1].v[2];
    superscalar[2][2] = ctx->superscalar[2].v[2];
    superscalar[3][2] = ctx->superscalar[3].v[2];
    superscalar[0][3] = ctx->superscalar[0].v[3];
    superscalar[1][3] = ctx->superscalar[1].v[3];
    superscalar[2][3] = ctx->superscalar[2].v[3];
    superscalar[3][3] = ctx->superscalar[3].v[3];

	a = ctx->superscalar[0].v[0];
	b = ctx->superscalar[1].v[0];
	c = ctx->superscalar[2].v[0];
	d = ctx->superscalar[3].v[0];
	a2 = ctx->superscalar[0].v[1];
	b2 = ctx->superscalar[1].v[1];
	c2 = ctx->superscalar[2].v[1];
	d2 = ctx->superscalar[3].v[1];
	a3 = ctx->superscalar[0].v[2];
	b3 = ctx->superscalar[1].v[2];
	c3 = ctx->superscalar[2].v[2];
	d3 = ctx->superscalar[3].v[2];
	a4 = ctx->superscalar[0].v[3];
	b4 = ctx->superscalar[1].v[3];
	c4 = ctx->superscalar[2].v[3];
	d4 = ctx->superscalar[3].v[3];

    int n = 0;
    for (; ip < ipend; ip += 4, n++) {
		a += ip[0];
		a2 += ip[1];
		a3 += ip[2];
		a4 += ip[3];
		b += a;
		b2 += a2;
		b3 += a3;
		b4 += a4;
		c += b;
		c2 += b2;
		c3 += b3;
		c4 += b4;
		d += c;
		d2 += c2;
		d3 += c3;
		d4 += c4;
#if defined(_KERNEL)
        if ((n == 0)||(n == 1) || (n == 2) || (n == 8189) || (n == 8190) || (n == 8191)){
            printk("n: %d, ip0-3: %x, %x, %x, %x", n, ip[0], ip[1], ip[2], ip[3]);
            printk("n: %d, a: %llx, b: %llx, c: %llx, d: %llx, a2: %llx, a3: %llx, a4: %llx", n, a, b, c, d, a2, a3, a4);
        }
#endif
	}

	ctx->superscalar[0].v[0] = a;
	ctx->superscalar[1].v[0] = b;
	ctx->superscalar[2].v[0] = c;
	ctx->superscalar[3].v[0] = d;
	ctx->superscalar[0].v[1] = a2;
	ctx->superscalar[1].v[1] = b2;
	ctx->superscalar[2].v[1] = c2;
	ctx->superscalar[3].v[1] = d2;
	ctx->superscalar[0].v[2] = a3;
	ctx->superscalar[1].v[2] = b3;
	ctx->superscalar[2].v[2] = c3;
	ctx->superscalar[3].v[2] = d3;
	ctx->superscalar[0].v[3] = a4;
	ctx->superscalar[1].v[3] = b4;
	ctx->superscalar[2].v[3] = c4;
	ctx->superscalar[3].v[3] = d4;
#if defined(_KERNEL)
    printk("==============C ori after===================");
    for (int k = 0; k < 4; k++){
        for (int j = 0; j < 4; j++){
            printk("[%d][%d]: %llx", k, j, ctx->superscalar[k].v[j]);
        }
    }
    printk("==============C ori after===================");
#endif

#if defined(_KERNEL)
    register unsigned char AA2 asm("v10") __attribute__((vector_size(16)));
    register unsigned char A3A4 asm("v11") __attribute__((vector_size(16)));
    register unsigned char BB2 asm("v12") __attribute__((vector_size(16)));
    register unsigned char B3B4 asm("v13") __attribute__((vector_size(16)));
    register unsigned char CC2 asm("v14") __attribute__((vector_size(16)));
    register unsigned char C3C4 asm("v15") __attribute__((vector_size(16)));
    register unsigned char DD2 asm("v16") __attribute__((vector_size(16)));
    register unsigned char D3D4 asm("v17") __attribute__((vector_size(16)));
    register unsigned char SRC1 asm("v18") __attribute__((vector_size(16)));
    register unsigned char SRC2 asm("v19") __attribute__((vector_size(16)));
    register unsigned char ZERO asm("v20") __attribute__((vector_size(16)));
    register unsigned char TMP1 asm("v21") __attribute__((vector_size(16)));
    register unsigned char TMP2 asm("v22") __attribute__((vector_size(16)));
#else
    unsigned char AA2 __attribute__((vector_size(16)));
    unsigned char A3A4 __attribute__((vector_size(16)));
    unsigned char BB2 __attribute__((vector_size(16)));
    unsigned char B3B4 __attribute__((vector_size(16)));
    unsigned char CC2 __attribute__((vector_size(16)));
    unsigned char C3C4 __attribute__((vector_size(16)));
    unsigned char DD2 __attribute__((vector_size(16)));
    unsigned char D3D4 __attribute__((vector_size(16)));
    unsigned char SRC1 __attribute__((vector_size(16)));
    unsigned char SRC2 __attribute__((vector_size(16)));
    unsigned char ZERO __attribute__((vector_size(16)));
    unsigned char TMP1 __attribute__((vector_size(16)));
    unsigned char TMP2 __attribute__((vector_size(16)));
#endif

    SUPERSCALAR4_NEON_INIT();
    int m = 0;
    for (; ip2 < ipend; ip2 += 4, m++) {
        SUPERSCALAR4_NEON_MAIN_LOAD();
        PREF1KL1(ip2, 1024)
        SUPERSCALAR4_NEON_MAIN();
#if defined(_KERNEL)
        if ((m == 0)||(m == 1) || (m == 2) || (m == 8189) || (m == 8190) || (m == 8191)){
            printk("m: %d, %x, %x, %x, %x", m, ip2[0], ip2[1], ip2[2], ip2[3]);
            printk("m: %d, TMP1-2: %llx, %llx, %llx, %llx, %llx, %llx", m, val0, val1, val2, val3, vala3b, vala4b);
            printk("m: %d, vala - a4: %llx, %llx, %llx, %llx", m, vala, vala2, vala3, vala4);
        }
#endif
    }
    SUPERSCALAR4_NEON_FINI_LOOP();
#if defined(_KERNEL)
//    printk("m, n: %d, %d", m, n);
    printk("==============ASM ori after===================");
    for (int k = 0; k < 4; k++){
        for (int j = 0; j < 4; j++){
            printk("[%d][%d]: %llx", k, j, superscalar[k][j]);
        }
    }
    printk("%llx, %llx, %llx, %llx", val00, val01, val02, val03);
    printk("%llx, %llx, %llx, %llx", val10, val11, val12, val13);
    printk("%llx, %llx, %llx, %llx", val20, val21, val22, val23);
    printk("%llx, %llx, %llx, %llx", val30, val31, val32, val33);
    printk("==============ASM ori after===================");
#endif
}

ZFS_NO_SANITIZE_UNDEFINED
static void
fletcher_4_superscalar4_byteswap(fletcher_4_ctx_t *ctx,
    const void *buf, uint64_t size)
{
	const uint32_t *ip = buf;
	const uint32_t *ipend = ip + (size / sizeof (uint32_t));
	uint64_t a, b, c, d;
	uint64_t a2, b2, c2, d2;
	uint64_t a3, b3, c3, d3;
	uint64_t a4, b4, c4, d4;

	a = ctx->superscalar[0].v[0];
	b = ctx->superscalar[1].v[0];
	c = ctx->superscalar[2].v[0];
	d = ctx->superscalar[3].v[0];
	a2 = ctx->superscalar[0].v[1];
	b2 = ctx->superscalar[1].v[1];
	c2 = ctx->superscalar[2].v[1];
	d2 = ctx->superscalar[3].v[1];
	a3 = ctx->superscalar[0].v[2];
	b3 = ctx->superscalar[1].v[2];
	c3 = ctx->superscalar[2].v[2];
	d3 = ctx->superscalar[3].v[2];
	a4 = ctx->superscalar[0].v[3];
	b4 = ctx->superscalar[1].v[3];
	c4 = ctx->superscalar[2].v[3];
	d4 = ctx->superscalar[3].v[3];

	for (; ip < ipend; ip += 4) {
		a += BSWAP_32(ip[0]);
		a2 += BSWAP_32(ip[1]);
		a3 += BSWAP_32(ip[2]);
		a4 += BSWAP_32(ip[3]);
		b += a;
		b2 += a2;
		b3 += a3;
		b4 += a4;
		c += b;
		c2 += b2;
		c3 += b3;
		c4 += b4;
		d += c;
		d2 += c2;
		d3 += c3;
		d4 += c4;
	}

	ctx->superscalar[0].v[0] = a;
	ctx->superscalar[1].v[0] = b;
	ctx->superscalar[2].v[0] = c;
	ctx->superscalar[3].v[0] = d;
	ctx->superscalar[0].v[1] = a2;
	ctx->superscalar[1].v[1] = b2;
	ctx->superscalar[2].v[1] = c2;
	ctx->superscalar[3].v[1] = d2;
	ctx->superscalar[0].v[2] = a3;
	ctx->superscalar[1].v[2] = b3;
	ctx->superscalar[2].v[2] = c3;
	ctx->superscalar[3].v[2] = d3;
	ctx->superscalar[0].v[3] = a4;
	ctx->superscalar[1].v[3] = b4;
	ctx->superscalar[2].v[3] = c4;
	ctx->superscalar[3].v[3] = d4;
#if defined(_KERNEL)
    printk("%llx", ctx->superscalar[0].v[0]);
    printk("%llx", ctx->superscalar[1].v[1]);
    printk("%llx", ctx->superscalar[2].v[2]);
#endif
}

static boolean_t fletcher_4_superscalar4_valid(void)
{
	return (B_TRUE);
}

const fletcher_4_ops_t fletcher_4_superscalar4_ops = {
	.init_native = fletcher_4_superscalar4_init,
	.compute_native = fletcher_4_superscalar4_native,
	.fini_native = fletcher_4_superscalar4_fini,
	.init_byteswap = fletcher_4_superscalar4_init,
	.compute_byteswap = fletcher_4_superscalar4_byteswap,
	.fini_byteswap = fletcher_4_superscalar4_fini,
	.valid = fletcher_4_superscalar4_valid,
	.name = "superscalar4"
};
