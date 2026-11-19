/*
 *  Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/* RVV SAD kernels, ported from the libvpx rvv-optimization work (G3).
 * Semantics match aom_sadWxH_c: sum of abs(src - ref).
 * abs(a-b) on u8 wraps, so compute max-min. Per-lane u16 accumulator
 * (max 255*128 per lane, no overflow), one reduction at the end.
 */
#include <riscv_vector.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_ports/mem.h"

static inline unsigned sad_rvv_big(const uint8_t *src, int src_stride,
                                   const uint8_t *ref, int ref_stride, int w,
                                   int h) {
  const size_t vlmax = __riscv_vsetvlmax_e16m2();
  vuint16m2_t acc = __riscv_vmv_v_x_u16m2(0, vlmax);
  int y;
  for (y = 0; y < h; y++) {
    int x = 0;
    while (x < w) {
      size_t vl = (size_t)(w - x) < vlmax ? (size_t)(w - x) : vlmax;
      const vuint8m1_t a = __riscv_vle8_v_u8m1(src + x, vl);
      const vuint8m1_t b = __riscv_vle8_v_u8m1(ref + x, vl);
      const vuint8m1_t hi = __riscv_vmaxu_vv_u8m1(a, b, vl);
      const vuint8m1_t lo = __riscv_vminu_vv_u8m1(a, b, vl);
      acc = __riscv_vwaddu_wv_u16m2(acc, __riscv_vsub_vv_u8m1(hi, lo, vl),
                                    vl);
      x += (int)vl;
    }
    src += src_stride;
    ref += ref_stride;
  }
  const vuint32m1_t s = __riscv_vwredsumu_vs_u16m2_u32m1(
      acc, __riscv_vmv_v_x_u32m1(0, vlmax), vlmax);
  return (unsigned)__riscv_vmv_x_s_u32m1_u32(s);
}

static inline unsigned sad_rvv_small(const uint8_t *src, int src_stride,
                                     const uint8_t *ref, int ref_stride, int w,
                                     int h) {
  const size_t vlmax = __riscv_vsetvlmax_e16m1();
  vuint16m1_t acc = __riscv_vmv_v_x_u16m1(0, vlmax);
  int y;
  for (y = 0; y < h; y++) {
    int x = 0;
    while (x < w) {
      size_t vl = (size_t)(w - x) < vlmax ? (size_t)(w - x) : vlmax;
      const vuint8mf2_t a = __riscv_vle8_v_u8mf2(src + x, vl);
      const vuint8mf2_t b = __riscv_vle8_v_u8mf2(ref + x, vl);
      const vuint8mf2_t hi = __riscv_vmaxu_vv_u8mf2(a, b, vl);
      const vuint8mf2_t lo = __riscv_vminu_vv_u8mf2(a, b, vl);
      acc = __riscv_vwaddu_wv_u16m1(acc, __riscv_vsub_vv_u8mf2(hi, lo, vl),
                                    vl);
      x += (int)vl;
    }
    src += src_stride;
    ref += ref_stride;
  }
  const vuint32m1_t s = __riscv_vwredsumu_vs_u16m1_u32m1(
      acc, __riscv_vmv_v_x_u32m1(0, vlmax), vlmax);
  return (unsigned)__riscv_vmv_x_s_u32m1_u32(s);
}

#define SAD_RVV(W, H)                                                        \
  uint32_t aom_sad##W##x##H##_rvv(const uint8_t *src, int src_stride,        \
                                  const uint8_t *ref, int ref_stride) {      \
    if ((W) <= 16)                                                           \
      return sad_rvv_small(src, src_stride, ref, ref_stride, (W), (H));      \
    return sad_rvv_big(src, src_stride, ref, ref_stride, (W), (H));          \
  }

SAD_RVV(4, 4)
SAD_RVV(4, 8)
SAD_RVV(4, 16)
SAD_RVV(8, 4)
SAD_RVV(8, 8)
SAD_RVV(8, 16)
SAD_RVV(8, 32)
SAD_RVV(16, 4)
SAD_RVV(16, 8)
SAD_RVV(16, 16)
SAD_RVV(16, 32)
SAD_RVV(16, 64)
SAD_RVV(32, 8)
SAD_RVV(32, 16)
SAD_RVV(32, 32)
SAD_RVV(32, 64)
SAD_RVV(64, 16)
SAD_RVV(64, 32)
SAD_RVV(64, 64)
SAD_RVV(64, 128)
SAD_RVV(128, 64)
SAD_RVV(128, 128)
