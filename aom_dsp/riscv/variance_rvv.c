/*
 *  Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/* RVV variance kernel, ported from the libvpx rvv-optimization work (G3).
 * Outputs both sum(diff) and sum(diff^2) as exact integers (the caller's
 * `sse - sum*sum/(w*h)` is rounding-sensitive). Per-lane i32 accumulators:
 * worst case 65025*128 ~ 8.3M, no overflow, so a single flush at the end.
 */
#include <riscv_vector.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"

static inline void variance_rvv_core(const uint8_t *a, int a_stride,
                                     const uint8_t *b, int b_stride, int w,
                                     int h, uint32_t *sse, int *sum) {
  const size_t vlmax = __riscv_vsetvl_e8m1((size_t)w);
  vint32m4_t acc_ss = __riscv_vmv_v_x_i32m4(0, vlmax);
  vint32m4_t acc_s = __riscv_vmv_v_x_i32m4(0, vlmax);
  int y;

  for (y = 0; y < h; ++y) {
    int x;
    for (x = 0; x < w; x += (int)vlmax) {
      const size_t vl = __riscv_vsetvl_e8m1((size_t)(w - x));
      const vuint8m1_t va = __riscv_vle8_v_u8m1(a + x, vl);
      const vuint8m1_t vb = __riscv_vle8_v_u8m1(b + x, vl);
      const vint16m2_t d = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vv_u16m2(__riscv_vzext_vf2_u16m2(vb, vl),
                                __riscv_vzext_vf2_u16m2(va, vl), vl));
      acc_s = __riscv_vwadd_wv_i32m4(acc_s, d, vl);
      acc_ss = __riscv_vwmacc_vv_i32m4(acc_ss, d, d, vl);
    }
    a += a_stride;
    b += b_stride;
  }

  {
    const vint32m1_t ss =
        __riscv_vredsum_vs_i32m4_i32m1(acc_ss, __riscv_vmv_v_x_i32m1(0, vlmax),
                                       vlmax);
    const vint32m1_t s =
        __riscv_vredsum_vs_i32m4_i32m1(acc_s, __riscv_vmv_v_x_i32m1(0, vlmax),
                                       vlmax);
    *sse = (uint32_t)__riscv_vmv_x_s_i32m1_i32(ss);
    *sum = __riscv_vmv_x_s_i32m1_i32(s);
  }
}

#define VAR_RVV(W, H)                                                        \
  uint32_t aom_variance##W##x##H##_rvv(const uint8_t *a, int a_stride,       \
                                       const uint8_t *b, int b_stride,       \
                                       uint32_t *sse) {                      \
    int sum;                                                                 \
    variance_rvv_core(a, a_stride, b, b_stride, (W), (H), sse, &sum);        \
    return *sse - (uint32_t)(((int64_t)sum * sum) / ((W) * (H)));            \
  }

VAR_RVV(8, 8)
VAR_RVV(8, 16)
VAR_RVV(16, 8)
VAR_RVV(16, 16)
VAR_RVV(16, 32)
VAR_RVV(16, 64)
VAR_RVV(32, 16)
VAR_RVV(32, 32)
VAR_RVV(32, 64)
VAR_RVV(64, 16)
VAR_RVV(64, 32)
VAR_RVV(64, 64)
VAR_RVV(64, 128)
VAR_RVV(128, 64)
VAR_RVV(128, 128)
