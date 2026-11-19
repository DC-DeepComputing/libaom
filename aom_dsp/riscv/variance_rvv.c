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
#include "aom_dsp/aom_filter.h"

static void variance_rvv_core(const uint8_t *a, int a_stride,
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

/* ---------------- sub-pixel variance (bilinear 2-tap + variance) ------- */

/* first pass: u8 * 2 taps -> u16, (a[i]*f0 + a[i+pstep]*f1 + 64) >> 7.
 * max value 255*128 = 32640, fits u16 with no clipping. */
static void bil_first_pass_rvv(const uint8_t *a, uint16_t *b, int sppl,
                               int pstep, int oh, int ow, const uint8_t *f) {
  const int f0 = f[0], f1 = f[1];
  int i;
  for (i = 0; i < oh; ++i) {
    int x = 0;
    while (x < ow) {
      const size_t vl = __riscv_vsetvl_e16m2((size_t)(ow - x));
      const vuint16m2_t wa = __riscv_vzext_vf2_u16m2(
          __riscv_vle8_v_u8m1(a + x, vl), vl);
      const vuint16m2_t wb = __riscv_vzext_vf2_u16m2(
          __riscv_vle8_v_u8m1(a + x + pstep, vl), vl);
      vuint16m2_t t = __riscv_vmul_vx_u16m2(wa, f0, vl);
      t = __riscv_vmacc_vx_u16m2(t, f1, wb, vl);
      t = __riscv_vsrl_vx_u16m2(__riscv_vadd_vx_u16m2(t, 64, vl), 7, vl);
      __riscv_vse16_v_u16m2(b + x, t, vl);
      x += (int)vl;
    }
    a += sppl;
    b += ow;
  }
}

/* second pass: u16 * 2 taps -> u8 (max 255, no clipping). */
static void bil_second_pass_rvv(const uint16_t *a, uint8_t *b, int sppl,
                                int pstep, int oh, int ow, const uint8_t *f) {
  const int f0 = f[0], f1 = f[1];
  int i;
  for (i = 0; i < oh; ++i) {
    int x = 0;
    while (x < ow) {
      const size_t vl = __riscv_vsetvl_e16m2((size_t)(ow - x));
      const vuint16m2_t va = __riscv_vle16_v_u16m2(a + x, vl);
      const vuint16m2_t vb = __riscv_vle16_v_u16m2(a + x + pstep, vl);
      vuint16m2_t t = __riscv_vmul_vx_u16m2(va, f0, vl);
      t = __riscv_vmacc_vx_u16m2(t, f1, vb, vl);
      t = __riscv_vsrl_vx_u16m2(__riscv_vadd_vx_u16m2(t, 64, vl), 7, vl);
      __riscv_vse8_v_u8m1(b + x, __riscv_vncvt_x_x_w_u8m1(t, vl), vl);
      x += (int)vl;
    }
    a += sppl;
    b += ow;
  }
}

/* comp_pred = (pred + ref + 1) >> 1, matching aom_comp_avg_pred_c. */
static void comp_avg_pred_rvv(uint8_t *comp, const uint8_t *pred, int w,
                              int h, const uint8_t *ref, int rs) {
  int i;
  for (i = 0; i < h; ++i) {
    int x = 0;
    while (x < w) {
      const size_t vl = __riscv_vsetvl_e8m1((size_t)(w - x));
      const vuint16m2_t t = __riscv_vadd_vx_u16m2(
          __riscv_vadd_vv_u16m2(__riscv_vzext_vf2_u16m2(
                                    __riscv_vle8_v_u8m1(pred + x, vl), vl),
                                __riscv_vzext_vf2_u16m2(
                                    __riscv_vle8_v_u8m1(ref + x, vl), vl),
                                vl),
          1, vl);
      __riscv_vse8_v_u8m1(
          comp + x,
          __riscv_vncvt_x_x_w_u8m1(__riscv_vsrl_vx_u16m2(t, 1, vl), vl), vl);
      x += (int)vl;
    }
    comp += w;
    pred += w;
    ref += rs;
  }
}

#define SUBPIX_VAR_RVV(W, H)                                                 \
  uint32_t aom_sub_pixel_variance##W##x##H##_rvv(                            \
      const uint8_t *a, int a_stride, int xoffset, int yoffset,              \
      const uint8_t *b, int b_stride, uint32_t *sse) {                       \
    DECLARE_ALIGNED(16, uint16_t, fdata3[(H + 1) * (W)]);                    \
    DECLARE_ALIGNED(16, uint8_t, temp2[(H) * (W)]);                          \
    bil_first_pass_rvv(a, fdata3, a_stride, 1, (H) + 1, (W),                 \
                       bilinear_filters_2t[xoffset]);                        \
    bil_second_pass_rvv(fdata3, temp2, (W), (W), (H), (W),                   \
                        bilinear_filters_2t[yoffset]);                       \
    {                                                                        \
      int sum;                                                               \
      variance_rvv_core(temp2, (W), b, b_stride, (W), (H), sse, &sum);       \
      return *sse - (uint32_t)(((int64_t)sum * sum) / ((W) * (H)));          \
    }                                                                        \
  }

#define SUBPIX_AVG_VAR_RVV(W, H)                                             \
  uint32_t aom_sub_pixel_avg_variance##W##x##H##_rvv(                        \
      const uint8_t *a, int a_stride, int xoffset, int yoffset,              \
      const uint8_t *b, int b_stride, uint32_t *sse,                         \
      const uint8_t *second_pred) {                                          \
    DECLARE_ALIGNED(16, uint16_t, fdata3[(H + 1) * (W)]);                    \
    DECLARE_ALIGNED(16, uint8_t, temp2[(H) * (W)]);                          \
    DECLARE_ALIGNED(16, uint8_t, comp_pred[(H) * (W)]);                      \
    bil_first_pass_rvv(a, fdata3, a_stride, 1, (H) + 1, (W),                 \
                       bilinear_filters_2t[xoffset]);                        \
    bil_second_pass_rvv(fdata3, temp2, (W), (W), (H), (W),                   \
                        bilinear_filters_2t[yoffset]);                       \
    comp_avg_pred_rvv(comp_pred, second_pred, (W), (H), temp2, (W));         \
    {                                                                        \
      int sum;                                                               \
      variance_rvv_core(comp_pred, (W), b, b_stride, (W), (H), sse, &sum);   \
      return *sse - (uint32_t)(((int64_t)sum * sum) / ((W) * (H)));          \
    }                                                                        \
  }

SUBPIX_VAR_RVV(8, 8)
SUBPIX_VAR_RVV(8, 16)
SUBPIX_VAR_RVV(16, 8)
SUBPIX_VAR_RVV(16, 16)
SUBPIX_VAR_RVV(16, 32)
SUBPIX_VAR_RVV(16, 64)
SUBPIX_VAR_RVV(32, 16)
SUBPIX_VAR_RVV(32, 32)
SUBPIX_VAR_RVV(32, 64)
SUBPIX_VAR_RVV(64, 16)
SUBPIX_VAR_RVV(64, 32)
SUBPIX_VAR_RVV(64, 64)
SUBPIX_VAR_RVV(64, 128)
SUBPIX_VAR_RVV(128, 64)
SUBPIX_VAR_RVV(128, 128)

SUBPIX_AVG_VAR_RVV(8, 8)
SUBPIX_AVG_VAR_RVV(8, 16)
SUBPIX_AVG_VAR_RVV(16, 8)
SUBPIX_AVG_VAR_RVV(16, 16)
SUBPIX_AVG_VAR_RVV(16, 32)
SUBPIX_AVG_VAR_RVV(16, 64)
SUBPIX_AVG_VAR_RVV(32, 16)
SUBPIX_AVG_VAR_RVV(32, 32)
SUBPIX_AVG_VAR_RVV(32, 64)
SUBPIX_AVG_VAR_RVV(64, 16)
SUBPIX_AVG_VAR_RVV(64, 32)
SUBPIX_AVG_VAR_RVV(64, 64)
SUBPIX_AVG_VAR_RVV(64, 128)
SUBPIX_AVG_VAR_RVV(128, 64)
SUBPIX_AVG_VAR_RVV(128, 128)

/* sum_sse_2d: i64 accumulation (single 128x128 block can reach 2^37,
 * exceeding i32), one chunked pass, row-level scalar accumulation. */
uint64_t aom_sum_sse_2d_i16_rvv(const int16_t *src, int src_stride, int width,
                                int height, int *sum) {
  uint64_t tss = 0;
  int32_t tsum = 0;
  int r, x;
  for (r = 0; r < height; ++r) {
    for (x = 0; x < width; x += 32) {
      const size_t vl = __riscv_vsetvl_e16m2((size_t)(width - x));
      const vint16m2_t v = __riscv_vle16_v_i16m2(src + x, vl);
      const vint64m8_t sq = __riscv_vsext_vf2_i64m8(
          __riscv_vwmul_vv_i32m4(v, v, vl), vl);
      const vint64m1_t ss = __riscv_vredsum_vs_i64m8_i64m1(
          sq, __riscv_vmv_v_x_i64m1(0, vl), vl);
      const vint32m1_t s = __riscv_vwredsum_vs_i16m2_i32m1(
          v, __riscv_vmv_v_x_i32m1(0, vl), vl);
      tss += (uint64_t)__riscv_vmv_x_s_i64m1_i64(ss);
      tsum += __riscv_vmv_x_s_i32m1_i32(s);
    }
    src += src_stride;
  }
  *sum = tsum;
  return tss;
}
