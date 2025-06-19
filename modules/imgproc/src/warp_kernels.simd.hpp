// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <numeric>
#include "precomp.hpp"
#include "warp_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

#define CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 M0 = vx_setall_f32(M[0]), \
              M3 = vx_setall_f32(M[3]); \
    v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])), \
              M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));
#define CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1() \
    CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 M6 = vx_setall_f32(M[6]); \
    v_float32 M_w = vx_setall_f32(static_cast<float>(y * M[7] + M[8]));
#define CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 dst_y = vx_setall_f32(float(y));

#define CV_WARP_VECTOR_GET_ADDR_C1() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0), \
            addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
#define CV_WARP_VECTOR_GET_ADDR_C3() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
#define CV_WARP_VECTOR_GET_ADDR_C4() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
#define CV_WARP_VECTOR_GET_ADDR(CN) \
    CV_WARP_VECTOR_GET_ADDR_##CN() \
    vx_store(addr, addr_0); \
    vx_store(addr + vlanes_32, addr_1);

#define CV_WARP_VECTOR_LINEAR_COMPUTE_COORD() \
    v_int32 src_ix0 = v_floor(src_x0), src_iy0 = v_floor(src_y0); \
    v_int32 src_ix1 = v_floor(src_x1), src_iy1 = v_floor(src_y1); \
    src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0)); \
    src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0)); \
    src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1)); \
    src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1));
#define CV_WARP_VECTOR_NEAREST_COMPUTE_COORD() \
    v_int32 src_ix0 = v_round(src_x0), src_iy0 = v_round(src_y0); \
    v_int32 src_ix1 = v_round(src_x1), src_iy1 = v_round(src_y1); \

#define CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN) \
    CV_WARP_VECTOR_##INTER##_COMPUTE_COORD() \
    v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols), \
             mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols); \
    mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows)); \
    mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows)); \
    v_uint16 inner_mask = v_pack(mask_0, mask_1); \
    CV_WARP_VECTOR_GET_ADDR(CN)

#define CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

#define CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_w0 = v_fma(M6, dst_x0, M_w), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y), \
              src_w1 = v_fma(M6, dst_x1, M_w); \
    src_x0 = v_div(src_x0, src_w0); \
    src_y0 = v_div(src_y0, src_w0); \
    src_x1 = v_div(src_x1, src_w1); \
    src_y1 = v_div(src_y1, src_w1); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

#define CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0, src_y0, \
              src_x1, src_y1; \
    if (map1 == map2) { \
        v_load_deinterleave(sx_data + 2*x, src_x0, src_y0); \
        v_load_deinterleave(sy_data + 2*(x+vlanes_32), src_x1, src_y1); \
    } else { \
        src_x0 = vx_load(sx_data+x); \
        src_y0 = vx_load(sy_data+x); \
        src_x1 = vx_load(sx_data+x+vlanes_32); \
        src_y1 = vx_load(sy_data+x+vlanes_32); \
    } \
    if (relative) { \
        src_x0 = v_add(src_x0, dst_x0); \
        src_y0 = v_add(src_y0, dst_y); \
        src_x1 = v_add(src_x1, dst_x1); \
        src_y1 = v_add(src_y1, dst_y); \
        dst_x0 = v_add(dst_x0, delta); \
        dst_x1 = v_add(dst_x1, delta); \
    } \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

namespace cv{
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void warpAffineNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void remapNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);

void warpAffineLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
// Approximate branch that uses FP16 intrinsics if possible
void warpAffineLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);

void warpPerspectiveLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
// Approximate branch that uses FP16 intrinsics if possible
void warpPerspectiveLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);

void remapLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
// Approximate branch that uses FP16 intrinsics if possible
void remapLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {
static inline int borderInterpolate_fast( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    return p;
}
} // anonymous

template <typename T, int channels>
class NearestInterpolation {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");
    using pixel_t = typename std::conditional<std::is_same<T, float>::value, float, int>::type;

public:
    static void run(const T *src, size_t srcstep, T *dstptr, int x, float sx, float sy, int srccols, int srcrows, int border_type, const T *bval, int border_type_x, int border_type_y) {
        int ix = cvRound(sx), iy = cvRound(sy);
        const T *srcptr = src + srcstep * iy + ix * channels;
        pixel_t p[channels];
        if ((((unsigned)ix < (unsigned)(srccols - 1)) & ((unsigned)iy < (unsigned)(srcrows - 1))) != 0) {
            for (int c = 0; c < channels; c++) {
                p[c] = srcptr[c];
            }
        } else {
            if ((border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) &&
                (((unsigned)(ix + 1) >= (unsigned)(srccols + 1)) | ((unsigned)(iy + 1) >= (unsigned)(srcrows + 1))) != 0) {
                if (border_type == BORDER_CONSTANT) {
                    for (int c = 0; c < channels; c++) {
                        dstptr[x * channels + c] = bval[c];
                    }
                }
                return;
            }

            if ((((unsigned)(ix) < (unsigned)srccols) & ((unsigned)(iy) < (unsigned)srcrows)) != 0) {
                for (int c = 0; c < channels; c++) {
                    p[c] = srcptr[c];
                }
            } else if (border_type == BORDER_CONSTANT) {
                for (int c = 0; c < channels; c++) {
                    p[c] = bval[c];
                }
            } else if (border_type == BORDER_TRANSPARENT) {
                for (int c = 0; c < channels; c++) {
                    p[c] = dstptr[x * channels + c];
                }
            } else {
                int _ix = borderInterpolate_fast(ix, srccols, border_type_x);
                int _iy = borderInterpolate_fast(iy, srcrows, border_type_y);
                size_t glob_ofs = _iy * srcstep + _ix * channels;
                for (int c = 0; c < channels; c++) {
                    p[c] = src[glob_ofs + c];
                }
            }
        }
        for (int c = 0; c < channels; c++) {
            dstptr[x * channels + c] = saturate_cast<T>(p[c]);
        }
    }
};

template <typename T, int channels>
class WarpAffineNearestInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    WarpAffineNearestInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                                 srccols(0), srcrows(0), dstcols(0), M{0}, bval{0},
                                 border_type(-1), border_type_x(-1), border_type_y(-1) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    const double dM[6], int border_type, const double border_value[4]) {
        WarpAffineNearestInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 6; i++) {
            worker.M[i] = static_cast<float>(dM[i]);
        }
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            int x = 0;
            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                NearestInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    float M[6];
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
};

template <typename T, int channels>
class WarpPerspectiveNearestInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    WarpPerspectiveNearestInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                                      srccols(0), srcrows(0), dstcols(0), M{0}, bval{0},
                                      border_type(-1), border_type_x(-1), border_type_y(-1) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    const double dM[6], int border_type, const double border_value[4]) {
        WarpPerspectiveNearestInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 9; i++) {
            worker.M[i] = static_cast<float>(dM[i]);
        }
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            int x = 0;
            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                NearestInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    float M[9];
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
};

template <typename T, int channels>
class RemapNearestInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    RemapNearestInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                            srccols(0), srcrows(0), dstcols(0), bval{0},
                            border_type(-1), border_type_x(-1), border_type_y(-1),
                            map1_data(nullptr), map2_data(nullptr),
                            map1_step(0), map2_step(0), relative(false) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    int border_type, const double border_value[4],
                    const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
        RemapNearestInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        worker.map1_data = map1_data;
        worker.map1_step = map1_step / sizeof(float);
        if (map2_data == nullptr) {
            worker.map2_data = worker.map1_data;
            worker.map2_step = worker.map1_step;
        } else {
            worker.map2_data = map2_data;
            worker.map2_step = map2_step / sizeof(float);
        }
        worker.relative = is_relative;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            const float *sx_data = map1_data + y * map1_step;
            const float *sy_data = map2_data + y * map2_step;
            int x = 0;
            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1_data == map2_data) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }
                NearestInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
    const float *map1_data;
    const float *map2_data;
    size_t map1_step;
    size_t map2_step;
    bool relative;
};

template <typename T, int channels>
class LinearInterpolation {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");
    using pixel_t = typename std::conditional<std::is_same<T, float>::value, float, int>::type;

public:
    static void run(const T *src, size_t srcstep, T *dstptr, int x, float sx, float sy, int srccols, int srcrows, int border_type, const T *bval, int border_type_x, int border_type_y) {
        int ix = cvFloor(sx), iy = cvFloor(sy);
        sx -= ix;
        sy -= iy;
        const T *srcptr = src + srcstep * iy + ix * channels;
        pixel_t p00[channels] = {0}, p01[channels] = {0}, p10[channels] = {0}, p11[channels] = {0};
        if ((((unsigned)ix < (unsigned)(srccols - 1)) & ((unsigned)iy < (unsigned)(srcrows - 1))) != 0) {
            for (int c = 0; c < channels; c++) {
                p00[c] = srcptr[c];
                p01[c] = srcptr[c + channels];
                p10[c] = srcptr[srcstep + c];
                p11[c] = srcptr[srcstep + c + channels];
            }
        } else {
            if ((border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) &&
                (((unsigned)(ix + 1) >= (unsigned)(srccols + 1)) | ((unsigned)(iy + 1) >= (unsigned)(srcrows + 1))) != 0) {
                if (border_type == BORDER_CONSTANT) {
                    for (int c = 0; c < channels; c++) {
                        dstptr[x * channels + c] = bval[c];
                    }
                }
                return;
            }

            // p00
            if ((((unsigned)(ix) < (unsigned)srccols) & ((unsigned)(iy) < (unsigned)srcrows)) != 0) {
                for (int c = 0; c < channels; c++) {
                    p00[c] = srcptr[c];
                }
            } else if (border_type == BORDER_CONSTANT) {
                for (int c = 0; c < channels; c++) {
                    p00[c] = bval[c];
                }
            } else if (border_type == BORDER_TRANSPARENT) {
                for (int c = 0; c < channels; c++) {
                    p00[c] = dstptr[x * channels + c];
                }
            } else {
                int _ix = borderInterpolate_fast(ix, srccols, border_type_x);
                int _iy = borderInterpolate_fast(iy, srcrows, border_type_y);
                size_t glob_ofs = _iy * srcstep + _ix * channels;
                for (int c = 0; c < channels; c++) {
                    p00[c] = src[glob_ofs + c];
                }
            }
            // p01
            if ((((unsigned)(ix+1) < (unsigned)srccols) & ((unsigned)(iy) < (unsigned)srcrows)) != 0) {
                for (int c = 0; c < channels; c++) {
                    p01[c] = srcptr[channels + c];
                }
            } else if (border_type == BORDER_CONSTANT) {
                for (int c = 0; c < channels; c++) {
                    p01[c] = bval[c];
                }
            } else if (border_type == BORDER_TRANSPARENT) {
                for (int c = 0; c < channels; c++) {
                    p01[c] = dstptr[x * channels + c];
                }
            } else {
                int _ix = borderInterpolate_fast(ix+1, srccols, border_type_x);
                int _iy = borderInterpolate_fast(iy, srcrows, border_type_y);
                size_t glob_ofs = _iy * srcstep + _ix * channels;
                for (int c = 0; c < channels; c++) {
                    p01[c] = src[glob_ofs + c];
                }
            }
            // p10
            if ((((unsigned)(ix) < (unsigned)srccols) & ((unsigned)(iy+1) < (unsigned)srcrows)) != 0) {
                for (int c = 0; c < channels; c++) {
                    p10[c] = srcptr[srcstep + c];
                }
            } else if (border_type == BORDER_CONSTANT) {
                for (int c = 0; c < channels; c++) {
                    p10[c] = bval[c];
                }
            } else if (border_type == BORDER_TRANSPARENT) {
                for (int c = 0; c < channels; c++) {
                    p10[c] = dstptr[x * channels + c];
                }
            } else {
                int _ix = borderInterpolate_fast(ix, srccols, border_type_x);
                int _iy = borderInterpolate_fast(iy+1, srcrows, border_type_y);
                size_t glob_ofs = _iy * srcstep + _ix * channels;
                for (int c = 0; c < channels; c++) {
                    p10[c] = src[glob_ofs + c];
                }
            }
            // p11
            if ((((unsigned)(ix+1) < (unsigned)srccols) & ((unsigned)(iy+1) < (unsigned)srcrows)) != 0) {
                for (int c = 0; c < channels; c++) {
                    p11[c] = srcptr[srcstep + channels + c];
                }
            } else if (border_type == BORDER_CONSTANT) {
                for (int c = 0; c < channels; c++) {
                    p11[c] = bval[c];
                }
            } else if (border_type == BORDER_TRANSPARENT) {
                for (int c = 0; c < channels; c++) {
                    p11[c] = dstptr[x * channels + c];
                }
            } else {
                int _ix = borderInterpolate_fast(ix+1, srccols, border_type_x);
                int _iy = borderInterpolate_fast(iy+1, srcrows, border_type_y);
                size_t glob_ofs = _iy * srcstep + _ix * channels;
                for (int c = 0; c < channels; c++) {
                    p11[c] = src[glob_ofs + c];
                }
            }
        }
        float v0[channels];
        for (int c = 0; c < channels; c++) {
            v0[c] = p00[c] + sx * (p01[c] - p00[c]);
            float v1 = p10[c] + sx * (p11[c] - p10[c]);
            v0[c] += sy * (v1 - v0[c]);
        }
        for (int c = 0; c < channels; c++) {
            dstptr[x * channels + c] = saturate_cast<T>(v0[c]);
        }
    }
};

template <typename T, int channels>
class WarpAffineLinearInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    WarpAffineLinearInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                                srccols(0), srcrows(0), dstcols(0), M{0}, bval{0},
                                border_type(-1), border_type_x(-1), border_type_y(-1) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    const double dM[6], int border_type, const double border_value[4]) {
        WarpAffineLinearInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 6; i++) {
            worker.M[i] = static_cast<float>(dM[i]);
        }
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            int x = 0;
            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                LinearInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    float M[6];
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
};

template <typename T, int channels>
class WarpPerspectiveLinearInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    WarpPerspectiveLinearInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                                      srccols(0), srcrows(0), dstcols(0), M{0}, bval{0},
                                      border_type(-1), border_type_x(-1), border_type_y(-1) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    const double dM[6], int border_type, const double border_value[4]) {
        WarpPerspectiveLinearInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 9; i++) {
            worker.M[i] = static_cast<float>(dM[i]);
        }
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            int x = 0;
            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                LinearInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    float M[9];
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
};

template <typename T, int channels>
class RemapLinearInvoker : public ParallelLoopBody {
    static_assert(std::is_same<T, uint8_t>::value ||
                  std::is_same<T, uint16_t>::value ||
                  std::is_same<T, float>::value, "T must be uint8_t, uint16_t or float");
    static_assert(channels == 1 || channels == 3 || channels == 4, "channels must be 1 or 3 or 4");

public:
    RemapLinearInvoker() : src(nullptr), dst(nullptr), srcstep(0), dststep(0),
                            srccols(0), srcrows(0), dstcols(0), bval{0},
                            border_type(-1), border_type_x(-1), border_type_y(-1),
                            map1_data(nullptr), map2_data(nullptr),
                            map1_step(0), map2_step(0), relative(false) {}

    static void run(const T *src_data, size_t src_step, int src_rows, int src_cols,
                    T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                    int border_type, const double border_value[4],
                    const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
        RemapLinearInvoker<T, channels> worker;

        worker.src = src_data;
        worker.dst = dst_data;
        worker.srcstep = src_step / sizeof(T);
        worker.dststep = dst_step / sizeof(T);
        worker.srccols = src_cols;
        worker.srcrows = src_rows;
        worker.dstcols = dst_cols;
        for (int i = 0; i < 4; i++) {
            worker.bval[i] = saturate_cast<T>(border_value[i]);
        }
        worker.border_type = border_type;
        worker.border_type_x = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_cols <= 1 ? BORDER_REPLICATE : border_type;
        worker.border_type_y = border_type != BORDER_CONSTANT &&
                               border_type != BORDER_TRANSPARENT &&
                               src_rows <= 1 ? BORDER_REPLICATE : border_type;

        worker.map1_data = map1_data;
        worker.map1_step = map1_step / sizeof(float);
        if (map2_data == nullptr) {
            worker.map2_data = worker.map1_data;
            worker.map2_step = worker.map1_step;
        } else {
            worker.map2_data = map2_data;
            worker.map2_step = map2_step / sizeof(float);
        }
        worker.relative = is_relative;

        parallel_for_(Range(0, dst_rows), worker);
    }

    void operator()(const Range &r) const CV_OVERRIDE {
        for (int y = r.start; y < r.end; y++) {
            T* dstptr = dst + y * dststep;
            const float *sx_data = map1_data + y * map1_step;
            const float *sy_data = map2_data + y * map2_step;
            int x = 0;
            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1_data == map2_data) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }
                LinearInterpolation<T, channels>::run(src, srcstep, dstptr, x, sx, sy, srccols, srcrows, border_type, bval, border_type_x, border_type_y);
            }
        }
    }

    const T *src;
    T *dst;
    size_t srcstep;
    size_t dststep;
    int srccols;
    int srcrows;
    int dstcols;
    T bval[4];
    int border_type;
    int border_type_x;
    int border_type_y;
    const float *map1_data;
    const float *map2_data;
    size_t map1_step;
    size_t map2_step;
    bool relative;
};

void warpAffineNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);

//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 int ix = cvRound(sx), iy = cvRound(sy);
//                 int p00g;
//                 const uint16_t *srcptr = src + srcstep * iy + ix;
//                 if ((((unsigned)ix < (unsigned)(srccols - 1)) &
//                      ((unsigned)iy < (unsigned)(srcrows - 1))) != 0) {
//                   p00g = srcptr[0];
//                 } else {
//                   if ((border_type == BORDER_CONSTANT ||
//                        border_type == BORDER_TRANSPARENT) &&
//                       (((unsigned)(ix + 1) >= (unsigned)(srccols + 1)) |
//                        ((unsigned)(iy + 1) >= (unsigned)(srcrows + 1))) != 0) {
//                     if (border_type == BORDER_CONSTANT) {
//                       dstptr[x] = bval[0];
//                     }
//                     continue;
//                   }
//                   if ((((unsigned)(ix + 0) < (unsigned)srccols) &
//                        ((unsigned)(iy + 0) < (unsigned)srcrows)) != 0) {
//                     size_t ofs = 0 * srcstep + 0;
//                     p00g = srcptr[ofs];
//                   } else if (border_type == BORDER_CONSTANT) {
//                     p00g = bval[0];
//                   } else if (border_type == BORDER_TRANSPARENT) {
//                     p00g = dstptr[x];
//                   } else {
//                     int ix_ =
//                         borderInterpolate_fast(ix + 0, srccols, border_type_x);
//                     int iy_ =
//                         borderInterpolate_fast(iy + 0, srcrows, border_type_y);
//                     size_t glob_ofs = iy_ * srcstep + ix_;
//                     p00g = src[glob_ofs];
//                   }
//                 };

//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);

//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);

//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);

//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineNearestInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);

//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             // CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             v_float32 dst_x0 = vx_load(start_indices.data());
//             v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
//             v_float32 M0 = vx_setall_f32(M[0]), M3 = vx_setall_f32(M[3]);
//             v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
//                     M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));
//             v_float32 M6 = vx_setall_f32(M[6]);
//             v_float32 M_w = vx_setall_f32(static_cast<float>(y * M[7] + M[8]));
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveNearestInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;
//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                        map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative)  {
    RemapNearestInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                        map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative)  {
    RemapNearestInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                        map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
//                 CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapNearestInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value,
                                       map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
//                     CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
//                     CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }
//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);
//                 CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
//     #endif
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00g, p01g, p10g, p11g;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
//                   greens = {1, 9, 17, 25, 4, 12, 20, 28},
//                   blues = {2, 10, 18, 26, 5, 13, 21, 29};
//     #endif
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00r, p01r, p10r, p11r,
//                           p00g, p01g, p10g, p11g,
//                           p00b, p01b, p10b, p11b;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
//             }
//         }
//     };

//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();

//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick

//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);

//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}


void warpAffineLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 int ix = cvFloor(sx), iy = cvFloor(sy);
//                 sx -= ix;
//                 sy -= iy;
//                 int p00r, p01r, p10r, p11r;
//                 int p00g, p01g, p10g, p11g;
//                 int p00b, p01b, p10b, p11b;
//                 const uint16_t *srcptr = src + srcstep * iy + ix * 3;
//                 if ((((unsigned)ix < (unsigned)(srccols - 1)) &
//                      ((unsigned)iy < (unsigned)(srcrows - 1))) != 0) {
//                   p00r = srcptr[0];
//                   p01r = srcptr[0 + 3];
//                   p10r = srcptr[srcstep + 0];
//                   p11r = srcptr[srcstep + 3 + 0];
//                   p00g = srcptr[1];
//                   p01g = srcptr[1 + 3];
//                   p10g = srcptr[srcstep + 1];
//                   p11g = srcptr[srcstep + 3 + 1];
//                   p00b = srcptr[2];
//                   p01b = srcptr[2 + 3];
//                   p10b = srcptr[srcstep + 2];
//                   p11b = srcptr[srcstep + 3 + 2];
//                 } else {
//                   if ((border_type == BORDER_CONSTANT ||
//                        border_type == BORDER_TRANSPARENT) &&
//                       (((unsigned)(ix + 1) >= (unsigned)(srccols + 1)) |
//                        ((unsigned)(iy + 1) >= (unsigned)(srcrows + 1))) != 0) {
//                     if (border_type == BORDER_CONSTANT) {
//                       dstptr[x * 3] = bval[0];
//                       dstptr[x * 3 + 1] = bval[1];
//                       dstptr[x * 3 + 2] = bval[2];
//                     }
//                     continue;
//                   }
//                   if ((((unsigned)(ix + 0) < (unsigned)srccols) &
//                        ((unsigned)(iy + 0) < (unsigned)srcrows)) != 0) {
//                     size_t ofs = 0 * srcstep + 0 * 3;
//                     p00r = srcptr[ofs];
//                     p00g = srcptr[ofs + 1];
//                     p00b = srcptr[ofs + 2];
//                   } else if (border_type == BORDER_CONSTANT) {
//                     p00r = bval[0];
//                     p00g = bval[1];
//                     p00b = bval[2];
//                   } else if (border_type == BORDER_TRANSPARENT) {
//                     p00r = dstptr[x * 3];
//                     p00g = dstptr[x * 3 + 1];
//                     p00b = dstptr[x * 3 + 2];
//                   } else {
//                     int ix_ =
//                         borderInterpolate_fast(ix + 0, srccols, border_type_x);
//                     int iy_ =
//                         borderInterpolate_fast(iy + 0, srcrows, border_type_y);
//                     size_t glob_ofs = iy_ * srcstep + ix_ * 3;
//                     p00r = src[glob_ofs];
//                     p00g = src[glob_ofs + 1];
//                     p00b = src[glob_ofs + 2];
//                   }
//                   if ((((unsigned)(ix + 1) < (unsigned)srccols) &
//                        ((unsigned)(iy + 0) < (unsigned)srcrows)) != 0) {
//                     size_t ofs = 0 * srcstep + 1 * 3;
//                     p01r = srcptr[ofs];
//                     p01g = srcptr[ofs + 1];
//                     p01b = srcptr[ofs + 2];
//                   } else if (border_type == BORDER_CONSTANT) {
//                     p01r = bval[0];
//                     p01g = bval[1];
//                     p01b = bval[2];
//                   } else if (border_type == BORDER_TRANSPARENT) {
//                     p01r = dstptr[x * 3];
//                     p01g = dstptr[x * 3 + 1];
//                     p01b = dstptr[x * 3 + 2];
//                   } else {
//                     int ix_ =
//                         borderInterpolate_fast(ix + 1, srccols, border_type_x);
//                     int iy_ =
//                         borderInterpolate_fast(iy + 0, srcrows, border_type_y);
//                     size_t glob_ofs = iy_ * srcstep + ix_ * 3;
//                     p01r = src[glob_ofs];
//                     p01g = src[glob_ofs + 1];
//                     p01b = src[glob_ofs + 2];
//                   }
//                   if ((((unsigned)(ix + 0) < (unsigned)srccols) &
//                        ((unsigned)(iy + 1) < (unsigned)srcrows)) != 0) {
//                     size_t ofs = 1 * srcstep + 0 * 3;
//                     p10r = srcptr[ofs];
//                     p10g = srcptr[ofs + 1];
//                     p10b = srcptr[ofs + 2];
//                   } else if (border_type == BORDER_CONSTANT) {
//                     p10r = bval[0];
//                     p10g = bval[1];
//                     p10b = bval[2];
//                   } else if (border_type == BORDER_TRANSPARENT) {
//                     p10r = dstptr[x * 3];
//                     p10g = dstptr[x * 3 + 1];
//                     p10b = dstptr[x * 3 + 2];
//                   } else {
//                     int ix_ =
//                         borderInterpolate_fast(ix + 0, srccols, border_type_x);
//                     int iy_ =
//                         borderInterpolate_fast(iy + 1, srcrows, border_type_y);
//                     size_t glob_ofs = iy_ * srcstep + ix_ * 3;
//                     p10r = src[glob_ofs];
//                     p10g = src[glob_ofs + 1];
//                     p10b = src[glob_ofs + 2];
//                   }
//                   if ((((unsigned)(ix + 1) < (unsigned)srccols) &
//                        ((unsigned)(iy + 1) < (unsigned)srcrows)) != 0) {
//                     size_t ofs = 1 * srcstep + 1 * 3;
//                     p11r = srcptr[ofs];
//                     p11g = srcptr[ofs + 1];
//                     p11b = srcptr[ofs + 2];
//                   } else if (border_type == BORDER_CONSTANT) {
//                     p11r = bval[0];
//                     p11g = bval[1];
//                     p11b = bval[2];
//                   } else if (border_type == BORDER_TRANSPARENT) {
//                     p11r = dstptr[x * 3];
//                     p11g = dstptr[x * 3 + 1];
//                     p11b = dstptr[x * 3 + 2];
//                   } else {
//                     int ix_ =
//                         borderInterpolate_fast(ix + 1, srccols, border_type_x);
//                     int iy_ =
//                         borderInterpolate_fast(iy + 1, srcrows, border_type_y);
//                     size_t glob_ofs = iy_ * srcstep + ix_ * 3;
//                     p11r = src[glob_ofs];
//                     p11g = src[glob_ofs + 1];
//                     p11b = src[glob_ofs + 2];
//                   }
//                 };
//                 float v0r = p00r + sx * (p01r - p00r);
//                 float v1r = p10r + sx * (p11r - p10r);
//                 float v0g = p00g + sx * (p01g - p00g);
//                 float v1g = p10g + sx * (p11g - p10g);
//                 float v0b = p00b + sx * (p01b - p00b);
//                 float v1b = p10b + sx * (p11b - p10b);
//                 v0r += sy * (v1r - v0r);
//                 v0g += sy * (v1g - v0g);
//                 v0b += sy * (v1b - v0b);
//                 ;
//                 dstptr[x * 3] = saturate_cast<uint16_t>(v0r);
//                 dstptr[x * 3 + 1] = saturate_cast<uint16_t>(v0g);
//                 dstptr[x * 3 + 2] = saturate_cast<uint16_t>(v0b);
//                 ;
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}


void warpAffineLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    WarpAffineLinearInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[6];
//         for (int i = 0; i < 6; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
//                     // CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx = x*M[0] + y*M[1] + M[2];
//                 float sy = x*M[3] + y*M[4] + M[5];

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpAffineLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);

                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }

    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpAffineLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;

                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
//     #endif
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00g, p01g, p10g, p11g;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
//                   greens = {1, 9, 17, 25, 4, 12, 20, 28},
//                   blues = {2, 10, 18, 26, 5, 13, 21, 29};
//     #endif
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00r, p01r, p10r, p11r,
//                           p00g, p01g, p10g, p11g,
//                           p00b, p01b, p10b, p11b;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step, dststep = dst_step;
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 16U);

//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);

//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);

//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);

//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    WarpPerspectiveLinearInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         float M[9];
//         for (int i = 0; i < 9; i++) {
//             M[i] = static_cast<float>(dM[i]);
//         }
//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float w = x*M[6] + y*M[7] + M[8];
//                 float sx = (x*M[0] + y*M[1] + M[2]) / w;
//                 float sy = (x*M[3] + y*M[4] + M[5]) / w;

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void remapLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<uchar, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4];

//         uint8_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
//     #endif
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00g, p01g, p10g, p11g;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<uchar, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint8_t pixbuf[max_uf*4*3];

//         uint8_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//         uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
//                   greens = {1, 9, 17, 25, 4, 12, 20, 28},
//                   blues = {2, 10, 18, 26, 5, 13, 21, 29};
//     #endif
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 uint8x8_t p00r, p01r, p10r, p11r,
//                           p00g, p01g, p10g, p11g,
//                           p00b, p01b, p10b, p11b;
//     #endif
//                 if (v_reduce_min(inner_mask) != 0) {
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
//     #else
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
//     #endif
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                     CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
//     #endif
//                 }
//     #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
//                 CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
//     #else
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
//     #endif
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<uchar, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         const auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step, dststep = dst_step,
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint8_t bval[] = {
//             saturate_cast<uint8_t>(border_value[0]),
//             saturate_cast<uint8_t>(border_value[1]),
//             saturate_cast<uint8_t>(border_value[2]),
//             saturate_cast<uint8_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_16 = VTraits<v_uint16>::vlanes();
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint8_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
//         v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
//         v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
//         v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint8_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
//     #endif
//                 } else {
//                     uint8_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<ushort, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4];

//         uint16_t bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<ushort, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         uint16_t pixbuf[max_uf*4*3];

//         uint16_t bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
//                 CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<ushort, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         uint16_t bval[] = {
//             saturate_cast<uint16_t>(border_value[0]),
//             saturate_cast<uint16_t>(border_value[1]),
//             saturate_cast<uint16_t>(border_value[2]),
//             saturate_cast<uint16_t>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         uint16_t bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
//         v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
//         v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
//         v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             uint16_t* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
//     #endif
//                 } else {
//                     uint16_t pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
//                     CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<float, 1>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4];

//         float bvalbuf[max_uf];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i] = bval[0];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
//                 CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<float, 3>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];
//         float pixbuf[max_uf*4*3];

//         float bvalbuf[max_uf*3];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*3] = bval[0];
//             bvalbuf[i*3+1] = bval[1];
//             bvalbuf[i*3+2] = bval[2];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
//                 } else {
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
//                 }
//                 CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
//                 CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
//                 CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
//                 CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    RemapLinearInvoker<float, 4>::run(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
//     auto worker = [&](const Range &r) {
//         CV_INSTRUMENT_REGION();

//         const auto *src = src_data;
//         auto *dst = dst_data;
//         auto *map1 = map1_data, *map2 = map2_data;
//         size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
//                map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
//         if (map2 == nullptr) {
//             map2 = map1;
//             map2step = map1step;
//         }
//         int srccols = src_cols, srcrows = src_rows;
//         int dstcols = dst_cols;
//         bool relative = is_relative;

//         float bval[] = {
//             saturate_cast<float>(border_value[0]),
//             saturate_cast<float>(border_value[1]),
//             saturate_cast<float>(border_value[2]),
//             saturate_cast<float>(border_value[3]),
//         };
//         int border_type_x = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srccols <= 1 ? BORDER_REPLICATE : border_type;
//         int border_type_y = border_type != BORDER_CONSTANT &&
//                             border_type != BORDER_TRANSPARENT &&
//                             srcrows <= 1 ? BORDER_REPLICATE : border_type;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//         constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
//         constexpr int max_uf{max_vlanes_32*2};
//         int vlanes_32 = VTraits<v_float32>::vlanes();
//         // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
//         int uf = vlanes_32 * 2;

//         std::array<float, max_vlanes_32> start_indices;
//         std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

//         v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
//                  inner_scols = vx_setall_u32((unsigned)srccols - 1),
//                  outer_srows = vx_setall_u32((unsigned)srcrows + 1),
//                  outer_scols = vx_setall_u32((unsigned)srccols + 1);
//         v_float32 delta = vx_setall_f32(static_cast<float>(uf));
//         v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
//         v_int32 v_srcstep = vx_setall_s32(int(srcstep));
//         int32_t addr[max_uf],
//                 src_ix[max_uf],
//                 src_iy[max_uf];

//         float bvalbuf[max_uf*4];
//         for (int i = 0; i < uf; i++) {
//             bvalbuf[i*4] = bval[0];
//             bvalbuf[i*4+1] = bval[1];
//             bvalbuf[i*4+2] = bval[2];
//             bvalbuf[i*4+3] = bval[3];
//         }
//         v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
//         v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
//         v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
//         v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
//         v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
//         v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
//         v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
//         v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
// #endif

//         for (int y = r.start; y < r.end; y++) {
//             float* dstptr = dst + y*dststep;
//             const float *sx_data = map1 + y*map1step;
//             const float *sy_data = map2 + y*map2step;
//             int x = 0;

// #if (CV_SIMD || CV_SIMD_SCALABLE)
//             CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
//             for (; x <= dstcols - uf; x += uf) {
//                 // [TODO] apply halide trick
//                 CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
//                 if (v_reduce_min(inner_mask) != 0) {
//                     float valpha[max_uf], vbeta[max_uf];
//                     vx_store(valpha, src_x0);
//                     vx_store(valpha+vlanes_32, src_x1);
//                     vx_store(vbeta, src_y0);
//                     vx_store(vbeta+vlanes_32, src_y1);
//     #if CV_SIMD256
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
//     #elif CV_SIMD128
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
//     #elif CV_SIMD_SCALABLE
//                     CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
//     #endif
//                 } else {
//                     float pixbuf[max_uf*4*4];
//                     CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
//                     CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
//                     CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
//                     CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
//                 }
//             }
// #endif // (CV_SIMD || CV_SIMD_SCALABLE)

//             for (; x < dstcols; x++) {
//                 float sx, sy;
//                 if (map1 == map2) {
//                     sx = sx_data[2*x];
//                     sy = sy_data[2*x+1];
//                 } else {
//                     sx = sx_data[x];
//                     sy = sy_data[x];
//                 }

//                 if (relative) {
//                     sx += x;
//                     sy += y;
//                 }

//                 CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
//                 CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
//                 CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
//             }
//         }
//     };
//     parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}
void remapLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}
void remapLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
