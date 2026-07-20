// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

inline int meanStdDev_8UC1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);
inline int meanStdDev_8UC4(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val);
inline int meanStdDev_32FC1(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

int meanStdDev(const uchar* src_data, size_t src_step, int width, int height, int src_type,
               double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {

    // The universal-intrinsics path is faster for unmasked mean-only calls.
    // See https://github.com/opencv/opencv/pull/29339#issuecomment-4778104352
    if (!mask && !stddev_val)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    switch (src_type)
    {
    case CV_8UC1:
        return meanStdDev_8UC1(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    case CV_8UC4:
        if (mask)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        return meanStdDev_8UC4(src_data, src_step, width, height, mean_val, stddev_val);
    case CV_32FC1:
        return meanStdDev_32FC1(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
}

static inline void reduce_8UC1(vuint32m4_t vec_sum, vuint32m4_t vec_sqsum,
                               uint64_t& sum, uint64_t& sqsum, int vlmax)
{
    auto zero = __riscv_vmv_s_x_u64m1(0, vlmax);
    auto vec_red = __riscv_vwredsumu(vec_sum, zero, vlmax);
    auto vec_sqred = __riscv_vwredsumu(vec_sqsum, zero, vlmax);
    sum += __riscv_vmv_x(vec_red);
    sqsum += __riscv_vmv_x(vec_sqred);
}

inline int meanStdDev_8UC1(const uchar* src_data, size_t src_step, int width, int height,
                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    constexpr int max_accumulations = std::numeric_limits<uint32_t>::max() / (255 * 255);
    int64_t nz = 0;
    int vlmax = __riscv_vsetvlmax_e32m4();
    const int block_size = max_accumulations * vlmax;
    vuint16m2_t vec_one = __riscv_vmv_v_x_u16m2(1, vlmax);
    uint64_t sum = 0, sqsum = 0;
    if (mask) {
        for (int i = 0; i < height; ++i) {
            const uchar* src_row = src_data + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int j = 0;
            while (j < width) {
                const int block_end = j + std::min(block_size, width - j);
                vuint32m4_t vec_sum = __riscv_vmv_v_x_u32m4(0, vlmax);
                vuint32m4_t vec_sqsum = __riscv_vmv_v_x_u32m4(0, vlmax);
                int vl;
                for ( ; j < block_end; j += vl) {
                    vl = __riscv_vsetvl_e8m1(block_end - j);
                    auto vec_pixel_u8 = __riscv_vle8_v_u8m1(src_row + j, vl);
                    auto vmask_u8 = __riscv_vle8_v_u8m1(mask_row + j, vl);
                    auto vec_pixel = __riscv_vzext_vf2(vec_pixel_u8, vl);
                    auto vmask = __riscv_vmsne_vx_u8m1_b8(vmask_u8, 0, vl);
                    vec_sum = __riscv_vwmaccu_vv_u32m4_tumu(vmask, vec_sum, vec_one, vec_pixel, vl);
                    vec_sqsum = __riscv_vwmaccu_vv_u32m4_tumu(vmask, vec_sqsum, vec_pixel, vec_pixel, vl);
                    nz += __riscv_vcpop_m_b8(vmask, vl);
                }
                reduce_8UC1(vec_sum, vec_sqsum, sum, sqsum, vlmax);
            }
        }
    } else {
        for (int i = 0; i < height; i++) {
            const uchar* src_row = src_data + i * src_step;
            int j = 0;
            while (j < width) {
                const int block_end = j + std::min(block_size, width - j);
                vuint32m4_t vec_sum = __riscv_vmv_v_x_u32m4(0, vlmax);
                vuint32m4_t vec_sqsum = __riscv_vmv_v_x_u32m4(0, vlmax);
                int vl;
                for ( ; j < block_end; j += vl) {
                    vl = __riscv_vsetvl_e8m1(block_end - j);
                    auto vec_pixel_u8 = __riscv_vle8_v_u8m1(src_row + j, vl);
                    auto vec_pixel = __riscv_vzext_vf2(vec_pixel_u8, vl);
                    vec_sum = __riscv_vwmaccu_vv_u32m4_tu(vec_sum, vec_one, vec_pixel, vl);
                    vec_sqsum = __riscv_vwmaccu_vv_u32m4_tu(vec_sqsum, vec_pixel, vec_pixel, vl);
                }
                reduce_8UC1(vec_sum, vec_sqsum, sum, sqsum, vlmax);
            }
        }
        nz = static_cast<int64_t>(height) * width;
    }
    if (nz == 0) {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    double mean = static_cast<double>(sum) / nz;
    if (mean_val)
        *mean_val = mean;
    if (stddev_val)
    {
        double variance = std::max((static_cast<double>(sqsum) / nz) - (mean * mean), 0.0);
        *stddev_val = std::sqrt(variance);
    }
    return CV_HAL_ERROR_OK;
}

static inline void reduce_8UC4(vuint32m4_t vec_sum, vuint32m4_t vec_sqsum,
                               uint64_t sum[4], uint64_t sqsum[4], int vlmax)
{
    auto channel = __riscv_vand(__riscv_vid_v_u32m4(vlmax), 3, vlmax);
    auto zero = __riscv_vmv_s_x_u64m1(0, vlmax);
    for (int i = 0; i < 4; ++i)
    {
        auto channel_mask = __riscv_vmseq(channel, static_cast<uint32_t>(i), vlmax);
        sum[i] += __riscv_vmv_x(__riscv_vwredsumu(channel_mask, vec_sum, zero, vlmax));
        sqsum[i] += __riscv_vmv_x(__riscv_vwredsumu(channel_mask, vec_sqsum, zero, vlmax));
    }
}

inline int meanStdDev_8UC4(const uchar* src_data, size_t src_step, int width, int height,
                             double* mean_val, double* stddev_val) {
    constexpr int max_accumulations = std::numeric_limits<uint32_t>::max() / (255 * 255);
    int64_t nz = static_cast<int64_t>(height) * width;
    int vlmax = __riscv_vsetvlmax_e32m4();
    const int block_size = max_accumulations * vlmax;
    vuint16m2_t vec_one = __riscv_vmv_v_x_u16m2(1, vlmax);
    uint64_t sum[4] = {0}, sqsum[4] = {0};
    for (int i = 0; i < height; i++) {
        const uchar* src_row = src_data + i * src_step;
        const int row_end = width * 4;
        int j = 0;
        while (j < row_end) {
            const int block_end = j + std::min(block_size, row_end - j);
            vuint32m4_t vec_sum = __riscv_vmv_v_x_u32m4(0, vlmax);
            vuint32m4_t vec_sqsum = __riscv_vmv_v_x_u32m4(0, vlmax);
            int vl;
            for ( ; j < block_end; j += vl) {
                vl = __riscv_vsetvl_e8m1(block_end - j);
                auto vec_pixel_u8 = __riscv_vle8_v_u8m1(src_row + j, vl);
                auto vec_pixel = __riscv_vzext_vf2(vec_pixel_u8, vl);
                vec_sum = __riscv_vwmaccu_vv_u32m4_tu(vec_sum, vec_one, vec_pixel, vl);
                vec_sqsum = __riscv_vwmaccu_vv_u32m4_tu(vec_sqsum, vec_pixel, vec_pixel, vl);
            }
            reduce_8UC4(vec_sum, vec_sqsum, sum, sqsum, vlmax);
        }
    }
    if (nz == 0) {
        for (int i = 0; i < 4; ++i)
        {
            if (mean_val) mean_val[i] = 0.0;
            if (stddev_val) stddev_val[i] = 0.0;
        }
        return CV_HAL_ERROR_OK;
    }
    double means[4];
    for (int i = 0; i < 4; ++i)
    {
        means[i] = static_cast<double>(sum[i]) / nz;
        if (mean_val)
            mean_val[i] = means[i];
        if (stddev_val)
            stddev_val[i] = std::sqrt(std::max((static_cast<double>(sqsum[i]) / nz) -
                                               (means[i] * means[i]), 0.0));
    }
    return CV_HAL_ERROR_OK;
}

inline int meanStdDev_32FC1(const uchar* src_data, size_t src_step, int width, int height,
                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    int64_t nz = 0;
    int vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vec_sum0 = __riscv_vfmv_v_f_f64m4(0, vlmax);
    vfloat64m4_t vec_sqsum0 = __riscv_vfmv_v_f_f64m4(0, vlmax);
    vfloat64m4_t vec_sum1 = __riscv_vfmv_v_f_f64m4(0, vlmax);
    vfloat64m4_t vec_sqsum1 = __riscv_vfmv_v_f_f64m4(0, vlmax);
    vfloat32m2_t vec_one = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    src_step /= sizeof(float);
    if (mask) {
        for (int i = 0; i < height; ++i) {
            const float* src_row = reinterpret_cast<const float*>(src_data) + i * src_step;
            const uchar* mask_row = mask + i * mask_step;
            int j = 0;
            for ( ; width - j >= 2 * vlmax; j += 2 * vlmax) {
                auto vec_pixel0 = __riscv_vle32_v_f32m2(src_row + j, vlmax);
                auto vec_pixel1 = __riscv_vle32_v_f32m2(src_row + j + vlmax, vlmax);
                auto vmask_u80 = __riscv_vle8_v_u8mf2(mask_row + j, vlmax);
                auto vmask_u81 = __riscv_vle8_v_u8mf2(mask_row + j + vlmax, vlmax);
                auto vmask0 = __riscv_vmsne_vx_u8mf2_b16(vmask_u80, 0, vlmax);
                auto vmask1 = __riscv_vmsne_vx_u8mf2_b16(vmask_u81, 0, vlmax);
                vec_sum0 = __riscv_vfwmacc_vv_f64m4_tumu(vmask0, vec_sum0, vec_one, vec_pixel0, vlmax);
                vec_sqsum0 = __riscv_vfwmacc_vv_f64m4_tumu(vmask0, vec_sqsum0, vec_pixel0, vec_pixel0, vlmax);
                vec_sum1 = __riscv_vfwmacc_vv_f64m4_tumu(vmask1, vec_sum1, vec_one, vec_pixel1, vlmax);
                vec_sqsum1 = __riscv_vfwmacc_vv_f64m4_tumu(vmask1, vec_sqsum1, vec_pixel1, vec_pixel1, vlmax);
                nz += __riscv_vcpop_m_b16(vmask0, vlmax) + __riscv_vcpop_m_b16(vmask1, vlmax);
            }
            int vl;
            for ( ; j < width; j += vl) {
                vl = __riscv_vsetvl_e32m2(width - j);
                auto vec_pixel = __riscv_vle32_v_f32m2(src_row + j, vl);
                auto vmask_u8 = __riscv_vle8_v_u8mf2(mask_row + j, vl);
                auto vmask = __riscv_vmsne_vx_u8mf2_b16(vmask_u8, 0, vl);
                vec_sum0 = __riscv_vfwmacc_vv_f64m4_tumu(vmask, vec_sum0, vec_one, vec_pixel, vl);
                vec_sqsum0 = __riscv_vfwmacc_vv_f64m4_tumu(vmask, vec_sqsum0, vec_pixel, vec_pixel, vl);
                nz += __riscv_vcpop_m_b16(vmask, vl);
            }
        }
    } else {
        for (int i = 0; i < height; i++) {
            const float* src_row = reinterpret_cast<const float*>(src_data) + i * src_step;
            int j = 0;
            for ( ; width - j >= 2 * vlmax; j += 2 * vlmax) {
                auto vec_pixel0 = __riscv_vle32_v_f32m2(src_row + j, vlmax);
                auto vec_pixel1 = __riscv_vle32_v_f32m2(src_row + j + vlmax, vlmax);
                vec_sum0 = __riscv_vfwmacc_vv_f64m4_tu(vec_sum0, vec_one, vec_pixel0, vlmax);
                vec_sqsum0 = __riscv_vfwmacc_vv_f64m4_tu(vec_sqsum0, vec_pixel0, vec_pixel0, vlmax);
                vec_sum1 = __riscv_vfwmacc_vv_f64m4_tu(vec_sum1, vec_one, vec_pixel1, vlmax);
                vec_sqsum1 = __riscv_vfwmacc_vv_f64m4_tu(vec_sqsum1, vec_pixel1, vec_pixel1, vlmax);
            }
            int vl;
            for ( ; j < width; j += vl) {
                vl = __riscv_vsetvl_e32m2(width - j);
                auto vec_pixel = __riscv_vle32_v_f32m2(src_row + j, vl);
                vec_sum0 = __riscv_vfwmacc_vv_f64m4_tu(vec_sum0, vec_one, vec_pixel, vl);
                vec_sqsum0 = __riscv_vfwmacc_vv_f64m4_tu(vec_sqsum0, vec_pixel, vec_pixel, vl);
            }
        }
        nz = static_cast<int64_t>(height) * width;
    }
    if (nz == 0) {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    vec_sum0 = __riscv_vfadd(vec_sum0, vec_sum1, vlmax);
    vec_sqsum0 = __riscv_vfadd(vec_sqsum0, vec_sqsum1, vlmax);
    auto zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    auto vec_red = __riscv_vfredusum(vec_sum0, zero, vlmax);
    auto vec_reddev = __riscv_vfredusum(vec_sqsum0, zero, vlmax);
    double sum = __riscv_vfmv_f(vec_red);
    double mean = sum / nz;
    if (mean_val)
        *mean_val = mean;
    if (stddev_val) {
        double sqsum = __riscv_vfmv_f(vec_reddev);
        double variance = std::max((sqsum / nz) - (mean * mean), 0.0);
        *stddev_val = std::sqrt(variance);
    }
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
