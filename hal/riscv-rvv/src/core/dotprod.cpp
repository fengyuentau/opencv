// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"
#include <algorithm>

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

static inline double dotProd_8u(const uchar *a, const uchar *b, int len) {
    constexpr int block_size0 = (1 << 15);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        vuint32m1_t s = __riscv_vmv_v_x_u32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int j = 0; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e8m4(block_size - j);

            auto va = __riscv_vle8_v_u8m4(a + j, vl);
            auto vb = __riscv_vle8_v_u8m4(b + j, vl);

            s = __riscv_vwredsumu(__riscv_vwmulu(va, vb, vl), s, vl);
        }
        r += (double)__riscv_vmv_x(s);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

static inline double dotProd_8s(const schar *a, const schar *b, int len) {
    constexpr int block_size0 = (1 << 14);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        vint32m1_t s = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
        int vl;
        for (int j = 0; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e8m4(block_size - j);

            auto va = __riscv_vle8_v_i8m4(a + j, vl);
            auto vb = __riscv_vle8_v_i8m4(b + j, vl);

            s = __riscv_vwredsum(__riscv_vwmul(va, vb, vl), s, vl);
        }
        r += (double)__riscv_vmv_x(s);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

static inline double dotProd_16u(const ushort *a, const ushort *b, int len) {
    constexpr int block_size0 = (1 << 24);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        int vlmax = __riscv_vsetvlmax_e16m4();
        vuint64m1_t s0 = __riscv_vmv_v_x_u64m1(0, __riscv_vsetvlmax_e64m1());
        vuint64m1_t s1 = s0;
        int j = 0;
        for (; j <= block_size - vlmax * 2; j += vlmax * 2) {
            auto va = __riscv_vle16_v_u16m4(a + j, vlmax);
            auto vb = __riscv_vle16_v_u16m4(b + j, vlmax);
            s0 = __riscv_vwredsumu(__riscv_vwmulu(va, vb, vlmax), s0, vlmax);
            va = __riscv_vle16_v_u16m4(a + j + vlmax, vlmax);
            vb = __riscv_vle16_v_u16m4(b + j + vlmax, vlmax);
            s1 = __riscv_vwredsumu(__riscv_vwmulu(va, vb, vlmax), s1, vlmax);
        }
        int vl;
        for (; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e16m4(block_size - j);

            auto va = __riscv_vle16_v_u16m4(a + j, vl);
            auto vb = __riscv_vle16_v_u16m4(b + j, vl);

            s0 = __riscv_vwredsumu(__riscv_vwmulu(va, vb, vl), s0, vl);
        }
        r += (double)__riscv_vmv_x(s0) + (double)__riscv_vmv_x(s1);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

static inline double dotProd_16s(const short *a, const short *b, int len) {
    constexpr int block_size0 = (1 << 24);

    double r = 0;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        int vlmax = __riscv_vsetvlmax_e16m4();
        vint64m1_t s0 = __riscv_vmv_v_x_i64m1(0, __riscv_vsetvlmax_e64m1());
        vint64m1_t s1 = s0;
        int j = 0;
        for (; j <= block_size - vlmax * 2; j += vlmax * 2) {
            auto va = __riscv_vle16_v_i16m4(a + j, vlmax);
            auto vb = __riscv_vle16_v_i16m4(b + j, vlmax);
            s0 = __riscv_vwredsum(__riscv_vwmul(va, vb, vlmax), s0, vlmax);
            va = __riscv_vle16_v_i16m4(a + j + vlmax, vlmax);
            vb = __riscv_vle16_v_i16m4(b + j + vlmax, vlmax);
            s1 = __riscv_vwredsum(__riscv_vwmul(va, vb, vlmax), s1, vlmax);
        }
        int vl;
        for (; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e16m4(block_size - j);

            auto va = __riscv_vle16_v_i16m4(a + j, vl);
            auto vb = __riscv_vle16_v_i16m4(b + j, vl);

            s0 = __riscv_vwredsum(__riscv_vwmul(va, vb, vl), s0, vl);
        }
        r += (double)__riscv_vmv_x(s0) + (double)__riscv_vmv_x(s1);

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

static inline double dotProd_32s(const int *a, const int *b, int len) {
    int vlmax = __riscv_vsetvlmax_e32m1();
    vfloat64m2_t s0 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s1 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s2 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s3 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    int j = 0;
    for (; j <= len - vlmax * 4; j += vlmax * 4) {
        auto va = __riscv_vle32_v_i32m1(a + j, vlmax);
        auto vb = __riscv_vle32_v_i32m1(b + j, vlmax);
        s0 = __riscv_vfadd(s0, __riscv_vfcvt_f(__riscv_vwmul(va, vb, vlmax), vlmax), vlmax);
        va = __riscv_vle32_v_i32m1(a + j + vlmax, vlmax);
        vb = __riscv_vle32_v_i32m1(b + j + vlmax, vlmax);
        s1 = __riscv_vfadd(s1, __riscv_vfcvt_f(__riscv_vwmul(va, vb, vlmax), vlmax), vlmax);
        va = __riscv_vle32_v_i32m1(a + j + vlmax * 2, vlmax);
        vb = __riscv_vle32_v_i32m1(b + j + vlmax * 2, vlmax);
        s2 = __riscv_vfadd(s2, __riscv_vfcvt_f(__riscv_vwmul(va, vb, vlmax), vlmax), vlmax);
        va = __riscv_vle32_v_i32m1(a + j + vlmax * 3, vlmax);
        vb = __riscv_vle32_v_i32m1(b + j + vlmax * 3, vlmax);
        s3 = __riscv_vfadd(s3, __riscv_vfcvt_f(__riscv_vwmul(va, vb, vlmax), vlmax), vlmax);
    }
    int vl;
    for (; j < len; j += vl) {
        vl = __riscv_vsetvl_e32m1(len - j);

        auto va = __riscv_vle32_v_i32m1(a + j, vl);
        auto vb = __riscv_vle32_v_i32m1(b + j, vl);

        s0 = __riscv_vfadd_vv_f64m2_tu(s0, s0, __riscv_vfcvt_f(__riscv_vwmul(va, vb, vl), vl), vl);
    }
    s0 = __riscv_vfadd(s0, s1, vlmax);
    s2 = __riscv_vfadd(s2, s3, vlmax);
    s0 = __riscv_vfadd(s0, s2, vlmax);
    return __riscv_vfmv_f(__riscv_vfredosum(s0, __riscv_vfmv_v_f_f64m1(0., __riscv_vsetvlmax_e64m1()), vlmax));
}

static inline double dotProd_32f(const float *a, const float *b, int len) {
    constexpr int block_size0 = (1 << 13);

    double r = 0.f;
    int i = 0;
    while (i < len) {
        int block_size = std::min(block_size0, len - i);

        int vlmax = __riscv_vsetvlmax_e32m2();
        vfloat32m2_t s0 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
        vfloat32m2_t s1 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
        vfloat32m2_t s2 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
        vfloat32m2_t s3 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
        int j = 0;
        for (; j <= block_size - vlmax * 4; j += vlmax * 4) {
            auto va = __riscv_vle32_v_f32m2(a + j, vlmax);
            auto vb = __riscv_vle32_v_f32m2(b + j, vlmax);
            s0 = __riscv_vfmacc(s0, va, vb, vlmax);
            va = __riscv_vle32_v_f32m2(a + j + vlmax, vlmax);
            vb = __riscv_vle32_v_f32m2(b + j + vlmax, vlmax);
            s1 = __riscv_vfmacc(s1, va, vb, vlmax);
            va = __riscv_vle32_v_f32m2(a + j + vlmax * 2, vlmax);
            vb = __riscv_vle32_v_f32m2(b + j + vlmax * 2, vlmax);
            s2 = __riscv_vfmacc(s2, va, vb, vlmax);
            va = __riscv_vle32_v_f32m2(a + j + vlmax * 3, vlmax);
            vb = __riscv_vle32_v_f32m2(b + j + vlmax * 3, vlmax);
            s3 = __riscv_vfmacc(s3, va, vb, vlmax);
        }
        int vl;
        for (; j < block_size; j += vl) {
            vl = __riscv_vsetvl_e32m2(block_size - j);

            auto va = __riscv_vle32_v_f32m2(a + j, vl);
            auto vb = __riscv_vle32_v_f32m2(b + j, vl);

            s0 = __riscv_vfmacc_vv_f32m2_tu(s0, va, vb, vl);
        }
        s0 = __riscv_vfadd(s0, s1, vlmax);
        s2 = __riscv_vfadd(s2, s3, vlmax);
        s0 = __riscv_vfadd(s0, s2, vlmax);
        r += (double)__riscv_vfmv_f(__riscv_vfredusum(s0, __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1()), vlmax));

        i += block_size;
        a += block_size;
        b += block_size;
    }

    return r;
}

static inline double dotProd_64f(const double *a, const double *b, int len) {
    int vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t s0 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s1 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s2 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    vfloat64m2_t s3 = __riscv_vfmv_v_f_f64m2(0., vlmax);
    int j = 0;
    for (; j <= len - vlmax * 4; j += vlmax * 4) {
        auto va = __riscv_vle64_v_f64m2(a + j, vlmax);
        auto vb = __riscv_vle64_v_f64m2(b + j, vlmax);
        s0 = __riscv_vfmacc(s0, va, vb, vlmax);
        va = __riscv_vle64_v_f64m2(a + j + vlmax, vlmax);
        vb = __riscv_vle64_v_f64m2(b + j + vlmax, vlmax);
        s1 = __riscv_vfmacc(s1, va, vb, vlmax);
        va = __riscv_vle64_v_f64m2(a + j + vlmax * 2, vlmax);
        vb = __riscv_vle64_v_f64m2(b + j + vlmax * 2, vlmax);
        s2 = __riscv_vfmacc(s2, va, vb, vlmax);
        va = __riscv_vle64_v_f64m2(a + j + vlmax * 3, vlmax);
        vb = __riscv_vle64_v_f64m2(b + j + vlmax * 3, vlmax);
        s3 = __riscv_vfmacc(s3, va, vb, vlmax);
    }
    int vl;
    for (; j < len; j += vl) {
        vl = __riscv_vsetvl_e64m2(len - j);
        auto va = __riscv_vle64_v_f64m2(a + j, vl);
        auto vb = __riscv_vle64_v_f64m2(b + j, vl);
        s0 = __riscv_vfmacc_vv_f64m2_tu(s0, va, vb, vl);
    }
    s0 = __riscv_vfadd(s0, s1, vlmax);
    s2 = __riscv_vfadd(s2, s3, vlmax);
    s0 = __riscv_vfadd(s0, s2, vlmax);
    return __riscv_vfmv_f(__riscv_vfredusum(s0, __riscv_vfmv_v_f_f64m1(0., __riscv_vsetvlmax_e64m1()), vlmax));
}

} // anonymous

using DotProdFunc = double (*)(const uchar *a, const uchar *b, int len);
int dotprod(const uchar *a_data, size_t a_step, const uchar *b_data, size_t b_step,
            int width, int height, int type, double *dot_val) {
    int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    static DotProdFunc dotprod_tab[CV_DEPTH_MAX] = {
        (DotProdFunc)dotProd_8u,  (DotProdFunc)dotProd_8s,
        (DotProdFunc)dotProd_16u, (DotProdFunc)dotProd_16s,
        (DotProdFunc)dotProd_32s, (DotProdFunc)dotProd_32f,
        (DotProdFunc)dotProd_64f, nullptr
    };
    DotProdFunc func = dotprod_tab[depth];
    if (func == nullptr) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    size_t elem_size1 = static_cast<size_t>(CV_ELEM_SIZE1(type));
    bool a_continuous = (a_step == width * elem_size1 * cn);
    bool b_continuous = (b_step == width * elem_size1 * cn);
    size_t nplanes = 1;
    size_t len = width * height;
    if (!a_continuous || !b_continuous) {
        nplanes = height;
        len = width;
    }
    len *= cn;

    double r = 0;
    auto _a = a_data;
    auto _b = b_data;
    for (size_t i = 0; i < nplanes; i++) {
        if (!a_continuous || !b_continuous) {
            _a = a_data + a_step * i;
            _b = b_data + b_step * i;
        }
        r += func(_a, _b, len);
    }
    *dot_val = r;

    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
