// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

static constexpr size_t sincos_mask = 0x3;

static constexpr float sincos_rad_scale = 2.f / CV_PI;
static constexpr float sincos_deg_scale = 2.f / 180.f;

// Taylor expansion coefficients for sin(x*pi/2) and cos(x*pi/2)
static constexpr double sincos_sin_p7 = -0.004681754135319;
static constexpr double sincos_sin_p5 = 0.079692626246167;
static constexpr double sincos_sin_p3 = -0.645964097506246;
static constexpr double sincos_sin_p1 = 1.570796326794897;

static constexpr double sincos_cos_p8 = 0.000919260274839;
static constexpr double sincos_cos_p6 = -0.020863480763353;
static constexpr double sincos_cos_p4 = 0.253669507901048;
static constexpr double sincos_cos_p2 = -1.233700550136170;
static constexpr double sincos_cos_p0 = 1.000000000000000;

// Taylor expansion and angle sum identity.
// Fused cosine uses 7 LMUL registers; the split form reduces register pressure.
template <bool lowRegisterCosine, typename RVV_T, typename T = typename RVV_T::VecType>
static inline void
    SinCos32f(T angle, T& sinval, T& cosval, float scale, T cos_p2, T cos_p0, size_t vl)
{
    angle = __riscv_vfmul(angle, scale, vl);
    auto round_angle = RVV_ToInt<RVV_T>::cast(angle, vl);
    auto delta_angle = __riscv_vfsub(angle, RVV_T::cast(round_angle, vl), vl);
    auto delta_angle2 = __riscv_vfmul(delta_angle, delta_angle, vl);

    auto sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sincos_sin_p7, vl), sincos_sin_p5, vl);
    sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sin, vl), sincos_sin_p3, vl);
    sin = __riscv_vfadd(__riscv_vfmul(delta_angle2, sin, vl), sincos_sin_p1, vl);
    sin = __riscv_vfmul(delta_angle, sin, vl);

    auto cos = __riscv_vfadd(__riscv_vfmul(delta_angle2, sincos_cos_p8, vl), sincos_cos_p6, vl);
    cos = __riscv_vfadd(__riscv_vfmul(delta_angle2, cos, vl), sincos_cos_p4, vl);
    if (lowRegisterCosine)
    {
        cos = __riscv_vfadd(__riscv_vfmul(cos, delta_angle2, vl), sincos_cos_p2, vl);
        cos = __riscv_vfadd(__riscv_vfmul(cos, delta_angle2, vl), sincos_cos_p0, vl);
    }
    else
    {
        cos = __riscv_vfmadd(cos, delta_angle2, cos_p2, vl);
        cos = __riscv_vfmadd(cos, delta_angle2, cos_p0, vl);
    }

    // idx = 0: sinval =  sin, cosval =  cos
    // idx = 1: sinval =  cos, cosval = -sin
    // idx = 2: sinval = -sin, cosval = -cos
    // idx = 3: sinval = -cos, cosval =  sin
    auto idx = __riscv_vand(round_angle, sincos_mask, vl);
    auto swap = __riscv_vmsne(__riscv_vand(idx, 1, vl), 0, vl);
    auto neg_sin = __riscv_vmsgt(idx, 1, vl);
    auto neg_cos = __riscv_vmxor(swap, neg_sin, vl);

    sinval = __riscv_vmerge(sin, cos, swap, vl);
    cosval = __riscv_vmerge(cos, sin, swap, vl);

    sinval = __riscv_vfneg_mu(neg_sin, sinval, sinval, vl);
    cosval = __riscv_vfneg_mu(neg_cos, cosval, cosval, vl);
}

template <bool withMagnitude, bool lowRegisterCosine, typename RVV_T,
          typename Elem = typename RVV_T::ElemType>
inline int polarToCartImpl(const Elem* mag, const Elem* angle, Elem* x, Elem* y, int len,
                           bool angleInDegrees)
{
    using T = RVV_F32M4;
    const auto sincos_scale = angleInDegrees ? sincos_deg_scale : sincos_rad_scale;

    size_t vl;
    auto cos_p2 = T::vmv(sincos_cos_p2, T::setvlmax());
    auto cos_p0 = T::vmv(sincos_cos_p0, T::setvlmax());
    for (; len > 0; len -= (int)vl, angle += vl, x += vl, y += vl)
    {
        vl = RVV_T::setvl(len);
        auto vangle = T::cast(RVV_T::vload(angle, vl), vl);
        T::VecType vsin, vcos;
        SinCos32f<lowRegisterCosine, T>(vangle, vsin, vcos, sincos_scale,
                                       cos_p2, cos_p0, vl);
        if (withMagnitude)
        {
            auto vmag = T::cast(RVV_T::vload(mag, vl), vl);
            vsin = __riscv_vfmul(vsin, vmag, vl);
            vcos = __riscv_vfmul(vcos, vmag, vl);
            mag += vl;
        }
        RVV_T::vstore(x, RVV_T::cast(vcos, vl), vl);
        RVV_T::vstore(y, RVV_T::cast(vsin, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

template <bool lowRegisterCosine, typename RVV_T, typename Elem = typename RVV_T::ElemType>
inline int polarToCart(const Elem* mag, const Elem* angle, Elem* x, Elem* y, int len,
                       bool angleInDegrees)
{
    return mag ? polarToCartImpl<true, lowRegisterCosine, RVV_T>(
                     mag, angle, x, y, len, angleInDegrees)
               : polarToCartImpl<false, lowRegisterCosine, RVV_T>(
                     mag, angle, x, y, len, angleInDegrees);
}

} // anonymous

int polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees) {
    return polarToCart<false, RVV_F32M4>(mag, angle, x, y, len, angleInDegrees);
}
int polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees) {
    return polarToCart<true, RVV_F64M8>(mag, angle, x, y, len, angleInDegrees);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
