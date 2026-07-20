// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"
#include "common.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

template <typename SQRT_T, typename T = typename SQRT_T::T::ElemType>
inline int magnitude(const T* x, const T* y, T* dst, int len)
{
    size_t vl;
    for (; len > 0; len -= (int)vl, x += vl, y += vl, dst += vl)
    {
        vl = SQRT_T::T::setvl(len);

        auto vx = SQRT_T::T::vload(x, vl);
        auto vy = SQRT_T::T::vload(y, vl);

        auto vmag = common::sqrt<SQRT_T::iter_times>(__riscv_vfmadd(vx, vx, __riscv_vfmul(vy, vy, vl), vl), vl);
        SQRT_T::T::vstore(dst, vmag, vl);
    }

    return CV_HAL_ERROR_OK;
}

template <typename SQRT_T, typename TAIL_SQRT_T, typename T = typename SQRT_T::T::ElemType>
inline int magnitude2x(const T* x, const T* y, T* dst, int len)
{
    using RVV_T = typename SQRT_T::T;

    const size_t vl = RVV_T::setvlmax();
    for (; len >= (int)(2 * vl); len -= (int)(2 * vl),
         x += 2 * vl, y += 2 * vl, dst += 2 * vl)
    {
        auto vx0 = RVV_T::vload(x, vl);
        auto vy0 = RVV_T::vload(y, vl);
        auto vx1 = RVV_T::vload(x + vl, vl);
        auto vy1 = RVV_T::vload(y + vl, vl);

        auto sum0 = __riscv_vfmadd(vx0, vx0, __riscv_vfmul(vy0, vy0, vl), vl);
        auto sum1 = __riscv_vfmadd(vx1, vx1, __riscv_vfmul(vy1, vy1, vl), vl);
        auto vmag0 = sum0;
        auto vmag1 = sum1;
        common::sqrt2x<SQRT_T::iter_times>(sum0, sum1, vmag0, vmag1, vl);

        RVV_T::vstore(dst, vmag0, vl);
        RVV_T::vstore(dst + vl, vmag1, vl);
    }

    return magnitude<TAIL_SQRT_T>(x, y, dst, len);
}

} // anonymous

int magnitude32f(const float *x, const float *y, float *dst, int len) {
    return magnitude2x<common::Sqrt32f<RVV_F32M4>, common::Sqrt32f<RVV_F32M8>>(x, y, dst, len);
}
int magnitude64f(const double *x, const double  *y, double *dst, int len) {
    return magnitude<common::Sqrt64f<RVV_F64M8>>(x, y, dst, len);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
