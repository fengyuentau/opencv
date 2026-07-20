// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

namespace {

constexpr RVV_LMUL getLMUL(size_t sz) {
    // c++11 only allows exactly one return statement inside the function body modified by constexpr
    return sz == 1 ? LMUL_8 : (sz == 2 ? LMUL_4 : (sz == 4 ? LMUL_2 : LMUL_1));
}

static inline vbool1_t vlt(const vuint8m8_t   &a, const vuint8m8_t   &b, const int vl) { return __riscv_vmsltu(a, b, vl); }
static inline vbool1_t vlt(const vint8m8_t    &a, const vint8m8_t    &b, const int vl) { return __riscv_vmslt(a, b, vl); }
static inline vbool2_t vlt(const vuint16m8_t  &a, const vuint16m8_t  &b, const int vl) { return __riscv_vmsltu(a, b, vl); }
static inline vbool2_t vlt(const vint16m8_t   &a, const vint16m8_t   &b, const int vl) { return __riscv_vmslt(a, b, vl); }
static inline vbool4_t vlt(const vint32m8_t   &a, const vint32m8_t   &b, const int vl) { return __riscv_vmslt(a, b, vl); }
static inline vbool4_t vlt(const vfloat32m8_t &a, const vfloat32m8_t &b, const int vl) { return __riscv_vmflt(a, b, vl); }
static inline vbool8_t vlt(const vint32m4_t   &a, const vint32m4_t   &b, const int vl) { return __riscv_vmslt(a, b, vl); }
static inline vbool8_t vlt(const vfloat32m4_t &a, const vfloat32m4_t &b, const int vl) { return __riscv_vmflt(a, b, vl); }

static inline vbool1_t vle(const vuint8m8_t   &a, const vuint8m8_t   &b, const int vl) { return __riscv_vmsleu(a, b, vl); }
static inline vbool1_t vle(const vint8m8_t    &a, const vint8m8_t    &b, const int vl) { return __riscv_vmsle(a, b, vl); }
static inline vbool2_t vle(const vuint16m8_t  &a, const vuint16m8_t  &b, const int vl) { return __riscv_vmsleu(a, b, vl); }
static inline vbool2_t vle(const vint16m8_t   &a, const vint16m8_t   &b, const int vl) { return __riscv_vmsle(a, b, vl); }
static inline vbool4_t vle(const vint32m8_t   &a, const vint32m8_t   &b, const int vl) { return __riscv_vmsle(a, b, vl); }
static inline vbool4_t vle(const vfloat32m8_t &a, const vfloat32m8_t &b, const int vl) { return __riscv_vmfle(a, b, vl); }
static inline vbool8_t vle(const vint32m4_t   &a, const vint32m4_t   &b, const int vl) { return __riscv_vmsle(a, b, vl); }
static inline vbool8_t vle(const vfloat32m4_t &a, const vfloat32m4_t &b, const int vl) { return __riscv_vmfle(a, b, vl); }

static inline vbool1_t veq(const vuint8m8_t   &a, const vuint8m8_t   &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool1_t veq(const vint8m8_t    &a, const vint8m8_t    &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool2_t veq(const vuint16m8_t  &a, const vuint16m8_t  &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool2_t veq(const vint16m8_t   &a, const vint16m8_t   &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool4_t veq(const vint32m8_t   &a, const vint32m8_t   &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool4_t veq(const vfloat32m8_t &a, const vfloat32m8_t &b, const int vl) { return __riscv_vmfeq(a, b, vl); }
static inline vbool8_t veq(const vint32m4_t   &a, const vint32m4_t   &b, const int vl) { return __riscv_vmseq(a, b, vl); }
static inline vbool8_t veq(const vfloat32m4_t &a, const vfloat32m4_t &b, const int vl) { return __riscv_vmfeq(a, b, vl); }

static inline vbool1_t vne(const vuint8m8_t   &a, const vuint8m8_t   &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool1_t vne(const vint8m8_t    &a, const vint8m8_t    &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool2_t vne(const vuint16m8_t  &a, const vuint16m8_t  &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool2_t vne(const vint16m8_t   &a, const vint16m8_t   &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool4_t vne(const vint32m8_t   &a, const vint32m8_t   &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool4_t vne(const vfloat32m8_t &a, const vfloat32m8_t &b, const int vl) { return __riscv_vmfne(a, b, vl); }
static inline vbool8_t vne(const vint32m4_t   &a, const vint32m4_t   &b, const int vl) { return __riscv_vmsne(a, b, vl); }
static inline vbool8_t vne(const vfloat32m4_t &a, const vfloat32m4_t &b, const int vl) { return __riscv_vmfne(a, b, vl); }

template <typename T> struct use_pair_relational : std::integral_constant<bool, sizeof(T) == 4> {};
template <typename T> struct use_pair_equality : std::is_same<T, float> {};

#define CV_HAL_RVV_COMPARE_OP(op_name, pair_trait) \
template <typename _Tps> \
struct op_name { \
    using in = RVV<_Tps, LMUL_8>; \
    using out = RVV<uint8_t, getLMUL(sizeof(_Tps))>; \
    using half_in = RVV<_Tps, LMUL_4>; \
    using half_out = RVV<uint8_t, LMUL_1>; \
    constexpr static uint8_t one = 255; \
    static inline int run_pair(const _Tps *, const _Tps *, uchar *, const int, std::false_type) { \
        return 0; \
    } \
    static inline int run_pair(const _Tps *src1, const _Tps *src2, uchar *dst, const int len, std::true_type) { \
        auto zero = half_out::vmv(0, half_out::setvlmax()); \
        const int vlmax = static_cast<int>(half_in::setvlmax()); \
        int i = 0; \
        for (; len - i >= 2 * vlmax; i += 2 * vlmax) { \
            auto v10 = half_in::vload(src1 + i, vlmax); \
            auto v20 = half_in::vload(src2 + i, vlmax); \
            auto m0 = v##op_name(v10, v20, vlmax); \
            auto v11 = half_in::vload(src1 + i + vlmax, vlmax); \
            auto v21 = half_in::vload(src2 + i + vlmax, vlmax); \
            auto m1 = v##op_name(v11, v21, vlmax); \
            half_out::vstore(dst + i, __riscv_vmerge(zero, one, m0, vlmax), vlmax); \
            half_out::vstore(dst + i + vlmax, __riscv_vmerge(zero, one, m1, vlmax), vlmax); \
        } \
        return i; \
    } \
    static inline void run(const _Tps *src1, const _Tps *src2, uchar *dst, const int len) { \
        auto zero = out::vmv(0, out::setvlmax()); \
        int i = run_pair(src1, src2, dst, len, pair_trait<_Tps>()); \
        int vl; \
        for (; i < len; i += vl) { \
            vl = in::setvl(len - i); \
            auto v1 = in::vload(src1 + i, vl); \
            auto v2 = in::vload(src2 + i, vl); \
            auto m = v##op_name(v1, v2, vl); \
            out::vstore(dst + i, __riscv_vmerge(zero, one, m, vl), vl); \
        } \
    } \
};

CV_HAL_RVV_COMPARE_OP(lt, use_pair_relational)
CV_HAL_RVV_COMPARE_OP(le, use_pair_relational)
CV_HAL_RVV_COMPARE_OP(eq, use_pair_equality)
CV_HAL_RVV_COMPARE_OP(ne, use_pair_equality)

template <template<typename _Tps> class op, typename _Tps> static inline
int compare_impl(const _Tps *src1_data, size_t src1_step, const _Tps *src2_data, size_t src2_step,
                 uchar *dst_data, size_t dst_step, int width, int height) {
    const size_t src_width = static_cast<size_t>(width) * sizeof(_Tps);
    if (src1_step == src_width && src2_step == src_width && dst_step == static_cast<size_t>(width)) {
        width *= height;
        height = 1;
    }

    for (int h = 0; h < height; h++) {
        const _Tps *src1 = reinterpret_cast<const _Tps*>((const uchar*)src1_data + h * src1_step);
        const _Tps *src2 = reinterpret_cast<const _Tps*>((const uchar*)src2_data + h * src2_step);
        uchar *dst = dst_data + h * dst_step;

        op<_Tps>::run(src1, src2, dst, width);
    }

    return CV_HAL_ERROR_OK;
}

template <typename _Tps> inline
int compare(const _Tps *src1_data, size_t src1_step, const _Tps *src2_data, size_t src2_step,
            uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    switch (operation) {
        case CMP_LT: return compare_impl<lt, _Tps>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height);
        case CMP_GT: return compare_impl<lt, _Tps>(src2_data, src2_step, src1_data, src1_step, dst_data, dst_step, width, height);
        case CMP_LE: return compare_impl<le, _Tps>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height);
        case CMP_GE: return compare_impl<le, _Tps>(src2_data, src2_step, src1_data, src1_step, dst_data, dst_step, width, height);
        case CMP_EQ: return compare_impl<eq, _Tps>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height);
        case CMP_NE: return compare_impl<ne, _Tps>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height);
        default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
}

} // namespace anonymous

int cmp8u(const uchar *src1_data, size_t src1_step, const uchar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<uchar>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}
int cmp8s(const schar *src1_data, size_t src1_step, const schar *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<schar>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}
int cmp16u(const ushort *src1_data, size_t src1_step, const ushort *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<ushort>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}
int cmp16s(const short *src1_data, size_t src1_step, const short *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<short>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}
int cmp32s(const int *src1_data, size_t src1_step, const int *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<int>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}
int cmp32f(const float *src1_data, size_t src1_step, const float *src2_data, size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height, int operation) {
    return compare<float>(src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width, height, operation);
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::core
