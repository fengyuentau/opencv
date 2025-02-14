// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copytright (C) 2025, SpaceMIT Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

double inline normL1_rvv(const double* src, int n) {
    const auto vle64m1 = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t r00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    int j = 0;
    for (; j <= n - vle64m1; j += vle64m1) {
        vfloat64m1_t v00 = __riscv_vle64_v_f64m1(src + j, vle64m1);
        v00 = __riscv_vfabs(v00, vle64m1);
        r00 = __riscv_vfadd(r00, v00, vle64m1);
    }
    printf("After loop\n");
    vfloat64m1_t s00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    printf("After setzero\n");
    s00 = __riscv_vfredusum(r00, __riscv_vfmv_v_f_f64m1(0.f, vle64m1), vle64m1);
    printf("After redsum1\n");
    double vs = __riscv_vfmv_f(s00);
    printf("After redsum2\n");
    double s = vs;
    for (; j < n; j++) {
        s += cv_abs(src[j]);
    }
    return s;
}

double inline normL2_rvv(const double* src, int n) {
    int j = 0;
    double s = 0.f;
    for (; j < n; j++) {
        double v = src[j];
        s += v * v;
    }
    return s;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // cv::
