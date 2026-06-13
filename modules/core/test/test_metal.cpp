// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if defined(HAVE_METAL)

#include "test_precomp.hpp"
#include "opencv2/core/metal.hpp"

namespace opencv_test { namespace {

TEST(Core_Metal, Add8U)
{
    Mat src1(32, 32, CV_8UC3);
    Mat src2(src1.size(), src1.type());
    randu(src1, 0, 255);
    randu(src2, 0, 255);

    Mat expected;
    cv::add(src1, src2, expected);

    Mat actual;
    cv::metal::add(src1, src2, actual);

    EXPECT_LE(cvtest::norm(actual, expected, NORM_INF), 0);
}

TEST(Core_Metal, Add32F)
{
    Mat src1(32, 32, CV_32FC1);
    Mat src2(src1.size(), src1.type());
    randu(src1, -10.0f, 10.0f);
    randu(src2, -10.0f, 10.0f);

    Mat expected;
    cv::add(src1, src2, expected);

    Mat actual;
    cv::metal::add(src1, src2, actual);

    EXPECT_LE(cvtest::norm(actual, expected, NORM_INF), 1e-6);
}

}} // namespace

#endif
