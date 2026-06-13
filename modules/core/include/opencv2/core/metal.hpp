// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#ifndef __cplusplus
#  error metal.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"

namespace cv { namespace metal {

//! @addtogroup core
//! @{

/** @brief Returns true when OpenCV was built with Metal support and a system Metal device is available. */
CV_EXPORTS_W bool haveMetal();

/** @brief Adds two matrices using the Metal backend.

This initial backend entry point supports continuous same-size matrices with identical type.
Supported depths are CV_8U and CV_32F. Unsupported inputs raise cv::Exception instead of
silently changing backend.
 */
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst);

//! @}

}} // namespace cv::metal

#endif // OPENCV_CORE_METAL_HPP
