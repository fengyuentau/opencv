// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#ifndef __cplusplus
#  error metal.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"

#include <memory>

namespace cv { namespace metal {

//! @addtogroup core
//! @{

/** @brief Returns true when OpenCV was built with Metal support and a system Metal device is available. */
CV_EXPORTS_W bool haveMetal();

/** @brief Metal-backed matrix storage.

MetalMat follows the same explicit-backend direction as cuda::GpuMat and uses a UMat-like
host/device ownership model internally. Data is uploaded once, operations consume MetalMat
buffers directly, and host synchronization happens only through download().
 */
class CV_EXPORTS MetalMat
{
public:
    MetalMat();
    MetalMat(int rows, int cols, int type);
    MetalMat(Size size, int type);
    explicit MetalMat(InputArray arr);
    MetalMat(const MetalMat& m);
    MetalMat& operator=(const MetalMat& m);
    ~MetalMat();

    void create(int rows, int cols, int type);
    void create(Size size, int type);
    void release();

    void upload(InputArray arr);
    void download(OutputArray dst) const;

    bool empty() const;
    int type() const;
    int depth() const;
    int channels() const;
    Size size() const;
    size_t step() const;
    size_t total() const;
    size_t elemSize() const;

    void* handle() const;

private:
    int flags_;
    int rows_;
    int cols_;
    size_t step_;
    std::shared_ptr<void> impl_;
};

/** @brief Adds two matrices using the Metal backend.

This backend entry point consumes MetalMat buffers directly. It supports same-size matrices with
identical type. Supported depths are CV_8U and CV_32F.
 */
CV_EXPORTS void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst);

/** @brief Convenience host wrapper for metal::add.

This overload uploads the inputs, calls add(const MetalMat&, const MetalMat&, MetalMat&), and
downloads the result. Prefer the MetalMat overload when chaining Metal operations.
 */
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst);

//! @}

}} // namespace cv::metal

#endif // OPENCV_CORE_METAL_HPP
