// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/metal.hpp"

namespace cv { namespace metal {

#ifdef HAVE_METAL
namespace impl {
bool metalHaveDevice();
void metalAddImpl(InputArray src1, InputArray src2, OutputArray dst);
}
#endif

bool haveMetal()
{
#ifdef HAVE_METAL
    return impl::metalHaveDevice();
#else
    return false;
#endif
}

void add(InputArray src1, InputArray src2, OutputArray dst)
{
#ifdef HAVE_METAL
    impl::metalAddImpl(src1, src2, dst);
#else
    CV_UNUSED(src1);
    CV_UNUSED(src2);
    CV_UNUSED(dst);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
#endif
}

}} // namespace cv::metal
