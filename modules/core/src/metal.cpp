// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/metal.hpp"

namespace cv { namespace metal {

#ifdef HAVE_METAL
namespace impl {
bool metalHaveDevice();
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

#ifndef HAVE_METAL

MetalMat::MetalMat() : flags_(0), rows_(0), cols_(0), step_(0) {}
MetalMat::MetalMat(int rows, int cols, int type) : flags_(0), rows_(0), cols_(0), step_(0)
{
    create(rows, cols, type);
}
MetalMat::MetalMat(Size size, int type) : flags_(0), rows_(0), cols_(0), step_(0)
{
    create(size, type);
}
MetalMat::MetalMat(InputArray arr) : flags_(0), rows_(0), cols_(0), step_(0)
{
    upload(arr);
}
MetalMat::MetalMat(const MetalMat& m) = default;
MetalMat& MetalMat::operator=(const MetalMat& m) = default;
MetalMat::~MetalMat() = default;

void MetalMat::create(int rows, int cols, int type)
{
    CV_UNUSED(rows);
    CV_UNUSED(cols);
    CV_UNUSED(type);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
}
void MetalMat::create(Size size, int type)
{
    create(size.height, size.width, type);
}
void MetalMat::release()
{
    flags_ = rows_ = cols_ = 0;
    step_ = 0;
    impl_.reset();
}
void MetalMat::upload(InputArray arr)
{
    CV_UNUSED(arr);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
}
void MetalMat::download(OutputArray dst) const
{
    CV_UNUSED(dst);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
}
bool MetalMat::empty() const { return true; }
int MetalMat::type() const { return CV_MAT_TYPE(flags_); }
int MetalMat::depth() const { return CV_MAT_DEPTH(flags_); }
int MetalMat::channels() const { return CV_MAT_CN(flags_); }
Size MetalMat::size() const { return Size(cols_, rows_); }
size_t MetalMat::step() const { return step_; }
size_t MetalMat::total() const { return static_cast<size_t>(rows_) * cols_; }
size_t MetalMat::elemSize() const { return CV_ELEM_SIZE(type()); }
void* MetalMat::handle() const { return 0; }

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    CV_UNUSED(src1);
    CV_UNUSED(src2);
    CV_UNUSED(dst);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
}

#endif

void add(InputArray src1, InputArray src2, OutputArray dst)
{
#ifdef HAVE_METAL
    MetalMat src1Metal(src1);
    MetalMat src2Metal(src2);
    MetalMat dstMetal;
    add(src1Metal, src2Metal, dstMetal);
    dstMetal.download(dst);
#else
    CV_UNUSED(src1);
    CV_UNUSED(src2);
    CV_UNUSED(dst);
    CV_Error(Error::StsNotImplemented, "Metal support is not enabled in this OpenCV build");
#endif
}

}} // namespace cv::metal
