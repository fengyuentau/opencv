// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/metal.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#if __has_feature(objc_arc)
#define CV_METAL_RELEASE(obj)
#else
#define CV_METAL_RELEASE(obj) [obj release]
#endif

namespace cv { namespace metal {

namespace {

struct MetalBufferImpl
{
    MetalBufferImpl() : buffer(nil), bytes(0) {}
    ~MetalBufferImpl()
    {
        CV_METAL_RELEASE(buffer);
    }

    id<MTLBuffer> buffer;
    size_t bytes;
};

static id<MTLDevice> getDevice()
{
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, int depth)
{
    static const char* source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add_u8(device const uchar* src1 [[buffer(0)]],\n"
        "                   device const uchar* src2 [[buffer(1)]],\n"
        "                   device uchar* dst [[buffer(2)]],\n"
        "                   constant uint& len [[buffer(3)]],\n"
        "                   uint gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if (gid < len)\n"
        "        dst[gid] = (uchar)min((uint)src1[gid] + (uint)src2[gid], 255u);\n"
        "}\n"
        "kernel void add_f32(device const float* src1 [[buffer(0)]],\n"
        "                    device const float* src2 [[buffer(1)]],\n"
        "                    device float* dst [[buffer(2)]],\n"
        "                    constant uint& len [[buffer(3)]],\n"
        "                    uint gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if (gid < len)\n"
        "        dst[gid] = src1[gid] + src2[gid];\n"
        "}\n";

    NSError* error = nil;
    NSString* nsSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [device newLibraryWithSource:nsSource options:nil error:&error];
    if (!library)
        CV_Error(Error::GpuApiCallError, cv::format("Metal library build failed: %s",
                 error ? [[error localizedDescription] UTF8String] : "unknown error"));

    NSString* functionName = depth == CV_8U ? @"add_u8" : @"add_f32";
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function)
    {
        CV_METAL_RELEASE(library);
        CV_Error(Error::GpuApiCallError, "Metal add kernel is not available");
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    CV_METAL_RELEASE(function);
    CV_METAL_RELEASE(library);
    if (!pipeline)
        CV_Error(Error::GpuApiCallError, cv::format("Metal pipeline creation failed: %s",
                 error ? [[error localizedDescription] UTF8String] : "unknown error"));

    return pipeline;
}

} // namespace

namespace impl {

bool metalHaveDevice()
{
    return getDevice() != nil;
}

} // namespace impl

MetalMat::MetalMat()
    : flags_(0), rows_(0), cols_(0), step_(0)
{
}

MetalMat::MetalMat(int rows, int cols, int type)
    : flags_(0), rows_(0), cols_(0), step_(0)
{
    create(rows, cols, type);
}

MetalMat::MetalMat(Size size, int type)
    : flags_(0), rows_(0), cols_(0), step_(0)
{
    create(size, type);
}

MetalMat::MetalMat(InputArray arr)
    : flags_(0), rows_(0), cols_(0), step_(0)
{
    upload(arr);
}

MetalMat::MetalMat(const MetalMat& m) = default;
MetalMat& MetalMat::operator=(const MetalMat& m) = default;
MetalMat::~MetalMat() = default;

void MetalMat::create(int rows, int cols, int type)
{
    CV_Assert(rows >= 0 && cols >= 0);
    const int depth = CV_MAT_DEPTH(type);
    if (depth != CV_8U && depth != CV_32F)
        CV_Error(Error::StsUnsupportedFormat, "MetalMat supports CV_8U and CV_32F inputs");

    id<MTLDevice> device = getDevice();
    if (!device)
        CV_Error(Error::StsNotImplemented, "No Metal device is available");

    const size_t elemSize = CV_ELEM_SIZE(type);
    const size_t bytes = static_cast<size_t>(rows) * cols * elemSize;

    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer && bytes != 0)
            CV_Error(Error::GpuApiCallError, "Metal buffer allocation failed");

        MetalBufferImpl* impl = new MetalBufferImpl();
        impl->buffer = buffer;
        impl->bytes = bytes;

        flags_ = Mat::MAGIC_VAL | CV_MAT_CONT_FLAG | type;
        rows_ = rows;
        cols_ = cols;
        step_ = cols * elemSize;
        impl_.reset(impl);
    }
}

void MetalMat::create(Size size, int type)
{
    create(size.height, size.width, type);
}

void MetalMat::release()
{
    flags_ = 0;
    rows_ = 0;
    cols_ = 0;
    step_ = 0;
    impl_.reset();
}

void MetalMat::upload(InputArray arr)
{
    CV_Assert(!arr.empty());
    Mat src = arr.getMat();
    Mat continuous = src.isContinuous() ? src : src.clone();

    create(continuous.rows, continuous.cols, continuous.type());

    MetalBufferImpl* impl = static_cast<MetalBufferImpl*>(impl_.get());
    CV_Assert(impl && impl->buffer);
    memcpy([impl->buffer contents], continuous.ptr(), impl->bytes);
}

void MetalMat::download(OutputArray dst) const
{
    CV_Assert(!empty());
    dst.create(rows_, cols_, type());
    Mat host = dst.getMat();
    CV_Assert(host.isContinuous());

    MetalBufferImpl* impl = static_cast<MetalBufferImpl*>(impl_.get());
    CV_Assert(impl && impl->buffer);
    memcpy(host.ptr(), [impl->buffer contents], impl->bytes);
}

bool MetalMat::empty() const
{
    return !impl_ || rows_ == 0 || cols_ == 0;
}

int MetalMat::type() const
{
    return CV_MAT_TYPE(flags_);
}

int MetalMat::depth() const
{
    return CV_MAT_DEPTH(flags_);
}

int MetalMat::channels() const
{
    return CV_MAT_CN(flags_);
}

Size MetalMat::size() const
{
    return Size(cols_, rows_);
}

size_t MetalMat::step() const
{
    return step_;
}

size_t MetalMat::total() const
{
    return static_cast<size_t>(rows_) * cols_;
}

size_t MetalMat::elemSize() const
{
    return CV_ELEM_SIZE(type());
}

void* MetalMat::handle() const
{
    MetalBufferImpl* impl = static_cast<MetalBufferImpl*>(impl_.get());
    return impl ? impl->buffer : nil;
}

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    CV_Assert(!src1.empty() && !src2.empty());
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    const int depth = src1.depth();
    if (depth != CV_8U && depth != CV_32F)
        CV_Error(Error::StsUnsupportedFormat, "Metal add supports CV_8U and CV_32F inputs");

    id<MTLDevice> device = getDevice();
    if (!device)
        CV_Error(Error::StsNotImplemented, "No Metal device is available");

    dst.create(src1.size(), src1.type());

    const uint len = (uint)(src1.total() * src1.channels());

    @autoreleasepool {
        id<MTLBuffer> src1Buffer = (id<MTLBuffer>)src1.handle();
        id<MTLBuffer> src2Buffer = (id<MTLBuffer>)src2.handle();
        id<MTLBuffer> dstBuffer = (id<MTLBuffer>)dst.handle();
        id<MTLBuffer> lenBuffer = [device newBufferWithBytes:&len length:sizeof(len) options:MTLResourceStorageModeShared];
        if (!src1Buffer || !src2Buffer || !dstBuffer || !lenBuffer)
            CV_Error(Error::GpuApiCallError, "Metal buffer allocation failed");

        id<MTLComputePipelineState> pipeline = makePipeline(device, depth);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!pipeline || !queue || !commandBuffer || !encoder)
        {
            CV_METAL_RELEASE(pipeline);
            CV_METAL_RELEASE(queue);
            CV_METAL_RELEASE(lenBuffer);
            CV_Error(Error::GpuApiCallError, "Metal command setup failed");
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src1Buffer offset:0 atIndex:0];
        [encoder setBuffer:src2Buffer offset:0 atIndex:1];
        [encoder setBuffer:dstBuffer offset:0 atIndex:2];
        [encoder setBuffer:lenBuffer offset:0 atIndex:3];

        const NSUInteger width = pipeline.threadExecutionWidth;
        MTLSize threadsPerThreadgroup = MTLSizeMake(width, 1, 1);
        MTLSize threadgroups = MTLSizeMake((len + width - 1) / width, 1, 1);
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if ([commandBuffer status] != MTLCommandBufferStatusCompleted)
            CV_Error(Error::GpuApiCallError, "Metal command buffer failed");

        CV_METAL_RELEASE(pipeline);
        CV_METAL_RELEASE(queue);
        CV_METAL_RELEASE(lenBuffer);
    }
}

}} // namespace cv::metal

#undef CV_METAL_RELEASE

#endif // HAVE_METAL
