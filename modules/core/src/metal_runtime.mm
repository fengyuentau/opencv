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

namespace cv { namespace metal { namespace impl {

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

bool metalHaveDevice()
{
    return getDevice() != nil;
}

void metalAddImpl(InputArray _src1, InputArray _src2, OutputArray _dst)
{
    CV_Assert(!_src1.empty() && !_src2.empty());

    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());
    CV_Assert(src1.isContinuous() && src2.isContinuous());

    const int depth = src1.depth();
    if (depth != CV_8U && depth != CV_32F)
        CV_Error(Error::StsUnsupportedFormat, "Metal add supports CV_8U and CV_32F inputs");

    id<MTLDevice> device = getDevice();
    if (!device)
        CV_Error(Error::StsNotImplemented, "No Metal device is available");

    _dst.create(src1.size(), src1.type());
    Mat dst = _dst.getMat();
    CV_Assert(dst.isContinuous());

    const size_t bytes = src1.total() * src1.elemSize();
    const uint len = (uint)(src1.total() * src1.channels());

    @autoreleasepool {
        id<MTLBuffer> src1Buffer = [device newBufferWithBytes:src1.ptr() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> src2Buffer = [device newBufferWithBytes:src2.ptr() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
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
            CV_METAL_RELEASE(src1Buffer);
            CV_METAL_RELEASE(src2Buffer);
            CV_METAL_RELEASE(dstBuffer);
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

        memcpy(dst.ptr(), [dstBuffer contents], bytes);
        CV_METAL_RELEASE(pipeline);
        CV_METAL_RELEASE(queue);
        CV_METAL_RELEASE(src1Buffer);
        CV_METAL_RELEASE(src2Buffer);
        CV_METAL_RELEASE(dstBuffer);
        CV_METAL_RELEASE(lenBuffer);
    }
}

}}} // namespace cv::metal::impl

#undef CV_METAL_RELEASE

#endif // HAVE_METAL
