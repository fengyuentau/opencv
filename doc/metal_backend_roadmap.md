# Metal Backend Roadmap for OpenCV 4.x

## Direction

Start with an explicit `cv::metal` backend surface in `core`, similar to CUDA's opt-in namespace, and avoid changing existing `cv::Mat` / `UMat` execution paths until ownership, synchronization, error handling, and test coverage are stable. Vulkan in `dnn` is a useful example for isolating platform runtime code behind `HAVE_*` guards; CUDA is the better example for a user-visible accelerator namespace.

## Phase 1: Core Runtime and First Kernels

- Add `WITH_METAL` / `HAVE_METAL` CMake detection for Apple platforms.
- Add `cv::metal::haveMetal()` and a small runtime wrapper around `MTLDevice`, command queue creation, buffers, and compute pipelines.
- Implement simple, correctness-first functions:
  - `cv::metal::add` for continuous `CV_8U` and `CV_32F`.
  - `cv::metal::multiply` for `CV_32F`.
  - `cv::metal::threshold` for `CV_8U`.
- Add CPU parity tests that skip cleanly when Metal is unavailable.

## Phase 2: Memory Model

- Introduce a Metal-owned matrix/buffer wrapper, likely `cv::metal::MetalMat`, after the first kernels prove the API shape.
- Support upload/download, ROI headers, step-aware kernels, and asynchronous command submission.
- Add allocator statistics and leak-focused tests.
- Decide how much of `cv::cuda::GpuMat` semantics should be mirrored versus kept smaller for Apple-only scope.

## Phase 3: Core and Imgproc Coverage

- Expand core arithmetic: `subtract`, `multiply`, `divide`, `absdiff`, `bitwise_*`, `compare`, `min`, `max`.
- Add reductions and scalar-producing ops: `sum`, `mean`, `norm`, `minMaxLoc`.
- Move to common image operations: `resize`, `cvtColor` RGB/BGR/BGRA/GRAY, `threshold`, `filter2D`, `boxFilter`, `GaussianBlur`.
- Add perf tests beside existing OpenCL perf coverage and compare against CPU, OpenCL where available, and Accelerate where relevant.

## Phase 4: Integration

- Evaluate optional dispatch from selected `cv::InputArray` functions only when the caller explicitly uses Metal storage.
- Consider G-API kernels before transparent `UMat` integration; the explicit backend model is easier to review.
- For DNN, follow the Vulkan backend shape first: backend nodes, wrappers, tensor storage, and denylist-based conformance.

## Phase 5: Production Hardening

- Cache compiled libraries and pipelines per device and data type.
- Add command queue reuse, events, and optional stream-like synchronization.
- Cover non-continuous matrices, ROI, large dimensions, multi-channel kernels, and overflow semantics.
- Add CI gates that build with `WITH_METAL=ON` on Apple runners and `WITH_METAL=OFF` elsewhere.

## Initial Patch

This branch starts Phase 1 with:

- `WITH_METAL` detection.
- `opencv2/core/metal.hpp`.
- `cv::metal::haveMetal()`.
- `cv::metal::add()` for continuous `CV_8U` and `CV_32F` matrices.
- Correctness tests against `cv::add`.
