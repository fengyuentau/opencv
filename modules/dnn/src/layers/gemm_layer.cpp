// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

// OpenCL backend
#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/inner_product.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// CANN backend
#include "../op_cann.hpp"

// OpenVINO backend
#include "../ie_ngraph.hpp"

// Vulkan backend
#include "../op_vkcom.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"

namespace cv { namespace dnn {

class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        const_B = params.get<bool>("constB", false); // true means blobs[0] is B
        const_C = params.get<bool>("constC", false); // true means blobs.back() is C
        have_bias = params.get<bool>("have_bias", false); // NOTE: have_bias being true does not mean bias is constant

        real_ndims_C = params.get<int>("real_ndims_C", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CUDA && const_B && !trans_a) ||
               backendId == DNN_BACKEND_CANN ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !have_bias && !trans_a);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = static_cast<int>(inputs.size() + blobs.size());
        CV_CheckGE(num_inputs, 2, "DNN/Gemm: Gemm takes at least two inputs");
        CV_CheckLE(num_inputs, 3, "DNN/Gemm: Gemm takes at most three inputs");

        // Check whether A and B are two dimensional
        const auto shape_A = inputs[0];
        const auto shape_B = const_B ? shape(blobs[0]) : inputs[1];
        CV_CheckGE(shape_A.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor A must be n-dimensional (n >= 2)");
        CV_CheckEQ(shape_B.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor B must be two dimensional");

        // Check legal matrix multiplication
        size_t dims_A = shape_A.size();
        int ma = shape_A[dims_A - 2], na = shape_A[dims_A - 1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K_a = trans_a ? ma : na;
        int K_b = trans_b ? nb : mb;
        CV_CheckEQ(K_a, K_b, "DNN/Gemm: Invalid dimension of dim K");

        // Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (have_bias) {
            const auto shape_C = const_C ? shape(blobs.back()) : inputs.back();

            auto ndims_C = shape_C.size();
            CV_CheckLE(ndims_C, static_cast<size_t>(2), "DNN/Gemm: C can only be 0d (scalar) / 1d / 2d tensor");

            if (real_ndims_C == 1) { // (1,) or (N,)
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/Gemm: invalid dimension of C");
            } else if (real_ndims_C == 2) { // (1, 1) or (1, N) or (M, 1) or (M, N)
                // printf("shape_C=[%d, %d]\n", shape_C[0], shape_C[1]);
                CV_Check(shape_C[0], (shape_C[0] == 1 && shape_C[1] == 1) ||
                                     (shape_C[0] == 1 && shape_C[1] == N) ||
                                     (shape_C[0] == M && shape_C[1] == 1) ||
                                     (shape_C[0] == M && shape_C[1] == N),
                                     "DNN/Gemm: C must be of shape (1, 1) or (1, N) or (M, 1) or (M, N)");
                if (shape_C[0] == 1) {
                    CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N, "DNN/Gemm: invalid dimension of C");
                } else if (shape_C[0] == M) {
                    CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N, "DNN/Gemm: invalid dimension of C");
                } else {
                    CV_Error(Error::StsBadSize, "DNN/Gemm: invalid dimension of C");
                }
            }
        }

        int batches = std::accumulate(shape_A.begin(), shape_A.end() - 2, 1, std::multiplies<int>());
        MatShape shape_y{M * batches, N};
        outputs.assign(1, shape_y);
        return false;
    }

    // TODO: replace with cv::broadcast() once 1d mat is supported
    // FIXME: fix if conditions if 1d mat is supported properly
    void broadcastCWtihBeta(int M, int N, const Mat &C) {
        if (beta != 0 && !C.empty()) {
            broadcast_C = Mat::zeros(M, N, CV_32FC1);

            const float *ptr_c = C.ptr<const float>();
            float *ptr_broadcast_c = broadcast_C.ptr<float>();
            const auto shape_C = shape(C);
            if ((real_ndims_C == 0) || (real_ndims_C == 1 && shape_C[0] == 1) ||
                (real_ndims_C == 2 && shape_C[0] == 1 && shape_C[1] == 1)) {
                // (), (1,), (1, 1)
                float c = *ptr_c;
                int total = M * N;
                for (int i = 0; i < total; ++i) {
                    ptr_broadcast_c[i] = beta * c;
                }
            } else if ((real_ndims_C == 1 && shape_C[0] == N) ||
                       (real_ndims_C == 2 && shape_C[0] == 1 && shape_C[1] == N)) {
                // (N,), (1, N)
                for (int i = 0; i < M; ++i) {
                    int step = i * N;
                    for (int j = 0; j < N; ++j) {
                        ptr_broadcast_c[step + j] = beta * ptr_c[j];
                    }
                }
            } else if (real_ndims_C == 2 && shape_C[0] == M && shape_C[1] == 1) {
                // (M, 1)
                for (int i = 0; i < M; ++i) {
                    int step = i * N;
                    for (int j = 0; j < N; ++j) {
                        ptr_broadcast_c[step + j] = beta * ptr_c[i];
                    }
                }
            } else {
                // (M, N)
                std::transform(ptr_c, ptr_c + M * N, ptr_broadcast_c, [this] (const float &c) {
                    return this->beta * c; });
            }
        }
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        // pack B if it is const
        if (const_B) {
            fastGemmPackB(blobs[0], packed_B, trans_b, opt);
        }

        // also pre-broadcast bias
        if (const_C) {
            const auto &C = blobs.back();

            std::vector<Mat> outputs;
            outputs_arr.getMatVector(outputs);
            const auto &Y = outputs[0];
            const auto shape_Y = shape(Y);
            size_t dims_Y = shape_Y.size();
            int M = shape_Y[dims_Y - 2], N = shape_Y[dims_Y - 1];

            // broadcast
            broadcastCWtihBeta(M, N, C);
        }

#ifdef HAVE_OPENCL
        ocl_op.release();
        umat_blobs.clear();
        umat_half_blobs.clear();
#endif
    }

    // Y = A * B + C, note that C is unidirectionaly broadcastable to (A * B).
    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget), forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto &A = inputs[0];
        auto &Y = outputs[0];

        const auto shape_A = shape(A), shape_Y = shape(Y);
        size_t dims_A = shape_A.size();
        int ma = shape_A[dims_A - 2], na = shape_A[dims_A - 1];
        size_t dims_Y = shape_Y.size();
        int M = shape_Y[dims_Y - 2], N = shape_Y[dims_Y - 1];
        int K = trans_a ? ma : na;

        // broadcast C and copy C to output
        if (have_bias) {
            if (!const_C) {
                broadcastCWtihBeta(M, N, inputs.back());
            }
            int step = M * N;
            CV_CheckEQ(broadcast_C.total(), static_cast<size_t>(step), "DNN/Gemm: C is not broadcast properly");
            float *ptr_y = Y.ptr<float>();
            std::memcpy(ptr_y, broadcast_C.ptr<float>(), step * sizeof(float));
        } else { // initialization
            float *ptr_y = Y.ptr<float>();
            size_t total = Y.total();
            std::memset(ptr_y, 0, total * sizeof(float));
        }

        if (const_B) {
            CV_CheckGT(packed_B.size(), static_cast<size_t>(0), "DNN/Gemm: constant B is not pre-packed");
            fastGemm(trans_a, M, N, K, alpha, A.ptr<const float>(), na, packed_B.data(), 1.f, Y.ptr<float>(), N, opt);
        } else {
            fastGemmBatch(trans_a, trans_b, alpha, A, inputs[1], 1.f, Y, opt);
        }
    }

#ifdef HAVE_OPENCL
    // Y = alpha * A * B + beta * C
    bool forward_ocl(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) {
        std::vector<UMat> inputs, outputs;

        bool use_half = (inputs_arr.depth() == CV_16S);
        inputs_arr.getUMatVector(inputs);
        outputs_arr.getUMatVector(outputs);

        const auto &A = inputs[0];
        auto &Y = outputs[0];

        std::cout << "use_half=" << use_half << ", A.depth()=" << A.depth() << ", inputs_arr.depth()=" << inputs_arr.depth() << std::endl;

        return false;

        // size_t num_blobs = blobs.size();
        // umat_blobs.resize(num_blobs);
        // for (int i = 0; i < num_blobs; i++) {
        //     blobs[i].copyTo(umat_blobs[i]);
        // }
        // if (use_half) {
        //     umat_half_blobs.resize(num_blobs);
        //     for (int i = 0; i < num_blobs; i++) {
        //         convertFp16(umat_blobs[i], umat_half_blobs[i]);
        //     }
        // }

        // const auto &A_shape = shape(A), &Y_shape = shape(Y);
        // int ma = A_shape[A_shape.size() - 2], na = A_shape.back(),
        //     M = trans_a ? na : ma,
        //     K = trans_a ? ma : na,
        //     N = Y_shape.back(),
        //     batch = static_cast<int>(Y_shape[0] / M);

        // if (ocl_op.empty()) {
        //     OCL4DNNInnerProductConfig config;
        //     config.M = M;          // M
        //     config.num_output = N; // N
        //     config.K = K;          // K
        //     config.bias_term = have_bias;
        //     config.use_half = use_half;

        //     ocl_op = Ptr<OCL4DNNInnerProduct<float>>(new OCL4DNNInnerProduct<float>(config));
        // }

        // const auto &B = const_B ? umat_blobs[0] : inputs[1],
        //            &C = have_bias ? umat_blobs.back() : UMat();

        // bool ret = ocl_op->Forward(A, B, C, Y);

        // if (ret) return true;

        // // In case OCL4DNNInnerProduct does not work
        // {
        //     UMat A_FP32, B_FP32, C_FP32, Y_FP32;
        //     if (use_half) {
        //         convertFp16(A, A_FP32);
        //         convertFp16(B, B_FP32);
        //     }
        // }

    }
#endif

#ifdef HAVE_CUDA
    // Y = A * B + C. B should be guaranteed as two dimensional.
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) CV_OVERRIDE {
        CV_CheckFalse(trans_a, "DNN/Gemm/Cuda: does not support transA");
        CV_CheckTrue(const_B, "DNN/Gemm/Cuda: input B (weight) is required to be constant");
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto wrapper_A = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto B = blobs[0];
        auto C = have_bias && const_C ? blobs[1] : Mat(); // in most cases C is constant

        if (!trans_b)
            cv::transpose(B, B);
        auto flatten_start_axis = normalize_axis(1, wrapper_A->getRank());
        return make_cuda_node<cuda4dnn::InnerProductOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), flatten_start_axis, B, C);
    }
#endif // HAVE_CUDA

#ifdef HAVE_CANN
    // Y = A * B + C.
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        auto x1 = inputs[0].dynamicCast<CannBackendWrapper>();
        auto desc_x1 = x1->getTensorDesc();
        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::MatMulV2>(name);

        // set attributes
        op->set_attr_transpose_x1(trans_a);
        op->set_attr_transpose_x2(trans_b);

        // set inputs
        // set inputs : x1
        op->set_input_x1_by_name(*op_x1, x1->name.c_str());
        op->update_input_desc_x1(*desc_x1);
        // set inputs : x2
        if (const_B) {
            auto B = blobs[0];
            auto op_const_B = std::make_shared<CannConstOp>(B.data, B.type(), shape(B), cv::format("%s_w", name.c_str()));
            op->set_input_x2_by_name(*(op_const_B->getOp()), "y");
            op->update_input_desc_x2(*(op_const_B->getTensorDesc()));
        } else {
            CV_CheckGE(inputs.size(), static_cast<size_t>(2), "DNN/Gemm/CANN: input B is required since it is not constant");
            CV_CheckGE(nodes.size(), static_cast<size_t>(2), "DNN/Gemm/CANN: input B is required since it is not constant");
            auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            auto desc_x2 = inputs[1].dynamicCast<CannBackendWrapper>()->getTensorDesc();
            op->set_input_x2_by_name(*op_x2, "y");
            op->update_input_desc_x2(*desc_x2);
        }
        // set inputs : bias
        auto mat_C = have_bias && const_C ? blobs.back() : Mat::zeros(1, 1, CV_32F);
        auto op_const_C = std::make_shared<CannConstOp>(mat_C.data, mat_C.type(), shape(mat_C), cv::format("%s_b", name.c_str()));
        op->set_input_bias(*(op_const_C->getOp()));
        op->update_input_desc_bias(*(op_const_C->getTensorDesc()));

        // set outputs
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);
        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ngraph::Node> matmul;

        if (nodes.size() == 2)
        {
            auto& inp2 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            matmul = std::make_shared<ngraph::op::MatMul>(ieInpNode, inp2, trans_a, trans_b);
        }
        else
        {
            std::shared_ptr<ngraph::Node> ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, getShape(blobs[0]), blobs[0].data);

            int flatten_axis = ieInpNode.get_shape().size() - ieWeights->get_shape().size();
            if (flatten_axis > 0) {
                std::vector<int> shape(1 + flatten_axis, 0);
                shape[shape.size() - 1] = -1;
                ieInpNode = std::make_shared<ngraph::op::v1::Reshape>(
                    ieInpNode,
                    std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{shape.size()}, shape.data()),
                    true
                );
            }
            matmul = std::make_shared<ngraph::op::MatMul>(ieInpNode, ieWeights, trans_a, trans_b);
        }
        if (alpha != 1.0f) {
            matmul = std::make_shared<ngraph::op::v1::Multiply>(matmul,
                std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1}, &alpha)
            );
        }

        if (have_bias && const_C) {
            Mat bias = blobs.back();
            auto shape = bias.total() == bias.size[0] ? ngraph::Shape{bias.total()} : getShape(bias);
            std::shared_ptr<ngraph::Node> bias_node = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, shape, bias.data);
            if (beta != 1.0f) {
                bias_node = std::make_shared<ngraph::op::v1::Multiply>(bias_node,
                    std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{1}, &beta)
                );
            }
            matmul = std::make_shared<ngraph::op::v1::Add>(matmul, bias_node, ngraph::op::AutoBroadcastType::NUMPY);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(matmul));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_VULKAN
    // Y = A * B + C. Currently support 2d matrix multiplication without bias.
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                       std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE
    {
        // does not support with bias; only 2d matmul
        auto wrapper_Y = outputs[0].dynamicCast<VkComBackendWrapper>();
        auto shape_Y = shape(*(wrapper_Y->getMat()));
        if (have_bias || shape_Y.size() > static_cast<size_t>(2)) {
            return Ptr<BackendNode>();
        }

        std::vector<Mat> vkBlobs;
        if (const_B) {
            vkBlobs.push_back(blobs[0]);
        }

        auto wrapper_A = inputs[0].dynamicCast<VkComBackendWrapper>();
        auto shape_A = shape(*wrapper_A->getMat());
        Ptr<vkcom::OpBase> op = (new vkcom::OpMatMul(vkBlobs, shape_A[0], shape_A[1], shape_Y[1]));
        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
    }
#endif

private:
    bool const_B;
    bool const_C;
    bool have_bias;
    std::vector<float> packed_B;
    Mat broadcast_C;
    int real_ndims_C;
    FastGemmOpt opt;

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNInnerProduct<float> > ocl_op;
    std::vector<UMat> umat_blobs;
    std::vector<UMat> umat_half_blobs;
#endif
};

Ptr<GemmLayer> GemmLayer::create(const LayerParams& params) {
    return makePtr<GemmLayerImpl>(params);
}

}} // namespace cv::dnn
