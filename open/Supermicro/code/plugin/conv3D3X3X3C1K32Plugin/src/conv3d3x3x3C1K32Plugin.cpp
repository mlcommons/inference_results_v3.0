/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdexcept>

#include "conv3d3x3x3C1K32Plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::conv3D3X3X3C1K32Plugin;
using nvinfer1::plugin::conv3D3X3X3C1K32PluginCreator;

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_));                     \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status_ = call;                                                                                  \
        if (status_ != CUDNN_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "CUDNN Error at line %d: %s\n", __LINE__, cudnnGetErrorString(status_));                   \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////

enum PluginMode
{
    FP32_LINEAR_FP32_LINEAR_MODE, // FP32 Linear -> FP32 Linear
    INT8_LINEAR_INT8_CDHW32_MODE, // INT8 Linear -> CDHW32 Linear
};

// Checks both intput/output format/datatypes
static PluginMode getPluginMode(
    const nvinfer1::PluginTensorDesc& inputDesc, const nvinfer1::PluginTensorDesc& outputDesc)
{
    if (inputDesc.format == nvinfer1::PluginFormat::kLINEAR && inputDesc.type == nvinfer1::DataType::kFLOAT
        && outputDesc.format == nvinfer1::PluginFormat::kLINEAR && outputDesc.type == nvinfer1::DataType::kFLOAT)
    {
        return FP32_LINEAR_FP32_LINEAR_MODE;
    }
    else if (inputDesc.format == nvinfer1::PluginFormat::kLINEAR && inputDesc.type == nvinfer1::DataType::kINT8
        && outputDesc.format == nvinfer1::PluginFormat::kCDHW32 && outputDesc.type == nvinfer1::DataType::kINT8)
    {
        return INT8_LINEAR_INT8_CDHW32_MODE;
    }
    else
    {
        ASSERT(false && "Unexpected input format");
    }
}

namespace
{
const char* CONV3D3X3X3C1K32_PLUGIN_VERSION{"1"};
const char* CONV3D3X3X3C1K32_PLUGIN_NAME{"CONV3D3X3X3C1K32_TRT"};
} // namespace

REGISTER_TENSORRT_PLUGIN(conv3D3X3X3C1K32PluginCreator);

PluginFieldCollection conv3D3X3X3C1K32PluginCreator::mFC{};
std::vector<PluginField> conv3D3X3X3C1K32PluginCreator::mPluginAttributes;

conv3D3X3X3C1K32Plugin::conv3D3X3X3C1K32Plugin(int inputChannels, const std::vector<float>& weights)
    : mInitialized(false)
    , mInputChannels(inputChannels)
    , mInActivationScale(-1.0F)
    , mOutActivationScale(-1.0F)
    , mWeights(weights)
{
}

conv3D3X3X3C1K32Plugin::conv3D3X3X3C1K32Plugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mInputChannels);
    deserialize_value(&serialData, &serialLength, &mWeights);
    deserialize_value(&serialData, &serialLength, &mWeightScale);
    deserialize_value(&serialData, &serialLength, &mInActivationScale);
    deserialize_value(&serialData, &serialLength, &mOutActivationScale);
}

conv3D3X3X3C1K32Plugin::~conv3D3X3X3C1K32Plugin()
{
    terminate();
}

// conv3D3X3X3C1K32Plugin returns one output.
int conv3D3X3X3C1K32Plugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs conv3D3X3X3C1K32Plugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *exprBuilder.constant(0), *exprBuilder.constant(32));

    return output;
}

int conv3D3X3X3C1K32Plugin::initialize() noexcept
{
    if (!mInitialized)
    {
        CHECK_CUDA(cudaMalloc(&mDeviceWeights, mWeights.size() * sizeof(float)));

        CHECK_CUDNN(cudnnCreate(&mCudnnHandle));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&mConvDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mImageDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&mFltDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&mOutDesc));

        const int padA[6] = {1, 1, 1, 1, 1, 1};
        const int dilationA[3] = {1, 1, 1};
        const int convstrideA[3] = {1, 1, 1};
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            mConvDesc, 3, padA, convstrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, device));

        memset(&mContext, 0, sizeof(mContext));
        mContext.sm_count = props.multiProcessorCount;
        mContext.sm_shared_size = props.sharedMemPerMultiprocessor;
        mContext.sm_version = props.major * 100 + props.minor * 10;

        memset(&mParams, 0, sizeof(mParams));

        mInitialized = true;
    }
    return 0;
}

void conv3D3X3X3C1K32Plugin::terminate() noexcept
{
    if (mInitialized)
    {
        cudaFree(mDeviceWeights);

        // Release cuDNN descriptors and handle.
        cudnnDestroyConvolutionDescriptor(mConvDesc);
        cudnnDestroyTensorDescriptor(mImageDesc);
        cudnnDestroyFilterDescriptor(mFltDesc);
        cudnnDestroyTensorDescriptor(mOutDesc);
        cudnnDestroy(mCudnnHandle);

        mInitialized = false;
    }
    return;
}

size_t conv3D3X3X3C1K32Plugin::setCudnnDescriptors(const nvinfer1::PluginTensorDesc* inputs) const
// returns workspace size needed by cuDNN
{
    nvinfer1::Dims input_dims = inputs[0].dims;
    const int n = input_dims.d[0];
    const int c = input_dims.d[1];
    const int d = input_dims.d[2];
    const int h = input_dims.d[3];
    const int w = input_dims.d[4];

    const int k = 32;

    const int nbDims = 5;
    const int dimA[nbDims] = {n, c, d, h, w};
    const int filterDimA[nbDims] = {k, c, 3, 3, 3};

    int strideA[nbDims];
    strideA[nbDims - 1] = 1;
    for (int dd = nbDims - 2; dd >= 0; dd--)
    {
        strideA[dd] = strideA[dd + 1] * dimA[dd + 1];
    }

    CHECK_CUDNN(cudnnSetConvolutionMathType(mConvDesc, CUDNN_DEFAULT_MATH));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(mImageDesc, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(mFltDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, filterDimA));
    const int outDimA[nbDims] = {n, k, d, h, w};
    int outStrideA[nbDims];
    outStrideA[nbDims - 1] = 1;
    for (int dd = nbDims - 2; dd >= 0; dd--)
    {
        outStrideA[dd] = outStrideA[dd + 1] * outDimA[dd + 1];
    }
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(mOutDesc, CUDNN_DATA_FLOAT, nbDims, outDimA, outStrideA));

    // Determine workspace sizes for the different convolutions.
    size_t workspace_sz = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(mCudnnHandle, mImageDesc, mFltDesc, mConvDesc, mOutDesc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_sz));
    return workspace_sz;
}

size_t conv3D3X3X3C1K32Plugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    const PluginMode pluginMode = getPluginMode(inputs[0], outputs[0]);
    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        return setCudnnDescriptors(inputs);
    }
    // uses zero workspace.
    return 0;
}

int conv3D3X3X3C1K32Plugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    ASSERT(mInitialized);
    ASSERT(mInputChannels == 1);

    const PluginMode pluginMode = getPluginMode(inputDesc[0], outputDesc[0]);

    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        // use cuDNN
        CHECK_CUDNN(cudnnSetStream(mCudnnHandle, stream));

        size_t workspace_sz = setCudnnDescriptors(inputDesc);

        const float one = 1.0F, zero = 0.0F;
        CHECK_CUDNN(cudnnConvolutionForward(mCudnnHandle, &one, mImageDesc, inputs[0], mFltDesc, mDeviceWeights,
            mConvDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace, workspace_sz, &zero, mOutDesc, outputs[0]));
    }
    else if (pluginMode == INT8_LINEAR_INT8_CDHW32_MODE)
    {
        const nvinfer1::Dims input_dims = inputDesc[0].dims;
        const int n = input_dims.d[0];
        const int c = input_dims.d[1];
        const int d = input_dims.d[2];
        const int h = input_dims.d[3];
        const int w = input_dims.d[4];

        const nvinfer1::Dims output_dims = outputDesc[0].dims;
        const int k = output_dims.d[1];

        mParams.gmem_in = const_cast<void*>(inputs[0]);
        mParams.gmem_flt = mDeviceWeights;
        mParams.gmem_out = outputs[0];
        mParams.img_n = n;
        mParams.img_c = c;
        mParams.img_d = d;
        mParams.img_h = h;
        mParams.img_w = w;
        mParams.flt_k = k;

        mParams.img_stride_w = 1;
        mParams.img_stride_h = mParams.img_stride_w * mParams.img_w;
        mParams.img_stride_d = mParams.img_stride_h * mParams.img_h;
        mParams.img_stride_c = mParams.img_stride_d * mParams.img_d;
        mParams.img_stride_n = mParams.img_stride_c * mParams.img_c;

        mParams.out_o = mParams.img_d;
        mParams.out_p = mParams.img_h;
        mParams.out_q = mParams.img_w;
        mParams.out_k = mParams.flt_k;

        // output in NC/32DHW32 format
        mParams.out_stride_q = 32;
        mParams.out_stride_p = mParams.out_stride_q * mParams.out_q;
        mParams.out_stride_o = mParams.out_stride_p * mParams.out_p;
        mParams.out_stride_k = mParams.out_stride_o * mParams.out_o;
        mParams.out_stride_n = mParams.out_stride_k / 32 * mParams.out_k;

        mParams.scale = mInActivationScale * mWeightScale / mOutActivationScale;
        mParams.is_fp32 = false;

        conv_3x3x3_c1_k32_dispatch(mContext, mParams, stream);
    }
    else
    {
        ASSERT(false && "Enqueue met unexpected case");
    }

    return 0;
}

size_t conv3D3X3X3C1K32Plugin::getSerializationSize() const noexcept
{
    return (serialized_size(mInputChannels) + serialized_size(mWeights) + serialized_size(mWeightScale)
        + serialized_size(mInActivationScale) + serialized_size(mOutActivationScale));
}

void conv3D3X3X3C1K32Plugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mInputChannels);
    serialize_value(&buffer, mWeights);
    serialize_value(&buffer, mWeightScale);
    serialize_value(&buffer, mInActivationScale);
    serialize_value(&buffer, mOutActivationScale);
}

bool conv3D3X3X3C1K32Plugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear
        = (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format);

    bool support_int8_linear_to_int8_cdhw32 = (pos < nbInputs)
        ? (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR)
        : (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32)
            && inOut[0].type == nvinfer1::DataType::kINT8 && inOut[0].format == nvinfer1::PluginFormat::kLINEAR;

    return support_fp32_linear || support_int8_linear_to_int8_cdhw32;
}

const char* conv3D3X3X3C1K32Plugin::getPluginType() const noexcept
{
    return CONV3D3X3X3C1K32_PLUGIN_NAME;
}

const char* conv3D3X3X3C1K32Plugin::getPluginVersion() const noexcept
{
    return CONV3D3X3X3C1K32_PLUGIN_VERSION;
}

void conv3D3X3X3C1K32Plugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* conv3D3X3X3C1K32Plugin::clone() const noexcept
{
    auto plugin = new conv3D3X3X3C1K32Plugin(mInputChannels, mWeights);
    plugin->setPluginNamespace(mPluginNamespace);
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void conv3D3X3X3C1K32Plugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* conv3D3X3X3C1K32Plugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType conv3D3X3X3C1K32Plugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);

    return nvinfer1::DataType::kFLOAT;
}

void conv3D3X3X3C1K32Plugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const PluginMode pluginMode = getPluginMode(in[0].desc, out[0].desc);

    if (pluginMode == FP32_LINEAR_FP32_LINEAR_MODE)
    {
        CHECK_CUDA(
            cudaMemcpy(mDeviceWeights, mWeights.data(), mWeights.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (pluginMode == INT8_LINEAR_INT8_CDHW32_MODE)
    {
        mInActivationScale = in[0].desc.scale;
        mOutActivationScale = out[0].desc.scale;
        std::vector<int8_t> weights(mWeights.size(), 0);
        float maxAbsVal = *std::max_element(mWeights.begin(), mWeights.end());
        float mult = 127.0F / maxAbsVal;
        mWeightScale = 1.0F / mult;
        std::transform(mWeights.begin(), mWeights.end(), weights.begin(),
            [=](float x) { return static_cast<int8_t>(roundf(std::max(std::min(x * mult, 127.0F), -127.0F))); });
        CHECK_CUDA(cudaMemcpy(mDeviceWeights, weights.data(), weights.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    }
}

// conv3D3X3X3C1K32PluginCreator methods
conv3D3X3X3C1K32PluginCreator::conv3D3X3X3C1K32PluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("inputChannels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* conv3D3X3X3C1K32PluginCreator::getPluginName() const noexcept
{
    return CONV3D3X3X3C1K32_PLUGIN_NAME;
}

const char* conv3D3X3X3C1K32PluginCreator::getPluginVersion() const noexcept
{
    return CONV3D3X3X3C1K32_PLUGIN_VERSION;
}

const PluginFieldCollection* conv3D3X3X3C1K32PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* conv3D3X3X3C1K32PluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    int inputChannels = 0;
    std::vector<float> weights;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "inputChannels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            inputChannels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights.resize(fields[i].length);
            memcpy(weights.data(), fields[i].data, fields[i].length * sizeof(float));
        }
    }

    conv3D3X3X3C1K32Plugin* obj = new conv3D3X3X3C1K32Plugin(inputChannels, weights);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* conv3D3X3X3C1K32PluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    conv3D3X3X3C1K32Plugin* obj = new conv3D3X3X3C1K32Plugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
