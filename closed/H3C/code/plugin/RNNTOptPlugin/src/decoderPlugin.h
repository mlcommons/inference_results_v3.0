/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef GNMT_DECODER_PLUGIN_H
#define GNMT_DECODER_PLUGIN_H

#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <cublasLt.h>


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>

#include <iostream>
#include <string>

#include "decoderPlugin.cuh"

namespace nvinfer1
{

namespace plugin
{

class RNNTDecoderPlugin : public IPluginV2DynamicExt  {
public:
    RNNTDecoderPlugin(const PluginFieldCollection* fc);

    RNNTDecoderPlugin(const void* data, size_t length);
    ~RNNTDecoderPlugin() override = default;

    // IPluginV2Ext fields
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    void destroy() noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    virtual void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2Ext fields
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override;
    virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override;
    virtual int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    void setCUDAInfo(
        cudaStream_t mStreamh, cublasHandle_t mCublas, void** mWeights_d, void** mBias_d, void* mWorkSpace_d);

    template <typename T>
    void write(char*& buffer, const T& val) const;

    template <typename T>
    void read(const char*& buffer, T& val) const;

private:
    int mInputSize;

    int mHiddenSize;
    int mNumLayers;
    
    nvinfer1::DataType mDataType;
        
    cudaStream_t mStreamh;
    
    cublasHandle_t mCublas;
    
    bool mInitialized{false};
    


    int mDevice;
    int mSMVersionMajor;
    int mSMVersionMinor;
    
    std::string mNamespace;
    
    size_t mWorkSpaceSize;
    void *mWorkSpace_d;
    
    void **mWeights_h;
    void **mWeights_d;
    
    void **mBias_h;
    void **mBias_d;
};

class RNNTDecoderPluginCreator : public IPluginCreator {
public:
    RNNTDecoderPluginCreator() = default;

    ~RNNTDecoderPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // GNMT_DECODER_PLUGIN_H
