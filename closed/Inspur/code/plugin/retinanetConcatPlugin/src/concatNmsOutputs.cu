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

#include <cuda.h>

#include "concatNmsOutputs.h"
#include "concatNmsOutputsKernel.cuh"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)




using namespace nvinfer1;
using nvinfer1::plugin::RetinanetConcatNmsOutputsPlugin;
using nvinfer1::plugin::RetinanetConcatNmsOutputsPluginCreator;

PluginFieldCollection RetinanetConcatNmsOutputsPluginCreator::mFC{};
REGISTER_TENSORRT_PLUGIN(RetinanetConcatNmsOutputsPluginCreator);



RetinanetConcatNmsOutputsPlugin::RetinanetConcatNmsOutputsPlugin(const PluginFieldCollection *fc) {
}

RetinanetConcatNmsOutputsPlugin::RetinanetConcatNmsOutputsPlugin(const void* data, size_t length) {
}

const char* RetinanetConcatNmsOutputsPlugin::getPluginType() const noexcept
{
    return "RetinanetConcatNmsOutputsPlugin";
}

const char* RetinanetConcatNmsOutputsPlugin::getPluginVersion() const noexcept
{
    return "1";
}

void RetinanetConcatNmsOutputsPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RetinanetConcatNmsOutputsPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void RetinanetConcatNmsOutputsPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* RetinanetConcatNmsOutputsPlugin::clone() const noexcept
{
    size_t sz = getSerializationSize();

    char* buff = (char*) malloc(getSerializationSize());

    // serialize is an assertion sanity check because SelectPlugin is sizeless
    serialize(buff);
    RetinanetConcatNmsOutputsPlugin* ret = new RetinanetConcatNmsOutputsPlugin(buff, sz);
    free(buff);

    return ret;
}

int RetinanetConcatNmsOutputsPlugin::getNbOutputs() const noexcept
{
    return 1;
}


DimsExprs RetinanetConcatNmsOutputsPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{

    assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    assert(nbInputs == 4);

    DimsExprs ret;
    ret.nbDims = 2;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.constant(7001);  // FIXME

    return(ret);

}

bool RetinanetConcatNmsOutputsPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (nbInputs != 4 || nbOutputs != 1 ) {
        printf("Wrong input or output count: %d and %d\n", nbInputs, nbOutputs);
        return false;
    }

    // Input 0 and Input 3 should be in INT32 linear format. The other inputs/outputs should be in FP32 linear format.
    DataType expectedDtype{DataType::kFLOAT};
    if (pos == 0 || pos == 3)
    {
        expectedDtype = DataType::kINT32;
    }

    if (inOut[pos].type != expectedDtype && inOut[pos].format == PluginFormat::kLINEAR)
    {
        return false;
    }

    return true;
}

void RetinanetConcatNmsOutputsPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

int RetinanetConcatNmsOutputsPlugin::initialize() noexcept
{
    return cudaSuccess;
}

void RetinanetConcatNmsOutputsPlugin::terminate() noexcept {
}

size_t RetinanetConcatNmsOutputsPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    size_t size = 0;

    return size;
}

// int RetinanetConcatNmsOutputsPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t
// stream) {
int RetinanetConcatNmsOutputsPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    int batchSize = inputDesc[0].dims.d[0];
    int C0 = inputDesc[1].dims.d[1];

    assert(C0==1000);

    launch_concat_nms_outputs_gpu(batchSize,C0,
        (float*) outputs[0],
        (float*) inputs[0],
        (float*) inputs[1],
        (float*) inputs[2],
        (float*) inputs[3],
        stream);

    return 0;
}

size_t RetinanetConcatNmsOutputsPlugin::getSerializationSize() const noexcept
{
    size_t sz = 0;

    return sz;
}

void RetinanetConcatNmsOutputsPlugin::serialize(void* buffer) const noexcept
{
// Use maybe_unused attribute when updating to CUDA_STANDARD C++17
#ifndef NDEBUG
    char* d = static_cast<char*>(buffer);
    auto *d_start = d;
#endif

    assert(d == d_start + getSerializationSize());
}

nvinfer1::DataType RetinanetConcatNmsOutputsPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return DataType::kFLOAT;
}

template <typename T>
void RetinanetConcatNmsOutputsPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void RetinanetConcatNmsOutputsPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* RetinanetConcatNmsOutputsPluginCreator::getPluginName() const noexcept
{
    return "RetinanetConcatNmsOutputsPlugin";
}

const char* RetinanetConcatNmsOutputsPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const PluginFieldCollection* RetinanetConcatNmsOutputsPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

void RetinanetConcatNmsOutputsPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RetinanetConcatNmsOutputsPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV2DynamicExt* RetinanetConcatNmsOutputsPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    return new RetinanetConcatNmsOutputsPlugin(fc);
}

IPluginV2DynamicExt* RetinanetConcatNmsOutputsPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new RetinanetConcatNmsOutputsPlugin(serialData, serialLength);
}
