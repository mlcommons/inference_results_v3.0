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

#ifndef LWIS_3DUNET_BUFFERS_H
#define LWIS_3DUNET_BUFFERS_H

#include "NvInfer.h"
#include "half.h"
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <numpy.hpp>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            printf("CUDA error \"%s\", at line %d\n", cudaGetErrorString(status), __LINE__);                           \
            cudaDeviceSynchronize();                                                                                   \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

namespace lwis
{

/* read in engine file into character array */
inline size_t GetModelStream(std::vector<char>& dst, std::string engineName)
{
    size_t size{0};
    std::ifstream file(engineName, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        dst.resize(size);
        file.read(dst.data(), size);
        file.close();
    }

    return size;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    // Fall through to error
    default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline int64_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format, const bool hasImplicitBatch)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    int channelDim{-1};
    switch (format)
    {
    case nvinfer1::TensorFormat::kCHW2:
        spv = 2;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW4:
        spv = 4;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kDHWC8:
        spv = 8;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kCHW16:
        spv = 16;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCHW32:
        spv = 32;
        channelDim = d_new.nbDims - 3;
        break;
    case nvinfer1::TensorFormat::kCDHW32:
        spv = 32;
        channelDim = d_new.nbDims - 4;
        break;
    case nvinfer1::TensorFormat::kLINEAR:
    default:
        spv = 1;
        channelDim = -1;
        break;
    }
    if (spv > 1)
    {
        assert(channelDim >= 0); // Make sure we have valid channel dimension.
        d_new.d[channelDim] = roundUp(d_new.d[channelDim], spv);
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const bool hasImplicitBatch)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the
//! allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be
//!          stored. size is the amount of memory in bytes to allocate. The boolean indicates
//!          whether or not the memory allocation was successful. FreeFunc must be a functor that
//!          takes in (void* ptr) and returns void. ptr is the allocated buffer address. It must
//!          work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer()
        : mByteSize(0)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size)
        : mByteSize(size)
    {
        if (!allocFn(&mBuffer, mByteSize))
            throw std::bad_alloc();
    }

    GenericBuffer(GenericBuffer&& buf)
        : mByteSize(buf.mByteSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mByteSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mByteSize = buf.mByteSize;
            mBuffer = buf.mBuffer;
            buf.mByteSize = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const
    {
        return mByteSize;
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mByteSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMallocHost(ptr, size) == cudaSuccess;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFreeHost(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int& batchSize, const int profileIdx)
        : mEngine(engine)
        , mBatchSize(batchSize)
        , mProfileIdx(profileIdx)
    {
        // Each optimization profile owns numBindings bindings.
        int numBindingsTotal = engine->getNbBindings();
        int numBindings = numBindingsTotal / engine->getNbOptimizationProfiles();
        int bindingOffset = profileIdx * numBindings;
        mHostBindings.resize(numBindingsTotal, nullptr);
        mDeviceBindings.resize(numBindingsTotal, nullptr);
        for (int i = 0; i < numBindings; i++)
        {
            // Create host and device buffers
            size_t vol = lwis::volume(
                mEngine->getBindingDimensions(i), mEngine->getBindingFormat(i), mEngine->hasImplicitBatchDimension());
            size_t elementSize = lwis::getElementSize(mEngine->getBindingDataType(i));
            size_t allocationSize = static_cast<size_t>(mBatchSize) * vol * elementSize;
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(allocationSize);
            manBuf->hostBuffer = HostBuffer(allocationSize);
            mHostBindings[bindingOffset + i] = manBuf->hostBuffer.data();
            mDeviceBindings[bindingOffset + i] = manBuf->deviceBuffer.data();
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns a vector of device buffers.
    //!
    const std::vector<void*>& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns a vector of host buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getHostBindings()
    {
        return mHostBindings;
    }

    //!
    //! \brief Returns a vector of host buffers.
    //!
    const std::vector<void*>& getHostBindings() const
    {
        return mHostBindings;
    }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(const std::string& tensorName) const
    {
        return getBuffer(false, tensorName);
    }

    //!
    //! \brief Returns the device buffer corresponding to index.
    //!
    void* getDeviceBuffer(const size_t index) const
    {
        return getBuffer(false, index);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getHostBuffer(const std::string& tensorName) const
    {
        return getBuffer(true, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to index.
    //!
    void* getHostBuffer(const size_t index) const
    {
        return getBuffer(true, index);
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->hostBuffer.size();
    }

    //!
    //! \brief Returns the volume of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t volume(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return lwis::volume(mEngine->getBindingDimensions(index), mEngine->getBindingFormat(index),
            mEngine->hasImplicitBatchDimension());
    }

    //!
    //! \brief Returns the elementSize of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t elementSize(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return lwis::getElementSize(mEngine->getBindingDataType(index));
    }

    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl;
            return;
        }
        void* buf = mManagedBuffers[index]->hostBuffer.data();
        size_t bufSize = mManagedBuffers[index]->hostBuffer.size();
        nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
        size_t rowCount = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize;
        for (int i = 0; i < bufDims.nbDims; i++)
            os << ", " << bufDims.d[i];
        os << "]" << std::endl;
        switch (mEngine->getBindingDataType(index))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported"); break;
        case nvinfer1::DataType::kBOOL: assert(0 && "Bool network-level input and output is not supported");
        }
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(const size_t index, const cudaStream_t& stream = 0, void* src = nullptr,
        const size_t size = 0, const size_t offset = 0)
    {
        memcpyBuffers(index, true, false, true, stream, src, size, offset);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(const size_t index, const cudaStream_t& stream = 0, void* dst = nullptr,
        const size_t size = 0, const size_t offset = 0)
    {
        memcpyBuffers(index, false, true, true, stream, dst, size, offset);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(const bool isHost, const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void* getBuffer(const bool isHost, const size_t index) const
    {
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const size_t index, const bool copyInput, const bool deviceToHost, const bool async,
        const cudaStream_t& stream = 0, void* buf = nullptr, const size_t size = 0, const size_t offset = 0)
    {
        auto numBindings = mManagedBuffers.size();
        CHECK(index < numBindings) << "Invalid binding index.";
        CHECK((copyInput && mEngine->bindingIsInput(index)) || (!copyInput && !mEngine->bindingIsInput(index)))
            << "Expecting binding " << index << " to be " << (copyInput ? "input" : "output")
            << "but get the opposite.";

        void* dstPtr = deviceToHost
            ? (buf ? buf : mManagedBuffers[index]->hostBuffer.data())
            : static_cast<char*>(mManagedBuffers[index]->deviceBuffer.data()) + (buf ? offset * size : 0);
        const void* srcPtr = deviceToHost
            ? static_cast<char*>(mManagedBuffers[index]->deviceBuffer.data()) + (buf ? 0 : offset * size)
            : (buf ? buf : mManagedBuffers[index]->hostBuffer.data());
        const size_t byteSize = buf && size ? size : mManagedBuffers[index]->hostBuffer.size();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        if (async)
        {
            CHECK_EQ(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream), cudaSuccess);
        }
        else
        {
            CHECK_EQ(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType), cudaSuccess);
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size
    int mProfileIdx;                                             //!< The optimization profile index
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mHostBindings;   //!< The vector of host buffers needed for engine execution
    std::vector<void*> mDeviceBindings; //!< The vector of device buffers needed for engine execution
};

class KiTS19ManagedBuffer
{
public:
    // holds KiTS19 sample copied over to device
    DeviceBuffer sampleDeviceInput;
    // holds device side output of the final response; patched SW output is aggregated on this
    // therefore, needs to be zeroed-out for each sample
    DeviceBuffer sampleDeviceOutput;
    // host copy of the sample response, i.e. final outcome
    DeviceBuffer sampleDeviceResponse;
    // holds sliced sample, i.e. input to engine, as many as # of inference streams
    std::vector<DeviceBuffer> stagedDeviceSWInputs;
    // holds sliding window inference output, as many as # of inference streams
    std::vector<DeviceBuffer> stagedDeviceSWOutputs;
    std::vector<DeviceBuffer> gaussianPatchesDevice;

    // dual of the sample buffers in Host
    HostBuffer sampleHostInput;
    HostBuffer sampleHostOutput;
    HostBuffer sampleHostResponse;
    std::vector<HostBuffer> stagedHostSWInputs;
    std::vector<HostBuffer> stagedHostSWOutputs;
    std::vector<HostBuffer> gaussianPatchesHost;
};

class KiTS19BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    // Create a KiTS19BufferManager for handling buffer interactions between host & device
    // NOTE: this has to be maintained properly for every KiTS19 sample as the size/shape becomes
    // different Batch size is always 1, as we cannot jive with different sized samples in a batch
    // NOTE: FIXME: SW window also is set up for batch size == 1 only
    KiTS19BufferManager(const int SW_dhw = 128, const int numInferStream = 1, const size_t maxSampleSize = 64225280,
        const std::string gpatch_path = "", const bool directHostAccess = false, const int deviceId = 0,
        const int NUMAidx = -1, const int numNUMA = 1, const int batchSize = 1)
        : mDirectHostAccess(directHostAccess)
        , mMaxSampleVol(maxSampleSize)
        , mDeviceId(deviceId)
        , mNUMAidx(NUMAidx)
        , mNumNUMA(numNUMA)
        , mBatchSize(batchSize)
    {
        CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
        if (mNumNUMA > 1)
        {
            bindNumaMemPolicy(mNUMAidx, mNumNUMA);
        }
        // Each optimization profile owns numBindings bindings.
        // Create host and device buffers
        size_t sw_vol = SW_dhw * SW_dhw * SW_dhw;
        // input is INT8 and 1-ch, always single sample (batch size of 1)
        size_t sw_input_alloc = sw_vol * 1 * 1 * batchSize;
        // output is FP16 and 3-ch, always single sample (batch size of 1)
        size_t sw_output_alloc = sw_vol * sizeof(__half) * 3 * batchSize;

        mManagedBuffers = std::make_shared<KiTS19ManagedBuffer>();

        mManagedBuffers->sampleDeviceInput = DeviceBuffer();
        mManagedBuffers->sampleDeviceOutput = DeviceBuffer();
        mManagedBuffers->sampleDeviceResponse = DeviceBuffer();
        mManagedBuffers->stagedDeviceSWInputs.resize(numInferStream);
        mManagedBuffers->stagedDeviceSWOutputs.resize(numInferStream);

        mManagedBuffers->sampleHostInput = HostBuffer();
        mManagedBuffers->sampleHostOutput = HostBuffer();
        mManagedBuffers->sampleHostResponse = HostBuffer();
        mManagedBuffers->stagedHostSWInputs.resize(numInferStream);
        mManagedBuffers->stagedHostSWOutputs.resize(numInferStream);

        for (int i = 0; i < numInferStream; ++i)
        {
            if (!mDirectHostAccess)
            {
                mManagedBuffers->stagedDeviceSWInputs[i] = DeviceBuffer(sw_input_alloc);
                CHECK_CUDA(cudaMemset(getSWInputDeviceBuffer(i), 0, sw_input_alloc));
                mManagedBuffers->stagedDeviceSWOutputs[i] = DeviceBuffer(sw_output_alloc);
            }
            else
            {
                mManagedBuffers->stagedHostSWInputs[i] = HostBuffer(sw_input_alloc);
                std::memset(getSWInputHostBuffer(i), 0, sw_input_alloc);
                mManagedBuffers->stagedHostSWOutputs[i] = HostBuffer(sw_output_alloc);
            }
        }

        m_SW_gaussian_patches_npy = std::make_unique<npy::NpyFile>(gpatch_path);
        std::vector<char> gpatch_data;
        m_SW_gaussian_patches_npy->loadAll(gpatch_data);

        m_SW_gaussian_patch_dim = m_SW_gaussian_patches_npy->getDims();
        m_SW_gaussian_patch_num = m_SW_gaussian_patch_dim[0];
        CHECK(m_SW_gaussian_patch_num == 27 && m_SW_gaussian_patch_dim[1] == SW_dhw
            && m_SW_gaussian_patch_dim[2] == SW_dhw && m_SW_gaussian_patch_dim[3] == SW_dhw)
            << "Unexpected 3DUNet KiTS19 gaussian patch shape";
        m_SW_gaussian_patch_size = m_SW_gaussian_patches_npy->getTensorSize() / m_SW_gaussian_patch_num;

        mManagedBuffers->gaussianPatchesDevice.resize(m_SW_gaussian_patch_num);
        mManagedBuffers->gaussianPatchesHost.resize(m_SW_gaussian_patch_num);
        for (int i = 0; i < m_SW_gaussian_patch_num; ++i)
        {
            if (!mDirectHostAccess)
            {
                mManagedBuffers->gaussianPatchesDevice[i] = DeviceBuffer(m_SW_gaussian_patch_size);
                CHECK_CUDA(cudaMemcpy(mManagedBuffers->gaussianPatchesDevice[i].data(),
                    gpatch_data.data() + i * m_SW_gaussian_patch_size, m_SW_gaussian_patch_size,
                    cudaMemcpyHostToDevice));
            }
            else
            {
                mManagedBuffers->gaussianPatchesHost[i] = HostBuffer(m_SW_gaussian_patch_size);
                std::memcpy(mManagedBuffers->gaussianPatchesHost[i].data(),
                    gpatch_data.data() + i * m_SW_gaussian_patch_size, m_SW_gaussian_patch_size);
            }
        }

        // optimized, max size buffer usage
        // preallocate buffer for max size sample
        // input is INT8 and 1-ch
        size_t sample_input_alloc = maxSampleSize * 1 * 1;
        // output is FP16 and 3-ch
        size_t sample_output_alloc = maxSampleSize * sizeof(__half) * 3;
        // response is INT8 and 1-ch
        size_t sample_resp_alloc = maxSampleSize * 1 * 1;
        mManagedBuffers->sampleHostInput = HostBuffer(sample_input_alloc);
        mManagedBuffers->sampleHostResponse = HostBuffer(sample_resp_alloc);
        if (!mDirectHostAccess)
        {
            mManagedBuffers->sampleDeviceInput = DeviceBuffer(sample_input_alloc);
            mManagedBuffers->sampleDeviceOutput = DeviceBuffer(sample_output_alloc);
            CHECK_CUDA(cudaMemset(getSampleOutputDeviceBuffer(), 0, sample_output_alloc));
            mManagedBuffers->sampleDeviceResponse = DeviceBuffer(sample_resp_alloc);
        }
        else
        {
            mManagedBuffers->sampleHostOutput = HostBuffer(sample_output_alloc);
            std::memset(getSampleOutputHostBuffer(), 0, sample_output_alloc);
        }

        if (mNumNUMA > 1)
        {
            resetNumaMemPolicy();
        }
    }

    // set up buffers for this
    // NOTE: sample batch size is always 1
    void setupBuffers(const int sample_d, const int sample_h, const int sample_w, const cudaStream_t& stream = 0,
        const bool async = false)
    {
        // KiTS19 sample is with 1ch LINEAR for input, 3ch LINEAR for output
        size_t sample_vol = sample_d * sample_h * sample_w;
        CHECK(sample_vol <= mMaxSampleVol) << "Sample Buffer Allocation error.";
        // input is INT8 and 1-ch, always single sample (batch size of 1)
        mSampleInputSize = sample_vol * 1 * 1;
        // output is FP16 and 3-ch, always single sample (batch size of 1)
        mSampleOutputSize = sample_vol * sizeof(__half) * 3;
        // response is INT8 and 1-ch, always single sample (batch size of 1)
        mSampleResponseSize = sample_vol * 1 * 1;

        // optimized, max size buffer usage
        if (!mDirectHostAccess)
        {
            if (async)
            {
                CHECK_CUDA(cudaMemsetAsync(getSampleInputDeviceBuffer(), 0, mMaxSampleVol * 1 * 1, stream));
                CHECK_CUDA(cudaMemsetAsync(getSampleOutputDeviceBuffer(), 0, mMaxSampleVol * 2 * 3, stream));
                CHECK_CUDA(cudaMemsetAsync(getSampleResponseDeviceBuffer(), 0, mMaxSampleVol * 1 * 1, stream));
            }
            else
            {
                CHECK_CUDA(cudaMemset(getSampleInputDeviceBuffer(), 0, mMaxSampleVol * 1 * 1));
                CHECK_CUDA(cudaMemset(getSampleOutputDeviceBuffer(), 0, mMaxSampleVol * 2 * 3));
                CHECK_CUDA(cudaMemset(getSampleResponseDeviceBuffer(), 0, mMaxSampleVol * 1 * 1));
            }
        }
        else
        {
            std::memset(getSampleInputHostBuffer(), 0, mMaxSampleVol * 1 * 1);
            std::memset(getSampleOutputHostBuffer(), 0, mMaxSampleVol * 2 * 3);
            std::memset(getSampleResponseHostBuffer(), 0, mMaxSampleVol * 1 * 1);
        }
    }

    bool isDirectHostAccessEnabled() const
    {
        return mDirectHostAccess;
    }

    // returns sample input buffer on device
    void* getSampleInputDeviceBuffer() const
    {
        return mManagedBuffers->sampleDeviceInput.data();
    }

    // returns sample output buffer on device
    void* getSampleOutputDeviceBuffer() const
    {
        return mManagedBuffers->sampleDeviceOutput.data();
    }

    // returns sample response buffer on device
    void* getSampleResponseDeviceBuffer() const
    {
        return mManagedBuffers->sampleDeviceResponse.data();
    }

    // returns sample SW input buffer on device
    void* getSWInputDeviceBuffer(const size_t index) const
    {
        return mManagedBuffers->stagedDeviceSWInputs[index].data();
    }

    // returns sample SW output buffer on device
    void* getSWOutputDeviceBuffer(const size_t index) const
    {
        return mManagedBuffers->stagedDeviceSWOutputs[index].data();
    }

    // returns sample input buffer on host
    void* getSampleInputHostBuffer() const
    {
        return mManagedBuffers->sampleHostInput.data();
    }

    // returns sample output buffer on host
    void* getSampleOutputHostBuffer() const
    {
        return mManagedBuffers->sampleHostOutput.data();
    }

    // returns sample response buffer on host
    void* getSampleResponseHostBuffer() const
    {
        return mManagedBuffers->sampleHostResponse.data();
    }

    // returns SW input buffer on host
    void* getSWInputHostBuffer(const size_t index) const
    {
        return mManagedBuffers->stagedHostSWInputs[index].data();
    }

    // returns SW output buffer on host
    void* getSWOutputHostBuffer(const size_t index) const
    {
        return mManagedBuffers->stagedHostSWOutputs[index].data();
    }

    // returns sample input buffer
    void* getSampleInputBuffer() const
    {
        return mDirectHostAccess ? getSampleInputHostBuffer() : getSampleInputDeviceBuffer();
    }

    // returns sample output buffer
    void* getSampleOutputBuffer() const
    {
        return mDirectHostAccess ? getSampleOutputHostBuffer() : getSampleOutputDeviceBuffer();
    }

    // returns sample response buffer
    void* getSampleResponseBuffer() const
    {
        return mDirectHostAccess ? getSampleResponseHostBuffer() : getSampleResponseDeviceBuffer();
    }

    // returns SW input buffer
    void* getSWInputBuffer(const size_t index) const
    {
        return mDirectHostAccess ? getSWInputHostBuffer(index) : getSWInputDeviceBuffer(index);
    }

    // returns SW output buffer
    void* getSWOutputBuffer(const size_t index) const
    {
        return mDirectHostAccess ? getSWOutputHostBuffer(index) : getSWOutputDeviceBuffer(index);
    }

    // returns sample input buffer on host
    size_t getSampleInputSize() const
    {
        return mSampleInputSize;
    }

    // returns sample output buffer on host
    size_t getSampleOutputSize() const
    {
        return mSampleOutputSize;
    }

    // returns sample response buffer on host
    size_t getSampleResponseSize() const
    {
        return mSampleResponseSize;
    }

    // returns SW input buffer on host
    size_t getSWInputSize(const size_t index) const
    {
        return mDirectHostAccess ? mManagedBuffers->stagedHostSWInputs[0].size()
                                 : mManagedBuffers->stagedDeviceSWInputs[0].size();
    }

    // returns SW output buffer on host
    size_t getSWOutputSize() const
    {
        return mDirectHostAccess ? mManagedBuffers->stagedHostSWOutputs[0].size()
                                 : mManagedBuffers->stagedDeviceSWOutputs[0].size();
    }

    // returns gaussian patch size
    size_t getGaussianPatchSize() const
    {
        return m_SW_gaussian_patch_size;
    }

    // returns gaussian patch buffer in Device
    void* getGaussianPatchDeviceBuffer(const size_t index) const
    {
        return mManagedBuffers->gaussianPatchesDevice[index].data();
    }

    // returns gaussian patch buffer in Host
    void* getGaussianPatchHostBuffer(const size_t index) const
    {
        return mManagedBuffers->gaussianPatchesHost[index].data();
    }

    // returns gaussian patch buffer
    void* getGaussianPatchBuffer(const size_t index) const
    {
        return mDirectHostAccess ? getGaussianPatchHostBuffer(index) : getGaussianPatchDeviceBuffer(index);
    }

    // copy the contents of input host buffers to input device buffers asynchronously
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0, void* src = nullptr, const size_t size = 0)
    {
        memcpyBuffers(true, true, stream, src, size);
    }

    // copy the contents of output device buffers to output host buffers asynchronously
    void copyOutputToHostAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(false, true, stream, nullptr, 0);
    }

    ~KiTS19BufferManager() = default;

private:
    // only need copy for: 1) KiTS19 sample from Host memory to mManagedBuffers->sampleInput -- H2D,
    // and
    //                     2) mManagedBuffers->sampleOutput to mManagedBuffers->sampleResponse --
    //                     D2H
    void memcpyBuffers(const bool H2D, // true: case 1), false: case 2)
        const bool async, const cudaStream_t& stream = 0,
        void* buf = nullptr, // if case 1), maybe provide Host addr of KiTS19 sample; else null
        const size_t size = 0)
    {
        const void* srcPtr = H2D ? (buf ? buf : getSampleInputHostBuffer()) : getSampleResponseDeviceBuffer();
        void* dstPtr = H2D ? getSampleInputDeviceBuffer() : getSampleResponseHostBuffer();
        const size_t byteSize = H2D ? (buf && (size > 0) ? size : getSampleInputSize()) : getSampleResponseSize();
        const cudaMemcpyKind memcpyType = H2D ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (async)
        {
            CHECK_EQ(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream), cudaSuccess);
        }
        else
        {
            CHECK_EQ(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType), cudaSuccess);
        }
    }

    std::shared_ptr<KiTS19ManagedBuffer> mManagedBuffers;
    const bool mDirectHostAccess;
    const int mNUMAidx;
    const int mDeviceId;
    const int mNumNUMA;
    const int mBatchSize;

    // Pre-conditioned Gaussian patches are coalesced, on first dim for different positions
    std::unique_ptr<npy::NpyFile> m_SW_gaussian_patches_npy;
    // dim/size for single Gaussian patch
    std::vector<size_t> m_SW_gaussian_patch_dim;
    size_t m_SW_gaussian_patch_num;
    size_t m_SW_gaussian_patch_size;
    // pointers for bufs/patches directly used in SW inference
    std::vector<void*> m_SW_gaussian_patches;
    // bookkeep current sample size
    size_t mMaxSampleVol{0};
    size_t mSampleInputSize{0};
    size_t mSampleOutputSize{0};
    size_t mSampleResponseSize{0};
};

} // namespace lwis

#endif // LWIS_3DUNET_BUFFERS_H
