/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * Modified by NEUCHIPS on 2023.
 */

#ifndef LWIS_BUFFERS_H
#define LWIS_BUFFERS_H

#include "NeuchipsInfer.h"
#include <cassert>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <new>
#include <random>
#include <fstream>

namespace lwis {

/* read in engine file into character array */
inline size_t GetModelStream(std::vector<char> &dst, std::string engineName) {
  size_t size{0};
  std::ifstream file(engineName, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    dst.resize(size);
    file.read(dst.data(), size);
    file.close();
  }

  return size;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline int64_t volume(const ncsinfer1::Dims& d, const ncsinfer1::TensorFormat& format, const bool hasImplicitBatch)
{
    ncsinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    int channelDim{-1};
    switch(format)
    {
        case ncsinfer1::TensorFormat::kLINEAR:
        default: spv = 1; channelDim = -1; break;
    }
    if (spv > 1)
    {
        assert(channelDim >= 0); // Make sure we have valid channel dimension.
        d_new.d[channelDim] = roundUp(d_new.d[channelDim], spv);
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const ncsinfer1::Dims& d, const bool hasImplicitBatch)
{
    return volume(d, ncsinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
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
    void* data() { return mBuffer; }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const { return mBuffer; }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const { return mByteSize; }

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
    //bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
    bool operator()(void** ptr, size_t size) const {

        *ptr = malloc(size);
        LOG(INFO) << "DeviceAllocator alloc size " << size << " , ptr " << reinterpret_cast<void *>(*ptr);
        if(!ptr) {
            LOG(INFO) << "DeviceAllocator malloc failed";
            return false;
        }
        else {
            return true;
        }

    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const {
        free(ptr);        
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
      *ptr = malloc(size);
      LOG(INFO) << "HostAllocator size " << size << ", ptr " << reinterpret_cast<void *>(*ptr);
      if(!ptr) {
          LOG(INFO) << "HostAllocator malloc failed";
	  return false;
      }
      else {
          return true;
      }
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const { 
        free(ptr);
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

} // namespace lwis

#endif // LWIS_BUFFERS_H
