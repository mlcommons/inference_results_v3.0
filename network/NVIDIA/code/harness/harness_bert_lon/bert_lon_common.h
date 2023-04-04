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

#ifndef __BERT_LON_COMMON_H__
#define __BERT_LON_COMMON_H__

#include <set>

#include "common.hpp"
#include "half.h"
#include "lwis_buffers.h"
#include "system_under_test.h"

namespace bert_lon
{

struct QuerySampleAndLength
{
    mlperf::QuerySample Sample;
    int Length;
};

constexpr size_t BERT_MAX_SEQ_LENGTH{384};
constexpr int NUM_RESPONSE_THREADS = 1;

template <typename T>
class BERTManagedBuffer;

using BERTInputType = int32_t;
using BERTInput = std::array<BERTInputType, BERT_MAX_SEQ_LENGTH>;
using BERTOutputType = half_float::half;
using BERTOutput = std::array<BERTOutputType, BERT_MAX_SEQ_LENGTH * 2>; // *2 for start & end

using BERTBufferIn = BERTManagedBuffer<BERTInputType>;
using BERTBufferOut = BERTManagedBuffer<BERTOutputType>;
using BERTTaskElem_t = std::pair<QuerySampleAndLength, std::chrono::high_resolution_clock::time_point>;
using BERTTask_t = std::vector<BERTTaskElem_t>;

// Use (copyStreamIdx, (max seqLen, batch size, total seqLen)) as key to get CUDA Graphs
using BERTGraphSpec_t = std::tuple<int, int, int>;
using BERTGraphKey_t = std::pair<int, BERTGraphSpec_t>;

bool operator==(const nvinfer1::Dims& d1, const nvinfer1 ::Dims& d2);

class CopyStream
{
public:
    CopyStream()
    {
        unsigned int flags = cudaEventDefault | cudaEventDisableTiming;
        CHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&h2d, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2h, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&infer, flags), cudaSuccess);
    }

    ~CopyStream()
    {
        CHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(h2d), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2h), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(infer), cudaSuccess);
    }

    void recordH2D()
    {
        CHECK_EQ(cudaEventRecord(h2d, s), cudaSuccess);
    }

    void recordD2H()
    {
        CHECK_EQ(cudaEventRecord(d2h, s), cudaSuccess);
    }

    void recordInferDone(cudaStream_t inferenceStream)
    {
        CHECK_EQ(cudaEventRecord(infer, inferenceStream), cudaSuccess);
    }

    void makeAwaitH2D(cudaStream_t inferenceStream)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferenceStream, h2d, 0), cudaSuccess);
    }

    void awaitInfer()
    {
        CHECK_EQ(cudaStreamWaitEvent(s, infer, 0), cudaSuccess);
    }

    void syncH2D()
    {
        CHECK_EQ(cudaEventSynchronize(h2d), cudaSuccess);
    }

    void syncD2H()
    {
        CHECK_EQ(cudaEventSynchronize(d2h), cudaSuccess);
    }

    cudaStream_t get() const
    {
        return s;
    }

    // private:
    cudaStream_t s;
    cudaEvent_t h2d;
    cudaEvent_t d2h;
    cudaEvent_t infer;
};

template <typename T>
class BERTManagedBuffer
{
public:
    BERTManagedBuffer(size_t sz)
        : mSize(sz)
        , mBytes(sizeof(T) * sz)
        , mDeviceBuffer(mBytes)
        , mHostBuffer(mBytes)
        , mHostPtr(static_cast<T*>(mHostBuffer.data()))
        , mDevicePtr(static_cast<T*>(mDeviceBuffer.data()))
    {
        CHECK_EQ(mHostPtr != nullptr, true);
        CHECK_EQ(mDevicePtr != nullptr, true);
        CHECK_EQ(mSize > 0, true);
        CHECK_EQ(mBytes > 0, true);
    }
    // memset device buffer
    void memsetD(int value)
    {
        CHECK_EQ(cudaMemset(mDeviceBuffer.data(), value, mBytes), cudaSuccess);
    }

    // transfer between pair of managed buffers
    void H2DAsync(size_t effectiveSize, cudaStream_t stream)
    {
        CHECK_EQ(cudaMemcpyAsync(mDeviceBuffer.data(), mHostBuffer.data(), effectiveSize * sizeof(T),
                     cudaMemcpyHostToDevice, stream),
            cudaSuccess);
    }

    void D2HAsync(size_t effectiveSize, cudaStream_t stream)
    {
        CHECK_EQ(cudaMemcpyAsync(mHostBuffer.data(), mDeviceBuffer.data(), effectiveSize * sizeof(T),
                     cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
    }

    // stage one sequence of length seqLen at batchIdx in the page-locked host memory
    void H2H(void* ptr, const size_t offset, const size_t elems)
    {
        memcpy(mHostPtr + offset, ptr, sizeof(T) * elems);
    }

    // // scan through the InputId pointed by ptr up to length, find [SEP] and reconstruct SegmentId accordingly
    // void CreateSegmentIdsH(void* ptr, std::size_t const offset, std::size_t const length)
    // {
    //     auto input_sample = static_cast<T*>(ptr) + offset;
    //     CHECK_EQ(*input_sample, 101) << "Input Token IDs Should start with [CLS]";
    //     int32_t seg = 0;
    //     for (std::size_t i = 0; i < length; ++i)
    //     {
    //         *(mHostPtr + offset + i) = seg;
    //         if (*(input_sample + i) == 102)
    //         {
    //             seg++;
    //         }
    //     }
    //     CHECK_EQ(seg, 1) << "Expecting segment Ids ending as 1";
    // }

    T* const HostData()
    {
        return mHostPtr;
    }
    T* const DeviceData()
    {
        return mDevicePtr;
    }

private:
    size_t mSize;
    size_t mBytes;
    lwis_lon::DeviceBuffer mDeviceBuffer;
    lwis_lon::HostBuffer mHostBuffer;
    T* const mHostPtr;
    T* const mDevicePtr;
};

// This is a data structure used to track all the accumulated sequence lengths appeared
// in each batch that has been or will be processed.
// A pointer will point to the `threshold`th percentile of all accumulated sequence lengths, so
// this threshold sequence length can be used as a reference to drop requests that are outside of
// this length in a coming batch.
class TotalLengthMultiSet
{
public:
    TotalLengthMultiSet(double threshold)
        : mThreshold(threshold)
        , mMinSampleCount(1000)
    {
        // default threshold of total length is INT_MAX
        mItrPos = 1;
        mSet.insert(INT_MAX);
        mThresholdItr = mSet.begin();
    }

    void InsertTasks(const BERTTask_t& tasks)
    {
        std::unique_lock<std::mutex> lck(mMtx);
        // insert each accumulated sequence length into the multiset
        std::size_t totalLength = 0;
        for (auto& t : tasks)
        {
            auto& [task, time] = t;
            auto& [sample, length] = task;
            totalLength += length;
            Insert(totalLength);
        }
    }

    inline int GetThresholdLength()
    {
        std::unique_lock<std::mutex> lck(mMtx);
        // return INT_MAX if samples are not enough
        if (mSet.size() < mMinSampleCount)
            return INT_MAX;
        return *mThresholdItr;
    }

private:
    void Insert(int totalLength)
    {
        if (totalLength <= *mThresholdItr)
        {
            mSet.insert(mThresholdItr, totalLength);
            ++mItrPos;
        }
        else
        {
            mSet.insert(totalLength);
        }
        // move the iterator to the correct position
        while (mItrPos <= mSet.size() * mThreshold)
        {
            ++mThresholdItr;
            ++mItrPos;
        }
        while (mItrPos - 1 > mSet.size() * mThreshold)
        {
            --mThresholdItr;
            --mItrPos;
        }
    }

private:
    double mThreshold;
    size_t mMinSampleCount;
    size_t mItrPos;
    std::multiset<int> mSet;
    std::multiset<int>::iterator mThresholdItr;

    // mutex to make the APIs thread-safe
    std::mutex mMtx;
};

}; // namespace bert_lon

#endif // __BERT_LON_COMMON_H__