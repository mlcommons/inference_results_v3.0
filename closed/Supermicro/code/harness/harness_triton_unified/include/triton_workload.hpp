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

#ifndef MLPINF_TRITON_WORKLOAD_HPP
#define MLPINF_TRITON_WORKLOAD_HPP

#include "lwis.hpp"
#include "pinned_memory_pool.hpp"
#include "qsl.hpp"
#include "triton_callbacks.hpp"
#include "triton_helpers.hpp"
#include "triton_request_pool.hpp"

#include "model_config.pb.h"

#include <memory>
#include <thread>
#include <vector>

namespace triton_frontend
{

class ITritonSUT;

using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;

struct HarnessConfigs
{
    bool m_BatchTritonRequests;
    size_t m_MaxBatchSize;
    bool m_CheckContiguity;
    bool m_StartFromDevice;
    bool m_EndOnDevice;
    size_t m_NumGPUs;
    bool m_PinnedInputs;

    HarnessConfigs(bool batchTritonRequests, size_t maxBatchSize, bool checkContiguity, bool startFromDevice,
        bool endOnDevice, size_t numGPUs, bool pinnedInputs)
        : m_BatchTritonRequests(batchTritonRequests)
        , m_MaxBatchSize(maxBatchSize)
        , m_CheckContiguity(checkContiguity)
        , m_StartFromDevice(startFromDevice)
        , m_EndOnDevice(endOnDevice)
        , m_NumGPUs(numGPUs)
        , m_PinnedInputs(pinnedInputs)
    {
    }
};

class ITritonWorkload
{
public:
    ITritonWorkload(const HarnessConfigs& harnessConfig)
        : m_HarnessConfig(harnessConfig)
    {
    }

    virtual std::shared_ptr<mlperf::QuerySampleLibrary> InitQSL(const std::string& mapPath,
        const std::vector<std::string>& tensorPaths, uint64_t performanceSampleCount, size_t padding,
        bool coalescedTensor, bool startFromDevice, const std::string& numaConfigStr = "");

    const std::vector<qsl::SampleLibraryPtr_t>& GetQSLs() const
    {
        return m_SampleLibraries;
    }

    virtual int32_t GetOutputMaxBytes(const inference::ModelConfig& config) const;

    /*!
     * Returns true if all the samples in the deque are contiguous
     */
    virtual bool CheckContiguity(const std::deque<mlperf::QuerySample>& samples) const;

    virtual std::chrono::microseconds GetBatcherTimeout() const
    {
        return m_BatcherTimeout;
    }

    virtual void IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device);

    virtual void IssueQueryServer(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device);

    // Update input shape and return input size in bytes
    virtual int32_t UpdateInputShape(RequestPool::Block* requestBlock, int32_t inputIdx,
        const InputMetaData& inputMetadata, mlperf::QuerySampleIndex sampleIdx)
    {
        return m_SampleLibraries[0]->GetSampleSize(inputIdx);
    }

protected:
    HarnessConfigs m_HarnessConfig;
    std::vector<qsl::SampleLibraryPtr_t> m_SampleLibraries;
    std::chrono::microseconds m_BatcherTimeout{1000};
};

struct BertConfig
{
    int32_t maxSeqLen;
};

class TritonWorkloadBERT : public ITritonWorkload
{
public:
    TritonWorkloadBERT(const HarnessConfigs& harnessConfig, const BertConfig& bertConfig)
        : ITritonWorkload(harnessConfig)
        , m_BertConfig(bertConfig)
    {
    }

    virtual int32_t GetOutputMaxBytes(const inference::ModelConfig& config) const override;

    int32_t GetSeqLen(const mlperf::QuerySampleIndex idx)
    {
        // Get seq length by checking where the input_mask change from 1 to 0
        const auto maxSeqLen = m_BertConfig.maxSeqLen;
        const auto inputMaskPtr = static_cast<int32_t*>(m_SampleLibraries[0]->GetSampleAddress(idx, 2));
        const auto first0It = std::lower_bound(inputMaskPtr, inputMaskPtr + maxSeqLen, 0,
            [](const auto a, const auto b) { return a > b; }); // returns first element equal to 0
        return std::distance(inputMaskPtr, first0It);
    }

    virtual int32_t UpdateInputShape(RequestPool::Block* requestBlock, int32_t inputIdx,
        const InputMetaData& inputMetadata, mlperf::QuerySampleIndex sampleIdx) override
    {
        const auto& [name, dtype, _] = inputMetadata;
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestRemoveInput(requestBlock->m_Data, name.c_str()), "failed to remove input");
        const auto seqLen = GetSeqLen(sampleIdx);
        std::array<int64_t, 2> shape{1, seqLen};
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                        requestBlock->m_Data, name.c_str(), dtype, shape.data(), shape.size()),
            "re-adding input");

        return seqLen * TRITONSERVER_DataTypeByteSize(dtype);
    }

    virtual void IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device);

public:
    BertConfig m_BertConfig;
};

} // namespace triton_frontend

#endif // MLPINF_TRITON_WORKLOAD_HPP