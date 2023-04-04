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

#ifndef MLPINF_TRITON_SUT_HPP
#define MLPINF_TRITON_SUT_HPP

// QSL
#include "qsl.hpp"

// LoadGen
#include "system_under_test.h"

// General C++
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <thread>

// NUMA utils
#include "pinned_memory_pool.hpp"
#include "utils.hpp"

#include "model_config.pb.h"

#include "triton_device.hpp"

namespace triton_frontend
{

using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;
using ResponseCallback = std::function<void(::mlperf::QuerySampleResponse* responses,
    std::vector<::mlperf::QuerySampleIndex>& sampleIds, size_t responseCount)>;

class ITritonSUT : public mlperf::SystemUnderTest
{
public:
    ITritonSUT(ITritonDevice* const device)
        : m_Device(device)
    {
    }
    ~ITritonSUT() {}

    // MLPerf SUT virtual interface
    /*!
     * Called by the loadgen to retrieve human-readable string for logging purposes
     */
    virtual const std::string& Name()
    {
        return m_Device->Name();
    }

    /*!
     * Called by the loadgen to issue samples to the SUT
     * Performs inference asynchronously
     */
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples) = 0;

    /*!
     * Called by the loadgen immediately after the last call to IssueQuery in a series is made
     * No-op
     */
    virtual void FlushQueries() {}

    // Helper Functions
    /*!
     * Initializes Triton server to be ready for warmup
     */
    virtual void Init(
        size_t bufferManagerThreadCount = 0, const std::string& numaConfigStr = "", const std::string& backendPath = "")
        = 0;

    /*!
     * Waits for all requests to be completed and cleans up threads/memory
     */
    virtual void Done()
    {
        m_Device->Done();
    }

    // // Triton helper functions
    // virtual void ModelMetadata();
    // virtual void ModelStats();
    // virtual void TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr,
    //                                    const std::string comment);

    // MLPerf side helper functions
    virtual void IncrementWarmupResponses()
    {
        m_Device->IncrementWarmupResponses();
    }

    // Wrapper around mlperf::QuerySamplesComplete
    // Creates callback for D2H output copy if m_EndOnDevice is true
    // virtual void QuerySamplesComplete(mlperf::QuerySampleResponse* responses, size_t
    // num_responses,
    //                                   int64_t pool_idx);
    virtual void Warmup(double duration_sec, double expected_qps) = 0;

protected:
    ITritonDevice* m_Device;
    ResponseCallback m_ResponseCallback;
};

} // namespace triton_frontend

#endif // MLPINF_TRITON_SUT_HPP
