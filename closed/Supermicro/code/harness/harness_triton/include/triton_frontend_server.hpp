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

#ifndef __TRITON_FRONTEND_SERVER_HPP__
#define __TRITON_FRONTEND_SERVER_HPP__

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
#include "triton_frontend_helpers.hpp"
#include "utils.hpp"

#include "model_config.pb.h"

#define TRITON_FRONTEND_TRACE 0
static_assert(TRITON_FRONTEND_TRACE == 0, "MLPINF-1690: Triton trace functionality broken");

namespace triton_frontend
{

using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;

class Triton_Server_SUT : public mlperf::SystemUnderTest
{
public:
    // Constructors and destructors
    Triton_Server_SUT(const std::string& name, const std::string& model_repo_path, const std::string& model_name,
        uint32_t model_version, bool use_dlrm_qsl, bool start_from_device, bool end_on_device, bool pinned_input,
        uint64_t request_count)
        : m_Name(name)
        , m_ModelRepositoryPath(model_repo_path)
        , m_ModelName(model_name)
        , m_ModelVersion(model_version)
        , m_UseDlrmQsl(use_dlrm_qsl)
        , m_StartFromDevice(start_from_device)
        , m_EndOnDevice(end_on_device)
        , m_RequestCount(request_count)
    {
        // Set input memory type accordingly, with start_from_device as highest priority
        m_InputMemoryType = pinned_input ? TRITONSERVER_MEMORY_CPU_PINNED : TRITONSERVER_MEMORY_CPU;
        m_InputMemoryType = start_from_device ? TRITONSERVER_MEMORY_GPU : m_InputMemoryType;
    }
    ~Triton_Server_SUT() {}

    // MLPerf SUT virtual interface
    virtual const std::string& Name()
    {
        return m_Name;
    }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples) = 0;
    virtual void FlushQueries() = 0;
    virtual void Done() = 0;

    // Init and warmup functions
    virtual void Init(size_t min_sample_size = 1, size_t max_sample_size = 1, size_t buffer_manager_thread_count = 0,
        bool batch_triton_requests = false, bool check_contiguity = false, const std::string& numa_config_str = "",
        const std::string& backend_path = "");
    virtual void Warmup(double duration_sec, double expected_qps) = 0;
    virtual void IncrementWarmupResponses()
    {
        m_NumWarmupResponses += 1;
    }

    // Triton helper functions
    virtual void ModelMetadata();
    virtual void ModelStats();
    virtual void TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr, const std::string comment);

    virtual void Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata) = 0;
    virtual void BatchCompletion(
        TRITONSERVER_InferenceResponse* response, const BatchResponseMetaData* response_metadata)
        = 0;

    // MLPerf side helper functions
    virtual void AddSampleLibrary(const qsl::SampleLibraryPtr_t& sl)
    {
        m_SampleLibraries.emplace_back(sl);
    }
    virtual void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
            std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
            callback)
    {
        m_ResponseCallback = callback;
    }

    // Wrapper around mlperf::QuerySamplesComplete
    // Creates callback for D2H output copy if m_EndOnDevice is true
    virtual void QuerySamplesComplete(mlperf::QuerySampleResponse* responses, size_t num_responses, int64_t pool_idx);

protected:
    // Triton Server-side Information
    const std::string m_ModelRepositoryPath;
    const std::string m_Name;
    std::shared_ptr<TRITONSERVER_Server> m_Server = nullptr;
    TRITONSERVER_ResponseAllocator* m_Allocator = nullptr;
    std::string m_ModelName;
    const uint32_t m_ModelVersion;

    inference::ModelConfig m_Config;

    TRITONSERVER_MemoryType m_InputMemoryType;
    std::atomic<uint64_t> m_NumWarmupResponses{0};
    std::shared_ptr<triton::server::TraceManager> m_TraceManager;

    // Host policy names that associates to the GPU devices
    std::vector<std::string> m_HostPolicyNames;
    // Query sample response callback.
    std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex>& sample_ids,
        size_t response_count)>
        m_ResponseCallback;

    // Harness settings
    bool m_IsDynamic = false;
    std::vector<InputMetaData> m_InputTensors;
    size_t m_OutputPaddingSize;
    bool m_UseDlrmQsl = false;
    size_t m_MaxBatchSize;
    bool m_BatchTritonRequests = false;
    bool m_CheckContiguity = false;
    bool m_StartFromDevice = false;
    bool m_EndOnDevice = false;
    size_t m_NumGPUs;

    uint64_t m_RequestCount;

    // Sample libraries
    std::vector<qsl::SampleLibraryPtr_t> m_SampleLibraries;

    // Pinned memory pool for output buffers:
    // we take advantage of the fact that the model has only one output and
    // the batch 1 size is fixed.
    // # buffer = 2 * # instance * (max_batch_size // min_sample_size + 1),
    // since each instance may ask for two sets of buffers in advance
    // in extreme case.
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_OutputBufferPool;

    // Pinned memory pool for output buffers for batched requests
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_BatchedOutputBufferPool;

    // In end on device mode, m_OutputBufferPool and m_BatchedOutputBufferPool are
    // device buffers, so separate host buffers are needed to store the results of the
    // D2H copy
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_OutputBufferPoolEndOnDevice;
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_BatchedOutputBufferPoolEndOnDevice;
    // Perform end on device D2H copy on separate stream
    ScopedCudaStream m_EndOnDeviceCopyStream;
};

typedef std::shared_ptr<Triton_Server_SUT> ServerSUTPtr_t;

} // namespace triton_frontend

#endif
