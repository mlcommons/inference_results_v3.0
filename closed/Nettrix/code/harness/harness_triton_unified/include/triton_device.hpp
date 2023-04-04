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

#ifndef MLPINF_TRITON_DEVICE_HPP
#define MLPINF_TRITON_DEVICE_HPP

#include "lwis.hpp"
#include "triton_helpers.hpp"
#include "triton_workload.hpp"

// TRITON
#include "triton/core/tritonserver.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace triton_frontend
{

class ITritonSUT; // forward declaration because SUT stores Device, but Device gets passed SUT in
                  // Init()

using ResponseCallback = std::function<void(::mlperf::QuerySampleResponse* responses,
    std::vector<::mlperf::QuerySampleIndex>& sampleIds, size_t responseCount)>;

class ITritonDevice
{
public:
    ITritonDevice(const std::string& name, const std::string& modelRepoPath, const std::string& modelName,
        uint32_t modelVersion, uint64_t requestCount, size_t numBatcherThreads, size_t numInferThreads,
        ITritonWorkload* const workload, HarnessConfigs harnessConfig)
        : m_Name(name)
        , m_Server(nullptr, TRITONSERVER_ServerDelete)
        , m_ModelRepositoryPath(modelRepoPath)
        , m_ModelName(modelName)
        , m_ModelVersion(modelVersion)
        , m_RequestCount(requestCount)
        , m_NumBatcherThreads(numBatcherThreads)
        , m_NumInferThreads(numInferThreads)
        , m_Workload(workload)
        , m_HarnessConfig(harnessConfig)
    {
        // Set input memory type accordingly, with start_from_device as highest priority
        m_InputMemoryType = m_HarnessConfig.m_PinnedInputs ? TRITONSERVER_MEMORY_CPU_PINNED : TRITONSERVER_MEMORY_CPU;
        m_InputMemoryType = m_HarnessConfig.m_StartFromDevice ? TRITONSERVER_MEMORY_GPU : m_InputMemoryType;
    }

    virtual ~ITritonDevice() = default;

    // MLPerf side helper functions
    // FIXME: This doesn't inherit mlperf::system_under_test::Name() right?
    virtual const std::string& Name()
    {
        return m_Name;
    }

    virtual void IncrementWarmupResponses()
    {
        m_NumWarmupResponses += 1;
    }

    virtual void Done();

    virtual void Init(
        size_t bufferManagerThreadCount, const std::string& numaConfigStr, const std::string& backendPath);

    virtual void Warmup(double durationSec, double expectedQps);

    virtual void Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* responseMetadata);

    virtual void SetResponseCallback(const ResponseCallback& responseCallback)
    {
        m_ResponseCallback = responseCallback;
    }

    virtual void AppendInputDataWithHostPolicy(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sampleIdx, const std::vector<qsl::SampleLibraryPtr_t>& qsls,
        const char* inputName, size_t inputSize, size_t inputIdx)
        = 0;

    void IssueTritonRequest();

    void EnqueueBatchForInference();

    template <typename T>
    void IssueTritonRequestInternal(const T& samples, bool isContiguous, TRITONSERVER_InferenceTrace* tracePtr,
        TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFunc);

public:
    ITritonWorkload* m_Workload;

    size_t m_NumBatcherThreads;
    lwis::SyncQueue<mlperf::QuerySample> m_ReadyToBatch;
    std::vector<std::thread> m_BatcherThreads;

    size_t m_NumInferThreads;
    lwis::SyncQueue<RequestInfo> m_ReadyToInfer;
    std::vector<std::thread> m_InferThreads;

    // Triton Server-side Information
    const std::string m_ModelRepositoryPath;
    const std::string m_Name;
    std::unique_ptr<TRITONSERVER_Server, decltype(&TRITONSERVER_ServerDelete)> m_Server;
    std::string m_ModelName;
    const uint32_t m_ModelVersion;

    inference::ModelConfig m_Config;
    HarnessConfigs m_HarnessConfig;

    std::atomic<uint64_t> m_NumWarmupResponses{0};
    std::shared_ptr<triton::server::TraceManager> m_TraceManager; // TODO: share with TritonSUT

    uint64_t m_RequestCount;

    // Perform end on device D2H copy on separate stream
    ScopedCudaStream m_EndOnDeviceCopyStream;

    // Host policy names that associates to the GPU devices
    std::vector<std::string> m_HostPolicyNames;

    // Pinned memory pool for output buffers:
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_OutputBufferPool;

    TRITONSERVER_MemoryType m_InputMemoryType;
    std::vector<InputMetaData> m_InputTensors;
    TRITONSERVER_ResponseAllocator* m_Allocator{nullptr};

    std::map<int, int> m_BatchStats;

    ResponseCallback m_ResponseCallback;
};

} // namespace triton_frontend

#endif // MLPINF_TRITON_DEVICE_HPP