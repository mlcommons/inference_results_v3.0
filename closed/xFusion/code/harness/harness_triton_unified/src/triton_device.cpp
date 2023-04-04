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

#include "triton_device.hpp"
#include "triton_request_pool.hpp"
#include "triton_workload.hpp"
#include <google/protobuf/util/json_util.h>

namespace triton_frontend
{

void ITritonDevice::Init(
    size_t bufferManagerThreadCount, const std::string& numaConfigStr, const std::string& backendPath)
{
    /* Create the options for the server */
    TRITONSERVER_ServerOptions* serverOptions = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&serverOptions), "creating server options");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(
                    serverOptions, TRITONSERVER_ModelControlMode::TRITONSERVER_MODEL_CONTROL_EXPLICIT),
        "Setting model control mode");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(serverOptions, m_ModelRepositoryPath.c_str()),
        "setting model repository path");

    // FIXME currently don't need to pass in valid directory as TensorRT backend
    // has not yet decoupled from Triton core, will need to fix once it is
    // separated as dynamically loaded library
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendDirectory(serverOptions, backendPath.c_str()), "setting backend directory");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(serverOptions, 1UL << 29),
        "setting pinned memory pool size");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(serverOptions, bufferManagerThreadCount),
        "setting buffer manager thread count");

    auto numaConfig = parseNumaConfig(numaConfigStr);
    for (size_t nodeIdx = 0; nodeIdx < numaConfig.size(); ++nodeIdx)
    {
        auto cpuIds = numaConfig[nodeIdx].second;
        std::sort(std::begin(cpuIds), std::end(cpuIds));
        std::string cpuStr = GenerateCPUStr(cpuIds);

        const auto& gpuIds = numaConfig[nodeIdx].first;
        for (const auto gpuId : gpuIds)
        {
            std::string policyName = "gpu_" + std::to_string(gpuId);
            FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetHostPolicy(
                            serverOptions, policyName.c_str(), "numa-node", std::to_string(nodeIdx).c_str()),
                "setting NUMA node for host policy");
            FAIL_IF_ERR(
                TRITONSERVER_ServerOptionsSetHostPolicy(serverOptions, policyName.c_str(), "cpu-cores", cpuStr.c_str()),
                "setting CPU cores for host policy");
        }
    }
    // Uncomment this to get detailed verbose logs from triton
    // FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 1),
    // "Setting verbose level 1");

    /* Actually create the server now */
    TRITONSERVER_Server* server = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerNew(&server, serverOptions), "creating server");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(serverOptions), "deleting server options");
    m_Server
        = std::unique_ptr<TRITONSERVER_Server, decltype(&TRITONSERVER_ServerDelete)>(server, TRITONSERVER_ServerDelete);

    FAIL_IF_ERR(TRITONSERVER_ServerLoadModel(m_Server.get(), m_ModelName.c_str()), "Loading model");
    /* Wait until the server is both live and ready, and the model is ready. */
    size_t healthIters = 0;
    while (true)
    {
        bool live;
        bool ready;
        bool modelReady;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_Server.get(), &live), "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_Server.get(), &ready), "unable to get server readiness");
        FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &modelReady),
            "unable to get model readiness");
        LOG(INFO) << "Server Health status: live " << live << ", ready " << ready << ", model ready " << modelReady
                  << std::endl;
        if (live && ready && modelReady)
        {
            LOG(INFO) << "Server is live and ready. Model is ready" << std::endl;
            break;
        }

        if (++healthIters >= 200)
        {
            FAIL("failed to find healthy inference server within 200 tries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Initalize enough pinned output buffer in front
    // We want to have instance # sets of output buffers, each set has
    // output # buffers, and each buffer has max batch bytes for the output
    TRITONSERVER_Message* modelConfigMessage = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(
                    m_Server.get(), m_ModelName.c_str(), m_ModelVersion, 1 /* config_version */, &modelConfigMessage),
        "unable to get model config message");
    auto lcm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        modelConfigMessage, TRITONSERVER_MessageDelete);

    const char* buffer;
    size_t byteSize;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(modelConfigMessage, &buffer, &byteSize),
        "unable to serialize model metadata message to JSON");

    ::google::protobuf::util::JsonStringToMessage({buffer, (int) byteSize}, &m_Config);
    LOG(INFO) << "Model Config:" << std::endl;
    LOG(INFO) << std::string(buffer, byteSize) << std::endl;

    /* Set the number of warmup responses to 0 to prepare for next warmup */
    m_NumWarmupResponses = 0;

    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    std::vector<std::string> outputNames;
    std::transform(std::begin(m_Config.output()), std::end(m_Config.output()), std::back_inserter(outputNames),
        [](auto a) { return a.name(); });
    const int32_t maxBatch1ByteSize = m_Workload->GetOutputMaxBytes(m_Config);
    const int32_t inferenceResponseBatchSize
        = m_HarnessConfig.m_BatchTritonRequests ? m_HarnessConfig.m_MaxBatchSize : 1;

    std::map<size_t, size_t> gpuInstanceCount;
    for (const auto& instanceGroup : m_Config.instance_group())
    {
        for (const auto& gpu : instanceGroup.gpus())
        {
            gpuInstanceCount[gpu] += instanceGroup.count();
        }
    }

    // Set the host policy names that maps to each GPU device,
    // here uses the default host policy name Triton would generate for
    // model instances on the corresponding device.
    for (size_t idx = 0; idx < gpuInstanceCount.size(); ++idx)
    {
        m_HostPolicyNames.emplace_back("gpu_" + std::to_string(idx));
    }
    m_HarnessConfig.m_NumGPUs = gpuInstanceCount.size();

    // FIXME from Madhu - need a different count for server and offline
    const int32_t nbBufsPerPool = m_HarnessConfig.m_BatchTritonRequests
        ? 8 * outputNames.size()
        : 2 * (m_HarnessConfig.m_MaxBatchSize + 1) * outputNames.size();
    const int32_t nbBytesPerBuf = inferenceResponseBatchSize * maxBatch1ByteSize;
    m_OutputBufferPool.reset(new PinnedMemoryPoolEnsemble(nbBufsPerPool, nbBytesPerBuf, gpuInstanceCount, numaConfig));
    LOG(INFO) << "Allocated Pinned memory pool of count: " << nbBufsPerPool << " size :" << nbBytesPerBuf
              << " bytes for batched requests.";

    for (const auto& io : m_Config.input())
    {
        InputMetaData input;
        auto& [name, dataType, shape] = input;
        name = io.name();
        dataType = DataTypeToTriton(io.data_type());
        if (m_Config.max_batch_size() != 0)
        {
            shape.push_back(m_HarnessConfig.m_BatchTritonRequests ? m_HarnessConfig.m_MaxBatchSize : 1);
        }
        for (const auto& dim : io.dims())
        {
            shape.push_back(dim);
        }
        m_InputTensors.emplace_back(input);
    }

    /*  Create the allocator that will be used to allocate buffers for
        the result tensors. */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&m_Allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
        "creating response allocator");

    // Pre-allocate a growable request pool for inference requests
    RequestPool::Create(m_RequestCount, m_Server.get(), this, m_ModelName, m_ModelVersion, m_InputTensors, outputNames);

    LOG(INFO) << "Creating " << m_NumBatcherThreads << " batcher threads " << std::endl;
    for (size_t i = 0; i < m_NumBatcherThreads; i++)
    {
        m_BatcherThreads.emplace_back(&ITritonDevice::EnqueueBatchForInference, this);
    }

    LOG(INFO) << "Creating " << m_NumInferThreads << " inference threads " << std::endl;

    for (size_t i = 0; i < m_NumInferThreads; i++)
    {
        m_InferThreads.emplace_back(&ITritonDevice::IssueTritonRequest, this);
    }
}

void ITritonDevice::EnqueueBatchForInference()
{
    // Batch queries and process them
    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        do
        {
            m_ReadyToBatch.acquire(samples, m_Workload->GetBatcherTimeout(), m_HarnessConfig.m_MaxBatchSize, true);
        } while (samples.empty());

        // Use a null (0) id to represent the end of samples
        if (samples.front().id == 0)
        {
            break;
        }

        RequestInfo req;
        m_BatchStats[samples.size()] += 1;
        req.isContiguous = m_Workload->CheckContiguity(samples);
        req.samples = std::move(samples);
        m_ReadyToInfer.emplace_back(req);
    }
}

void ITritonDevice::IssueTritonRequest()
{
    while (true)
    {
        RequestInfo req = m_ReadyToInfer.front_then_pop();

        // req.batchSize== 0 denotes done()
        if (req.samples.size() == 0)
        {
            break;
        }

        IssueTritonRequestInternal(req.samples, req.isContiguous, nullptr, ResponseComplete);
    }
}

template <typename T>
void ITritonDevice::IssueTritonRequestInternal(const T& samples, bool isContiguous,
    TRITONSERVER_InferenceTrace* tracePtr, TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFunc)
{
    auto requestBlock = RequestPool::Obtain(0);
    requestBlock->m_ResponseMetadata.m_TracePtr = tracePtr;
    requestBlock->m_ResponseMetadata.m_ResponseId.clear();
    requestBlock->m_ResponseMetadata.m_QuerySampleIdxList.clear();
    requestBlock->m_ResponseMetadata.m_RequestBatchSize = samples.size();
    requestBlock->m_ResponseMetadata.m_DevPtr = this;

    for (const auto& sample : samples)
    {
        requestBlock->m_ResponseMetadata.m_ResponseId.emplace_back(sample.id);
        requestBlock->m_ResponseMetadata.m_QuerySampleIdxList.emplace_back(sample.index);
    }

    const auto& qsls = m_Workload->GetQSLs();

    auto appendInputData
        = [this, &qsls, &requestBlock](size_t inputIdx, size_t numSamples, const mlperf::QuerySampleIndex& sampleIdx) {
              const int32_t inputBatch1Size
                  = m_Workload->UpdateInputShape(requestBlock, inputIdx, m_InputTensors[inputIdx], sampleIdx);
              const int32_t bufSize = inputBatch1Size * numSamples;
              const int8_t* inputData = (int8_t*) qsls[0]->GetSampleAddress(sampleIdx, inputIdx);
              const auto& name = std::get<0>(m_InputTensors[inputIdx]);

              FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                              requestBlock->m_Data, name.c_str(), inputData, bufSize, m_InputMemoryType, 0),
                  "appending input data");
              AppendInputDataWithHostPolicy(
                  requestBlock->m_Data, sampleIdx, m_Workload->GetQSLs(), name.c_str(), bufSize, inputIdx);
          };

    auto begin = std::cbegin(samples);
    auto end = isContiguous ? begin + 1 : std::cend(samples);
    size_t numSamples = isContiguous ? samples.size() : 1;

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                        requestBlock->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
            "removing input data");

        for (auto b = begin; b < end; b++)
        {
            appendInputData(idx, numSamples, begin->index);
        }
    }

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(requestBlock->m_Data, m_Allocator,
                    m_OutputBufferPool.get(), responseCompleteFunc, &requestBlock->m_ResponseMetadata),
        "appending input data");

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), requestBlock->m_Data, tracePtr), "running inference");
}

template void ITritonDevice::IssueTritonRequestInternal<std::vector<mlperf::QuerySample>>(
    const std::vector<mlperf::QuerySample>& samples, bool isContiguous, TRITONSERVER_InferenceTrace* tracePtr,
    TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFunc);

template void ITritonDevice::IssueTritonRequestInternal<std::deque<mlperf::QuerySample>>(
    const std::deque<mlperf::QuerySample>& samples, bool isContiguous, TRITONSERVER_InferenceTrace* tracePtr,
    TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFunc);

template void ITritonDevice::IssueTritonRequestInternal<std::array<mlperf::QuerySample, 1>>(
    const std::array<mlperf::QuerySample, 1>& samples, bool isContiguous, TRITONSERVER_InferenceTrace* tracePtr,
    TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFunc);

void ITritonDevice::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    LOG(INFO) << "Starting Triton warmup for concurrent harness" << std::endl;
    auto batchSize = m_HarnessConfig.m_BatchTritonRequests ? m_HarnessConfig.m_MaxBatchSize : 1;
    auto num_inferences = static_cast<int>((duration_sec * expected_qps) / batchSize);

    const auto& qsls = m_Workload->GetQSLs();
    const int32_t dummySample = 0;
    qsls[0]->LoadSamplesToRam({dummySample});

    RequestInfo req;
    req.samples = std::deque<mlperf::QuerySample>(batchSize, mlperf::QuerySample{dummySample, 0});
    req.isContiguous = false;
    for (int32_t i = 0; i < num_inferences; i++)
    {
        IssueTritonRequestInternal(req.samples, false, nullptr, WarmupResponseComplete);
    }
    while (m_NumWarmupResponses < num_inferences)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    qsls[0]->UnloadSamplesFromRam({dummySample});
    m_NumWarmupResponses = 0;

    LOG(INFO) << "Finished Triton warmup" << std::endl;
}

void ITritonDevice::Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Extract and process the response data */
    const char* name;
    TRITONSERVER_DataType datatype;
    void* userp;
    const int64_t* shape;
    uint64_t dim_count;
    const void* output0_content;
    size_t output0_byte_size;
    TRITONSERVER_MemoryType output0_memory_type;
    int64_t output0_memory_type_id;
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(response, 0 /* index */, &name, &datatype, &shape, &dim_count,
                    &output0_content, &output0_byte_size, &output0_memory_type, &output0_memory_type_id, &userp),
        "getting output0 result");

    // Recast the output pointer as a uintptr_t (for LoadGen)
    uintptr_t output0_result = reinterpret_cast<uintptr_t>(output0_content);

    // Construct response list from Inference response
    std::vector<mlperf::QuerySampleResponse> loadgen_responses;
    // output buffers used by Triton are allocated for max batch size
    // the actual response may contain fewer than max batch elements
    // (e.g. for last inference)
    int32_t responseBatchSize = m_HarnessConfig.m_BatchTritonRequests ? m_HarnessConfig.m_MaxBatchSize : 1;
    int32_t batch1_output_size = output0_byte_size / responseBatchSize;

    auto buffer_ptr = static_cast<const int8_t*>(output0_content);

    for (int i = 0; i < (response_metadata->m_ResponseId).size(); i++)
    {
        loadgen_responses.emplace_back(
            mlperf::QuerySampleResponse{(response_metadata->m_ResponseId)[i], output0_result, batch1_output_size});
        const void* buffer_ptr_inc = buffer_ptr + (batch1_output_size * (i + 1));
        output0_result = reinterpret_cast<uintptr_t>(buffer_ptr_inc);
    }

    // callback if it exists
    if (m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> sample_indices = response_metadata->m_QuerySampleIdxList;
        m_ResponseCallback(&loadgen_responses[0], sample_indices, (response_metadata->m_QuerySampleIdxList).size());
    }

    mlperf::QuerySamplesComplete(&loadgen_responses[0], response_metadata->m_ResponseId.size());
}

void ITritonDevice::Done()
{
    // Wait for all real requests to be completed
    while (!m_ReadyToBatch.empty())
    {
    }

    LOG(INFO) << "Async server SUT Done 1" << std::endl;
    for (int i = 0; i < m_NumBatcherThreads; i++)
    {
        // Add a dummy request for each batcher thread
        m_ReadyToBatch.emplace_back(mlperf::QuerySample{0, 0});
        // wait for dummy request to be consumed before adding another request
        while (!m_ReadyToBatch.empty())
        {
        }
    }

    for (auto& thread : m_BatcherThreads)
    {
        thread.join();
    }

    for (int i = 0; i < m_NumInferThreads; i++)
    {
        RequestInfo req;
        m_ReadyToInfer.emplace_back(req);
    }

    for (auto& thread : m_InferThreads)
    {
        thread.join();
    }

    RequestPool::Destroy();

    /* Delete the response allocator since we are done with it */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(m_Allocator), "deleting response allocator");
}

} // namespace triton_frontend
