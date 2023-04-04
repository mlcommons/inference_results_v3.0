

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

#include "dla_concurrent_frontend.hpp"
#include "dla_triton_frontend_helpers.hpp"
#include "loadgen.h"

namespace triton_frontend
{

void DLA_Triton_SUT::InitDlaBatcherThreads()
{
    LOG(INFO) << "Creating " << m_DlaNumBatcherThreads << " DLA batching threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        for (int i = 0; i < m_DlaNumBatcherThreads; i++)
            m_DlaBatcherThreads.emplace_back(std::thread(&DLA_Triton_SUT::ProcessDlaBatch, this));
    }
    else
    {
        for (int i = 0; i < m_DlaNumBatcherThreads; i++)
            m_DlaBatcherThreads.emplace_back(std::thread(&DLA_Triton_SUT::ProcessDlaSample, this));
    }
}

void DLA_Triton_SUT::InitDlaIssueThreads()
{
    LOG(INFO) << "Creating " << m_DlaNumIssueThreads << " DLA issue threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        for (int i = 0; i < m_DlaNumIssueThreads; i++)
            m_DlaIssueThreads.emplace_back(std::thread(&DLA_Triton_SUT::IssueTritonDlaBatchRequest, this));
    }
    else
    {
        for (int i = 0; i < m_DlaNumIssueThreads; i++)
            m_DlaIssueThreads.emplace_back(std::thread(&DLA_Triton_SUT::IssueTritonDlaRequest, this));
    }
}

void DLA_Triton_SUT::Init(size_t min_sample_size, size_t max_sample_size, size_t buffer_manager_thread_count,
    bool batch_triton_requests, bool check_contiguity, const std::string& numa_config_str,
    const std::string& backend_path)
{
    Concurrent_Frontend_SUT::Init(min_sample_size, max_sample_size, buffer_manager_thread_count, batch_triton_requests,
        check_contiguity, numa_config_str, backend_path);

    // Get the DLA model config
    TRITONSERVER_Message* model_config_message = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerLoadModel(m_Server.get(), m_DlaModelName.c_str()), "Failed to load DLA model");

    size_t health_iters = 0;
    while (true)
    {
        bool live, ready, model_ready;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_Server.get(), &live), "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_Server.get(), &ready), "unable to get server readiness");
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelIsReady(m_Server.get(), m_DlaModelName.c_str(), m_ModelVersion, &model_ready),
            "unable to get model readiness");
        std::cout << "Server Health status: live " << live << ", ready " << ready << ", model ready " << model_ready
                  << std::endl;
        if (live && ready && model_ready)
        {
            std::cout << "Server is live and ready. Model is ready" << std::endl;
            break;
        }

        if (++health_iters >= 200)
        {
            FAIL("failed to find healthy inference server within 200 tries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(m_Server.get(), m_DlaModelName.c_str(), m_DlaModelVersion,
                    1 /* config_version */, &model_config_message),
        "unable to get DLa model config message");
    auto lcm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_config_message, TRITONSERVER_MessageDelete);

    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_config_message, &buffer, &byte_size),
        "unable to serialize model metadata message to JSON");

    ::google::protobuf::util::JsonStringToMessage({buffer, (int) byte_size}, &m_DlaConfig);
    std::cout << "DLA Model Config:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;

    // TODO: Create DLA output pool. For now use gpu output pool for DLA as well.
    // TODO: Create separate DLA request pool. For now use gpu request pool.
    std::vector<InputMetaData> inputs;
    m_IsDynamic = false;
    for (const auto& io : m_DlaConfig.input())
    {
        InputMetaData input;
        std::get<0>(input) = io.name();
        std::get<1>(input) = DataTypeToTriton(io.data_type());
        auto& shape = std::get<2>(input);
        if (m_DlaConfig.max_batch_size() != 0)
        {
            shape.push_back(m_BatchTritonRequests ? m_DlaBatchSize : 1);
        }
        for (const auto& dim : io.dims())
        {
            m_IsDynamic |= (dim == -1);
            shape.push_back(dim);
        }
        inputs.emplace_back(std::move(input));
    }

    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    auto output_names = getOutputNames(m_DlaConfig);
    size_t max_batch1_byte_size = GetBatchOneByteSize(m_DlaConfig, output_names);

    std::map<size_t, size_t> dla_instance_count;
    for (const auto& instance_group : m_DlaConfig.instance_group())
    {
        for (const auto& gpu : instance_group.gpus())
        {
            dla_instance_count[gpu] += instance_group.count();
        }
    }

    int max_batchn_byte_size = max_batch1_byte_size * m_DlaBatchSize;
    auto numa_config = parseNumaConfig(numa_config_str);

    if (m_BatchTritonRequests)
    {
        size_t pool_item_size_batch = max_batchn_byte_size;
        // FIXME Madhu - need a different count for server and offline
        size_t pool_item_count_batch = 8 * output_names.size();
        m_DlaBatchedOutputBufferPool.reset(
            new PinnedMemoryPoolEnsemble(pool_item_count_batch, pool_item_size_batch, dla_instance_count, numa_config));
        LOG(INFO) << "Allocated Pinned memory pool of count: " << pool_item_count_batch
                  << " size :" << pool_item_size_batch << " bytes for DLA batched requests.";
    }
    else
    {
        size_t pool_item_size = max_batch1_byte_size;
        // FIXME Madhu - need a different count for server and offline
        size_t pool_item_count = 8 * m_DlaBatchSize * output_names.size();
        m_DlaOutputBufferPool.reset(
            new PinnedMemoryPoolEnsemble(pool_item_count, pool_item_size, dla_instance_count, numa_config));
        LOG(INFO) << "Allocated Pinned memory pool of count: " << pool_item_count << " size :" << pool_item_size
                  << " bytes for DLA requests.";
    }

    DLARequestPool::Create(m_RequestCount /* initial_element_count */, m_Server.get(), this, m_DlaModelName,
        m_DlaModelVersion, inputs, output_names);

    InitDlaBatcherThreads();
    InitDlaIssueThreads();
}

void DLA_Triton_SUT::ProcessDlaSample() // Process single query at a
                                        // time
{
    while (true)
    {
        mlperf::QuerySample sample = m_DlaWorkQueue.front_then_pop();

        // sample.index == 0 and sample.id == 0 denoted dummy sample
        if (sample.index == 0 && sample.id == 0)
        {
            break;
        }
        auto req = std::make_shared<RequestInfo>();
        req->batchSize = 1;
        req->samples.emplace_back(std::move(sample));
        req->isContiguous = true;
        m_DlaIssueQueue.emplace_back(req);
    }
}

void DLA_Triton_SUT::IssueTritonDlaRequest()
{
    while (true)
    {
        auto req = m_DlaIssueQueue.front_then_pop();

        // req.batchSize== 0 denotes done()
        if (req->batchSize == 0)
        {
            break;
        }

        TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
        if (m_TraceManager != nullptr)
        {
            trace = m_TraceManager->SampleTrace();
            TraceCaptureTimeStamp(trace, "MLPerf Request START");
        }
#endif // TRITON_FRONTEND_TRACE

        auto request_block = DLARequestPool::Obtain(0);
        request_block->m_ResponseMetadata.m_ResponseId = req->samples.front().id;
        request_block->m_ResponseMetadata.m_QuerySampleIdx = req->samples.front().index;
        request_block->m_ResponseMetadata.m_TracePtr = trace;
        request_block->m_ResponseMetadata.m_ServerPtr = this;
        request_block->m_BatchResponseMetadata.m_ServerPtr = this;

        for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
        {
            // Get a pointer to the input data
            int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(
                req->samples.front().index, idx); // Get address of the query
            size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx);

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                            request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                "removing input data");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                "appending input data");

            AppendInputDataForAllGPUs(request_block->m_Data, req->samples.front().index,
                std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
            AppendInputDataForAllQSLs(request_block->m_Data, req->samples.front().index,
                std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
        }

        /* Actually perform inference (asynchronously) */
        // For the userp field, pass in a pointer to a tuple with a pointer to the
        // SUT, a pointer to
        // the request provider, and the LoadGen response ID
        /* Set response callback for this request */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                        m_DlaOutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
            "appending input data");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
        TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
    }
}

void DLA_Triton_SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    // Only offline case
    if (samples.size() > m_DlaBatchSize + m_MaxBatchSize)
    {
        m_DlaWorkQueue.insert(samples, 0, samples.size() * m_DlaRequestRatio);
        m_WorkQueue.insert(samples, samples.size() * m_DlaRequestRatio, samples.size());
    }
    else
    {
        m_WorkQueue.insert(samples);
    }
}

void DLA_Triton_SUT::ModelStats()
{
    // Print gathered batch stats
    LOG(INFO) << "DLA Batch stats" << std::endl;
    for (auto i : m_DlaBatchStats)
    {
        LOG(INFO) << "Batch size: " << i.first << " ,Count: " << i.second << std::endl;
    }
    LOG(INFO) << "GPU Batch stats" << std::endl;
    for (auto i : m_BatchStats)
    {
        LOG(INFO) << "Batch size: " << i.first << " ,Count: " << i.second << std::endl;
    }

    Triton_Server_SUT::ModelStats();
    TRITONSERVER_Message* model_stats_message = nullptr;
    auto err = TRITONSERVER_ServerModelStatistics(
        m_Server.get(), m_DlaModelName.c_str(), m_DlaModelVersion, &model_stats_message);
    if (err != nullptr)
    {
        std::cerr << "failed to obtain stats message: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    auto lms = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_stats_message, TRITONSERVER_MessageDelete);
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(model_stats_message, &buffer, &byte_size), "serializing stats message");

    std::cout << "DLA Model Stats:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
}

void DLA_Triton_SUT::Done()
{
    while (!m_DlaWorkQueue.empty())
    {
    }

    for (int i = 0; i < m_DlaNumBatcherThreads; i++)
    {
        // Add a dummy request for each batcher thread
        m_DlaWorkQueue.emplace_back(mlperf::QuerySample{0, 0});
        // wait for dummy request to be consumed before adding another request
        while (!m_DlaWorkQueue.empty())
        {
        }
    }

    for (auto& thread : m_DlaBatcherThreads)
        thread.join();

    for (int i = 0; i < m_DlaNumIssueThreads; i++)
    {
        auto req = std::make_shared<RequestInfo>();
        req->batchSize = 0;
        m_DlaIssueQueue.emplace_back(req);
    }

    for (auto& thread : m_DlaIssueThreads)
        thread.join();

    DLARequestPool::Destroy();

    Concurrent_Frontend_SUT::Done();
}

void DLA_Triton_SUT::ProcessDlaBatch() // Batch queries according to dla_batch_size and process them
{
    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        do
        {
            m_DlaWorkQueue.acquire(samples, workQueueTimeout, m_DlaBatchSize, true);
        } while (samples.empty());

        // Use a null (0) id to represent the end of samples
        if (samples.front().id == 0)
        {
            break;
        }

        auto req = std::make_shared<RequestInfo>();
        req->batchSize = samples.size();
        m_DlaBatchStats[samples.size()] += 1;
        req->isContiguous = m_CheckContiguity ? CheckContiguity(samples) : false;
        req->samples = std::move(samples);
        m_DlaIssueQueue.emplace_back(req);
    }
}

void DLA_Triton_SUT::IssueTritonDlaBatchRequest()
{
    while (true)
    {
        auto req = m_DlaIssueQueue.front_then_pop();

        // req.batchSize== 0 denotes done()
        if (req->batchSize == 0)
        {
            break;
        }

        TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
        if (m_TraceManager != nullptr)
        {
            trace = m_TraceManager->SampleTrace();
            TraceCaptureTimeStamp(trace, "MLPerf Request START");
        }
#endif // TRITON_FRONTEND_TRACE

        auto request_block = DLARequestPool::Obtain(0);
        request_block->m_BatchResponseMetadata.m_TracePtr = trace;
        request_block->m_BatchResponseMetadata.m_ResponseId.clear();
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_DlaBatchSize;

        for (auto sample : req->samples)
        {
            request_block->m_BatchResponseMetadata.m_ResponseId.emplace_back(sample.id);
            request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.emplace_back(sample.index);
        }
        if (req->isContiguous)
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(
                    req->samples.front().index, idx); // Get address of the query
                size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx) * req->samples.size();

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                    "removing input data");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                    "appending input data");

                AppendInputDataForAllGPUs(request_block->m_Data, req->samples.front().index,
                    std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
                AppendInputDataForAllQSLs(request_block->m_Data, req->samples.front().index,
                    std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
            }
        }
        else
        {
            for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                    "removing input data");
                for (auto sample : req->samples)
                {
                    size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx);
                    // Get a pointer to the input data
                    int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(
                        sample.index, idx); // Get address of the query

                    FAIL_IF_ERR(
                        TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                            std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
                        "appending input data");

                    AppendInputDataForAllGPUs(
                        request_block->m_Data, sample.index, std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
                    AppendInputDataForAllQSLs(
                        request_block->m_Data, sample.index, std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
                }
            }
        }
        /* Actually perform inference (asynchronously) */
        // For the userp field, pass in a pointer to a tuple with a pointer to the
        // SUT, a pointer to
        // the request provider, and the LoadGen response ID
        /* Set response callback for this request */
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                m_DlaBatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
            "appending input data");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
        TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
    }
}
} // namespace triton_frontend
