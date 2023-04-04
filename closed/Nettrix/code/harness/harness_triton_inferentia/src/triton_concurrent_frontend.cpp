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

#include "triton_concurrent_frontend.hpp"
#include "loadgen.h"
#include "qsl_cpu.hpp"

namespace triton_frontend
{

size_t Concurrent_Frontend_SUT::GetBatchOneByteSize(std::vector<std::string>& output_names)
{
    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    size_t max_batch1_byte_size = 0;

    for (const auto& output : m_Config.output())
    {
        int batch1_byte_size = 1;
        for (const auto& dim : output.dims())
        {
            // FIXME: hard-coded value for variable dims
            if (dim == -1)
            {
                batch1_byte_size *= BERT_MAX_SEQ_LENGTH;
            }
            else
            {
                batch1_byte_size *= dim;
            }
        }
        batch1_byte_size *= GetDataTypeByteSize(output.data_type());
        if (batch1_byte_size <= 0)
        {
            FAIL("can't preallocate memory for variable size data type");
        }
        max_batch1_byte_size = std::max(max_batch1_byte_size, (size_t) batch1_byte_size);
        output_names.emplace_back(output.name());
    }

    return max_batch1_byte_size;
}

void Concurrent_Frontend_SUT::Init(size_t min_sample_size, size_t max_sample_size, size_t buffer_manager_thread_count,
    bool batch_triton_requests, bool check_contiguity, const std::string& numa_config_str,
    const std::string& backend_path)
{
    Triton_Server_SUT::Init(min_sample_size, max_sample_size, buffer_manager_thread_count, batch_triton_requests,
        check_contiguity, numa_config_str, backend_path);

    std::vector<std::string> output_names;
    m_MaxBatchSize = m_Config.max_batch_size() == 0 ? 1 : m_Config.max_batch_size();

    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    size_t max_batch1_byte_size = GetBatchOneByteSize(output_names);

    int max_batchn_byte_size = max_batch1_byte_size * m_MaxBatchSize;

    // # buffer = 2 * # instance * (max_batch_size // min_sample_size + 1),
    // since each instance may ask for two sets of buffers in advance
    // in extreme case for each output
    size_t pool_item_count_per_instance = 8 * (2) * output_names.size();
    size_t pool_item_size = max_batch1_byte_size * max_sample_size;

    // auto numa_config = parseNumaConfig(numa_config_str);
    size_t instance_count = 0;
    for (const auto& instance_group : m_Config.instance_group())
    {
        instance_count += instance_group.count();
    }

    if (!m_BatchTritonRequests)
    {
        m_OutputBufferPool.reset(new PinnedMemoryPool(
            pool_item_count_per_instance * instance_count, pool_item_size, TRITONSERVER_MEMORY_CPU));
    }
    LOG(INFO) << "Allocated Pinned memory pool of count " << pool_item_count_per_instance << " size:" << pool_item_size
              << " bytes for every instance : " << instance_count;

    // For batched requests, we need 2 sets of output buffers, one of them is
    // used
    // for the
    // batched requests and the other for single-sample requests. The size and
    // numbers of these
    // buffers are different.
    if (m_BatchTritonRequests)
    {
        size_t pool_item_size_batch = max_batchn_byte_size;
        size_t pool_item_count_batch = 8 * instance_count * output_names.size();
        m_BatchedOutputBufferPool.reset(
            new PinnedMemoryPool(pool_item_count_batch, pool_item_size_batch, TRITONSERVER_MEMORY_CPU));
        LOG(INFO) << "Allocated Pinned memory pool of count: " << pool_item_count_batch
                  << " size :" << pool_item_size_batch << " bytes for batched requests.";
    }

    std::vector<InputMetaData> inputs;
    m_IsDynamic = false;
    for (const auto& io : m_Config.input())
    {
        InputMetaData input;
        std::get<0>(input) = io.name();
        std::get<1>(input) = DataTypeToTriton(io.data_type());
        auto& shape = std::get<2>(input);
        if (m_Config.max_batch_size() != 0)
        {
            shape.push_back(m_BatchTritonRequests ? m_MaxBatchSize : 1);
        }
        for (const auto& dim : io.dims())
        {
            m_IsDynamic |= (dim == -1);
            shape.push_back(dim);
        }
        m_InputTensors.emplace_back(input);
        inputs.emplace_back(std::move(input));
    }

    /*  Create the allocator that will be used to allocate buffers for
        the result tensors. */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&m_Allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
        "creating response allocator");

    // Pre-allocate a growable request pool for inference requests
    RequestPool::Create(m_RequestCount /* initial_element_count */, m_Server.get(), this, m_ModelName, m_ModelVersion,
        inputs, output_names);

    // Prepare padding buffer in the case of DLRM. The model assumes
    // even batch size but some sample has odd batch size
    if (m_UseDlrmQsl)
    {
        for (const auto& io : m_Config.output())
        {
            int64_t batch1_byte_size = TRITONSERVER_DataTypeByteSize(DataTypeToTriton(io.data_type()));
            for (const auto& dim : io.dims())
            {
                batch1_byte_size *= dim;
            }
            m_OutputPaddingSize = (size_t) batch1_byte_size;
        }
    }

    /* Set the number of warmup responses to 0 to prepare for next warmup */
    m_NumWarmupResponses = 0;

    InitBatcherThreads();
    InitIssueThreads();
}

void Concurrent_Frontend_SUT::InitBatcherThreads()
{
    LOG(INFO) << "Creating " << m_NumBatcherThreads << " batching threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        LOG(INFO) << "Creating " << m_NumBatcherThreads << " batch batching threads " << std::endl;
        for (int i = 0; i < m_NumBatcherThreads; i++)
            m_BatcherThreads.emplace_back(std::thread(&Concurrent_Frontend_SUT::ProcessBatch, this));
    }
    else
    {
        LOG(INFO) << "Creating " << m_NumBatcherThreads << " single sample batching threads " << std::endl;
        for (int i = 0; i < m_NumBatcherThreads; i++)
            m_BatcherThreads.emplace_back(std::thread(&Concurrent_Frontend_SUT::ProcessSample, this));
    }
}

void Concurrent_Frontend_SUT::InitIssueThreads()
{
    LOG(INFO) << "Creating " << m_NumIssueThreads << " issue threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        for (int i = 0; i < m_NumIssueThreads; i++)
            m_IssueThreads.emplace_back(std::thread(&Concurrent_Frontend_SUT::IssueTritonBatchRequest, this));
    }
    else
    {
        for (int i = 0; i < m_NumIssueThreads; i++)
            m_IssueThreads.emplace_back(std::thread(&Concurrent_Frontend_SUT::IssueTritonRequest, this));
    }
}

void Concurrent_Frontend_SUT::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    LOG(INFO) << "Starting Triton warmup for concurrent harness" << std::endl;
    auto num_inferences
        = static_cast<int>((duration_sec * expected_qps) / (m_BatchTritonRequests ? m_MaxBatchSize : 1));

    /* Keep track of the number of inferences that we have sent so far */
    int inferences_sent = 0;

    // Load a sample to RAM to use
    mlperf::QuerySampleIndex index{0}; // Arbitrary sample index
    std::vector<mlperf::QuerySampleIndex> samples;
    samples.push_back(index);
    for (auto& qsl : m_SampleLibraries)
    {
        qsl->LoadSamplesToRam(samples);
    }

    while (inferences_sent < num_inferences)
    {
        IssueTritonRequestHelper(index, 0, nullptr, true);
        inferences_sent += 1;
    }
    while (m_NumWarmupResponses < inferences_sent)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    /* Unload sample from RAM */
    for (auto& qsl : m_SampleLibraries)
    {
        qsl->UnloadSamplesFromRam(samples);
    }

    /* Reset the number of warmup responses */
    m_NumWarmupResponses = 0;

    /* Notify user that we are done with the warmup */
    LOG(INFO) << "Finished Triton warmup" << std::endl;
}

void Concurrent_Frontend_SUT::IssueTritonRequestHelper(mlperf::QuerySampleIndex index, mlperf::ResponseId responseID,
    TRITONSERVER_InferenceTrace* m_TracePtr, bool isWarmup)
{
    /* Create the inference request provider, which provides the request
         header information as well as the actual data. */
    auto request_block = RequestPool::Obtain(0);
    if (!isWarmup)
    {
        request_block->m_ResponseMetadata.m_ResponseId = responseID;
        request_block->m_ResponseMetadata.m_QuerySampleIdx = index;
        request_block->m_ResponseMetadata.m_TracePtr = m_TracePtr;
    }
    request_block->m_ResponseMetadata.m_ServerPtr = this;
    request_block->m_BatchResponseMetadata.m_ServerPtr = this;

    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(index, idx); // Get address of the query
        size_t input_size = m_SampleLibraries[0]->GetSampleSize(idx);

        FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                        request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
            "removing input data");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
            "appending input data");
    }

    /* Actually perform inference (asynchronously) */
    // For the userp field, pass in a pointer to a tuple with a pointer to the
    // SUT, a pointer to
    // the request provider, and the LoadGen response ID
    /* Set response callback for this request */
    if (isWarmup)
    {
        if (!m_BatchTritonRequests)
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                            request_block->m_Data, m_Allocator, m_OutputBufferPool.get(), WarmupResponseComplete, this),
                "appending input data");
        }
        else
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                            m_BatchedOutputBufferPool.get(), WarmupResponseComplete, this),
                "appending input data");
        }
    }
    else
    {
        /* Set response callback for this request */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                        m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
            "appending input data");
    }

    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, m_TracePtr), "running inference");

#if TRITON_FRONTEND_TRACE
    TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
}

void Concurrent_Frontend_SUT::Completion(
    TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata)
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

    /* Call QuerySamplesComplete */
    mlperf::QuerySampleResponse loadgen_response{
        response_metadata->m_ResponseId, output0_result, output0_byte_size - response_metadata->m_PaddingSize};
    // callback if it exists
    if (m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> response_indices = {response_metadata->m_QuerySampleIdx};
        m_ResponseCallback(&loadgen_response, response_indices, 1);
    }

    mlperf::QuerySamplesComplete(&loadgen_response,
        1); // We always send one inference response at a time
}

void Concurrent_Frontend_SUT::BatchCompletion(
    TRITONSERVER_InferenceResponse* response, const BatchResponseMetaData* response_metadata)
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
    // std::cout << shape[0] << std::endl;
    size_t batch1_output_size = output0_byte_size / response_metadata->m_RequestBatchSize;

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

void Concurrent_Frontend_SUT::Done()
{
    // Wait for all real requests to be completed
    while (!m_WorkQueue.empty())
    {
    }

    LOG(INFO) << "Async server SUT Done 1" << std::endl;
    for (int i = 0; i < m_NumBatcherThreads; i++)
    {
        // Add a dummy request for each batcher thread
        m_WorkQueue.emplace_back(mlperf::QuerySample{0, 0});
        // wait for dummy request to be consumed before adding another request
        while (!m_WorkQueue.empty())
        {
        }
    }

    for (auto& thread : m_BatcherThreads)
        thread.join();

    for (int i = 0; i < m_NumIssueThreads; i++)
    {
        auto req = std::make_shared<RequestInfo>();
        req->batchSize = 0;
        m_IssueQueue.emplace_back(req);
    }

    for (auto& thread : m_IssueThreads)
        thread.join();

    RequestPool::Destroy();

    /* Delete the response allocator since we are done with it */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(m_Allocator), "deleting response allocator");

    /* Reset the server pointer to nullptr to ensure Init() is called before the
     * server is used
     * again */
    m_Server = nullptr;
}

void Concurrent_Frontend_SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    if (m_IsSingleStream)
    {
        TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
        if (m_TraceManager != nullptr)
        {
            trace = m_TraceManager->SampleTrace();
            TraceCaptureTimeStamp(trace, "MLPerf Request START");
        }
#endif // TRITON_FRONTEND_TRACE
        IssueTritonRequestHelper(samples[0].index, samples[0].id, trace, false);
    }
    else
    {
        m_WorkQueue.insert(samples);
    }
}

void Concurrent_Frontend_SUT::FlushQueries() {}

void Concurrent_Frontend_SUT::ProcessSample() // Process single query at a
                                              // time
{
    while (true)
    {
        mlperf::QuerySample sample = m_WorkQueue.front_then_pop();

        // sample.index == 0 and sample.id == 0 denoted dummy sample
        if (sample.index == 0 && sample.id == 0)
        {
            break;
        }
        auto req = std::make_shared<RequestInfo>();
        req->batchSize = 1;
        req->samples.emplace_back(std::move(sample));
        req->isContiguous = true;
        m_IssueQueue.emplace_back(req);
    }
}

void Concurrent_Frontend_SUT::ProcessBatch()
{
    // Batch queries and process them
    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        do
        {
            m_WorkQueue.acquire(samples, workQueueTimeout, m_MaxBatchSize, true);
        } while (samples.empty());

        // Use a null (0) id to represent the end of samples
        if (samples.front().id == 0)
        {
            break;
        }

        auto req = std::make_shared<RequestInfo>();
        req->batchSize = samples.size();
        m_BatchStats[samples.size()] += 1;
        req->isContiguous = m_CheckContiguity ? CheckContiguity(samples) : false;
        req->samples = std::move(samples);
        m_IssueQueue.emplace_back(req);
        // std::cout << "Processes batch of size " << req->batchSize << std::endl;
    }
}

void Concurrent_Frontend_SUT::IssueTritonRequest()
{
    while (true)
    {
        auto req = m_IssueQueue.front_then_pop();

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

        IssueTritonRequestHelper(req->samples.front().index, req->samples.front().id, trace, false);
    }
}

bool Concurrent_Frontend_SUT::CheckContiguity(std::deque<mlperf::QuerySample>& samples)
{
    bool contiguous = true;

    for (size_t i = 0; i < m_InputTensors.size() && contiguous; i++)
    {
        auto prev = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress(samples.front().index, i));

        auto sample_size = m_SampleLibraries[0]->GetSampleSize(i);
        auto iter = samples.begin();
        iter++;
        for (auto j = iter; j < samples.end(); j++)
        {
            auto next = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress((*j).index, i));

            if (next != prev + sample_size)
            {
                contiguous = false;
                break;
            }
            prev = next;
        }
    }
    return contiguous;
}

void Concurrent_Frontend_SUT::IssueTritonBatchRequest()
{
    while (true)
    {
        auto req = m_IssueQueue.front_then_pop();

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

        auto request_block = RequestPool::Obtain(0);
        request_block->m_BatchResponseMetadata.m_TracePtr = trace;
        request_block->m_BatchResponseMetadata.m_ResponseId.clear();
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;

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
                m_BatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
            "appending input data");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");

#if TRITON_FRONTEND_TRACE
        TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif // TRITON_FRONTEND_TRACE
    }
}
} // namespace triton_frontend
