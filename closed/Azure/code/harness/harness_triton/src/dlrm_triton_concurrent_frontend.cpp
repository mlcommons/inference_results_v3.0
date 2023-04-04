
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

#include "dlrm_triton_concurrent_frontend.hpp"
#include "loadgen.h"

namespace triton_frontend
{

void DLRM_Triton_SUT::Init(size_t min_sample_size, size_t max_sample_size, size_t buffer_manager_thread_count,
    bool batch_triton_requests, bool check_contiguity, const std::string& numa_config_str,
    const std::string& backend_path)
{
    Concurrent_Frontend_SUT::Init(min_sample_size, max_sample_size, buffer_manager_thread_count, batch_triton_requests,
        check_contiguity, numa_config_str, backend_path);
    m_MinSampleSize = min_sample_size;
    m_MaxSampleSize = max_sample_size;

    // Compute output padding size to keep even number of pairs
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

void DLRM_Triton_SUT::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    LOG(INFO) << "Starting Triton warmup" << std::endl;

    /* Calculate the number of inferences to send
       An "inference" can either be a single sample or a batch, depending on BatchTritonRequests.
       We should scale our num_inferences appropriately.
    */
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
        /* Create the inference request provider, which provides the request
            header information as well as the actual data. */
        auto request_block = RequestPool::Obtain(0);
        // Inputs will need to be re-added as the shape is different from run
        // to run
        TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
        // new batch size for the request
        auto num_pairs = qsl->GetNumUserItemPairs(index);

        // Set default input buffer
        for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
        {
            // Get a pointer to the input data
            int8_t* input_data = (int8_t*) qsl->GetSampleAddress(index, idx); // Get address of the query
            size_t single_sample_size = qsl->GetSampleSize(idx);

            auto& [name, type, shape] = m_InputTensors[idx];
            shape[0] = num_pairs;
            if (num_pairs % 2)
            {
                shape[0] += 1;
            }
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, name.c_str(),
                            std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data, name.c_str(), input_data,
                            single_sample_size * num_pairs, m_InputMemoryType, 0),
                "appending input data");

            AppendInputDataForAllGPUs(request_block->m_Data, index, name.c_str(), single_sample_size * num_pairs, idx);
            AppendInputDataForAllQSLs(request_block->m_Data, index, name.c_str(), single_sample_size * num_pairs, idx);

            // Add padding buffer
            if (num_pairs % 2)
            {
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data, name.c_str(),
                                input_data, single_sample_size, m_InputMemoryType, 0),
                    "appending input data padding");
                AppendInputDataForAllGPUs(request_block->m_Data, index, name.c_str(), single_sample_size, idx);
                AppendInputDataForAllQSLs(request_block->m_Data, index, name.c_str(), single_sample_size, idx);
            }
        }

        /* Set response callback for warmup */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(), WarmupResponseComplete, this),
            "appending input data");

        /* Actually perform inferences (asynchronously) */
        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, nullptr), "running inference");
        inferences_sent += 1;
    }

    /* Wait for all the warmup inferences to complete */
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

void DLRM_Triton_SUT::BatchCompletion(
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

    int numQueryResponses = response_metadata->m_ResponseId.size();
    auto buffer_ptr = static_cast<const int8_t*>(output0_content);
    int cumulativeNumPairs = 0;
    for (int i = 0; i < numQueryResponses; i++)
    {
        loadgen_responses.emplace_back(mlperf::QuerySampleResponse{(response_metadata->m_ResponseId)[i], output0_result,
            response_metadata->m_DlrmNumPairsList[i] * TRITONSERVER_DataTypeByteSize(datatype)});
        cumulativeNumPairs += response_metadata->m_DlrmNumPairsList[i];
        const void* buffer_ptr_inc = buffer_ptr + cumulativeNumPairs * TRITONSERVER_DataTypeByteSize(datatype);
        output0_result = reinterpret_cast<uintptr_t>(buffer_ptr_inc);
    }

    mlperf::QuerySamplesComplete(&loadgen_responses[0], response_metadata->m_ResponseId.size());
}

void DLRM_Triton_SUT::ProcessSample() // Process single query at a time
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
        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
        // In the case of DLRM we use the batchsize to denote number of pairs
        req->batchSize = qsl->GetNumUserItemPairs(sample.index);
        req->samples.emplace_back(std::move(sample));
        req->isContiguous = true;
        m_IssueQueue.emplace_back(req);
    }
}

// Trying to capture the heuristic part of batching in the function
size_t DLRM_Triton_SUT::GetNumQuerySamplesToAcquire(size_t num_pairs, int num_successful_tries, int num_empty_tries)
{
    // If samples are contiguous, better to create bigger batches
    if (m_CheckContiguity)
    {
        return (m_MaxBatchSize - num_pairs) / (maxSampleSizeMultiplierOffline * m_MaxSampleSize);
    }
    return (m_MaxBatchSize - num_pairs) / (maxSampleSizeMultiplierServer * m_MaxSampleSize);
}

void DLRM_Triton_SUT::ProcessBatch()
{
    bool endSample = false;
    bool ready = false;

    std::deque<mlperf::QuerySample> remainingSamples;

    // Each iteration of this while loop creates a request
    while (true)
    {
        auto req = std::make_shared<RequestInfo>();
        req->isContiguous = false;

        size_t num_pairs = 0;
        int num_empty_tries = 0;
        int num_successful_tries = 0;
        std::deque<mlperf::QuerySample> samples;

        // Wait till there is some work on the work queue,
        // This is to avoid accessing the qsl till the initialization is done
        while (!ready)
        {
            mlperf::QuerySample peekSample = m_WorkQueue.front();
            ready = (peekSample.index != 0 && peekSample.id != 0);
        }

        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());

        if (remainingSamples.size() > 0)
        {
            req->samples.insert(req->samples.begin(), remainingSamples.begin(), remainingSamples.end());
            num_successful_tries = 1;
        }
        for (const auto& sample : remainingSamples)
        {
            num_pairs += qsl->GetNumUserItemPairs(sample.index);
        }
        remainingSamples.clear();

        do
        {
            // Here we try to acquire as many samples as possible to form a complete batch
            // assuming that each query sample has maxSampleSize
            samples.clear();
            size_t numQuerySamplesToAcquire
                = GetNumQuerySamplesToAcquire(num_pairs, num_successful_tries, num_empty_tries);

            m_WorkQueue.acquire(samples, workQueueTimeout, numQuerySamplesToAcquire, true);
            // calculate total number of num_pairs acquired this time
            if (samples.size() > 0 && samples[0].index == 0 && samples[0].id == 0)
            {
                endSample = true;
                break;
            }

            size_t num_pairs_acquired = std::accumulate(samples.begin(), samples.end(), 0,
                [&](size_t i, const mlperf::QuerySample& s) { return qsl->GetNumUserItemPairs(s.index) + i; });

            if (num_pairs_acquired > 0)
            {
                num_successful_tries++;
                if (num_pairs_acquired + num_pairs < m_MaxBatchSize)
                {
                    num_pairs += num_pairs_acquired;
                    req->samples.insert(req->samples.end(), samples.begin(), samples.end());
                }
                else
                {
                    auto itr = samples.begin();
                    while (num_pairs < m_MaxBatchSize)
                    {
                        auto new_num_pairs = num_pairs + qsl->GetNumUserItemPairs((*itr).index);
                        if (new_num_pairs > m_MaxBatchSize)
                        {
                            remainingSamples.insert(remainingSamples.begin(), itr, samples.end());
                            break;
                        }
                        else
                        {
                            num_pairs = new_num_pairs;
                            req->samples.push_back(*itr);
                            itr++;
                        }
                    }
                }
            }

            if (req->samples.size() > 0 && num_pairs_acquired == 0)
            {
                num_empty_tries++;
            }
        } while ((num_pairs < (m_MaxBatchSize - m_MaxSampleSize) || req->samples.size() == 0)
            && (num_empty_tries <= m_NumEmptyTries && num_successful_tries <= m_NumSuccessfulAcquires));

        if (endSample)
        {
            break;
        }

        req->batchSize = num_pairs;
        if (m_CheckContiguity)
        {
            req->isContiguous = CheckContiguity(req->samples, num_pairs);
        }
        m_IssueQueue.emplace_back(req);
        m_BatchStats[num_pairs] += 1;
    }
}

void DLRM_Triton_SUT::IssueTritonRequest()
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
       // Set the Request Provider
        auto request_block = RequestPool::Obtain(0);

        request_block->m_ResponseMetadata.m_ResponseId = req->samples.front().id;
        request_block->m_ResponseMetadata.m_QuerySampleIdx = req->samples.front().index;
        request_block->m_ResponseMetadata.m_TracePtr = trace;
        // Inputs will need to be re-added as the shape is different from run
        // to run
        TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());
        // new batch size for the request
        auto num_pairs = req->batchSize;
        auto l_InputTensors = m_InputTensors;

        for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
        {
            // Get a pointer to the input data
            int8_t* input_data = (int8_t*) qsl->GetSampleAddress(req->samples.front().index,
                idx); // Get address of the query
            const size_t single_sample_size = qsl->GetSampleSize(idx);

            auto& [name, type, shape] = l_InputTensors[idx];

            shape[0] = num_pairs;
            if (num_pairs % 2)
            {
                shape[0] += 1;
            }
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                            request_block->m_Data, name.c_str(), type, shape.data(), shape.size()),
                "re-adding input");
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data, name.c_str(), input_data,
                            single_sample_size * num_pairs, m_InputMemoryType, 0),
                "appending input data");
            AppendInputDataForAllGPUs(
                request_block->m_Data, req->samples.front().index, name.c_str(), single_sample_size * num_pairs, idx);
            AppendInputDataForAllQSLs(
                request_block->m_Data, req->samples.front().index, name.c_str(), single_sample_size * num_pairs, idx);
            if (num_pairs % 2)
            {
                // Add padding buffer
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size,
                                TRITONSERVER_MEMORY_CPU, 0),
                    "appending input data padding");

                AppendInputDataForAllGPUs(
                    request_block->m_Data, req->samples.front().index, name.c_str(), single_sample_size, idx);
                AppendInputDataForAllQSLs(
                    request_block->m_Data, req->samples.front().index, name.c_str(), single_sample_size, idx);
            }
        }
        request_block->m_ResponseMetadata.m_PaddingSize = (num_pairs % 2) ? m_OutputPaddingSize : 0;

        /* Actually perform inference (asynchronously) */
        // For the userp field, pass in a pointer to a tuple with a pointer to the SUT, a pointer to
        // the request provider, and the LoadGen response ID
        auto buffer_pool = m_OutputBufferPool.get();
        /* Set response callback for this request */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                        m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
            "appending input data");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");
#if TRITON_FRONTEND_TRACE
        TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif //
    }
}

void DLRM_Triton_SUT::Done()
{
    Concurrent_Frontend_SUT::Done();
    // Print batch stats
    for (auto batch : m_BatchStats)
    {
        LOG(INFO) << "Batch Size: " << batch.first << " Batch Count: " << batch.second << std::endl;
    }
}

void DLRM_Triton_SUT::IssueTritonBatchRequest()
{
    while (true)
    {
        auto req = m_IssueQueue.front_then_pop();

        TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRITON_FRONTEND_TRACE
        if (m_TraceManager != nullptr)
        {
            trace = m_TraceManager->SampleTrace();
            TraceCaptureTimeStamp(trace, "MLPerf Request START");
        }
#endif // TRITON_FRONTEND_TRACE

        // req.batchSize== 0 denotes done()
        if (req->batchSize == 0)
        {
            break;
        }
        auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());

        auto request_block = RequestPool::Obtain(0);
        request_block->m_BatchResponseMetadata.m_TracePtr = trace;
        request_block->m_BatchResponseMetadata.m_ResponseId.clear();
        request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.clear();
        request_block->m_BatchResponseMetadata.m_DlrmNumPairsList.clear();
        request_block->m_BatchResponseMetadata.m_RequestBatchSize = m_MaxBatchSize;
        request_block->m_BatchResponseMetadata.m_ServerPtr = this;
        request_block->m_ResponseMetadata.m_ServerPtr = this;

        TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

        for (const auto& sample : req->samples)
        {
            request_block->m_BatchResponseMetadata.m_ResponseId.emplace_back(sample.id);
            request_block->m_BatchResponseMetadata.m_QuerySampleIdxList.emplace_back(sample.index);
            request_block->m_BatchResponseMetadata.m_DlrmNumPairsList.emplace_back(
                qsl->GetNumUserItemPairs(sample.index));
        }

        auto l_InputTensors = m_InputTensors;
        for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
        {
            const size_t single_sample_size = qsl->GetSampleSize(idx);
            auto& shape = std::get<2>(l_InputTensors[idx]);
            shape[0] = req->batchSize;
            if (req->batchSize % 2)
            {
                shape[0] += 1;
            }
            FAIL_IF_ERR(
                TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                "re-adding input");

            if (!req->isContiguous)
            {
                for (const auto& sample : req->samples)
                {
                    // Get a pointer to the input data
                    int8_t* input_data = (int8_t*) qsl->GetSampleAddress(sample.index, idx); // Get address of the query

                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                    std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                    single_sample_size * qsl->GetNumUserItemPairs(sample.index), m_InputMemoryType, 0),
                        "appending input data");

                    AppendInputDataForAllGPUs(request_block->m_Data, sample.index,
                        std::get<0>(m_InputTensors[idx]).c_str(),
                        single_sample_size * qsl->GetNumUserItemPairs(sample.index), idx);
                    AppendInputDataForAllQSLs(request_block->m_Data, sample.index,
                        std::get<0>(m_InputTensors[idx]).c_str(),
                        single_sample_size * qsl->GetNumUserItemPairs(sample.index), idx);
                }
            }
            else
            {
                int8_t* input_data
                    = (int8_t*) qsl->GetSampleAddress(req->samples[0].index, idx); // Get address of the query

                const size_t single_sample_size = qsl->GetSampleSize(idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                                std::get<0>(m_InputTensors[idx]).c_str(), input_data,
                                single_sample_size * req->batchSize, m_InputMemoryType, 0),
                    "appending input data");
                AppendInputDataForAllGPUs(request_block->m_Data, req->samples[0].index,
                    std::get<0>(m_InputTensors[idx]).c_str(), single_sample_size * req->batchSize, idx);
                AppendInputDataForAllQSLs(request_block->m_Data, req->samples[0].index,
                    std::get<0>(m_InputTensors[idx]).c_str(), single_sample_size * req->batchSize, idx);
            }
            if (req->batchSize % 2)
            {
                int8_t* input_data
                    = (int8_t*) qsl->GetSampleAddress(req->samples[0].index, idx); // Get address of the query

                FAIL_IF_ERR(
                    TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, single_sample_size, m_InputMemoryType, 0),
                    "Appending padding input");

                AppendInputDataForAllGPUs(request_block->m_Data, req->samples[0].index,
                    std::get<0>(m_InputTensors[idx]).c_str(), single_sample_size, idx);
                AppendInputDataForAllQSLs(request_block->m_Data, req->samples[0].index,
                    std::get<0>(m_InputTensors[idx]).c_str(), single_sample_size, idx);
            }
        }
        request_block->m_BatchResponseMetadata.m_PaddingSize = (req->batchSize % 2) ? m_OutputPaddingSize : 0;
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                m_BatchedOutputBufferPool.get(), BatchResponseComplete, &request_block->m_BatchResponseMetadata),
            "Setting response callback");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace), "running inference");
    }
}
void DLRM_Triton_SUT::InitBatcherThreads()
{
    LOG(INFO) << "Creating DLRM " << m_NumBatcherThreads << " batching threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        for (int i = 0; i < m_NumBatcherThreads; i++)
            m_BatcherThreads.emplace_back(std::thread(&DLRM_Triton_SUT::ProcessBatch, this));
    }
    else
    {
        for (int i = 0; i < m_NumBatcherThreads; i++)
            m_BatcherThreads.emplace_back(std::thread(&DLRM_Triton_SUT::ProcessSample, this));
    }
}

void DLRM_Triton_SUT::InitIssueThreads()
{
    LOG(INFO) << "Creating DLRM " << m_NumIssueThreads << " issue threads " << std::endl;
    if (m_BatchTritonRequests)
    {
        for (int i = 0; i < m_NumIssueThreads; i++)
            m_IssueThreads.emplace_back(std::thread(&DLRM_Triton_SUT::IssueTritonBatchRequest, this));
    }
    else
    {
        for (int i = 0; i < m_NumIssueThreads; i++)
            m_IssueThreads.emplace_back(std::thread(&DLRM_Triton_SUT::IssueTritonRequest, this));
    }
}
bool DLRM_Triton_SUT::CheckContiguity(const std::deque<mlperf::QuerySample>& samples, int num_pairs)
{
    auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());

    for (size_t i = 0; i < m_InputTensors.size(); i++)
    {
        int8_t* first_task_start = static_cast<int8_t*>(qsl->GetSampleAddress(samples[0].index, i));

        int8_t* last_task_start = static_cast<int8_t*>(qsl->GetSampleAddress(samples[samples.size() - 1].index, i));

        auto num_pairs_last_task = qsl->GetNumUserItemPairs(samples[samples.size() - 1].index);
        auto single_sample_size = qsl->GetSampleSize(i);

        // new batch size for the request
        auto sample_size = single_sample_size * num_pairs;

        if (first_task_start + sample_size != last_task_start + (num_pairs_last_task * single_sample_size))
        {
            return false;
        }
    }
    return true;
}

void DLRM_Triton_SUT::AppendInputDataForAllGPUs(TRITONSERVER_InferenceRequest* request,
    const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx)
{
    if (m_StartFromDevice && m_NumGPUs > 1)
    {
        for (int i = 0; i < m_NumGPUs; i++)
        {
            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[0].get());

            int8_t* input_data = (int8_t*) qsl->GetSampleAddress(sample_idx, input_idx, 0,
                i); // Get address of the query for device

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request, input_name, input_data,
                            input_size, m_InputMemoryType, i, m_HostPolicyNames[i].c_str()),
                "appending input data with host policy");
        }
    }
}

void DLRM_Triton_SUT::AppendInputDataForAllQSLs(TRITONSERVER_InferenceRequest* request,
    const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx)
{
    // If there are more than one QSL, we want to add input buffer with device affinity,
    // use the proper host policy name recognized by Triton.
    if (m_SampleLibraries.size() > 1)
    {
        for (size_t qsl_idx = 0; qsl_idx < m_SampleLibraries.size(); ++qsl_idx)
        {
            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibraries[qsl_idx].get());

            // Get a pointer to the input data
            int8_t* input_data = (int8_t*) qsl->GetSampleAddress(sample_idx, input_idx,
                0); // Get address of the query

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request, input_name, input_data,
                            input_size, m_InputMemoryType, 0, m_HostPolicyNames[qsl_idx].c_str()),
                "appending input data");
        }
    }
}

} // namespace triton_frontend
