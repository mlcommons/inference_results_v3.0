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

#include "bert_concurrent_frontend.hpp"
#include "loadgen.h"

namespace triton_frontend
{

void BERT_Triton_SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    if (samples.size() > 1)
    {
        std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
        for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
        {
            sequenceSamplePosAndLength[samplePos] = std::make_pair(
                samplePos, static_cast<int>(GetSampleLength(m_SampleLibraries[0], samples[samplePos].index)));
        }
        // Sort samples in the descending order of sentence length
        // Plugin kernels are specialized for specific max sequence lengths (64, 128, 256 etc),
        // sorting samples improves the efficiency of running these kernels because neighbouring
        // samples will have the same max_seq_lengths
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });

        for (auto samplePair : sequenceSamplePosAndLength)
        {
            m_WorkQueue.insert({samples[samplePair.first]});
        }
    }
    else if (m_IsSingleStream)
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

void BERT_Triton_SUT::IssueTritonRequestHelper(
    mlperf::QuerySampleIndex index, mlperf::ResponseId responseID, TRITONSERVER_InferenceTrace* trace, bool isWarmup)
{
    /* Create the inference request provider, which provides the request
          header information as well as the actual data. */
    auto request_block = RequestPool::Obtain(0);

    if (!isWarmup)
    {
        request_block->m_ResponseMetadata.m_ResponseId = responseID;
        request_block->m_ResponseMetadata.m_QuerySampleIdx = index;
        request_block->m_ResponseMetadata.m_TracePtr = trace;
    }
    // Special handling as BERT is the only model uses dynamic shape
    //
    // Inputs will need to be re-added as the shape is different from run
    // to run
    TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

    size_t seq_len = GetSampleLength(m_SampleLibraries[0], index);
    for (size_t idx = 0; idx < m_InputTensors.size(); idx++)
    {
        // Get a pointer to the input data
        int8_t* input_data = (int8_t*) m_SampleLibraries[0]->GetSampleAddress(index, idx); // Get address of the query
        // Need to calculate the shape from data for dynamic case
        size_t input_size = seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

        thread_local std::vector<int64_t> shape{1, 0};
        shape[1] = seq_len;
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestAddInput(request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
            "re-adding input");
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(request_block->m_Data,
                        std::get<0>(m_InputTensors[idx]).c_str(), input_data, input_size, m_InputMemoryType, 0),
            "appending input data");

        AppendInputDataForAllGPUs(
            request_block->m_Data, index, std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
        AppendInputDataForAllQSLs(
            request_block->m_Data, index, std::get<0>(m_InputTensors[idx]).c_str(), input_size, idx);
    }

    if (isWarmup)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(), WarmupResponseComplete, this),
            "appending input data");
    }
    else
    {
        /* Set response callback for this request */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(request_block->m_Data, m_Allocator,
                        m_OutputBufferPool.get(), ResponseComplete, &request_block->m_ResponseMetadata),
            "appending input data");
    }
    /* Actually perform inferences (asynchronously) */
    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, nullptr), "running inference");
}
void BERT_Triton_SUT::Warmup(double duration_sec, double expected_qps)
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

void BERT_Triton_SUT::IssueTritonRequest()
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

        IssueTritonRequestHelper(req->samples.front().index, req->samples.front().id, trace, false);

#if TRITON_FRONTEND_TRACE
        TraceCaptureTimeStamp(trace, "Called Infer Async");
#endif //
    }
}

void BERT_Triton_SUT::InitBatcherThreads()
{
    LOG(INFO) << "Creating BERT " << m_NumBatcherThreads << " batching threads " << std::endl;
    for (int i = 0; i < m_NumBatcherThreads; i++)
        m_BatcherThreads.emplace_back(std::thread(&Concurrent_Frontend_SUT::ProcessSample, this));
}

void BERT_Triton_SUT::InitIssueThreads()
{
    LOG(INFO) << "Creating BERT " << m_NumIssueThreads << " issue threads " << std::endl;
    for (int i = 0; i < m_NumIssueThreads; i++)
        m_IssueThreads.emplace_back(std::thread(&BERT_Triton_SUT::IssueTritonRequest, this));
}

}; // namespace triton_frontend