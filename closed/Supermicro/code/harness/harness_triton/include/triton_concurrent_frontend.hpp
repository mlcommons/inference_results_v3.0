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
#ifndef __TRITON_FRONTEND_SERVER_V2_HPP__
#define __TRITON_FRONTEND_SERVER_V2_HPP__

#include "lwis.hpp"
#include "triton_frontend_server.hpp"
#include <chrono>

namespace triton_frontend
{

struct RequestInfo
{
    std::deque<mlperf::QuerySample> samples;
    bool isContiguous;
    size_t batchSize;
};

class Concurrent_Frontend_SUT : public Triton_Server_SUT
{
public:
    Concurrent_Frontend_SUT(const std::string& name, const std::string& model_repo_path, const std::string& model_name,
        uint32_t model_version, bool use_dlrm_qsl, bool start_from_device, bool pinned_input, uint64_t request_count,
        size_t num_batcher_threads, size_t num_issue_threads, bool is_single_stream)
        : Triton_Server_SUT(name, model_repo_path, model_name, model_version, use_dlrm_qsl, start_from_device, false,
            pinned_input, request_count)
        , m_NumBatcherThreads(num_batcher_threads)
        , m_NumIssueThreads(num_issue_threads)
        , m_IsSingleStream(is_single_stream)
    {
    }
    ~Concurrent_Frontend_SUT() {}

    virtual void Init(size_t min_sample_size = 1, size_t max_sample_size = 1, size_t buffer_manager_thread_count = 0,
        bool batch_triton_requests = false, bool check_contiguity = false, const std::string& numa_config_str = "",
        const std::string& backend_path = "");
    virtual void Warmup(double duration_sec, double expected_qps);

    virtual void Completion(TRITONSERVER_InferenceResponse* response, const ResponseMetaData* response_metadata);
    virtual void BatchCompletion(
        TRITONSERVER_InferenceResponse* response, const BatchResponseMetaData* response_metadata);
    virtual void Done();

    // SUT virtual interface
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);
    virtual void FlushQueries();

    // Functions that read off work queue, optionally form a batch and add to
    // issue queue
    virtual void ProcessSample(); // Process single query at a time
    virtual void ProcessBatch();  // Batch queries and process them

    // Functions that read off of issue queue and issue triton requests
    virtual void IssueTritonRequest();
    virtual void IssueTritonBatchRequest();

    // Avoid queuing query/request when running SingleStream
    virtual void IssueTritonRequestHelper(mlperf::QuerySampleIndex index, mlperf::ResponseId responseID,
        TRITONSERVER_InferenceTrace* m_TracePtr, bool isWarmup = false);

protected:
    // SUT settings
    std::chrono::microseconds workQueueTimeout{1000};
    bool m_IsSingleStream{false};

    // Query management
    size_t m_NumBatcherThreads{1};
    lwis::SyncQueue<mlperf::QuerySample> m_WorkQueue;
    std::vector<std::thread> m_BatcherThreads;
    virtual void InitBatcherThreads();

    // Batch management
    size_t m_NumIssueThreads{1};
    lwis::SyncQueue<std::shared_ptr<RequestInfo>> m_IssueQueue;
    std::vector<std::thread> m_IssueThreads;
    virtual void InitIssueThreads();

    // stats
    std::map<int, int> m_BatchStats;

    // Helper

    /* Returns true if all the samples in the deque are contiguous */
    virtual bool CheckContiguity(std::deque<mlperf::QuerySample>& samples);

    /* If the input is on the GPU i.e start_from_device is true, this function
    calls Triton's
    function AppendInputDataWithHostPolicy for each copy of the input on different
    GPUs  */
    virtual void AppendInputDataForAllGPUs(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx);

    /* If there are multiple QSLs, this function calls Triton's function
    AppendInputDataWithHostPolicy for each copy of the input in each of the QSL
    copies. This is
    useful for NUMA support  */
    virtual void AppendInputDataForAllQSLs(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx);
    virtual size_t GetBatchOneByteSize(
        const inference::ModelConfig& config, const std::vector<std::string>& output_names);
};

} // namespace triton_frontend

#endif
