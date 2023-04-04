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
#ifndef __DLRM_TRITON_CONCURRENT_FRONTEND_HPP__
#define __DLRM_TRITON_CONCURRENT_FRONTEND_HPP__

#include "dlrm_qsl.hpp"
#include "lwis.hpp"
#include "triton_concurrent_frontend.hpp"

namespace triton_frontend
{

class DLRM_Triton_SUT : public Concurrent_Frontend_SUT
{
public:
    DLRM_Triton_SUT(const std::string& name, const std::string& model_repo_path, const std::string& model_name,
        uint32_t model_version, bool use_dlrm_qsl, bool start_from_device, bool pinned_input, uint64_t request_count,
        size_t num_batcher_threads, size_t num_issue_threads)
        : Concurrent_Frontend_SUT(name, model_repo_path, model_name, model_version, use_dlrm_qsl, start_from_device,
            pinned_input, request_count, num_batcher_threads, num_issue_threads, false)
    {
    }
    ~DLRM_Triton_SUT() {}

    virtual void Init(size_t min_sample_size = 1, size_t max_sample_size = 1, size_t buffer_manager_thread_count = 0,
        bool batch_triton_requests = false, bool check_contiguity = false, const std::string& numa_config_str = "",
        const std::string& backend_path = "");
    virtual void Warmup(double duration_sec, double expected_qps);

    virtual void BatchCompletion(
        TRITONSERVER_InferenceResponse* response, const BatchResponseMetaData* response_metadata);
    virtual void Done();

    // Functions that read off work queue, optionally form a batch and add to
    // issue queue
    virtual void ProcessSample(); // Process single query at a time
    virtual void ProcessBatch();  // Batch queries and process them

    // Functions that read off of issue queue and issue triton requests
    virtual void IssueTritonRequest();
    virtual void IssueTritonBatchRequest();

protected:
    // Min and max sample size in each loadgen sample
    // 1 for everything apart from DLRM
    size_t m_MinSampleSize = 1;
    size_t m_MaxSampleSize = 1;

    // m_NumEmptyTries denotes number of times m_QueuedSample.acquire can time out
    // with 0 samples acquired before batch is queued
    size_t m_NumEmptyTries = 1;

    // m_NumSuccessfulAcquired denotes number of times m_QueuedSample.acquire will
    // be called
    // in attempt to get a batch as close to maxBatchSize as possible
    size_t m_NumSuccessfulAcquires = 3;

    float maxSampleSizeMultiplierOffline = 0.6;
    float maxSampleSizeMultiplierServer = 0.8;
    // A map of batch size to count.
    // Note that batch size here refers to number of user item pairs
    std::map<int, int> m_BatchStats;

    /* Returns true if all the samples in the deque are contiguous */
    virtual bool CheckContiguity(const std::deque<mlperf::QuerySample>& samples, int num_pairs);

    virtual void InitBatcherThreads();
    virtual void InitIssueThreads();

    size_t GetNumQuerySamplesToAcquire(size_t num_pairs, int num_successful_tries, int num_empty_tries);

    virtual void AppendInputDataForAllGPUs(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx);

    virtual void AppendInputDataForAllQSLs(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sample_idx, const char* input_name, size_t input_size, size_t input_idx);
};

} // namespace triton_frontend

#endif