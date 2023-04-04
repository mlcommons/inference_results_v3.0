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

#ifndef __DLA_TRITON_CONCURRENT_FRONTEND_HPP__
#define __DLA_TRITON_CONCURRENT_FRONTEND_HPP__

#include "dlrm_qsl.hpp"
#include "lwis.hpp"
#include "triton_concurrent_frontend.hpp"
#include <google/protobuf/util/json_util.h>

namespace triton_frontend
{

class DLA_Triton_SUT : public Concurrent_Frontend_SUT
{
public:
    DLA_Triton_SUT(const std::string& name, const std::string& model_repo_path, const std::string& model_name,
        uint32_t model_version, const std::string& dla_model_name, uint32_t dla_model_version, bool start_from_device,
        bool pinned_input, uint64_t request_count, size_t num_batcher_threads, size_t num_issue_threads,
        size_t dla_batch_size, size_t dla_num_batchers, size_t dla_num_issuers, float dla_ratio)
        : Concurrent_Frontend_SUT(name, model_repo_path, model_name, model_version, false, start_from_device,
            pinned_input, request_count, num_batcher_threads, num_issue_threads, false)
        , m_DlaBatchSize(dla_batch_size)
        , m_DlaModelName(dla_model_name)
        , m_DlaModelVersion(dla_model_version)
        , m_DlaNumBatcherThreads(dla_num_batchers)
        , m_DlaNumIssueThreads(dla_num_issuers)
        , m_DlaRequestRatio(dla_ratio)
    {
    }
    ~DLA_Triton_SUT() {}

    virtual void Init(size_t min_sample_size = 1, size_t max_sample_size = 1, size_t buffer_manager_thread_count = 0,
        bool batch_triton_requests = false, bool check_contiguity = false, const std::string& numa_config_str = "",
        const std::string& backend_path = "");
    // TODO: Add warmup for DLA+GPU
    virtual void Warmup(double duration_sec, double expected_qps) {}

    virtual void Done();

    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);

    // Functions that read off work queue, optionally form a batch and add to
    // issue queue
    virtual void ProcessDlaBatch();  // Batch queries according to dla_batch_size and process them
    virtual void ProcessDlaSample(); // Batch queries according to dla_batch_size and process them

    // Functions that read off of issue queue and issue triton requests
    virtual void IssueTritonDlaBatchRequest();
    virtual void IssueTritonDlaRequest();

protected:
    TRITONSERVER_ResponseAllocator* m_DlaAllocator = nullptr;
    std::string m_DlaModelName;
    const uint32_t m_DlaModelVersion;

    inference::ModelConfig m_DlaConfig;
    size_t m_DlaBatchSize;

    void ModelStats();

    // Ratio of samples for DLA versus total
    float m_DlaRequestRatio;

    // Threads to batch DLA requests
    size_t m_DlaNumBatcherThreads{1};
    std::vector<std::thread> m_DlaBatcherThreads;

    // Use a separate Queue for DLA work to ensure reproducible distribution of work between GPU and
    // DLA
    lwis::SyncQueue<mlperf::QuerySample> m_DlaWorkQueue;

    // Threads that create and issue requests to Triton DLA model
    size_t m_DlaNumIssueThreads{1};
    lwis::SyncQueue<std::shared_ptr<RequestInfo>> m_DlaIssueQueue;
    std::vector<std::thread> m_DlaIssueThreads;

    // A map of batch size to count.
    std::map<int, int> m_DlaBatchStats;

    virtual void InitDlaBatcherThreads();
    virtual void InitDlaIssueThreads();

    // Pinned memory pool for output buffers for batched requests
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_DlaOutputBufferPool;
    std::unique_ptr<PinnedMemoryPoolEnsemble> m_DlaBatchedOutputBufferPool;
};

} // namespace triton_frontend

#endif