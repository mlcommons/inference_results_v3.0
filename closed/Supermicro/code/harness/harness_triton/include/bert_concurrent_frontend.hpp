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
#ifndef __BERT_TRITON_CONCURRENT_FRONTEND_HPP__
#define __BERT_TRITON_CONCURRENT_FRONTEND_HPP__

#include "lwis.hpp"
#include "triton_concurrent_frontend.hpp"

namespace triton_frontend
{

class BERT_Triton_SUT : public Concurrent_Frontend_SUT
{
public:
    BERT_Triton_SUT(const std::string& name, const std::string& model_repo_path, const std::string& model_name,
        uint32_t model_version, bool start_from_device, bool pinned_input, uint64_t request_count,
        size_t num_batcher_threads, size_t num_issue_threads, bool is_single_stream)
        : Concurrent_Frontend_SUT(name, model_repo_path, model_name, model_version, false, start_from_device,
            pinned_input, request_count, num_batcher_threads, num_issue_threads, is_single_stream)
    {
    }
    ~BERT_Triton_SUT() {}

    virtual void Warmup(double duration_sec, double expected_qps);

    // Functions that read off of issue queue and issue triton requests
    virtual void IssueTritonRequest();

    // Need to override this to sort samples by seq length before putting it on to the queue
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);

protected:
    virtual void InitBatcherThreads();
    virtual void InitIssueThreads();
    void IssueTritonRequestHelper(mlperf::QuerySampleIndex index, mlperf::ResponseId responseID,
        TRITONSERVER_InferenceTrace* m_TracePtr, bool isWarmup = false);
};

} // namespace triton_frontend

#endif