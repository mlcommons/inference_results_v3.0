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

#include "triton_concurrent_sut.hpp"

namespace triton_frontend
{

void TritonConcurrentSUT::Init(
    size_t buffer_manager_thread_count, const std::string& numa_config_str, const std::string& backend_path)
{
    // #if TRITON_FRONTEND_TRACE
    //     /* Set up trace manager */
    //     ni::TraceManager* manager = nullptr;
    //     FAIL_IF_ERR(triton::server::TraceManager::Create(
    //                     &manager, TRITONSERVER_TRACE_LEVEL_MAX, 80 /* rate, one sample per
    //                     batch*/, "triton_trace.log"),
    //                 "creating trace manger");
    //     m_TraceManager.reset(manager);
    // #endif // TRITON_FRONTEND_TRACE
    m_Device->Init(buffer_manager_thread_count, numa_config_str, backend_path);
}

void TritonConcurrentSUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    switch (m_Scenario)
    {
    case mlperf::TestScenario::SingleStream:
    case mlperf::TestScenario::MultiStream:
    {
        // pass to Triton immediately
        // no benefit to enqueueing since loadgen serializes sending the queries
        m_Device->IssueTritonRequestInternal(samples, true, nullptr, ResponseComplete);
        break;
    }
    case mlperf::TestScenario::Offline:
    {
        m_Device->m_Workload->IssueQueryOffline(samples, m_Device);
        break;
    }
    case mlperf::TestScenario::Server:
    {
        m_Device->m_Workload->IssueQueryServer(samples, m_Device);
        break;
    }
    }
}

void TritonConcurrentSUT::Warmup(double duration_sec, double expected_qps)
{
    m_Device->Warmup(duration_sec, expected_qps);
}

} // namespace triton_frontend