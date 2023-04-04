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

#ifndef MLPINF_TRITON_CONCURRENT_SUT_HPP
#define MLPINF_TRITON_CONCURRENT_SUT_HPP

#include "triton_sut.hpp"

#include "lwis.hpp"
#include <chrono>

namespace triton_frontend
{

class TritonConcurrentSUT : public ITritonSUT
{
public:
    TritonConcurrentSUT(mlperf::TestScenario scenario, ITritonDevice* const device)
        : m_Scenario(scenario)
        , ITritonSUT(device)
    {
    }

    ~TritonConcurrentSUT() {}

    virtual void Init(size_t bufferManagerThreadCount = 0, const std::string& numaConfigStr = "",
        const std::string& backendPath = "") override;

    // SUT virtual interface
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

    void Warmup(double duration_sec, double expected_qps) override;

private:
    mlperf::TestScenario m_Scenario;
};

} // namespace triton_frontend

#endif // MLPINF_TRITON_CONCURRENT_SUT_HPP