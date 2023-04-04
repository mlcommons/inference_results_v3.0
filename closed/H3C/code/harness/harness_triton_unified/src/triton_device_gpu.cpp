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

#include "include/triton_device_gpu.hpp"
#include "include/pinned_memory_pool.hpp"
#include "include/triton_callbacks.hpp"
#include "include/triton_helpers.hpp"

namespace triton_frontend
{

void TritonDeviceGPU::AppendInputDataWithHostPolicy(TRITONSERVER_InferenceRequest* request,
    const mlperf::QuerySampleIndex& sampleIdx, const std::vector<qsl::SampleLibraryPtr_t>& qsls, const char* inputName,
    size_t inputSize, size_t inputIdx)
{
    // append input data for all GPUs
    if (m_HarnessConfig.m_StartFromDevice && m_HarnessConfig.m_NumGPUs > 1)
    {
        for (int i = 0; i < m_HarnessConfig.m_NumGPUs; i++)
        {
            int8_t* inputData = static_cast<int8_t*>(qsls[0]->GetSampleAddress(sampleIdx, inputIdx,
                i)); // Get address of the query for device

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request, inputName, inputData,
                            inputSize, m_InputMemoryType, i, m_HostPolicyNames[i].c_str()),
                "appending input data with host policy");
        }
    }

    // If there are more than one QSL, we want to add input buffer with device
    // affinity
    if (qsls.size() > 1)
    {
        for (size_t qslIdx = 0; qslIdx < qsls.size(); ++qslIdx)
        {
            // Get a pointer to the input data
            int8_t* inputData
                = static_cast<int8_t*>(qsls[qslIdx]->GetSampleAddress(sampleIdx, inputIdx)); // Get address of the query

            CHECK(m_InputMemoryType != TRITONSERVER_MEMORY_GPU);
            // Note: the memory_type_id parameter is unused when memory type is not GPU
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputDataWithHostPolicy(request, inputName, inputData,
                            inputSize, m_InputMemoryType, 0, m_HostPolicyNames[qslIdx].c_str()),
                "appending input data");
        }
    }
}

} // namespace triton_frontend
