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

#ifndef MLPINF_TRITON_DEVICE_GPU_HPP
#define MLPINF_TRITON_DEVICE_GPU_HPP

#include "lwis.hpp"
#include "triton_device.hpp"

#include <memory>
#include <thread>
#include <vector>

namespace triton_frontend
{

class ITritonSUT;

class TritonDeviceGPU : public ITritonDevice
{
public:
    TritonDeviceGPU(const std::string& name, const std::string& modelRepoPath, const std::string& modelName,
        uint32_t modelVersion, uint64_t requestCount, size_t numBatcherThreads, size_t numInferThreads,
        ITritonWorkload* const workload, HarnessConfigs harnessConfig)
        : ITritonDevice(name, modelRepoPath, modelName, modelVersion, requestCount, numBatcherThreads, numInferThreads,
            workload, harnessConfig)
    {
    }

    void AppendInputDataWithHostPolicy(TRITONSERVER_InferenceRequest* request,
        const mlperf::QuerySampleIndex& sampleIdx, const std::vector<qsl::SampleLibraryPtr_t>& qsls,
        const char* inputName, size_t inputSize, size_t inputIdx) override;

    virtual ~TritonDeviceGPU() = default;
};

} // namespace triton_frontend

#endif // MLPINF_TRITON_DEVICE_GPU_HPP