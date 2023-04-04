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

#include "triton_workload.hpp"
#include "triton_device.hpp"

namespace triton_frontend
{

bool ITritonWorkload::CheckContiguity(const std::deque<mlperf::QuerySample>& samples) const
{
    auto nbInputs = m_SampleLibraries[0]->GetNbInputs();
    for (size_t i = 0; i < nbInputs; i++)
    {
        auto sampleSize = m_SampleLibraries[0]->GetSampleSize(i);

        auto start = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress(std::begin(samples)->index, i));
        auto end = static_cast<int8_t*>(m_SampleLibraries[0]->GetSampleAddress(std::prev(std::end(samples))->index, i));
        if ((end - start) != sampleSize * (samples.size() - 1))
        {
            return false;
        }
    }
    return true;
}

int32_t ITritonWorkload::GetOutputMaxBytes(const inference::ModelConfig& config) const
{
    const auto& outputs = config.output();
    CHECK(outputs.size() > 0);

    return std::transform_reduce(
        std::begin(outputs), std::end(outputs), (size_t) 0, [](auto a, auto b) { return std::max(a, b); },
        [](const auto& output) -> size_t {
            const auto dtypeBytes = GetDataTypeByteSize(output.data_type());
            const auto batch1ByteSize = std::accumulate(
                std::begin(output.dims()), std::end(output.dims()), (size_t) dtypeBytes, std::multiplies<int32_t>());
            if (batch1ByteSize <= 0)
            {
                FAIL("can't preallocate memory for variable size data type");
            }
            return batch1ByteSize;
        });
}

std::shared_ptr<mlperf::QuerySampleLibrary> ITritonWorkload::InitQSL(const std::string& mapPath,
    const std::vector<std::string>& tensorPaths, uint64_t performanceSampleCount, size_t padding, bool coalescedTensor,
    bool startFromDevice, const std::string& numaConfigStr)
{
    auto numaConfig = parseNumaConfig(numaConfigStr);
    std::vector<bool> startFromDeviceVec(tensorPaths.size(), startFromDevice);
    std::shared_ptr<mlperf::QuerySampleLibrary> qsl;
    if (numaConfig.empty())
    {
        std::vector<bool> startFromDeviceVec(tensorPaths.size(), startFromDevice);
        auto lib = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", mapPath, tensorPaths,
            performanceSampleCount, padding, coalescedTensor, startFromDeviceVec);
        m_SampleLibraries.emplace_back(lib);
        qsl = lib;
    }
    else
    {
        std::vector<qsl::SampleLibraryPtr_t> qsls;
        const int32_t nbNumas = numaConfig.size();
        auto gpuNumaVec = getGpuToNumaMap(numaConfig);
        for (int32_t numaIdx : gpuNumaVec)
        {
            // Use a thread to construct QSL so that the allocated memory is closer to that NUMA
            // node.
            auto constructQsl = [&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                bindNumaMemPolicy(numaIdx, nbNumas);
                auto oneQsl = std::make_shared<qsl::SampleLibrary>("Triton_SampleLibrary", mapPath, tensorPaths,
                    performanceSampleCount, padding, coalescedTensor, startFromDeviceVec);
                resetNumaMemPolicy();
                qsls.emplace_back(oneQsl);
            };
            std::thread t(constructQsl);
            bindThreadToCpus(t, numaConfig[numaIdx].second);
            t.join();
            m_SampleLibraries.emplace_back(qsls.back());
        }
        qsl.reset(new qsl::SampleLibraryEnsemble(qsls));
    }
    return qsl;
}

void ITritonWorkload::IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device)
{
    CHECK(m_HarnessConfig.m_BatchTritonRequests);
    device->m_ReadyToBatch.insert(samples);
}
void ITritonWorkload::IssueQueryServer(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device)
{
    for (const auto& sample : samples)
    {
        std::array<mlperf::QuerySample, 1> sampleArr = {sample};
        device->IssueTritonRequestInternal(sampleArr, false, nullptr, ResponseComplete);
    }
}

int32_t TritonWorkloadBERT::GetOutputMaxBytes(const inference::ModelConfig& config) const
{
    const auto& outputs = config.output();
    CHECK(outputs.size() > 0);

    return std::transform_reduce(
        std::begin(outputs), std::end(outputs), (size_t) 0, [](auto a, auto b) { return std::max(a, b); },
        [&](const auto& output) -> size_t {
            const auto dtypeBytes = GetDataTypeByteSize(output.data_type());
            const auto batch1ByteSize = std::accumulate(std::begin(output.dims()), std::end(output.dims()),
                static_cast<size_t>(dtypeBytes), [&](const auto a, const auto b) {
                    if (b == -1)
                        return a * m_BertConfig.maxSeqLen;
                    else
                        return a * b;
                });
            if (batch1ByteSize <= 0)
            {
                FAIL("can't preallocate memory for variable size data type");
            }
            return batch1ByteSize;
        });
}
void TritonWorkloadBERT::IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples, ITritonDevice* device)
{
    std::vector<std::pair<int32_t, int32_t>> samplesSorted(samples.size());
    for (int32_t i = 0; i < samples.size(); i++)
    {
        samplesSorted[i] = std::make_pair(i, GetSeqLen(samples[i].index));
    }
    // Sort samples in the descending order of sentence length
    // Plugin kernels are specialized for specific max sequence lengths (64, 128, 256 etc),
    // sorting samples improves the efficiency of running these kernels because neighbouring
    // samples will have the same max_seq_lengths
    std::sort(std::begin(samplesSorted), std::end(samplesSorted),
        [this](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& p : samplesSorted)
    {
        // TODO: There is a loadgen bug here; loadgen should only be sending
        // one sample per IssueQuery() call
        std::array<mlperf::QuerySample, 1> sampleArr = {samples[p.first]};
        device->IssueTritonRequestInternal(sampleArr, false, nullptr, ResponseComplete);
    }
}

} // namespace triton_frontend