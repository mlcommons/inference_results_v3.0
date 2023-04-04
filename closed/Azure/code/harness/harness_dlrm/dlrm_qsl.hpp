/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __DLRM_QSL_HPP__
#define __DLRM_QSL_HPP__

#include "qsl.hpp"
#include "utils.hpp"

class DLRMSampleLibrary : public qsl::SampleLibrary
{
public:
    DLRMSampleLibrary(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
        std::vector<int> samplePartition, size_t perfSampleCount, size_t perfPairCount, size_t padding = 0,
        bool coalesced = false, bool startFromDevice = false, int32_t numaIdx = -1, int32_t numNuma = 0)
        : mSampleStartIdxs(samplePartition)
        , mPerfSampleCount(perfSampleCount)
        , mTotalSampleCount(samplePartition.size() - 1)
        , qsl::SampleLibrary(
              name, mapPath, tensorPaths, perfPairCount, padding, coalesced, {startFromDevice, startFromDevice})
        , mNumIndividualPairs(std::min(perfPairCount, qsl::SampleLibrary::TotalSampleCount()))
        , mNumaIdx(numaIdx)
        , mNumNuma(numNuma)
    {
        CHECK_EQ(mSampleStartIdxs.back(), mNumIndividualPairs);
        LOG(INFO) << "PerformanceSampleCount: " << mPerfSampleCount;
        LOG(INFO) << "TotalSampleCount: " << mTotalSampleCount << " (" << mNumIndividualPairs << " pairs).";

        if (startFromDevice)
            CHECK_LE(numNuma, 1);
    }

    size_t TotalSampleCount() override
    {
        return mTotalSampleCount;
    }
    size_t PerformanceSampleCount() override
    {
        return mPerfSampleCount;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        LOG(INFO) << "Calling LoadSamplesToRam() for QSL[" << (UseNuma() ? mNumaIdx : 0) << "] of " << samples.size()
                  << " samples...";

        // Set memory allocation setting
        if (UseNuma())
        {
            bindNumaMemPolicy(mNumaIdx, mNumNuma);
        }

        qsl::SampleLibrary::LoadSamplesToRam(mapToIndividualIndices(samples));

        auto num_dev = UseNuma() ? 1 : m_NumDevices;

        sampleAddresses.resize(num_dev * mTotalSampleCount * m_NumInputs);
        for (int deviceIdx = 0; deviceIdx < num_dev; ++deviceIdx)
            for (const auto& sampleIdx : samples)
                for (int inputIdx = 0; inputIdx < m_NumInputs; ++inputIdx)
                    sampleAddresses[(deviceIdx * mTotalSampleCount + sampleIdx) * m_NumInputs + inputIdx]
                        = qsl::SampleLibrary::GetSampleAddress(mSampleStartIdxs[sampleIdx], inputIdx, deviceIdx);

        // Reset memory allocation setting
        if (UseNuma())
        {
            resetNumaMemPolicy();
        }

        {
            m_SampleAddressMapHost.clear();
            m_SampleAddressMapDevice.clear();
        }

        LOG(INFO) << "Completed LoadSamplesToRam() for QSL[" << (UseNuma() ? mNumaIdx : 0) << "]";
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        LOG(INFO) << "Calling UnloadSamplesFromRam() for QSL[" << (UseNuma() ? mNumaIdx : 0) << "] of "
                  << samples.size() << " samples...";

        // qsl::SampleLibrary::UnloadSamplesFromRam(mapToIndividualIndices(samples));
        {
            sampleAddresses.clear();
        }

        LOG(INFO) << "Completed UnloadSamplesFromRam() for QSL[" << (UseNuma() ? mNumaIdx : 0) << "]";
    }

    void* GetSampleAddress(
        mlperf::QuerySampleIndex sampleIdx, size_t inputIdx, size_t pairIdx = 0, size_t deviceIdx = 0)
    {
        auto d_idx = UseNuma() ? 0 : deviceIdx;
        auto addr
            = static_cast<char*>(sampleAddresses[(d_idx * mTotalSampleCount + sampleIdx) * m_NumInputs + inputIdx])
            + pairIdx * qsl::SampleLibrary::GetSampleSize(inputIdx);
        return addr;
    }

    size_t GetNumUserItemPairs(size_t index)
    {
        size_t start = mSampleStartIdxs[index];
        size_t end = mSampleStartIdxs[index + 1];
        return end - start;
    }

    inline bool UseNuma()
    {
        return mNumNuma > 0;
    };

private:
    std::vector<mlperf::QuerySampleIndex> mapToIndividualIndices(const std::vector<mlperf::QuerySampleIndex>& samples)
    {
        std::vector<mlperf::QuerySampleIndex> remappedIndices;

        for (auto index : samples)
        {
            CHECK_EQ((index >= 0) && (index < mTotalSampleCount), true);
            int start = mSampleStartIdxs[index];
            int end = mSampleStartIdxs[index + 1];
            for (int i = start; i < end; ++i)
            {
                remappedIndices.push_back(i);
            }
        }
        return remappedIndices;
    }

    int mNumIndividualPairs;
    size_t mPerfSampleCount;
    size_t mTotalSampleCount;
    std::vector<int> mSampleStartIdxs;
    std::vector<void*> sampleAddresses;
    int32_t mNumNuma;
    int32_t mNumaIdx;
};

typedef std::shared_ptr<DLRMSampleLibrary> DLRMSampleLibraryPtr_t;

class DLRMSampleLibraryEnsemble : public mlperf::QuerySampleLibrary
{
public:
    DLRMSampleLibraryEnsemble(const std::vector<DLRMSampleLibraryPtr_t> qsls)
        : m_qsls(qsls){};
    const std::string& Name() override
    {
        return m_qsls[0]->Name();
    }
    size_t TotalSampleCount() override
    {
        return m_qsls[0]->TotalSampleCount();
    }
    size_t PerformanceSampleCount() override
    {
        return m_qsls[0]->PerformanceSampleCount();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        LOG(INFO) << "Calling LoadSamplesToRam() for QSL ensemble...";
        for (auto qsl : m_qsls)
        {
            qsl->LoadSamplesToRam(samples);
        }
        LOG(INFO) << "Completed LoadSamplesToRam() for QSL ensemble.";
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        LOG(INFO) << "Calling UnloadSamplesFromRam() for QSL ensemble...";
        for (auto qsl : m_qsls)
        {
            qsl->UnloadSamplesFromRam(samples);
        }
        LOG(INFO) << "Completed UnloadSamplesFromRam() for QSL ensemble.";
    }

private:
    std::vector<DLRMSampleLibraryPtr_t> m_qsls;
};

#endif // __DLRM_QSL_HPP__
