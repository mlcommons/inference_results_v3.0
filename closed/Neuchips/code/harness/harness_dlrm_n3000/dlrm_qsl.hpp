/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "config.h"
#include "qsl.hpp"
#include "utils.hpp"

class DLRMSampleLibrary : public qsl::SampleLibrary
{
 public:
    DLRMSampleLibrary(std::string name, std::string mapPath,
        std::vector<std::string> tensorPaths,
        std::vector<int> samplePartition, size_t perfSampleCount,
        size_t perfPairCount, size_t padding = 0,
        bool coalesced = false, bool startFromDevice = false,
        int32_t numaIdx = -1, int32_t numNuma = 0,
        std::vector<int> deviceIDs = {})
        : mSampleStartIdxs(samplePartition)
        , mPerfSampleCount(perfSampleCount)
        , mTotalSampleCount(samplePartition.size() - 1)
        , qsl::SampleLibrary(
              name, mapPath, tensorPaths, perfPairCount, padding,
              coalesced, {startFromDevice, startFromDevice})
        , mNumIndividualPairs(std::min(perfPairCount,
                                qsl::SampleLibrary::TotalSampleCount()))
        , mNumaIdx(numaIdx)
        , mNumNuma(numNuma)
        , mDeviceIDs(std::move(deviceIDs)) {
        LOG(INFO) << "DLRMSampleLibrary mSampleStartIdxs.size(): "
                  << mSampleStartIdxs.size()
                  << " perfPairCount" << perfPairCount
                  << " qsl::SampleLibrary::TotalSampleCount() "
                  << qsl::SampleLibrary::TotalSampleCount();

        CHECK_EQ(mSampleStartIdxs.back(), mNumIndividualPairs);
        LOG(INFO) << "DLRMSampleLibrary PerformanceSampleCount: "
                  << mPerfSampleCount;
        LOG(INFO) << "DLRMSampleLibrary TotalSampleCount: "
                  << mTotalSampleCount << " ("
                  << mNumIndividualPairs << " pairs).";
        LOG(INFO) << "DLRMSampleLibrary mDeviceIDs in mNumaIdx " << mNumaIdx;
        for (auto i : mDeviceIDs) {
            LOG(INFO) << "ID " << i;
        }
    }

    size_t TotalSampleCount() override {
        return mTotalSampleCount;
    }

    size_t PerformanceSampleCount() override {
        return mPerfSampleCount;
    }

    u32 get_1st_sample_size() {
        return qsl::SampleLibrary::get_1st_sample_size();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        // Set memory allocation setting
        if (UseNuma()) {
            LOG(INFO) << "DLRMSampleLibrary LoadSamplesToRam, UseNuma "
                      << UseNuma()
                      << " mNumaIdx " << mNumaIdx << " mNumNuma " << mNumNuma
                      << " mDeviceIDs[0]" << mDeviceIDs[0];
            bindNumaMemPolicy(mDeviceIDs[0], NUM_NUMA_ZONES);
        }

        LOG(INFO) << "qsl::SampleLibrary::LoadSamplesToRam+ samples.size() "
                  << samples.size();
        qsl::SampleLibrary::LoadSamplesToRamNeuchips(
                mapToIndividualIndices(samples),
                samples, mSampleStartIdxs, mTotalSampleCount, mNumaIdx,
                mNumNuma, mDeviceIDs);
        LOG(INFO) << "qsl::SampleLibrary::LoadSamplesToRam-";

        sampleAddresses.resize(m_NumDevices * mTotalSampleCount * m_NumInputs);
        DLOG(INFO) << "DLRMSampleLibrary LoadSamplesToRam "
                   << m_NumDevices * mTotalSampleCount * m_NumInputs;
        DLOG(INFO) << "mTotalSampleCount " << mTotalSampleCount;
        DLOG(INFO) << "m_NumDevices " << m_NumDevices;
        DLOG(INFO) << "m_NumInputs " << m_NumInputs;

        for (int deviceIdx = 0; deviceIdx < m_NumDevices; ++deviceIdx) {
            LOG(INFO) << "DLRMSampleLibrary::LoadSamplesToRam deviceIdx "
                      << deviceIdx;
            for (const auto& sampleIdx : samples) {
                for (int inputIdx = 0; inputIdx < m_NumInputs; ++inputIdx) {
                    sampleAddresses[
                        (deviceIdx * mTotalSampleCount + sampleIdx)
                        * m_NumInputs + inputIdx]
                        = qsl::SampleLibrary::GetSampleAddress(
                            mSampleStartIdxs[sampleIdx], inputIdx, deviceIdx);
                }
            }
        }

        // Reset memory allocation setting
        if (UseNuma()) {
            DLOG(INFO) <<
                "LoadSamplesToRam Reset memory allocation setting";
            resetNumaMemPolicy();
        }
    }

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex>& samples) override {
        DLOG(INFO) << "UnloadSamplesFromRam: samples.size() "
                   << samples.size() << " mNumaIdx " << mNumaIdx;
        qsl::SampleLibrary::UnloadSamplesFromRamNeuchips(
            samples, mSampleStartIdxs, mNumaIdx, mDeviceIDs);
    }

    void* GetSampleAddress(
        mlperf::QuerySampleIndex sampleIdx, size_t inputIdx,
        size_t pairIdx = 0, size_t deviceIdx = 0) {
        return static_cast<char*>(sampleAddresses[
            (deviceIdx * mTotalSampleCount + sampleIdx) *
            m_NumInputs + inputIdx])
            + pairIdx * qsl::SampleLibrary::GetSampleSize(inputIdx);
    }

    size_t GetNumUserItemPairs(size_t index) {
        size_t start = mSampleStartIdxs[index];
        size_t end = mSampleStartIdxs[index + 1];
        return end - start;
    }

    inline bool UseNuma() {
        return mNumNuma > 0;
    }

 private:
    std::vector<mlperf::QuerySampleIndex> mapToIndividualIndices(
        const std::vector<mlperf::QuerySampleIndex>& samples) {
        std::vector<mlperf::QuerySampleIndex> remappedIndices;
        LOG(INFO) << "mapToIndividualIndices samples.size() "
                  << samples.size();
        for (auto index : samples) {
            CHECK_EQ((index >= 0) && (index < mTotalSampleCount), true);
            int start = mSampleStartIdxs[index];
            int end = mSampleStartIdxs[index + 1];
            for (int i = start; i < end; ++i) {
                remappedIndices.push_back(i);
            }
        }
        LOG(INFO) << "remappedIndices size() " << remappedIndices.size();
        return remappedIndices;
    }

    int mNumIndividualPairs;
    size_t mPerfSampleCount;
    size_t mTotalSampleCount;
    std::vector<int> mSampleStartIdxs;
    std::vector<void*> sampleAddresses;
    int32_t mNumNuma;
    int32_t mNumaIdx;
    std::vector<int> mDeviceIDs;
};

typedef std::shared_ptr<DLRMSampleLibrary> DLRMSampleLibraryPtr_t;

class DLRMSampleLibraryEnsemble : public mlperf::QuerySampleLibrary
{
 public:
    DLRMSampleLibraryEnsemble(const std::vector<DLRMSampleLibraryPtr_t> qsls)
        : m_qsls(qsls){};
    const std::string& Name() override {
        return m_qsls[0]->Name();
    }

    size_t TotalSampleCount() override {
        return m_qsls[0]->TotalSampleCount();
    }
    size_t PerformanceSampleCount() override {
        return m_qsls[0]->PerformanceSampleCount();
    }

    void LoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex>& samples) override {
        for (auto qsl : m_qsls) {
            qsl->LoadSamplesToRam(samples);
        }
    }
    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex>& samples) override {
        for (auto qsl : m_qsls) {
            qsl->UnloadSamplesFromRam(samples);
        }
    }

 private:
    std::vector<DLRMSampleLibraryPtr_t> m_qsls;
};

#endif  // CLOSED_NEUCHIPS_CODE_HARNESS_HARNESS_DLRM_N3000_DLRM_QSL_HPP_
