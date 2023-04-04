#pragma once
#include "qsl.hpp"
#include "util.hpp"

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
    }

    size_t TotalSampleCount() override
    {
        return mTotalSampleCount;
    }
    size_t PerformanceSampleCount() override
    {
        return mPerfSampleCount;
    }

    size_t NumIndividualPairs() {
        return mNumIndividualPairs;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        // Set memory allocation setting
        if (UseNuma())
        {
            bindNumaMemPolicy(mNumaIdx, mNumNuma);
        }

        qsl::SampleLibrary::LoadSamplesToRam(mapToIndividualIndices(samples));

        sampleAddresses.resize(m_NumDevices * mTotalSampleCount * m_NumInputs);
        for (int deviceIdx = 0; deviceIdx < m_NumDevices; ++deviceIdx)
            for (const auto& sampleIdx : samples)
                for (int inputIdx = 0; inputIdx < m_NumInputs; ++inputIdx)
                    sampleAddresses[(deviceIdx * mTotalSampleCount + sampleIdx) * m_NumInputs + inputIdx]
                        = qsl::SampleLibrary::GetSampleAddress(mSampleStartIdxs[sampleIdx], inputIdx, deviceIdx);

        // Reset memory allocation setting
        if (UseNuma())
        {
            resetNumaMemPolicy();
        }
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        qsl::SampleLibrary::UnloadSamplesFromRam(mapToIndividualIndices(samples));
    }

    void* GetSampleAddress(
        mlperf::QuerySampleIndex sampleIdx, size_t inputIdx, size_t pairIdx = 0, size_t deviceIdx = 0)
    {
        return static_cast<char*>(sampleAddresses[(deviceIdx * mTotalSampleCount + sampleIdx) * m_NumInputs + inputIdx])
            + pairIdx * qsl::SampleLibrary::GetSampleSize(inputIdx);
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
        for (auto qsl : m_qsls)
        {
            qsl->LoadSamplesToRam(samples);
        }
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        for (auto qsl : m_qsls)
        {
            qsl->UnloadSamplesFromRam(samples);
        }
    }

private:
    std::vector<DLRMSampleLibraryPtr_t> m_qsls;
};
