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

#pragma once

#include "half.h"
// #include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "bert_lon_common.h"
#include "bert_lon_core_vs.h"
#include "lwis_buffers.h"
// #include "common.h"
// #include "util.h"

// SUT_DEBUG_DISABLE_INFERENCE disables all device operations including data copies, most events, and compute
// SUT_DEBUG_DISABLE_COMPUTE disables the compute (TRT enqueue) portion of inference
// #define SUT_DEBUG_DISABLE_INFERENCE
// #define SUT_DEBUG_DISABLE_COMPUTE

namespace bert_lon
{

using namespace std::chrono_literals;

class BERTServer;
using ServerPtr_t = std::shared_ptr<BERTServer>;

struct ServerSettings
{
    bool EnableDequeLimit{false};

    bool SutUsesHostMemForRdma{false};
    bool SutRecvWithHostMemForRdma{false};
    bool SutSendWithHostMemForRdma{false};

    int GPUBatchSize{1};
    int GPUCopyStreams{1};
    int GPUInferStreams{1};

    int NumIBQPsPerNIC{1};

    std::chrono::microseconds Timeout{10000us};
    lwis_lon::NumaConfig NumaConfig;
    lwis_lon::GpuNumaMap GpuNumaMap;

    int GraphMaxSeqLen;

    double SoftDrop;
    double SoftDropTargetLatencyPercentile;
};

class BERTServer : public mlperf::SystemUnderTest
{
public:
    BERTServer(const std::string name, const std::string enginePath, const std::vector<int>& gpus, bool useGraphs,
        const std::string& graphSpecs, const ServerSettings sut_settings);

    ~BERTServer()
    {
        mWorkerThreads.clear();
        mIssueThreads.clear();
        mCompletionThreads.clear();
    };

    void Done();
    void Warmup(double duration);

    const std::string& Name() override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;

    void IssueQuery(std::deque<mlperf::QuerySample> const& samples, int const gpu_idx);

    std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>> get_work_queue(int gpu_idx)
    {
        return mDevWorkQueue[gpu_idx];
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>> get_completion_queue(int gpu_idx, int qp_idx)
    {
        return mDevCompletionQueue[gpu_idx][qp_idx];
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>> get_completion_unified_queue(int gpu_idx)
    {
        return mDevCompletionUnifiedQueue[gpu_idx];
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::CudaResource>> get_resource_return_queue(int gpu_idx, int bc_idx)
    {
        return mDevResourceReturnQueue[gpu_idx][bc_idx];
    }

    int get_num_bert_cores()
    {
        return mNumBERTCores;
    }

private:
    // If the function returns empty vector then there are no tasks remained and the caller should
    // exit
    BERTTask_t GetTasks(int maxSampleCount, int qThreadIdx);

    template <typename T>
    void ProcessTasks(std::shared_ptr<T>, int deviceId, int qThreadIdx);

    void ProcessSamples(int deviceId);
    void ProcessCompletions(int deviceId);

    void ProcessWarmup(std::shared_ptr<BERTCoreVS> bc, double duration);

    int GetSampleLength(void* TokenAddr, int TokenSize);

    void CreateEnginesPerGPU(
        int deviceId, std::shared_ptr<std::mutex> pMtx, const std::vector<std::vector<char>>& trtModelStreams);

    // Check if NUMA is used
    bool UseNuma()
    {
        return !mServerSettings.GpuNumaMap.empty();
    };

    // Get number of NUMA nodes
    int GetNbNumas()
    {
        return mServerSettings.GpuNumaMap.size();
    };

    // Get NUMA node index of a GPU
    int GetNumaIdxByGpuId(const int deviceId)
    {
        return UseNuma() ? mServerSettings.GpuNumaMap[deviceId] : 0;
    }

    // Get closest CPUs to a GPU
    std::vector<int> GetClosestCpusToGpu(const int deviceId)
    {
        CHECK(UseNuma());
        return std::get<2>(mServerSettings.NumaConfig[GetNumaIdxByGpuId(deviceId)]);
    }

    const std::string mName;
    const std::string mEnginePath;

    // For each GPU device id, create a vector of ptrs to ICudaEngine
    std::unordered_map<int, std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>> mEnginesPerGPU;

    int mMaxBatchSize;

    // Each query sample is accompanied by the time query arrived
    std::vector<std::deque<BERTTaskElem_t>> mTasksVec;

    // mutex to serialize access to mTasks member variable
    std::unique_ptr<std::vector<std::mutex>> mMtxs;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::unique_ptr<std::vector<std::condition_variable>> mCondVars;

    // Completion management
    std::vector<std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>>> mDevCompletionUnifiedQueue;
    std::vector<std::vector<std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>>>> mDevCompletionQueue;
    std::vector<std::vector<std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::CudaResource>>>> mDevResourceReturnQueue;
    std::vector<std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>>> mDevWorkQueue;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopGetTasks;
    bool mStopProcessResponse;

    // Max seqlen to be used for creating CUDA graphs
    int mGraphMaxSeqLen;

    // Whether to apply soft drop policy
    double mSoftDrop;
    double mTargetLatencyPercentile;
    TotalLengthMultiSet mTotalLengthSet;
    uint64_t mTotalTasksCount;
    uint64_t mSoftDropCount;

    // SUT
    std::vector<int> mGpusIndices;
    int mNumBERTCores;
    int mNumCopyStreams;
    std::vector<BERTCoreVSPtrVec> mBERTCores;

    // server settings
    ServerSettings mServerSettings;

    // mutex for both mTotalTasksCount and mSoftDropCount
    std::mutex mSoftDropMtx;

    std::vector<std::thread> mWorkerThreads;
    std::vector<std::thread> mIssueThreads;
    std::vector<std::thread> mCompletionThreads;
};

}; // namespace bert_lon
