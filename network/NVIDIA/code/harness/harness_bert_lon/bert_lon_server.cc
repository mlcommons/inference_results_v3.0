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

#include "bert_lon_server.h"
#include "bert_lon_core_vs.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>

#include <nvtx3/nvToolsExt.h> // For NVTX annotations

// ================================
//     Debug support: nvtx ranges
// ================================

// #define NVTX_ON

#ifdef NVTX_ON
enum nvtx_color
{
    COLOR_BLUE_0 = 255,
    COLOR_BLUE_1 = 200,
    COLOR_BLUE_2 = 150,
    COLOR_BLUE_3 = 100,
    COLOR_GREEN_0 = 255 << 8,
    COLOR_GREEN_1 = 200 << 8,
    COLOR_GREEN_2 = 150 << 8,
    COLOR_GREEN_3 = 100 << 8,
    COLOR_RED_0 = 255 << 16,
    COLOR_RED_1 = 200 << 16,
    COLOR_RED_2 = 150 << 16,
    COLOR_RED_3 = 100 << 16,
};
#define NVTX_GLOBAL_START(A, B, C) nvtxRangeId_t A = global_event_start(B, C)
#define NVTX_GLOBAL_END(A) global_event_end(A)
#define NVTX_THREAD_START(B, C) thread_event_start(B, C)
#define NVTX_THREAD_END() thread_event_end()
#define NVTX_MARK(B, C) mark_event(B, C)
#else
#define NVTX_GLOBAL_START(A, B, C)
#define NVTX_GLOBAL_END(A)
#define NVTX_THREAD_START(B, C)
#define NVTX_THREAD_END()
#define NVTX_MARK(B, C)
#endif

#ifdef NVTX_ON
#define CREAT_NVTX_EVENT_ATTRIB(A)                                                                                     \
    nvtxEventAttributes_t A = {0};                                                                                     \
    A.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                            \
    A.version = NVTX_VERSION;                                                                                          \
    A.colorType = NVTX_COLOR_ARGB;                                                                                     \
    A.color = eventColor;                                                                                              \
    A.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                                           \
    A.message.ascii = eventName.data()

nvtxRangeId_t global_event_start(const std::string& eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxRangeId_t corrId = nvtxRangeStartEx(&eventAttrib);
    return (corrId);
}

void global_event_end(nvtxRangeId_t corrId)
{
    nvtxRangeEnd(corrId);
}

void thread_event_start(const std::string& eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxRangeId_t corrId = nvtxRangePushEx(&eventAttrib);
}

void thread_event_end()
{
    nvtxRangePop();
}

void mark_event(const std::string& eventName, const nvtx_color eventColor)
{
    CREAT_NVTX_EVENT_ATTRIB(eventAttrib);
    nvtxMarkEx(&eventAttrib);
}
#endif

namespace bert_lon
{

template <typename T>
void BERTServer::ProcessTasks(std::shared_ptr<T> bertCore, int deviceId, int qThreadIdx)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    uint64_t totalCountInThread = 0;

    // hold soft drop tasks if any
    BERTTask_t holdedTasks;

    // Process samples in batches
    NVTX_THREAD_START("GetTasks", COLOR_BLUE_1);
    auto tasks = GetTasks(mMaxBatchSize, qThreadIdx);
    NVTX_THREAD_END();

    while (!tasks.empty())
    {
        totalCountInThread += tasks.size();
        if (mSoftDrop < 1.0)
        {
            std::unique_lock<std::mutex> lck(mSoftDropMtx);
            mTotalTasksCount += tasks.size();
            mTotalLengthSet.InsertTasks(tasks);

            // Drop requests until the total length is not greater than the threshold
            // Use target latency percentile as a hard limit on how many requests we
            // can drop
            while (BERTCoreVS::CountTotalLength(tasks) > mTotalLengthSet.GetThresholdLength()
                && mSoftDropCount
                    <= std::floor(static_cast<double>(mTotalTasksCount) * (1.0 - mTargetLatencyPercentile)) - 1)
            {
                holdedTasks.push_back(tasks.front());
                tasks.erase(tasks.begin());
                ++mSoftDropCount;
            }
        }
        NVTX_THREAD_START("infer:" + std::to_string(tasks.size()) + " tasks", COLOR_BLUE_0);
        bertCore->infer(tasks);
        NVTX_THREAD_END();

        NVTX_THREAD_START("GetTasks", COLOR_BLUE_1);
        tasks = GetTasks(mMaxBatchSize, qThreadIdx);
        NVTX_THREAD_END();
    }

    if (mSoftDrop < 1.0)
    {
        // Process soft drop tasks if any
        LOG(INFO) << "Total number of soft drop tasks: " << holdedTasks.size() << " out of " << totalCountInThread
                  << " total tasks";
        while (holdedTasks.size() != 0)
        {
            std::vector<std::pair<QuerySampleAndLength, std::chrono::high_resolution_clock::time_point>> tasks;
            tasks.reserve(mMaxBatchSize);
            // Consume up to mMaxBatchSize tasks
            for (int i = 0; (i < mMaxBatchSize) && !holdedTasks.empty(); ++i)
            {
                tasks.push_back(holdedTasks.back());
                holdedTasks.pop_back();
            }
            NVTX_THREAD_START("infer:" + std::to_string(tasks.size()) + " tasks", COLOR_BLUE_0);
            bertCore->infer(tasks);
            NVTX_THREAD_END();
        }
        // This is necessary to avoid a race condition if the bertCore is destructed
        // before we process all responses
        bertCore->WaitUntilQueueEmpty();
    }

    using CLK = std::chrono::high_resolution_clock;
    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << "End of ProcessTasks: "
        << std::chrono::duration_cast<std::chrono::microseconds>(CLK::now().time_since_epoch()).count() << std::endl;
}

static void createModelStreams(const std::string& enginePath, std::vector<std::vector<char>>& trtModelStreams)
{
    // we get a comma-separated list of engine paths
    std::vector<std::string> paths;
    int from = 0;
    int to;
    while ((to = enginePath.find(',', from)) != std::string::npos)
    {
        paths.emplace_back(enginePath.substr(from, to - from));
        from = to + 1;
    }

    if (from < enginePath.size())
    {
        paths.emplace_back(enginePath.substr(from, enginePath.size() - from));
    }

    for (auto& p : paths)
    {
        LOG(INFO) << "Engine Path: " << p;
    }

    trtModelStreams.resize(paths.size());
    for (size_t i = 0; i < trtModelStreams.size(); ++i)
    {
        lwis_lon::GetModelStream(trtModelStreams[i], paths[i]);
    }
}

void BERTServer::CreateEnginesPerGPU(
    int deviceId, std::shared_ptr<std::mutex> pMtx, const std::vector<std::vector<char>>& trtModelStreams)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));

    // load all the engines
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> inferObjects(trtModelStreams.size());
    std::transform(trtModelStreams.begin(), trtModelStreams.end(), inferObjects.begin(),
        [&](const std::vector<char>& trtModelStream) {
            return InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size(), nullptr));
        });

    {
        std::unique_lock<std::mutex> lck(*pMtx.get());
        mEnginesPerGPU[deviceId] = std::move(inferObjects);
    }
}

void BERTServer::ProcessSamples(int devId)
{
    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << "BERTServer::ProcessSamples GPU[" << devId << "] -- on CPU " << sched_getcpu() << std::endl;

    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        TIMER_START(m_WorkQueue_acquire_total);
        do
        {
            get_work_queue(devId)->acquire(
                samples, mServerSettings.Timeout, mMaxBatchSize, mServerSettings.EnableDequeLimit);
        } while (samples.empty());
        TIMER_END(m_WorkQueue_acquire_total);

        if (samples.front().id == UINT64_MAX || samples.front().index == UINT32_MAX)
            return;

        IssueQuery(samples, devId);
    }
}

void BERTServer::ProcessCompletions(int devId)
{
    DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
        << "BERTServer::ProcessCompletions GPU[" << devId << "] -- on CPU " << sched_getcpu() << std::endl;

    while (true)
    {
        std::deque<lwis_lon::Batch> resps;
        TIMER_START(m_CompletionUnifiedQueue_acquire_total);
        do
        {
            get_completion_unified_queue(devId)->acquire(
                resps, mServerSettings.Timeout, mMaxBatchSize, mServerSettings.EnableDequeLimit);
        } while (resps.empty());
        TIMER_END(m_CompletionUnifiedQueue_acquire_total);

        if (resps.front().QueuePairId == UINT32_MAX)
        {
            break;
        }

        for (auto& r : resps)
        {
            get_completion_queue(devId, r.QueuePairId)->push_back(r);
        }
    }
}

BERTServer::BERTServer(const std::string name, const std::string enginePath, const std::vector<int>& gpus,
    bool useGraphs, const std::string& graphSpecs, ServerSettings const sut_settings)
    : mName{name}
    , mGpusIndices{gpus}
    , mStopGetTasks{false}
    , mStopProcessResponse{false}
    , mMaxBatchSize{sut_settings.GPUBatchSize}
    , mNumBERTCores{sut_settings.GPUInferStreams}
    , mNumCopyStreams{sut_settings.GPUCopyStreams}
    , mGraphMaxSeqLen{sut_settings.GraphMaxSeqLen}
    , mSoftDrop{sut_settings.SoftDrop}
    , mTargetLatencyPercentile{sut_settings.SoftDropTargetLatencyPercentile}
    , mTotalLengthSet{mSoftDrop}
    , mTotalTasksCount{0}
    , mSoftDropCount{0}
    , mServerSettings{sut_settings}
{
    auto numGPUs = gpus.size();
    auto numQPs = sut_settings.NumIBQPsPerNIC;
    mDevWorkQueue.resize(numGPUs);
    mDevCompletionQueue.resize(numGPUs);
    mDevCompletionUnifiedQueue.resize(numGPUs);
    mDevResourceReturnQueue.resize(numGPUs);
    for (int idx = 0; idx < numGPUs; ++idx)
    {
        auto deviceId = gpus[idx];
        auto numa_node = GetNumaIdxByGpuId(deviceId);
        auto num_numa = GetNbNumas();

        if (UseNuma())
        {
            bindNumaMemPolicy(numa_node, num_numa);
        }

        mDevWorkQueue[deviceId] = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySample>>();
        mDevCompletionUnifiedQueue[deviceId] = std::make_shared<lwis_lon::SyncQueue<lwis_lon::Batch>>();
        mDevCompletionQueue[deviceId].resize(numQPs);
        for (int i = 0; i < numQPs; i++)
        {
            mDevCompletionQueue[deviceId][i] = std::make_shared<lwis_lon::SyncQueue<lwis_lon::Batch>>();
        }

        CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
        mDevResourceReturnQueue[deviceId].resize(mNumBERTCores);
        for (int profileIdx = 0; profileIdx < mNumBERTCores; ++profileIdx)
        {
            mDevResourceReturnQueue[deviceId][profileIdx]
                = std::make_shared<lwis_lon::SyncQueue<lwis_lon::CudaResource>>();
        }
        if (UseNuma())
        {
            resetNumaMemPolicy();
        }
    }

    {
        // only create one model streams
        std::vector<std::vector<char>> trtModelStreams;
        createModelStreams(enginePath, trtModelStreams);

        // create TRT engines in parallel
        std::shared_ptr<std::mutex> pMtx = std::make_shared<std::mutex>();
        std::vector<std::thread> engineCreationThreads;
        for (auto deviceId : gpus)
        {
            engineCreationThreads.emplace_back(&BERTServer::CreateEnginesPerGPU, this, deviceId, pMtx, trtModelStreams);
        }
        for (auto& thread : engineCreationThreads)
        {
            thread.join();
        }
        LOG(INFO) << "Engines Creation Completed";
    }

    if (useGraphs)
    {
        LOG(INFO) << "Use CUDA graphs";
    }

    // Create BERTCoreVS and store in vector, capture CUDA graphs in parallel
    mBERTCores.resize(numGPUs);
    for (int profileIdx = 0; profileIdx < mNumBERTCores; ++profileIdx)
    {
        std::vector<std::thread> cudaGraphsCapturingThreads;
        for (int idx = 0; idx < numGPUs; ++idx)
        {
            auto deviceId = gpus[idx];
            CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
            mBERTCores[deviceId].push_back(std::make_shared<BERTCoreVS>(mEnginesPerGPU.at(deviceId), mNumCopyStreams,
                mNumBERTCores, profileIdx, useGraphs, deviceId, mDevCompletionUnifiedQueue[deviceId],
                mDevResourceReturnQueue[deviceId][profileIdx], numQPs));

            // Capture CUDA graphs in parallel
            if (useGraphs)
            {
                cudaGraphsCapturingThreads.emplace_back(&BERTCoreVS::BuildGraphs,
                    mBERTCores[deviceId][profileIdx].get(), mGraphMaxSeqLen, graphSpecs, mNumBERTCores);
            }
        }
        if (useGraphs)
        {
            for (auto& thread : cudaGraphsCapturingThreads)
            {
                thread.join();
            }
        }
    }

    if (mSoftDrop < 1.0)
    {
        LOG(INFO) << "Apply soft drop policy with threshold = " << mSoftDrop;
    }

    mTasksVec.resize(numGPUs);
    mMtxs = std::make_unique<std::vector<std::mutex>>(numGPUs);
    mCondVars = std::make_unique<std::vector<std::condition_variable>>(numGPUs);

    // Warm up BERTCoreVS and launch threads for processing tasks
    int counter = 0;
    int qThreadIdx = 0;

    mWorkerThreads.reserve(numGPUs);
    mIssueThreads.reserve(numGPUs);
    mCompletionThreads.reserve(numGPUs);
    for (int idx = 0; idx < numGPUs; ++idx)
    {
        auto deviceId = gpus[idx];
        auto numa_node = GetNumaIdxByGpuId(deviceId);
        auto num_numa = GetNbNumas();

        if (UseNuma())
        {
            bindNumaMemPolicy(numa_node, num_numa);
        }

        CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
        for (int profileIdx = 0; profileIdx < mNumBERTCores; ++profileIdx)
        {
            auto bertCore = mBERTCores[deviceId][profileIdx];
            CHECK_EQ(mMaxBatchSize <= bertCore->GetMaxBatchSize(), true);

            mWorkerThreads.emplace_back(&BERTServer::ProcessTasks<BERTCoreVS>, this, bertCore, deviceId, qThreadIdx);

            ++counter;
            if (counter == mNumBERTCores)
            {
                ++qThreadIdx;
                counter = 0;
            }
        }

        mIssueThreads.emplace_back(std::thread(&BERTServer::ProcessSamples, this, deviceId));
        mCompletionThreads.emplace_back(std::thread(&BERTServer::ProcessCompletions, this, deviceId));

        if (UseNuma())
        {
            // CPU to map the thread
            auto cpus = GetClosestCpusToGpu(deviceId);
            bindThreadToCpus(mWorkerThreads.back(), cpus);
            bindThreadToCpus(mIssueThreads.back(), cpus);
            bindThreadToCpus(mCompletionThreads.back(), cpus);

            // and reset the mem policy
            resetNumaMemPolicy();
        }
    }
}

void BERTServer::Warmup(double duration)
{
    std::vector<std::thread> warmupThreads;
    for (auto& i : mBERTCores)
    {
        for (auto& j : i)
        {
            warmupThreads.emplace_back(std::thread(&BERTServer::ProcessWarmup, this, j, duration));
        }
    }
    for (auto& thread : warmupThreads)
    {
        thread.join();
    }
}

void BERTServer::ProcessWarmup(std::shared_ptr<BERTCoreVS> bc, double duration)
{
    bc->WarmUp(duration);
}

void BERTServer::Done()
{
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i)
        {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        mStopProcessResponse = true;
        for (int i = 0; i < mCondVars->size(); ++i)
        {
            (*mCondVars)[i].notify_all();
        }
    }
    // send terminate signal
    for (auto& gpu : mGpusIndices)
    {
        for (int bc = 0; bc < mNumBERTCores; ++bc)
        {
            get_resource_return_queue(gpu, bc)->push_back(lwis_lon::CudaResource{0, 0, 0, true});
        }

        while (!get_completion_unified_queue(gpu)->empty())
        {
        }
        lwis_lon::Batch termination_batch;
        termination_batch.QueuePairId = UINT32_MAX;
        get_completion_unified_queue(gpu)->emplace_back(termination_batch);

        get_work_queue(gpu)->emplace_back(mlperf::QuerySample{UINT64_MAX, UINT32_MAX});
    }
    for (auto& t : mCompletionThreads)
    {
        t.join();
    }
    for (auto& t : mIssueThreads)
    {
        t.join();
    }
    for (auto& t : mWorkerThreads)
    {
        t.join();
    }
}

const std::string& BERTServer::Name()
{
    return mName;
}

void BERTServer::IssueQuery(std::deque<mlperf::QuerySample> const& samples, int const gpu_idx)
{
    NVTX_MARK("IssueQuery:" + std::to_string(samples.size()) + " tasks", COLOR_BLUE_2);
    auto queryArrivedTime = std::chrono::high_resolution_clock::now();

    // Sort samples in the descending order of sentence length
    std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
    for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        auto TokenPtr = reinterpret_cast<void*>(samples[samplePos].index);
        auto sampleLen = static_cast<int>(GetSampleLength(TokenPtr, sizeof(BERTInput) * BERT_MAX_SEQ_LENGTH));
        sequenceSamplePosAndLength[samplePos] = std::make_pair(samplePos, sampleLen);
    }

    std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });

    for (int beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size(); beginSamplePos += mMaxBatchSize)
    {
        int actualBatchSize
            = std::min(mMaxBatchSize, static_cast<int>(sequenceSamplePosAndLength.size()) - beginSamplePos);
        static int totalBatchSize = 0;
        totalBatchSize += actualBatchSize;
        {
            std::unique_lock<std::mutex> lck((*mMtxs)[gpu_idx]);
            for (int i = 0; i < actualBatchSize; ++i)
            {
                int samplePosInOriginalRequest = sequenceSamplePosAndLength[beginSamplePos + i].first;
                auto sampleLength = sequenceSamplePosAndLength[beginSamplePos + i].second;
                QuerySampleAndLength sl{samples[samplePosInOriginalRequest], sampleLength};
                mTasksVec[gpu_idx].push_back({sl, queryArrivedTime});
            }

            // Let some worker thread to consume tasks
            (*mCondVars)[gpu_idx].notify_one();
        }
    }
}

void BERTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    // unused
    CHECK(false);
}

void BERTServer::FlushQueries()
{
    if (mSoftDrop < 1.0)
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i)
        {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        for (int i = 0; i < mCondVars->size(); ++i)
        {
            (*mCondVars)[i].notify_all();
        }
    }
}

BERTTask_t BERTServer::GetTasks(int maxSampleCount, int qThreadIdx)
{
    BERTTask_t res;
    res.reserve(maxSampleCount);
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck((*mMtxs)[qThreadIdx]);
    (*mCondVars)[qThreadIdx].wait(lck, [&] { return (!mTasksVec[qThreadIdx].empty()) || mStopGetTasks; });

    // Consume up to maxSampleCount tasks
    for (int i = 0; (i < maxSampleCount) && !mTasksVec[qThreadIdx].empty(); ++i)
    {
        res.push_back(mTasksVec[qThreadIdx].front());
        mTasksVec[qThreadIdx].pop_front();
    }

    // Let some other thread consume remaining tasks
    if (!mTasksVec[qThreadIdx].empty())
    {
        (*mCondVars)[qThreadIdx].notify_one();
    }

    return res;
}

int BERTServer::GetSampleLength(void* TokenAddr, int TokenSize)
{
    // Get sample length by checking where the input_mask change from nonzero to
    // zero
    int start{0};
    int end{BERT_MAX_SEQ_LENGTH};
    int cursor{(start + end) / 2};

    BERTInput* input_sample = static_cast<BERTInput*>(TokenAddr);
    while (cursor != start)
    {
        if ((*input_sample)[cursor])
        {
            start = cursor;
        }
        else
        {
            end = cursor;
        }
        cursor = (start + end) / 2;
    }
    return end;
}

}; // namespace bert_lon