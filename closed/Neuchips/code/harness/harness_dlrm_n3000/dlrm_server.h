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
 *
 * Modified by NEUCHIPS on 2023.
 *
 */

#pragma once

#include "config.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "dlrm_qsl.hpp"
#include "lwis_buffers.h"

#include "batch_maker.hpp"
#include <immintrin.h>

struct DLRMResult
{
    std::shared_ptr<std::vector<DLRMOutputType>> outputs;
    std::vector<DLRMTask> tasks;
    DLRMInferCallback callback;
    int batchSize;
};

using DLRMResultProcessingCallback = std::function<void(const DLRMResult& r)>;

using DLRMNumericPtrBuffer = DLRMDataBuffer<DLRMNumericInputType*>;
using DLRMCategoricalPtrBuffer = DLRMDataBuffer<DLRMCategoricalInputType*>;
using DLRMSampleSizeBuffer = DLRMDataBuffer<size_t>;

struct DLRMDeferredResult
{
    size_t batchSize;
    const DLRMOutputType* outputs;
    std::vector<DLRMTask> tasks;
    DLRMInferCallback callback;
    DLRMResultProcessingCallback resultCallback;
};

class DLRMResultHandlerPool {
 public:
    DLRMResultHandlerPool(size_t numThreads)
        : mStopWork(false) {
        LOG(INFO) << "DLRMResultHandlerPool  numThreads " << numThreads;
        for (int i = 0; i < numThreads; ++i) {
            DLOG(INFO) << "Creating DLRMResultHandlerPool::HandleResult thread "
                       << i;
            mThreads.emplace_back(&DLRMResultHandlerPool::HandleResult, this);
        }

        if (NUM_INF_PER_PIPE_STAGE == 192) {
            mTempInt8InputBuf =  (unsigned char*) malloc(
                    NUM_INF_PER_PIPE_STAGE * 1000 * 2);
            LOG(INFO) << "DLRMResultHandlerPool allocate  mTempInt8InputBuf "
                      << reinterpret_cast<void *>(mTempInt8InputBuf);
            mTempFp32OutputBuf = (float*) malloc(
                    NUM_INF_PER_PIPE_STAGE * 1000 * 2 * sizeof(float));
        } else {
            mTempInt8InputBuf =  (unsigned char*) malloc(
                    NUM_INF_PER_PIPE_STAGE * 750 * 2);
            LOG(INFO) << "DLRMResultHandlerPool allocate  mTempInt8InputBuf "
                    << reinterpret_cast<void *>(mTempInt8InputBuf);
            mTempFp32OutputBuf = (float*) malloc(
                    NUM_INF_PER_PIPE_STAGE * 750 * 2 * sizeof(float));
        }
    }

    ~DLRMResultHandlerPool() {
        {
            std::unique_lock<std::mutex> lock(mMtx);
            mStopWork = true;
            mCondVar.notify_all();
        }
        for (auto& t : mThreads) {
            t.join();
        }

        delete mTempInt8InputBuf;
        delete mTempFp32OutputBuf;
    }

    void Enqueue(const DLRMResult& r) {
        static unsigned int enqueued_tasks = 0;
        {
            std::unique_lock<std::mutex> lock(mMtx);
            enqueued_tasks += r.tasks.size();
            mResultQ.emplace_back(r);
            mCondVar.notify_one();
        }
    }

    inline void ConvertFromUint8_AVX2(float* mO, unsigned char* mI,
                                size_t numElements, float scaling) {
        size_t ii;
        __m256 vscaling, tmp;

        vscaling = _mm256_set1_ps(scaling);

        for (ii = 0; ii < numElements && ((uintptr_t)(mO + ii) & 31); ii++) {
            mO[ii] = (float)(mI[ii]) * scaling;
        }
        // main loop
        if (numElements >= 32) {
            for (; ii < numElements - 31; ii += 32) {
                tmp = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(mI + ii))));
                _mm256_store_ps(mO + ii, _mm256_mul_ps(tmp, vscaling));
                tmp = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(mI + ii + 8))));
                _mm256_store_ps(mO + ii + 8, _mm256_mul_ps(tmp, vscaling));
                tmp = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(mI + ii + 16))));
                _mm256_store_ps(mO + ii + 16, _mm256_mul_ps(tmp, vscaling));
                tmp = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(mI + ii + 24))));
                _mm256_store_ps(mO + ii + 24, _mm256_mul_ps(tmp, vscaling));
            }
        }
        // epilogue
        for (; ii < numElements; ii++) {
            mO[ii] = (float)(mI[ii]) * scaling;
        }
    }

    void HandleResult() {
        bool freshStart = true;

        while (true) {
            DLRMResult res;
            {
                std::unique_lock<std::mutex> lock(mMtx);
                mCondVar.wait(lock,
                    [&]() { return (!mResultQ.empty()) || mStopWork; });

                if (mStopWork) {
                    break;
                }

                res = mResultQ.front();
                mResultQ.pop_front();
                mCondVar.notify_one();
            }

            std::vector<mlperf::QuerySampleResponse> responses;
            int offset = 0;
            int valid_len;

            offset = 0;
            valid_len = N3000_MAX_BATCH_SIZE;

            ConvertFromUint8_AVX2(mTempFp32OutputBuf,
                    reinterpret_cast<unsigned char*>(
                    &res.outputs->at(0)), valid_len, m_output_q);

            int count = 0;
            for (const auto& task : res.tasks) {
                DLOG(INFO) << "pop result current offset " << offset
                        << " task.dbgSeq " << task.dbgSeq
                        << " qsi " << task.querySample.index
                        << " skipSamples " << task.skipSamples
                        << " numIndividualPairs " << task.numIndividualPairs;
                if (task.numIndividualPairs > 1000) {
                    DLOG(INFO) << "#HandleResult mResultLeftTasks dbgSeq "
                               << task.dbgSeq << " IGNORE.";
                    break;
                } else {
                    if (offset + task.numIndividualPairs <= valid_len) {
                        DLOG(INFO) << "response* mResultLeftTasks offset "
                        << offset << " task.dbgSeq " <<  task.dbgSeq
                        << " qsi " << task.querySample.index
                        << " skipSamples " << task.skipSamples
                        << " numIndividualPairs " << task.numIndividualPairs;

                        mlperf::QuerySampleResponse response{
                            task.querySample.id,
                            (uintptr_t)(mTempFp32OutputBuf + offset),
                            sizeof(float) * task.numIndividualPairs};

                        responses.emplace_back(response);

                        offset = offset + task.numIndividualPairs;
                        if (task.skipSamples != 0) {
                            DLOG(INFO) << "skipSample != 0 prevRes task.dbgSeq "
                                << task.dbgSeq << " skipSamples "
                                << task.skipSamples << " numIndividualPairs "
                                << task.numIndividualPairs;
                            offset = offset + task.skipSamples;
                        }
                    }
                }
            }
            res.callback(responses);
        }
    }

 private:
    std::vector<std::thread> mThreads;
    std::deque<DLRMResult> mResultQ;
    /* N3000_DLRM output handling */
    unsigned char* mTempInt8InputBuf;
    float* mTempFp32OutputBuf;
    size_t mTempInt8InputIdx;
    const float m_output_q = (float)1 / (float)(1 << 8);
    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork;
};

class DLRMEventBufferBundle
{
 public:
    DLRMEventBufferBundle(size_t bundleIdx, size_t numInVol,
                          size_t catInVol, size_t outVol, size_t maxBatchSize)
        : idx(bundleIdx)
        , numericInputBuf(numInVol, maxBatchSize, false)
        , categoricalInputBuf(catInVol, maxBatchSize, false)
        , outputBuf(outVol, maxBatchSize, true)
        , outputBufReFormat(outVol, maxBatchSize, true)
        , numericInputPtrBuf(1, maxBatchSize, true)
        , categoricalInputPtrBuf(1, maxBatchSize, true)
        , sampleSizesBuf(1, maxBatchSize, true)
        , sampleOffsetsBuf(1, maxBatchSize, true) {
    }

    size_t idx;
    DLRMNumericBuffer numericInputBuf;
    DLRMCategoricalBuffer categoricalInputBuf;
    DLRMOutputBuffer outputBuf;
    DLRMOutputBuffer outputBufReFormat;

    // The next 4 buffers are used only in start_from_device mode
    DLRMNumericPtrBuffer numericInputPtrBuf;
    DLRMCategoricalPtrBuffer categoricalInputPtrBuf;
    DLRMSampleSizeBuffer sampleSizesBuf;
    DLRMSampleSizeBuffer sampleOffsetsBuf;

};

class DLRMCore
{
 public:
    DLRMCore(int maxBatchSize, int numBundles, int numCompleteThreads,
        int profileIdx);
    ~DLRMCore();
    void infer(std::shared_ptr<DLRMEventBufferBundle> ebBundle,
               size_t batchSize,
               std::vector<DLRMTask>& tasks,
               Batch* batch, void (*h2dCallBack)(void*),
               DLRMInferCallback resultCallback,
               DLRMNumericInputType* numericInputPtr,
               DLRMCategoricalInputType* categoricalInputPtr,
               int fdN3000, int deviceId, int profileIdx);

    void inferFromDevice(std::shared_ptr<DLRMEventBufferBundle> ebBundle,
                         size_t batchSize,
                         std::vector<DLRMTask>& tasks,
                         DLRMInferCallback resultCallback);

    size_t GetMaxBatchSize() {
        return mMaxBatchSize;
    };

    size_t mNumInVol;
    size_t mCatInVol;
    size_t mOutVol;

    void* mN3000PingOutAddr;
    void* mN3000PongOutAddr;
    int ncs_setup_output_ping_pong_addr(u32 device_id, int fd);
    int CalculateStages(int batch_size);

    std::shared_ptr<DLRMEventBufferBundle> NextForegroundBundle();

    inline void h2d_renew_sgl_init(
        struct ncs_arg_h2d_renew_sgl* arg_h2d_renew_sgl,
        u32 chan_idx, Batch* batch, bool useDummyIndex,
         size_t batchSize);

    std::shared_ptr<DLRMEventBufferBundle> getBundleByIdx(size_t idx);
    int getStageNo() {
        int ret;
        mMtxStageNo.lock();
        ret = mStageNo;
        mMtxStageNo.unlock();
        return ret;
    }

    void increaseStageNo() {
        mMtxStageNo.lock();
        mStageNo++;
        mMtxStageNo.unlock();
    }

    void resetStageNo() {
        DLOG(INFO) << "resetStageNo().";
        mMtxStageNo.lock();
        mStageNo = 0;
        mMtxStageNo.unlock();
    }

    bool is_the_other_thread_infering(int cur_pingpong) {
        int the_other;
        int is_infering;
        if (cur_pingpong == 0) {
            the_other = 1;
        } else {
            the_other = 0;
        }
        mMtxStageNo.lock();
        is_infering = mInferThreadRunning[the_other];
        mMtxStageNo.unlock();
        return is_infering;
    }

    u32 get_1st_sample_size() {
        u32 mysize;
        size_t n;
        if (m_1st_sample_size == 0) {
            FILE *fp;
            fp = fopen("mlperf_sample.bin", "rb");
            if (fp) {
                n = fread(&mysize, sizeof(mysize), 1, fp);
                DLOG(INFO) << "dlrm_server::get_1st_sample_size " << mysize;
                fclose(fp);

                m_1st_sample_size = mysize;
            }
        }

        return m_1st_sample_size;
    }

    void set_current_thread_infering(int cur_pingpong, bool is_infering) {
        mMtxStageNo.lock();
        mInferThreadRunning[cur_pingpong] = is_infering;
        mMtxStageNo.unlock();
    }

    void set_1st_sample_size(u32 sample_size) {
        if (sample_size > 0) {
            m_1st_sample_size = sample_size;
        }
    }

 private:
    void SetBatchSize(int batchSize);
    size_t mMaxBatchSize;

    DLRMResultHandlerPool mResHandlerPool;
    std::vector<std::shared_ptr<DLRMEventBufferBundle>> mEventBufferBundle;
    size_t mBundleCounter;

    std::vector<std::vector<void*>> mBindings;

    bool mNewStartPingPong;
    int mStageNo;
    std::mutex mMtxStageNo;
    bool mInferThreadRunning[2];
    u32 m_1st_sample_size;
};

class DLRMServer : public mlperf::SystemUnderTest {
 public:
    DLRMServer(const std::string name, const std::string enginePath,
               std::vector<DLRMSampleLibraryPtr_t> qsls,
               const std::vector<int>& gpus, int maxBatchSize,
               int numBundles, int numCompleteThreads, int numDLRMCores,
               double warmupDuration, int numStagingThreads,
               int numStagingBatches, int maxPairsPerThread,
               int splitThreshold, bool checkContiguity, bool startFromDevice,
               NumaConfig numaConfig, const std::string scenario,
               int num_issue_query_threads);

    virtual ~DLRMServer();
    const std::string& Name() override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;

    void StartIssueThread(int threadIdx);

    bool UseNuma() {
        return !mNumaConfig.empty();
    }

 private:
    void IssueQueryOffline(const std::vector<mlperf::QuerySample>& samples);
    void IssueQueryServer(const std::vector<mlperf::QuerySample>& samples);
    void ProcessTasks(std::shared_ptr<DLRMCore>, int deviceId, int profileIdx);
    void ProcessTasksFromDevice(std::shared_ptr<DLRMCore>,
                                    int deviceId, int profileIdx);
    void SetupDevice(const std::string enginePath, int numBundles,
                     int numCompleteThreads, int numDLRMCores,
                     int warmupDuration, int deviceId);

    std::vector<DLRMTask> GetBatch();
    std::vector<std::shared_ptr<BatchMaker>> mBatchMakers;

    const std::string mName;
    int mMaxBatchSize;
    std::vector<DLRMSampleLibraryPtr_t> mQsls;
    bool mStartFromDevice;

    // Queue to be used for the start_from_device case, each sample is accompanied by the pair count
    std::deque<DLRMTask> mTasks;

    // mutex to serialize access to mTasks member variable
    std::mutex mMtx;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::condition_variable mCondVar;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopWork;

    // if sample size is smaller than this threshold, don't do round-robin on batchMakers
    const int64_t mSplitThreshold;

    std::vector<std::thread> mWorkerThreads;
    std::vector<std::shared_ptr<DLRMCore>> mDLRMCores;
    size_t mNumInVol;
    size_t mCatInVol;

    // NUMA configs of the machine: list of CPUs for each NUMA node, assuming each GPU corresponds to one NUMA node.
    NumaConfig mNumaConfig;
    GpuToNumaMap mGpuToNumaMap;
    // When NUMA is used, we issue queries around NUMA nodes in round-robin.
    int mPrevBatchMakerIdx{-1};

    // data members to support multiple IssueQuery() threads if server_num_issue_query_threads != 0
    std::mutex mMtxIssueQuery;
    std::map<std::thread::id, int> mThreadMap;
    std::vector<std::thread> mIssueQueryThreads;
    bool mServerMode;
};
