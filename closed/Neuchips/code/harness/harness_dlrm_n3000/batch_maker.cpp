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

#include <csignal>
#include "batch_maker.hpp"
#include "utils.hpp"

Batch::Batch(std::string id, size_t minBatchSize, size_t maxBatchSize, size_t numericVolume, size_t categoricalVolume,
    BatchMaker* batchMaker)
    : mDebugId(id)
    , mBatchMaker(batchMaker)
    , mMinBatchSize(minBatchSize)
    , mMaxBatchSize(maxBatchSize)
    , mNumericInputBuf(numericVolume, maxBatchSize, true, false)
    , mCategoricalInputBuf(categoricalVolume, maxBatchSize, true, false)
    , mH2DIndex(0)
    , mH2DIndicesMaxSize(mMaxBatchSize)
{
    DLOG(INFO) << "Batch::Batch constructor. mMinBatchSize:" << mMinBatchSize
               << " mMaxBatchSize:" << mMaxBatchSize;

    u32  mH2DIndexSize = mH2DIndicesMaxSize/100;
    mH2DIndices = new u32[mH2DIndexSize];
    DLOG(INFO) << "Batch::Batch allocate mH2DIndices " << mH2DIndexSize
               << " bytes.";

    reset();
}

void Batch::IndividualH2H(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy)
{
    auto qsl = mBatchMaker->mQsl;
    size_t copiedPairs = 0;

    while (copiedPairs < pairsToCopy) {
        const auto& task = mTasks[tasksOffset++];
        auto qslIdx = task.querySample.index;
        auto numIndividualPairs = task.numIndividualPairs;

        unsigned long my_index =
            (unsigned long)(qsl->GetSampleAddress(qslIdx, 0));
        if(my_index > 204799) {
            LOG(INFO) << "ERROR: IndividualH2H my_index > 204799, " << my_index;
        }

        pushH2DIndex(my_index);
        copiedPairs += numIndividualPairs;
    }
}

void Batch::doCopy(size_t tasksOffset, size_t pairsOffset, size_t pairsToCopy)
{
    DLOG(INFO) << "Batch::doCopy tasksOffset " << tasksOffset <<" pairsOffset "
                 << pairsOffset << " pairsToCopy " << pairsToCopy;
    DLOG(INFO) << mDebugId << " doCopy : " << pairsToCopy;
    IndividualH2H(tasksOffset, pairsOffset, pairsToCopy);
}

void Batch::reset()
{
    mCommittedCopies = 0;
    mCompletedCopies = 0;
    mReadyWhenComplete = false;
    mOddBatch = false;
    mTasks.clear();
    mTasks.reserve(mMaxBatchSize);
    mNumericInputBuf.ResetHostPtr();
    mCategoricalInputBuf.ResetHostPtr();

    clearH2DIndices();
}

void Batch::padToEvenSize()
{
    size_t actualBatchSize = mCommittedCopies;
    mOddBatch = (actualBatchSize % 2) == 1;
}

BatchMaker::BatchMaker(size_t numStagingThreads, size_t numStagingBatches, 
    size_t maxBatchSize, size_t maxPairsPerThread, size_t numericVolume,
    size_t categoricalVolume, bool checkContiguity,
    std::shared_ptr<DLRMSampleLibrary> qsl, int64_t numaIdx, int64_t numaNum,
    std::vector<int> cpus, const std::string scenario)
    : mMaxBatchSize(maxBatchSize)
    , mMaxPairsPerThread(maxPairsPerThread)
    , mCheckContiguity(checkContiguity)
    , mQsl(qsl)
    , mStagingBatch(nullptr)
    , mIssuedPairs(0)
    , mReadiedPairs(0)
    , mFlushQueries(false)
    , mStopWork(false)
    , mWaitingBatches(0)
    , mNumaIdx(numaIdx)
    , mNumaNum(numaNum)
    , mCpus(cpus) {
    if (UseNuma()) {
        bindNumaMemPolicy(mNumaIdx, NUM_NUMA_ZONES);
    }

    LOG(INFO) << "BatchMaker - numStagingThreads = " << numStagingThreads;
    LOG(INFO) << "BatchMaker - numStagingBatches = " << numStagingBatches;
    LOG(INFO) << "BatchMaker - maxPairsPerThread = " << maxPairsPerThread;
    LOG(INFO) << "BatchMaker - numericVolume     = " << numericVolume
                        << " categoricalVolume = " << categoricalVolume;
    LOG(INFO) << "BatchMaker - mNumaIdx = " << mNumaIdx
                        << " mNumaNum " << mNumaNum;

    if (mCheckContiguity) {
        LOG(INFO) << "Contiguity-Aware H2H : ON";
        if (mMaxPairsPerThread == 0) {
            mMaxPairsPerThread = mMaxBatchSize;
        }
    } else {
        LOG(INFO) << "BatchMaker::BatchMaker Contiguity-Aware H2H : OFF";
        if (mMaxPairsPerThread == 0) {
            mMaxPairsPerThread = mMaxBatchSize / numStagingThreads;
            LOG(INFO) << "BatchMaker::BatchMaker mMaxPairsPerThread " <<
                    mMaxPairsPerThread << "= (mMaxBatchSize " << mMaxBatchSize
                    << " / numStagingThreads)" << numStagingThreads;
        } else {
            LOG(INFO) << "BatchMaker::BatchMaker mMaxPairsPerThread "
                        << mMaxPairsPerThread;
        }
    }

    LOG(INFO) << "mInputBatches.reserve " << numStagingBatches
                << " numStagingBatches";
    mInputBatches.reserve(numStagingBatches);
    for (int i = 0; i < numStagingBatches; ++i) {
        mInputBatches.emplace_back("Batch#" + std::to_string(i),
                1, mMaxBatchSize, numericVolume, categoricalVolume, this);
    }

    // All batches idle initially
    mIdleBatches.reserve(numStagingBatches);
    for (auto& batch : mInputBatches) {
        mIdleBatches.push_back(&batch);
    }

    // start StageBatchthreads
    LOG(INFO) << "start StageBatchthreads numStagingThreads:  "
                << numStagingThreads;
    mStagingThreads.reserve(numStagingThreads);
    for (int i = 0; i < numStagingThreads; ++i) {
        LOG(INFO) << "mStagingThreads.emplace_back() StageBatch " << i;
        mStagingThreads.emplace_back(&BatchMaker::StageBatch, this, i);
    }

    // Limit the staging threads to the closest CPUs.
    if (UseNuma()) {
        LOG(INFO) << "UseNuma bindThreadToCpus";
        for (auto& stagingThread : mStagingThreads) {
            bindThreadToCpus(stagingThread, mCpus);
        }
    }

    if (UseNuma()) {
        LOG(INFO) << "UseNuma resetNumaMemPolicy";
        resetNumaMemPolicy();
    }

    if (0 == scenario.compare("Server")) {
        LOG(INFO) << "BatchMaker scenario is Server.";
        mServerMode = true;
    } else {
        mServerMode = false;
    }
}

BatchMaker::~BatchMaker()
{
    DLOG(INFO) << "~BatchMaker";
    StopWork();
    for (auto& stagingThread : mStagingThreads) {
        stagingThread.join();
    }
}

inline bool BatchMaker::isServerMode()
{
    return mServerMode;
}

void BatchMaker::IssueQuery(const std::vector<mlperf::QuerySample>& samples,
                                        int offset, int count)
{
    mDbgSeq = offset;
    DLOG(INFO) << "BatchMaker(" << mNumaIdx << ")IssueQuery: "
                    << " offset " << offset << " , count " << count;
    mFlushQueries = false;
    for (size_t i = 0; i < count; ++i) {
        {
            const auto& sample = samples[offset + i];
            std::unique_lock<std::mutex> lock(mMutex);

            size_t numPairs = mQsl->GetNumUserItemPairs(sample.index);
            mTasksQ.push_back({sample, numPairs, 0, mDbgSeq});
            mDbgSeq++;

            mIssuedPairs += numPairs;
            if (!isServerMode()) {
                if (i == (count -1)) {
                    LOG(INFO) <<
                    "IssueQuery 1 dummy task for last stage. dbgSeq "<< (i+1);
                    mTasksQ.push_back({sample, N3000_MAX_BATCH_SIZE,
                                        N3000_MAX_BATCH_SIZE, 99999});
                    mIssuedPairs += N3000_MAX_BATCH_SIZE;
                }
            }
        }
    }
    mProducerCV.notify_one();
    DLOG(INFO) << "BatchMaker::IssueQuery- " << " offset " << offset
                    << " , count " << count;
}

void BatchMaker::FlushQueries()
{
    std::unique_lock<std::mutex> mMutex;

    if (isServerMode()) {
        LOG(INFO) << "*FlushQuries 1 dummy task for last stage. dbgSeq 99999";

        mlperf::QuerySample dummySample;
        dummySample.index = 0;
        mTasksQ.push_back({dummySample, N3000_MAX_BATCH_SIZE,
                                N3000_MAX_BATCH_SIZE, 99999});
        mIssuedPairs += N3000_MAX_BATCH_SIZE;
    }

    LOG(INFO) << "BatchMaker::FlushQuries mTasksQ.empty() " << mTasksQ.empty();
    if (mStagingBatch) {
        DLOG(INFO) << "FlushQuries mStagingBatch->getCommittedCopies() "
                    << mStagingBatch->getCommittedCopies();
    }

    if (mTasksQ.empty() && mStagingBatch &&
                    mStagingBatch->getCommittedCopies() > 0) {
        // close this buffer on my own,
        // since no thread will do it until new tasks arrive
        CloseBatch(mStagingBatch);
    } else {
        // no CloseBatch
        mProducerCV.notify_one();
    }

    mFlushQueries = true;
}

void BatchMaker::StopWork()
{
    DLOG(INFO) << "Stop Work";
    std::unique_lock<std::mutex> lock(mMutex);
    mStopWork = true;
    mProducerCV.notify_all();
    mConsumerCV.notify_all();
}

Batch* BatchMaker::GetBatch()
{
    std::unique_lock<std::mutex> lock(mMutex);

    if (mReadyBatches.empty()) {
        if (mStagingBatch) {
            CloseBatch(mStagingBatch);
        }
        ++mWaitingBatches;
        mConsumerCV.wait(lock,
            [&] { return !mReadyBatches.empty() || mStopWork; });
        --mWaitingBatches;
    }

    if (mStopWork) {
        DLOG(INFO) << "GetBatch Done";
        return nullptr;
    }

    Batch* readyBatch = mReadyBatches.front();
    mReadyBatches.pop_front();
    auto completedCopies = readyBatch->getCompletedCopies();
    mReadiedPairs += readyBatch->isOddBatch()
                        ? (completedCopies - 1) : completedCopies;

    CHECK_LE(mReadiedPairs, mIssuedPairs);

    if (!mReadyBatches.empty()) {
        mConsumerCV.notify_one();
    }

    return readyBatch;
}

void BatchMaker::HandleReadyBatch(Batch* batch)
{
    DLOG(INFO) << "Ready : " << batch->mDebugId
                << " " << batch->getCompletedCopies();

    auto actualBatchSize = batch->getCompletedCopies();
    CHECK(batch->isComplete());
    CHECK_GT(actualBatchSize, 0);

    batch->padToEvenSize();

    mReadyBatches.push_back(batch);
    mConsumerCV.notify_one();
}

void BatchMaker::NotifyH2D(Batch* batch)
{
    DLOG(INFO) << "Notify " << batch->mDebugId << "; flush = " << mFlushQueries;
    batch->reset();

    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIdleBatches.push_back(batch);
    }

    mProducerCV.notify_one();
}

void BatchMaker::StageBatch(int threadIdx)
{
    char temp[256];
    sprintf(temp, "StageBatch thread%i", threadIdx);

    if (UseNuma()) {
        bindNumaMemPolicy(mNumaIdx, mNumaNum);
    }

    LOG(INFO) << "StageBatch(" << mNumaIdx << "/" << threadIdx << ")";

    while (true) {
        Batch* batchPtr;
        std::vector<DLRMTask> committedTasks;
        size_t pairsToCopy = 0;
        size_t tasksOffset = 0;
        size_t pairsOffset = 0;

        // de-queue tasks and commit copies, under lock
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mProducerCV.wait(
                lock,
                [&] {
                    return (
                        (mStagingBatch || !mIdleBatches.empty())
                        && !mTasksQ.empty()) || mStopWork;
                    });

            if (mStopWork) {
                DLOG(INFO) << "BatchMaker::StageBatch mStopWork";
                return;
            }

            if (!mStagingBatch) {
                CHECK(!mIdleBatches.empty());
                mStagingBatch = mIdleBatches.back();
                mIdleBatches.pop_back();
            }

            // mStagingBatch may change once lock is released, so store a copy
            batchPtr = mStagingBatch;
            tasksOffset = batchPtr->getTasks().size();
            pairsOffset = batchPtr->getCommittedCopies();
            size_t maxPairs = std::min(mMaxPairsPerThread,
                                batchPtr->getFreeSpace());

            while (!mTasksQ.empty()) {
                auto task = mTasksQ.front();
                size_t numPairs = task.numIndividualPairs;

                DLOG(INFO) << "StageBatch(" << mNumaIdx << "/" << threadIdx
                           << ") dbgSeq " << task.dbgSeq << " pairsToCopy "
                           << pairsToCopy << " numPairs " << numPairs
                           << " maxPairs " << maxPairs;

                if (numPairs + pairsToCopy > maxPairs) {
                    if (numPairs >= batchPtr->getFreeSpace()) {
                        // batch can't fit next sample
                        CloseBatch(batchPtr);
                    } else {
                        // DLOG(INFO) << pairsToCopy
                        //            << ":Break because maxPairs";
                    }
                    // Let some other thread commit remaining tasks
                    break;
                }

                pairsToCopy += numPairs;
                batchPtr->commitCopies(numPairs);
                batchPtr->pushTask(task);
                mTasksQ.pop_front();
            }

            if (pairsToCopy == 0) {
                continue;
            }

            if ((mTasksQ.empty() && mFlushQueries) /*|| mWaitingBatches*/) {
                DLOG(INFO) << "StageBatch() *CloseBatch mTasksQ.empty() "
                            << mTasksQ.empty() << " mFlushQueries "
                            << mFlushQueries << " mWaitingBatches "
                            << mWaitingBatches << " batch tasks sz "
                            << batchPtr->getTasks().size();
                // no more queries
                CloseBatch(batchPtr);
            }

            mProducerCV.notify_one();
        }

        batchPtr->doCopy(tasksOffset, pairsOffset, pairsToCopy);
        // mark copy as complete, under lock
        {
            std::unique_lock<std::mutex> lock(mMutex);
            batchPtr->completeCopies(pairsToCopy);
            ReadyBatchIfComplete(batchPtr);
            mConsumerCV.notify_one();
        }
    }

    if (UseNuma()) {
        resetNumaMemPolicy();
    }
}

void BatchMaker::CloseBatch(Batch* batch) {
    if (!batch->isReadyWhenComplete()) {
        DLOG(INFO) << batch->mDebugId << " closing";
        // batch will move to mReadyBatches once copies complete
        batch->markReadyWhenComplete();
        // if already complete, move to ready
        ReadyBatchIfComplete(batch);
        // next thread will set new staging batch
        mStagingBatch = nullptr;
    }
}

void BatchMaker::ReadyBatchIfComplete(Batch* batch)
{
    if (batch->isComplete()) {
        DLOG(INFO) << "BatchMaker::ReadyBatchIfComplete batch->isComplete";
        HandleReadyBatch(batch);
    }
}
