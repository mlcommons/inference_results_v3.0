#pragma once
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <glog/logging.h>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include "system_under_test.h"
#include "dlrm_qsl.hpp"

struct DLRMTask {
    mlperf::QuerySample sample_;
    size_t numSample_;
};

struct DLRMBatch {
    std::shared_ptr<std::list<DLRMTask>> tasks_;
    size_t batchSize_;
};

using DLRMOutDtype = float;

struct DLRMResult {
    std::shared_ptr<std::vector<DLRMOutDtype>> outputs_;
    std::shared_ptr<std::list<DLRMTask>> tasks_;
};

class DLRMResultHandler {
public:
    DLRMResultHandler(int numThreads);
    ~DLRMResultHandler();
    void Enqueue(const DLRMResult &r);
    void HandleResult(int i);
    void StopWork();
private:
    const int numThreads_;
    std::vector<std::thread> threads_;
    std::deque<DLRMResult> resultQ_;
    std::mutex mtx_;
    std::condition_variable condVar_;
    bool stopWork_;
};

class DLRMBatchMaker {
public:
    DLRMBatchMaker(int maxBatchSize,
                   DLRMSampleLibraryPtr_t &qsl);

    ~DLRMBatchMaker();

    DLRMBatch GetTask(int i);

    void ServerEnqueue(const std::vector<mlperf::QuerySample> &samples);

    void OfflineEnqueue(const mlperf::QuerySample &sample);

    void StopWork();

    void ServerFlush();

    size_t TotalQueriesLeft();

    size_t TotalSampleLeft();
private:
    void ServerIssue();

    // query sample library
    DLRMSampleLibraryPtr_t &qsl_;
    // batch size
    int batchSize_;
    // mutex to protect batch maker
    std::mutex mtx_;
    // dlrm batch queue
    std::deque<DLRMBatch> dlrmBatch_;
    // dlrm staging tasks
    std::list<DLRMTask> dlrmTasks_;
    // cumulative num of samples in queue
    size_t queueSize_;
    //conditional variable on which thread will wait to produce batches
    std::condition_variable condVar_;
    //control the status of thread
    bool stopWork_;
};
