#include <cassert>
#include "dlrm_proc.hpp"
#include "loadgen.h"
#include "util.hpp"

DLRMResultHandler::DLRMResultHandler(int numThreads)
    :numThreads_(numThreads),
     stopWork_(false) {
    for (int i = 0; i < numThreads; ++i) {
        threads_.emplace_back(&DLRMResultHandler::HandleResult, this, i);
    }
}

DLRMResultHandler::~DLRMResultHandler() {
    {
        std::unique_lock<std::mutex> lock(mtx_);
        stopWork_ = true;
        condVar_.notify_all();
    }
    for (auto &t : threads_) {
        t.join();
    }
}

void DLRMResultHandler::Enqueue(const DLRMResult &r) {
    std::unique_lock<std::mutex> lock(mtx_);
    VLOG(5) << "Enqueue";
    resultQ_.emplace_back(r);
    condVar_.notify_one();
}

void DLRMResultHandler::HandleResult(int i) {
    bindThreadToCpus(0, 1);
    bindNumaMemPolicy(0, 2);
    VLOG(1) << "DLRMResultHandler " << i << " start";
    while (true) {
        DLRMResult res;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            condVar_.wait(lock,
                          [&](){return (!resultQ_.empty() | stopWork_);
                          });
            if (stopWork_)
                break;
            if (resultQ_.empty())
                continue;
            res = resultQ_.front();
            resultQ_.pop_front();
            condVar_.notify_one();
        }
        CHECK_GT(res.tasks_->size(), 0);
        std::vector<mlperf::QuerySampleResponse> responses;
        int offset = 0;
        for (auto &task : *res.tasks_) {
            mlperf::QuerySampleResponse r{task.sample_.id,
                reinterpret_cast<uintptr_t>(&res.outputs_->at(offset)),
                sizeof(DLRMOutDtype) * task.numSample_};
            responses.emplace_back(r);
            offset = offset + task.numSample_;
        }
        VLOG(5) << "DLRMResultHandler " << i << " processing";
        // notify loadgen that sample is done
        mlperf::QuerySamplesComplete(responses.data(), responses.size());
        VLOG(2) << "Handled " << offset << " pairs";
    }
    VLOG(1) << "DLRMResultHandler " << i << " end";
    resetNumaMemPolicy();
}

void DLRMResultHandler::StopWork() {
    std::unique_lock<std::mutex> lock(mtx_);
    stopWork_ = true;
    condVar_.notify_all();
}

DLRMBatchMaker::DLRMBatchMaker(int maxBatchSize,
                               DLRMSampleLibraryPtr_t &qsl)
    :qsl_(qsl),
     batchSize_(maxBatchSize),
     stopWork_(false),
     queueSize_(0) {
}

DLRMBatchMaker::~DLRMBatchMaker() {
}

void DLRMBatchMaker::ServerIssue() {
    dlrmBatch_.push_back({
            std::make_shared<std::list<DLRMTask>>(this->dlrmTasks_),
            this->queueSize_});
    this->dlrmTasks_.clear();
    this->queueSize_ = 0;
    VLOG(5) << "Server Issue";
}

void DLRMBatchMaker::ServerEnqueue(const std::vector<mlperf::QuerySample> &samples) {
    std::unique_lock<std::mutex> lock(mtx_);
    for (int i = 0; i < samples.size(); ++i) {
        size_t n = qsl_->GetNumUserItemPairs(samples[i].index);
        dlrmTasks_.push_back({samples[i], n});
        queueSize_ += n;
        if (batchSize_ <= queueSize_)
            ServerIssue();
    }
    condVar_.notify_one();
}

void DLRMBatchMaker::OfflineEnqueue(const mlperf::QuerySample &sample) {
    std::unique_lock<std::mutex> lock(mtx_);
    size_t n = qsl_->GetNumUserItemPairs(sample.index);
    dlrmTasks_.push_back({sample, n});
    queueSize_ += n;
    if (batchSize_ <= queueSize_)
        ServerIssue();
    condVar_.notify_one();
}

void DLRMBatchMaker::ServerFlush() {
    if (queueSize_ > 0)
        ServerIssue();
}

DLRMBatch DLRMBatchMaker::GetTask(int i) {
    std::unique_lock<std::mutex> lock(mtx_);
    std::vector<DLRMTask> res;
    condVar_.wait(lock,
                  [&]{
                      return (!dlrmBatch_.empty() || stopWork_);
                  });
    if (stopWork_)
        return DLRMBatch();
    CHECK_EQ(dlrmBatch_.empty(), false);
    VLOG(5) << "GetTask start " << i;
    auto b = dlrmBatch_.front();
    dlrmBatch_.pop_front();
    if (!dlrmBatch_.empty())
        condVar_.notify_one();
    VLOG(5) << "GetTask done " << i;
    return b;
}

void DLRMBatchMaker::StopWork() {
    std::unique_lock<std::mutex> lock(mtx_);
    stopWork_ = true;
    condVar_.notify_all();
}

size_t DLRMBatchMaker::TotalQueriesLeft() {
    size_t sum = dlrmBatch_.size();
    for (auto &b : dlrmBatch_) {
        sum += b.tasks_->size();
    }
    return sum;
}

size_t DLRMBatchMaker::TotalSampleLeft() {
    size_t sum = queueSize_;
    for (auto &b : dlrmBatch_)
        sum += b.batchSize_;
    return sum;
}
