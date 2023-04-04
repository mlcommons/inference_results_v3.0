#pragma once
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "dlrm_core.hpp"
#include "dlrm_qsl.hpp"
#include "system_under_test.h"
#include "cnpy.h"
#include "dlrm_proc.hpp"

class DLRMServer : public mlperf::SystemUnderTest {
public:
     DLRMServer (
        const std::string &name,
        std::vector<DLRMSampleLibraryPtr_t> &qsls,
        std::vector<std::vector<float>> &scales,
        const std::string &model_path,
        const std::string &output_path,
        const int maxBatchSize,
        const int numSockets,
        const int coresPerSocket,
        const int num_producers,
        const int consumer_per_producer,
        const int start_consumer_core,
        bool accuracy,
        bool server);

    ~DLRMServer ();

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

    void FlushQueries() override;

    void ProcessTasks(int i);

    void ConcatSamples(size_t batchsize,
                       std::list<DLRMTask> &tasks,
                       std::unordered_map<size_t, std::vector<void *>> &bufferStore,
                       void **dsx_ptr,
                       void **lsi_ptr,
                       void **y_ptr);

    void SaveResult(float *actual, float *pred, size_t len);

    void CalcAccuracy(std::vector<DLRMOutDtype> &y_true,
                      DLRMOutDtype *y_pred);

    void BindUncoreThreadToCpus(int start_index, int len);
    void BindCoreThreadToCpus(int start_index, int len);

    const std::string& Name() override;

    std::shared_ptr<cnpy::npz_t> LoadModel(const std::string &model_path);

private:
    QParam_t PrepareParam(std::shared_ptr<cnpy::npz_t> params);

    const std::string name_;
    std::vector<DLRMSampleLibraryPtr_t> &qsls_;
    std::vector<std::shared_ptr<DLRMBatchMaker>> makers_;
    std::vector<std::shared_ptr<DLRMCore>> cores_;
    std::vector<std::shared_ptr<DLRMResultHandler>> reshandls_;
    std::vector<std::thread> consumer_threads_;
    std::vector<float> scales_;
    // parameters for each numa
    std::vector<QParam_t> params_;
    // Task queue
    // std::deque<DLRMTask> tasks_;
    // mutex to protect tasks_
    std::mutex mtx_;
    // conditional variable to control the thread pool
    std::condition_variable condVar_;
    // control the status of all threads
    bool stopWork_;
    // index of batch maker to be delivered
    int batchMakerIdx_;
    const int numSockets_;
    const int coresPerSocket_;
    const int numProducers_;
    const int numConsumerPerProducer_;
    const int numConsumers_;
    const int coresPerConsumer_;
    const int startConsumerCore_;
    const int batchSize_;
    // mode of running
    bool accuracy_;
    // scenario of running
    bool server_;
    // num of result;
    size_t numResult_;
    // prediction
    std::vector<float> prediction_;
    // actual value
    std::vector<float> actual_;
    // num of issued queries
    size_t numQueriesIssued_;
    // num of samples processed;
    size_t numSamplesComplete_;
    size_t numSamplesGood_;
    // num of queries processed;
    size_t numQueriesComplete_;

    std::string output_path_;
};

using DLRMServerPtr_t = std::shared_ptr<DLRMServer>;
