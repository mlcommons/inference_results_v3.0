#include <cassert>
#include <math.h>
#include <numeric>
#include <fstream>
#include "dlrm_server.hpp"
#include "kernels.hpp"

DLRMServer::DLRMServer(
    const std::string &name,
    std::vector<DLRMSampleLibraryPtr_t> &qsls,
    std::vector<std::vector<float>> &scales,
    const std::string &model_path,
		const std::string &output_path,
    const int maxBatchSize,
    const int num_sockets,
    const int cores_socket,
    const int num_producers,
    const int consumer_per_producer,
    const int start_consumer_core,
    bool accuracy,
    bool server)
    : name_(name),
      qsls_(qsls),
      output_path_(output_path),
      stopWork_(false),
      numSockets_(num_sockets),
      coresPerSocket_(cores_socket),
      numProducers_(num_producers),
      numConsumerPerProducer_(consumer_per_producer),
      numConsumers_(num_producers * consumer_per_producer),
      coresPerConsumer_(num_sockets * cores_socket / consumer_per_producer / num_producers),
      startConsumerCore_(start_consumer_core),
      batchMakerIdx_(0),
      batchSize_(maxBatchSize),
      accuracy_(accuracy),
      server_(server),
      numResult_(0),
      numQueriesIssued_(0),
      numSamplesComplete_(0),
      numSamplesGood_(0),
      numQueriesComplete_(0) {

    BindUncoreThreadToCpus(0, start_consumer_core);

    // batch maker
    for (int i = 0; i < numProducers_; ++i) {
        makers_.emplace_back(std::make_shared<DLRMBatchMaker>(maxBatchSize, qsls[i]));
    }

    // scales
    scales_.resize(18);
    int offset = 0;
    for (auto v : scales) {
        scales_[offset] = v[0];
        offset++;
        scales_[offset] = v[1];
        offset++;
    }

    // result handler
    for (int i = 0; i < numConsumers_; ++i) {
        // 1 thread per result handler
        reshandls_.emplace_back(std::make_shared<DLRMResultHandler>(1));
    }

    for (int i = 0; i < numProducers_; ++i) {
        BindUncoreThreadToCpus(i*cores_socket, 1);
        bindNumaMemPolicy(i, numProducers_);
        // load quantize model, numa aware
        auto model = LoadModel(model_path);
        auto params = PrepareParam(model);
        params_.push_back(params);
        for (int j = 0; j < numConsumerPerProducer_; ++j) {
            // dlrm model
            cores_.emplace_back(std::make_shared<DLRMCore>(
                                    DLRMCore(params,
                                             scales_,
                                             i)));
        }
        // clear model after quantization
        model->clear();
        resetNumaMemPolicy();
    }

    BindUncoreThreadToCpus(0, start_consumer_core);

    // start consumer thread
    for (int i = start_consumer_core; i < numConsumers_; ++i) {
        consumer_threads_.emplace_back(&DLRMServer::ProcessTasks, this, i);
    }

    // prepare accuracy result buffer
    if (accuracy_) {
        // reserve space for accuracy calculation
        prediction_.resize(qsls_[0]->NumIndividualPairs());
        actual_.resize(qsls_[0]->NumIndividualPairs());
    }
}

void DLRMServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    numQueriesIssued_ += samples.size();
    // VLOG(0) << "Issued queries " << numQueriesIssued_;
    if (server_) {
        // for server mode
        int nextBatchMaker = batchMakerIdx_ % numProducers_;
        batchMakerIdx_++;
        makers_[nextBatchMaker]->ServerEnqueue(samples);
    } else {
        // for offline mode
        for (size_t i = 0; i < samples.size(); ++i) {
            int nextBatchMaker = batchMakerIdx_ % numProducers_;
            batchMakerIdx_++;
            makers_[nextBatchMaker]->OfflineEnqueue(samples[i]);
        }
    }
}

void DLRMServer::FlushQueries() {
    VLOG(5) << "FlushQueries";
    for (auto maker : makers_)
        maker->ServerFlush();
}

void verify_lsi(cnpy::npz_t &pool,
                int idx,
                int32_t *res,
                size_t inner_dim, size_t outer_dim) {
    std::string key = "lsi" + std::to_string(idx);
    cnpy::NpyArray &refe_npy = pool[key];
    int32_t * refe = refe_npy.data<int32_t>();
    size_t err_no = 0;
    for (size_t i = 0; i < outer_dim; ++i) {
        for (size_t j = 0; j < inner_dim; ++j) {
            size_t id = i * inner_dim + j;
            if (refe[id] != res[id]) {
                err_no++;
                printf("lsi%d diff at (%zu, %zu)  value(%d: %d)\n",
                       idx, i, j, refe[id], res[id]);
            }
            CHECK_LT(err_no, 3);
        }
    }
}

void verify_dsx(cnpy::npz_t &pool,
                int idx,
                int8_t *res, float scale,
                size_t inner_dim, size_t outer_dim) {
    std::string key = "dsx" + std::to_string(idx);
    cnpy::NpyArray &refe_npy = pool[key];
    float *refe = refe_npy.data<float>();
    size_t err_no = 0;
    for (size_t i = 0; i < outer_dim; ++i) {
        for (size_t j = 0; j < inner_dim; ++j) {
            size_t id = i * inner_dim + j;
            float refe_f = round(refe[id] * scale);
            if (refe_f > 127.5)
                refe_f = 127.0;
            int8_t refe_s8 = (int8_t)round(refe_f);
            if (refe_s8 != res[id]) {
                err_no++;
                printf("dsx%d diff at (%zu, %zu) value (%d: %d)\n",
                       idx, i, j, refe_s8, res[id]);
            }
            CHECK_LT(err_no, 3);
        }
    }
}

void verify_res(cnpy::npz_t &pool,
                int idx,
                float *res,
                size_t outer_dim) {
    std::string key = "res" + std::to_string(idx);
    cnpy::NpyArray &refe_npy = pool[key];
    float *refe = refe_npy.data<float>();
    size_t err_no = 0;
    for (size_t i = 0; i < outer_dim; ++i) {
        if (std::abs(refe[i] - res[i]) > 0.001 * std::abs(res[i]) + 0.001) {
            err_no++;
            printf("res%d diff at (%zu) value (%f: %f)\n",
                   idx, i, refe[i], res[i]);
        }
        CHECK_LT(err_no, 8);
    }
}

void DLRMServer::ProcessTasks(int i) {
    BindCoreThreadToCpus(coresPerConsumer_*i, coresPerConsumer_);
    int numaIndex = i / numConsumerPerProducer_;
    bindNumaMemPolicy(numaIndex, numProducers_);
    VLOG(1) << "DLRMConsumer " << i << " start";
    int batch_id = i / numConsumerPerProducer_;
    float *y_ptr = nullptr;
    int8_t *dsx_ptr = nullptr;
    int32_t *lsi_ptr = nullptr;
    // cnpy::npz_t pool = cnpy::npz_load("/DataDisk_2/syk/pool.npz");

    std::unordered_map<size_t, std::vector<void *>> bufferStore;
    while (!stopWork_) {
        // get batch
        assert(batch_id >= 0);
        assert(batch_id < makers_.size());
        // CHECK_EQ(numQueriesIssued_, numQueriesComplete_ + makers_[batch_id]->TotalQueriesLeft());
        DLRMBatch batch = makers_[batch_id]->GetTask(i);
        // break the loop if no available batch
        if ((!batch.tasks_) || (batch.tasks_->size() == 0))
            continue;

        // VLOG(5) << "DLRMConsumer " << i << " running";
        // concat samples
        size_t BatchSize = batch.batchSize_;
        ConcatSamples(BatchSize,
                      *batch.tasks_,
                      bufferStore,
                      (void **)&dsx_ptr,
                      (void **)&lsi_ptr,
                      (void **)&y_ptr);

        // inference
        DLRMOutDtype *output = cores_[i]->infer(BatchSize, (int8_t *)dsx_ptr, lsi_ptr); // todo

        // VLOG(5) << "DLRMConsumer batch " << BatchSize;
        std::vector<DLRMOutDtype> outputs(BatchSize, 0.0);
        // copy true value to predict, for debug only!!!
        memcpy(outputs.data(), output, BatchSize * sizeof(DLRMOutDtype));
        DLRMResult r{
            std::make_shared<std::vector<DLRMOutDtype>>(outputs),
            batch.tasks_}; // = dlrm->infer();
        if (i == 0)
            VLOG(1) << "Complete " << batch.tasks_->size() << " queries";
        // VLOG(0) << "Completed queries " << numQueriesComplete_;
        reshandls_[i]->Enqueue(r);
        if (accuracy_) {
            CalcAccuracy(outputs, y_ptr);
            SaveResult(y_ptr, outputs.data(), BatchSize);
        }
    }
    VLOG(1) << "Num of buffer used " << 3 * bufferStore.size();

    for (auto &i : bufferStore) {
        freeHost(&i.second[0]);
        freeHost(&i.second[1]);
        freeHost(&i.second[2]);
    }
    resetNumaMemPolicy();
    VLOG(1) << "DLRMConsumer " << i << " end";
    // VLOG(0) << "Sample unprocessed " << makers_[batch_id]->TotalSampleLeft();
}

void DLRMServer::ConcatSamples(size_t BatchSize,
                               std::list<DLRMTask> &tasks,
                               std::unordered_map<size_t, std::vector<void *>> &bufferStore,
                               void **dsx_ptr,
                               void **lsi_ptr,
                               void **y_ptr) {
    auto it = bufferStore.find(BatchSize);
    if (it == bufferStore.end()) {
        CHECK_EQ(mallocHost(dsx_ptr, BatchSize * 32 * sizeof(int8_t)), cpuSucc);
        CHECK_EQ(mallocHost(lsi_ptr, BatchSize * 26 * sizeof(int32_t)), cpuSucc);
        CHECK_EQ(mallocHost(y_ptr, BatchSize * sizeof(float)), cpuSucc);
        bufferStore[BatchSize] = {*dsx_ptr, *lsi_ptr, *y_ptr};
    } else {
        *dsx_ptr = it->second[0];
        *lsi_ptr = it->second[1];
        *y_ptr = it->second[2];
    }
    // copy data
    size_t offset = 0;
    for (auto &t : tasks) {
        auto index = t.sample_.index;
        int32_t *y_sample = (int32_t *)qsls_[0]->GetSampleAddress(index, 0, 0, 0);
        int32_t *int_sample = (int32_t *)qsls_[0]->GetSampleAddress(index, 1, 0, 0);
        int32_t *cat_sample = (int32_t *)qsls_[0]->GetSampleAddress(index, 2, 0, 0);
        memcpy((char *)(*lsi_ptr) + offset * 26 * sizeof(int32_t),
               cat_sample,
               t.numSample_ * 26 * sizeof(int32_t));
        cvtS32toF32((float *)(*y_ptr) + offset,
                    y_sample,
                    t.numSample_);
        intAdd1LogScalePad32((int8_t *)(*dsx_ptr) + offset * 32,
                             int_sample,
                             t.numSample_,
                             scales_[0]);
        offset += t.numSample_;
    }
    return;
}

void
DLRMServer::CalcAccuracy(std::vector<DLRMOutDtype> &y_pred,
                         DLRMOutDtype *y_true) {
    size_t BatchSize = y_pred.size();
    size_t good = 0;
    for (int i = 0; i < BatchSize; ++i) {
        float f = y_pred[i];
        if (f < 0.0f) {
            f = 0.0f;
        } else if (f > 1.0f) {
            f = 1.0f;
        } else {
            f = round(f);
        }
        if (f == y_true[i]) {
            good++;
        }
    }
    std::unique_lock<std::mutex> lock(mtx_);
    numSamplesGood_ += good;
    numSamplesComplete_ += BatchSize;
}

void
DLRMServer::SaveResult(float *actual, float *pred, size_t len) {
    std::unique_lock<std::mutex> lock(mtx_);
    // condVar_.wait(lock,
    //               [&]{
    //                   return stopWork_;});
    // for (size_t i = 0; i < 20 ; ++i) {
    //     printf("%f %f\n", actual[i], pred[i]);
    // }
    // printf("len %d id %ld\n", len, tasks[0].sample_.index);
    size_t oldlen = numResult_;
    memcpy(prediction_.data() + oldlen, pred, len * sizeof(float));
    memcpy(actual_.data() + oldlen, actual, len * sizeof(float));
    numResult_ += len;
    // condVar_.notify_one();
}

DLRMServer::~DLRMServer() {
    // clean up
    VLOG(1) << "~DLRMServer";
    {
        std::unique_lock<std::mutex> lock(mtx_);
        stopWork_ = true;
        condVar_.notify_all();
    }

    for (auto b: makers_) {
        b->StopWork();
    }

    for (auto r: reshandls_) {
        r->StopWork();
    }

    for (auto &t : consumer_threads_) {
        t.join();
    }

    if (accuracy_) {
        std::vector<double> result = roc_auc_score(actual_.data(),
                                                   prediction_.data(),
                                                   numResult_,
                                                   true);
        LOG(INFO) << "roc_auc " << result[0] * 100.0 << "% "
                  << " accuracy " << float(numSamplesGood_) / float(numSamplesComplete_) * 100.0 << "% "
                  << " on " << numResult_ << " samples";
        std::fstream fout;
        fout.open(output_path_ + "/accuracy.txt", std::ios::out);
        fout << "AUC=" << result[0] * 100.0 << "%";

    }
}

const std::string &DLRMServer::Name() {
    return name_;
}

std::shared_ptr<cnpy::npz_t> DLRMServer::LoadModel(const std::string &model_path) {
    LOG(INFO) << "Loading model " << model_path;
    cnpy::npz_t model = cnpy::npz_load(model_path);
    for (auto &p : model) {
        if (p.second.shape.size() == 1)
            LOG(INFO) << "Loaded " << p.first << " size ("
                      << p.second.shape[0] << ")";
        else
            LOG(INFO) << "Loaded " << p.first << " size ("
                      << p.second.shape[0] << ","<< p.second.shape[1] << ")";
    }
    LOG(INFO) << "Load model complete";
    return std::make_shared<cnpy::npz_t>(model);
}

QParam_t DLRMServer::PrepareParam(std::shared_ptr<cnpy::npz_t> params) {
    QParam_t res;
    for (auto &p : *params) {
        std::string name = p.first;
        float *ptr = p.second.data<float>();
        size_t len = p.second.num_vals;
        std::shared_ptr<DLRMParam> param(new DLRMParam(ptr, len));
        if (name.find("emb_l") != std::string::npos) {
            // LOG(INFO) << "quantize " << name << " per tensor";
            param->quantize(p.second.shape, true);
        } else if (name.find("weight") != std::string::npos) {
            // LOG(INFO) << "quantize " << name << " per row";
            param->quantize(p.second.shape, false);
        }
        res[p.first] = std::move(param);
    }
    return res;
}

void DLRMServer::BindUncoreThreadToCpus(int start_index, int len) {
    if (startConsumerCore_ > 0) {
        bindThreadToCpus(start_index, len);
    }
}

void DLRMServer::BindCoreThreadToCpus(int start_index, int len) {
    assert(start_index >= startConsumerCore_);
    bindThreadToCpus(start_index, len);
}
