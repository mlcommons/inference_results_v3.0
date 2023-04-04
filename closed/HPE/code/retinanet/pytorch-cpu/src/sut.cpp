//#include <pthread.h>
#include <chrono>
#include "sut.hpp"
#include "sut_server.hpp"
#include "backend.hpp"
#include "utils.hpp"
//#include "kmp_launcher.hpp"
#include "sut_server.hpp"

#ifndef CHANNELS
#define CHANNELS 3
#endif

#ifndef IM_WIDTH
#define IM_WIDTH 800
#endif

#ifndef IM_HEIGHT
#define IM_HEIGHT 800
#endif

SUT::SUT(std::string pth_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
        int batch_size
    ) : 
    ds_(new RetinaNetDataset(data_path, total_sample_count)){;
    this->model_filename_ = pth_model;
    this->data_path_ = data_path;

    this->instances_ = num_instances;
    this->cpus_per_instance_ = cpus_per_instance;
    this->workers_.resize(this->instances_);
    this->batch_size_ = batch_size;
    this->backend_ = SUT::createInstance();
    SUT::startSUT();
}


// 'Join' all instances
SUT::~SUT(){
    
    {
        std::unique_lock<std::mutex> lck(this->mutex_);
        this->completed_ = true;
        this->cond_var_.notify_all();
    }
    
    for (auto& worker : this->workers_){
        worker.join();
    }

}


//All queries completion flag
void SUT::setCompletion(){
    this->completed_ = true;
}


// Backend which has model and inference apis
std::shared_ptr<Backend> SUT::createInstance(){

    std::shared_ptr<Backend> model( new Backend(this->model_filename_));
    model->load();
    return model;

}

// An instance/worker
void SUT::instanceThread(){

    this->mutex_.lock();
    std::thread::id threadNum = std::this_thread::get_id();
    int start_core = this->start_core_;

    // Instance KMP controller
    kmp::KMPLauncher thCtrl;
    std::vector<int> places(this->cpus_per_instance_);
    for (int i = 0; i < this->cpus_per_instance_; ++i){
        places[i] = start_core + i;
    }

    int instance_id = this->instance_id_++;
    std::cout << " [SUT] Creating instance " << instance_id << std::endl;
    thCtrl.setAffinityPlaces(places).pinThreads();

    // Load model for the instance
    //std::unique_ptr<Backend> net = this->backend_;//createInstance();

    std::cout << " [SUT] Instance " << instance_id << " created on cores " << places <<".\n";
    
    this->start_core_ += this->cpus_per_instance_;
    this->idle_instances_.fetch_add(1);
    this->mutex_.unlock();

    std::vector<mlperf::QuerySampleIndex> batch_idxs;
    batch_idxs.resize(this->batch_size_);
    for (size_t j=0; j < this->batch_size_; j++) {
        batch_idxs[j] = j;
    }

    // Warmup section
    at::Tensor input_tensor = torch::zeros({this->batch_size_, CHANNELS, IM_HEIGHT, IM_WIDTH}).to(torch::kInt8).contiguous();
    
    int8_t* inp_data_ptr = input_tensor.data_ptr<int8_t>();
    {
        std::unique_lock<std::mutex> lock(this->warmup_mutex_);
        this->w_cond_var_.wait(lock, [&](){return this->start_warmup_;});
        lock.unlock();
        if (this->num_warmup_==0){
            this->w_cond_var_.notify_one();
            this->idle_instances_.fetch_add(1);
        } else {
            this->mutex_.lock();
            std::cout << " Inference warmup for instance " << instance_id <<std::endl;
            this->mutex_.unlock();
            for (int i=0; i < this->num_warmup_; i++){
                // SUT::ds_->GetSamplesTensor(batch_idxs, input_tensor);
                input_tensor = SUT::ds_->GetSamples(batch_idxs);
                // SUT::ds_->GetSamplesPtr(batch_idxs, inp_data_ptr);
                //SUT::ds_->GetSamplesCopyTensor(batch_idxs, input_tensor);
                torch::jit::IValue out = this->backend_->predict(input_tensor);
            }
        
            this->idle_instances_.fetch_add(1);
            this->w_cond_var_.notify_one();
        }

    }
    

    while (true){
        std::unique_lock<std::mutex> lock(this->mutex_);
        
        this->cond_var_.wait(lock, [&](){return !this->query_queue_.empty() || this->completed_;});

        if (this->completed_){
            std::cout << " Exiting thread " << threadNum << std::endl;
            break;
        }

        Item qitem = std::move(this->query_queue_.front());
        this->query_queue_.pop();
        lock.unlock();
        this->cond_var_.notify_one();
        // SUT::ds_->GetSamplesTensor(qitem.sample_idxs_, input_tensor);
        input_tensor = SUT::ds_->GetSamples(qitem.sample_idxs_);
        // SUT::ds_->GetSamplesPtr(qitem.sample_idxs_, inp_data_ptr);
        //SUT::ds_->GetSamplesCopyTensor(qitem.sample_idxs_, input_tensor);
        // Inference step
        torch::jit::IValue outputs = this->backend_->predict(input_tensor);

        SUT::ds_->PostProcess( qitem, outputs);
    }
 
}


// Initiate SUTs with instances and their affinities
void SUT::startSUT(){
    this->idle_instances_.store(0);
    for (int i = 0; i < this->instances_; i++){
        workers_[i] = std::thread(&SUT::instanceThread, this);
    }

}


//TODO: Complete this warmup method

void SUT::doWarmup(){

    while(this->idle_instances_.load() < this->instances_){
        int waits = 1; // TODO: Do nothing??
    }

    std::vector<mlperf::QuerySampleIndex> batch_idxs;
    batch_idxs.resize(this->batch_size_);
    for (size_t j=0; j < this->batch_size_; j++) {
        batch_idxs[j] = j;
    }
    this->ds_->LoadSamplesToRam(batch_idxs);

    this->idle_instances_.store(0);
    
    this->start_warmup_ = true;

    this->w_cond_var_.notify_all();
    std::unique_lock<std::mutex> lock(this->warmup_mutex_);
    this->w_cond_var_.wait(lock, [&](){return this->idle_instances_.load()==this->instances_;});

    this->ds_->UnloadSamplesFromRam(batch_idxs);
}


// Receive query from loadgen to workers
void SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples){
    std::unique_lock<std::mutex> lock(this->mutex_);
    size_t sample_size = samples.size();
    size_t num_batches = sample_size / this->batch_size_;
    std::vector<mlperf::QuerySampleIndex> batch_idxs; // Contains batch samples indexes
    std::vector<mlperf::ResponseId> batch_resp_ids; // Contains batch samples response id
    batch_idxs.resize(this->batch_size_);
    batch_resp_ids.resize(this->batch_size_);

    for (size_t i = 0; i < num_batches; i++){
        size_t offset = i*this->batch_size_;
        for (size_t j=0; j < this->batch_size_; j++){
            batch_idxs[j] = samples[j+offset].index;
            batch_resp_ids[j] = samples[j+offset].id;
        }
        Item batch_item(batch_resp_ids, batch_idxs);
        this->query_queue_.push(batch_item);
    }

    // Remainder
    size_t rem_start = num_batches * this->batch_size_;
    if (rem_start < sample_size){
        std::vector<mlperf::QuerySampleIndex> rem_batch_idxs;
        std::vector<mlperf::ResponseId> rem_batch_resp_ids;
        rem_batch_idxs.resize(sample_size - rem_start);
        rem_batch_resp_ids.resize(sample_size - rem_start);
        size_t j = 0;

        for( ; rem_start+j < sample_size; j++){
            rem_batch_idxs[j] = samples[rem_start+j].index;
            rem_batch_resp_ids[j] = samples[rem_start+j].id;
        }
        int num_dummies = this->batch_size_ - rem_batch_idxs.size();
        for(int i=0;i<num_dummies;i++)
        {
            rem_batch_idxs.push_back(samples[0].index);
            rem_batch_resp_ids.push_back(samples[0].id);
        }
        Item batch_item(rem_batch_resp_ids, rem_batch_idxs,num_dummies);
        this->query_queue_.push(batch_item);
    }
    
    lock.unlock();
    this->cond_var_.notify_one();
    //printf("Notified instances\n");
}

// Get SUT name
const std::string& SUT::Name() {
    return this->name_;
}


// Flush queries after 'IssueQuery' call series
void SUT::FlushQueries(){

}


// Definition of SUT server Member functions
SUTServer::SUTServer(std::string pth_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
        int batch_size
    ) :
    ds_(new RetinaNetDataset(data_path, total_sample_count)){;
    this->model_filename_ = pth_model;
    this->data_path_ = data_path;

    this->instances_ = num_instances;
    this->cpus_per_instance_ = cpus_per_instance;
    this->workers_.resize(this->instances_);
    this->batch_size_ = batch_size;
    
    SUTServer::startSUTServer();
}

// 'Join' all instances
SUTServer::~SUTServer(){
    
    {
        std::unique_lock<std::mutex> lck(this->mutex_);
        this->completed_ = true;
        this->cond_var_.notify_all();
        this->batcher_cv_.notify_all();
    }
    
    for (auto& worker : this->workers_){
        worker.join();
    }

    batcher_thread_.join();

}

void SUTServer::setCompletion(){
    this->completed_ = true;
}

// Backend which has model and inference apis
std::unique_ptr<Backend> SUTServer::createInstance(){

    std::unique_ptr<Backend> model( new Backend(this->model_filename_));
    model->load();
    return model;

}

// A Server instance/worker
void SUTServer::instanceThread(){

    this->mutex_.lock();
    std::thread::id threadNum = std::this_thread::get_id();
    int start_core = this->start_core_;

    // Instance KMP controller
    kmp::KMPLauncher thCtrl;
    std::vector<int> places(this->cpus_per_instance_);
    for (int i = 0; i < this->cpus_per_instance_; ++i){
        places[i] = start_core + i;
    }

    int instance_id = this->instance_id_++;
    std::cout << " [SUTServer] Creating instance " << instance_id << std::endl;
    thCtrl.setAffinityPlaces(places).pinThreads();

    // Load model for the instance
    //TODO: Enable weight sharing - use single model across instances
    std::unique_ptr<Backend> net = createInstance();

    std::cout << " [SUTServer] Instance " << instance_id << " created on cores " << places <<".\n";
    
    this->start_core_ += this->cpus_per_instance_;
    this->idle_instances_.fetch_add(1);
    this->mutex_.unlock();

    std::vector<mlperf::QuerySampleIndex> batch_idxs;
    batch_idxs.resize(this->batch_size_);
    for (size_t j=0; j < this->batch_size_; j++) {
        batch_idxs[j] = j;
    }

    // Warmup section
    at::Tensor input_tensor = torch::zeros({this->batch_size_, CHANNELS, IM_HEIGHT, IM_WIDTH}).to(torch::kFloat).contiguous();
    {
        std::unique_lock<std::mutex> lock(this->warmup_mutex_);
        this->w_cond_var_.wait(lock, [&](){return this->start_warmup_;});
        lock.unlock();
        this->w_cond_var_.notify_one();
        if (this->num_warmup_==0){
            this->idle_instances_.fetch_add(1);
        } else {
            this->mutex_.lock();
            std::cout << " Inference warmup for instance " << instance_id <<std::endl;
            this->mutex_.unlock();
            for (int i=0; i < this->num_warmup_; i++){
                //SUTServer::ds_->GetSamplesTensor(batch_idxs, input_tensor);
                input_tensor = SUTServer::ds_->GetSamples(batch_idxs);
                
                torch::jit::IValue out = net->predict(input_tensor);
            }
        
            this->idle_instances_.fetch_add(1);
            this->w_cond_var_.notify_one();
        }

    }
    

    while (true){
        std::unique_lock<std::mutex> lock(this->mutex_);
        this->cond_var_.wait(lock, [&](){return !this->query_queue_.empty() || this->completed_;});

        if (this->completed_){
            std::cout << " Exiting thread " << threadNum << std::endl;
            break;
        }

        Item qitem = std::move(this->query_queue_.front());
        this->query_queue_.pop();
        lock.unlock();
        this->cond_var_.notify_one();
        //SUTServer::ds_->GetSamplesTensor(qitem.sample_idxs_, input_tensor);
        input_tensor = SUTServer::ds_->GetSamples(qitem.sample_idxs_);

        // Inference step
        torch::jit::IValue outputs = net->predict(input_tensor);

        // Post-process
        
        SUTServer::ds_->PostProcess( qitem, outputs);
    }
 
}


// Initiate Server SUTs with instances and their affinities
void SUTServer::startSUTServer(){
    this->idle_instances_.store(0);
    for (int i = 0; i < this->instances_; i++){
        workers_[i] = std::thread(&SUTServer::instanceThread, this);
    }
    // Start batcher thread
    batcher_thread_ = std::thread(&SUTServer::QueryBatchingThread, this);
}


//TODO: Complete this warmup method

void SUTServer::doWarmup(){

    while(this->idle_instances_.load() < this->instances_){
        int waits = 1; // TODO: Do nothing??
    }

    std::vector<mlperf::QuerySampleIndex> batch_idxs;
    batch_idxs.resize(this->batch_size_);
    for (size_t j=0; j < this->batch_size_; j++) {
        batch_idxs[j] = j;
    }
    this->ds_->LoadSamplesToRam(batch_idxs);

    this->idle_instances_.store(0);
    
    this->start_warmup_ = true;

    this->w_cond_var_.notify_all();
    std::unique_lock<std::mutex> lock(this->warmup_mutex_);
    this->w_cond_var_.wait(lock, [&](){return this->idle_instances_.load()==this->instances_;});

    this->ds_->UnloadSamplesFromRam(batch_idxs);

}

void SUTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    std::unique_lock<std::mutex> lock(this->batcher_mutex_);
    this->minibatch_.push_back(samples[0]);
    lock.unlock();
    this->batcher_cv_.notify_one();
}

void SUTServer::QueryBatchingThread(){
    using namespace std::chrono_literals;
    Item batch_items;
    batch_items.response_ids_.resize(this->batch_size_);
    batch_items.sample_idxs_.resize(this->batch_size_);
    int minibatch = 0;
    bool timer_flag = false;
    while (true){
        if (minibatch == this->batch_size_) { // If we have enough samples to create a batch
            std::unique_lock<std::mutex> qlock(this->mutex_);
            this->query_queue_.push(batch_items);
            qlock.unlock();
            this->cond_var_.notify_one();

            minibatch = 0; // reset batch count
            batch_items.response_ids_.clear();
            batch_items.sample_idxs_.clear();
            batch_items.response_ids_.resize(this->batch_size_);
            batch_items.sample_idxs_.resize(this->batch_size_);

            timer_flag = false;
        }

        std::unique_lock<std::mutex> block(this->batcher_mutex_);
        auto res = this->batcher_cv_.wait_for(block, 100ms, [&](){return !this->minibatch_.empty() || this->completed_;});
        if (this->completed_){
            std::cout << " Exiting Batcher thread" << std::endl;
            break;
        }
        if (res ==  false){// Some hiatus or end of query issuing by loadgen. Attach dummies, and batch
            if (minibatch > 0){
                // Attach dummy samples for the remainder
                batch_items.number_dummies = this->batch_size_ - minibatch;
                for (; minibatch < this->batch_size_; minibatch++){
                    batch_items.sample_idxs_[minibatch] = batch_items.sample_idxs_[0];
                    batch_items.response_ids_[minibatch] = batch_items.response_ids_[0];
                }
                std::unique_lock<std::mutex> qlock(this->mutex_);
                this->query_queue_.push(batch_items);
                qlock.unlock();
                this->cond_var_.notify_one();

                minibatch=0;
            }
        } else { // Add sample to minibatch
            mlperf::QuerySample qsample = std::move(this->minibatch_.front());
            this->minibatch_.pop_front();
            block.unlock();
            this->batcher_cv_.notify_one();
            batch_items.sample_idxs_[minibatch] = qsample.index;
            batch_items.response_ids_[minibatch] = qsample.id;
            minibatch++;
        }
    }
}

// Get SUT name
const std::string& SUTServer::Name() {
    return this->name_;
}


// Flush queries after 'IssueQuery' call series
void SUTServer::FlushQueries(){

}

