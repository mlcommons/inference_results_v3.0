#ifndef SUT_H_
#define SUT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <unistd.h>
#include <atomic>
#include <chrono>

#include <torch/torch.h>

#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

#include "dataset.hpp"
#include "backend.hpp"
#include "item.hpp"
#include "kmp_launcher.hpp"
//

using namespace std::chrono;

class SUT : public mlperf::SystemUnderTest {
public:

       SUT(std::string rn50_part1,
        std::string rn50_part3,
        std::string full_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
        int batch_size
    );

    SUT(std::string rn50_part1,
        std::string rn50_part3,
        std::string full_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
	    int warmup_count,
        int batch_size
	) :  SUT(rn50_part1,
        rn50_part3,
        full_model,
        data_path,
        total_sample_count,
        num_instances,
        cpus_per_instance,
        mlperf_user_conf,
        batch_size
    	) {

	this->num_warmup_ = warmup_count;
	}

    ~SUT();

    // Initiate SUT instances
    void startSUT();


    const std::string& Name() override;

    // Interfaces with Loadgen to receive queries
    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;


    void FlushQueries() override;

    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies);// override;

    void setCompletion();

    // Instance thread is responsible for performing inference
    void instanceThread();

    // Creates Instance (for models)
    // TODO: Make flexible to share weights across all instances
    std::unique_ptr<Backend> createInstance();

    // Returns qsl object shared with loadgen
    mlperf::QuerySampleLibrary* GetQsl(){
        return ds_;
    }

    // Perform warmup
    void doWarmup();

    

private:
    Dataset* ds_;
    std::unique_ptr<Backend> backend_;

    std::string rn50_part1_, rn50_part3_, full_model_;
    std::string data_path_;
    std::string name_ = "SUT_Offline";
    time_point<steady_clock> query_issue_time_;
    time_point<steady_clock> query_final_time_ = steady_clock::now();
    std::unordered_map<int, std::vector<double> > E2E_latency_map_;
    std::unordered_map<int, std::vector<double> > inference_latency_map_;
    std::unordered_map<int, std::vector<double> > dataload_latency_map_;
    std::unordered_map<int, std::vector<std::tuple<double,double,double, double, double, double>> > latency_splits_;
    
    size_t total_query_samples_;

    size_t total_count_ = 1024;
    size_t perf_count_ = 1024;    

    int instances_ = 0;
    int cpus_per_instance_;
    int batch_size_=1;
     at::Tensor input_tensor = torch::zeros({batch_size_, Start_In_C, Start_I_H, Start_I_W}).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    at::Tensor backbone_output = torch::zeros({ batch_size_, 2048, 7, 7 }).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    torch::jit::IValue outputs;

    // Comm vars
    std::queue<Item> query_queue_;
    std::condition_variable cond_var_;
    std::mutex mutex_;
    bool completed_ = false;
    std::vector<std::thread> workers_;

    int start_core_ = 0;
    int instance_id_ = 0;
    mlperf::TestSettings settings_;
    int num_warmup_ = 0;

    //void doWarmup();
    bool start_warmup_ = false;
    std::atomic<int> idle_instances_;
    std::mutex warmup_mutex_;
    std::condition_variable w_cond_var_;

}; // SUT

#endif