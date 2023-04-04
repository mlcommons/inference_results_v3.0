#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <tuple>
#include <map>

#include <torch/torch.h>
#include <ATen/ATen.h>

#include <query_sample_library.h>
//#include <test_settings.h>

#include "item.hpp"

class Dataset : public mlperf::QuerySampleLibrary {

public:

    //Dataset( const std::string data_path );
    Dataset( const std::string data_path, size_t total_sample_count);
    /*
    Dataset( const std::string data_path, size_t total_sample_count) : Dataset(data_path){
        this->total_sample_count_ = total_sample_count; 
    };
    */
    virtual ~Dataset(){};

    // Load samples requestd by loadgen
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

    // Unload samples as requested by
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

    size_t PerformanceSampleCount() override;

    size_t TotalSampleCount() override;

    const std::string& Name()  override;

    // Load all available dataset into memory (each input is preprocessed)
    void LoadDataset();

    // Post process inference output and made ready to send to loadgen
    /*
    virtual std::vector<mlperf::QuerySampleResponse>
        PostProcess(
            Item qitem,
            torch::jit::IValue& output
        );
    */
    
    virtual std::vector<mlperf::QuerySampleResponse> PostProcess(
        Item qitem, torch::jit::IValue& output){
        printf("NOT IMPLEMENTED\n");
        // TODO: Implement post-processing
        std::vector<float> results = {0.1,0.6,1.35,8.5};
        results[0] = 0.5;
        mlperf::QuerySampleResponse response{ qitem.response_ids_[0], reinterpret_cast<std::uintptr_t>(&results[0]), sizeof(float)*results.size()};
        return {response};
    }

    at::Tensor GetSamples(std::vector<mlperf::QuerySampleIndex>& samples);

    void GetSamplesTensor(std::vector<mlperf::QuerySampleIndex>& samples, at::Tensor& tensor);

    void GetSamplesPointer(std::vector<mlperf::QuerySampleIndex>& samples, int8_t* tensor_pointer);

    void GetPointerToSamples(std::vector<mlperf::QuerySampleIndex>& samples, int64_t* input_pointers);

    std::string GetFileAtIndex(size_t index);

    void SetName(std::string dataset_name);

private:
    std::string data_path_;
    std::vector<std::string> image_list_;
    std::vector<std::tuple<int, int> > image_dims_;
    std::vector< unsigned long > ids_;

    std::map<mlperf::QuerySampleIndex, at::Tensor> data_in_memory_;

    short height_=800, width_=800, num_channels_=3;
    size_t total_sample_count_ = 5000;
    size_t perf_count_ = 64;
    std::string name_ = "Dataset";
    
    
}; // Dataset

#endif
