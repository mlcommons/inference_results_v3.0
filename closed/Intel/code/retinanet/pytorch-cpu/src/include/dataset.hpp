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

    virtual ~Dataset(){};

    // Load samples requestd by loadgen
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

    // Unload samples as requested by
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

    size_t PerformanceSampleCount() override;

    size_t TotalSampleCount() override;

    const std::string& Name() override;

    // Load all available dataset into memory (each input is preprocessed)
    void LoadDataset();

    virtual void PostProcess(
        Item qitem, torch::jit::IValue& output){
        printf("NOT IMPLEMENTED\n");
    }

    at::Tensor GetSamples(std::vector<mlperf::QuerySampleIndex>& samples);

    void GetSamplesTensor(std::vector<mlperf::QuerySampleIndex>& samples, at::Tensor& tensor);
    void GetSamplesPtr(std::vector<mlperf::QuerySampleIndex>&, int8_t* );
    void GetSamplesCopyTensor(std::vector<mlperf::QuerySampleIndex>&, at::Tensor&);

    std::string GetFileAtIndex(size_t index);

    void SetName(std::string dataset_name);

private:
    std::string data_path_;
    std::vector<std::string> image_list_;
    std::vector<std::tuple<int, int> > image_dims_;
    std::vector< unsigned long > ids_;

    std::map<mlperf::QuerySampleIndex, at::Tensor> data_in_memory_;
    std::map<mlperf::QuerySampleIndex, std::vector<int8_t>> data_buffer_;

    short height_=800, width_=800, num_channels_=3;
    size_t total_sample_count_ = 5000;
    size_t perf_count_ = 128;
    std::string name_ = "Dataset";
    
    
}; // Dataset

#endif
