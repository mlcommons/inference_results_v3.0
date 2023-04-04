#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "dataset.hpp"
#include <stdlib.h>  
#include <fstream>
#include "utils.hpp"
#include <torch/script.h>
#include <experimental/filesystem>
#include "omp.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/embed.h>  // python interpreter
// #include <pybind11/stl.h> 

#define RESIZE_HEIGHT 256
#define RESIZE_WIDTH 256
#define CROP_HEIGHT 224
#define CROP_WIDTH 224

namespace py = pybind11;
namespace fs = std::experimental::filesystem;
using namespace cv;

using namespace std;

void resizeWithAspectratio(cv::Mat* image, cv::Mat* resized_image,
            int out_height, int out_width, int interpol, float scale = 87.5) {
    int width = (*image).cols;
    int height = (*image).rows;
    int new_height = int(100. * out_height / scale);
    int new_width = int(100. * out_width / scale);

    int h, w = 0;
    if (height > width) {
        w = new_width;
        h = int(new_height * height / width);
    } else {
        h = new_height;
        w = int(new_width * width / height);
    }

    cv::resize((*image), (*resized_image), cv::Size(w, h), interpol);
}


void centerCrop(cv::Mat* image, int out_height, int out_width, cv::Mat* cropped_image) {
    int width = (*image).cols;
    int height = (*image).rows;
    int left = int((width - out_width) / 2);
    int top = int((height - out_height) / 2);
    cv::Rect customROI(left, top, out_width, out_height);

    (*cropped_image) = (*image)(customROI);
}

// Preprocess image file
const at::Tensor PreProcessSample(std::string filename, int resize_height, int resize_width, int crop_height, int crop_width){
    cv::Mat image = cv::imread(filename);

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int width = (image).cols;
    int height = (image).rows;
    double ratio_height = ((double) resize_height) / height;
    double ratio_width = ((double) resize_width) / width;

    int margin = 19;
    int opt_width, opt_height, iwidth, iheight;
    if (ratio_height < 1.0 || ratio_width < 1.0) {
      if (ratio_height < 1.0 && ratio_width < 1.0) {
        if( width > height ) {   //case 1
          opt_width = resize_width + 23;
          opt_height = resize_height + 10;
          cv::resize(image, image, cv::Size(opt_width, opt_height), 0.5, 0.5, cv::INTER_AREA);
        }
        else {  //case 2
          opt_width = resize_width + 19;
          opt_height = resize_height + 13;
          cv::resize(image, image, cv::Size(opt_width, opt_height), 0.5, 0.5, cv::INTER_AREA);
        }
      }
      else if (ratio_height < 1.0 && ratio_width > 1.0) { //case 3
        cv::resize(image, image, cv::Size(resize_width+margin, resize_height+19), 0.5, 0.5, cv::INTER_AREA);
      }
      else { //case 4
        opt_width = resize_width + 17;
        opt_height = resize_height + 24;
        cv::resize(image, image, cv::Size(opt_width, opt_height), 0.5, 0.5, cv::INTER_AREA);
      }

      centerCrop(&image, crop_width, crop_height, &image);
    }
    else { //case 5
      if (height >= crop_height && width >= crop_width) {
        opt_width = resize_width + 10;
        opt_height = resize_height + 10;
        cv::resize(image, image, cv::Size(opt_width, opt_height), 0.5, 0.5, cv::INTER_LINEAR);
        centerCrop(&image, crop_width, crop_height, &image);
      }
      else { //case 6
        opt_width = resize_width + iwidth;
        opt_height = resize_height + iheight;
        cv::resize(image, image, cv::Size(resize_width+15, resize_height+15), 0.5, 0.5, cv::INTER_LINEAR);
        centerCrop(&image, crop_width, crop_height, &image);
      }
    }

    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    cv::Mat mean(crop_height, crop_width, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(crop_height, crop_width, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::subtract(image, mean, image);
    cv::divide(image, std, image);

    at::Tensor tensor = torch::from_blob(image.data, {1, crop_height, crop_width, 3}).clone();
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(at::kFloat);
    tensor = tensor.to(torch::MemoryFormat::ChannelsLast);

    // Pre-quantize tensors
    tensor = torch::quantize_per_tensor(tensor, 0.02070588245987892, 0, torch::kQInt8);

    return tensor;
}

// Instantiate dataset and prefetches the image lists as well as ids, and sizes
Dataset::Dataset(const std::string data_path, size_t total_sample_count){

    std::ifstream file_handle;
    file_handle.open(data_path + "/val_map.txt");
    this->total_sample_count_ = total_sample_count;
    std::string image;
    int label;
    while (file_handle >> image >> label) {
        Dataset::image_list_.push_back( data_path + "/" + image);
        Dataset::ids_.push_back(label);
        if(image_list_.size()  >= Dataset::TotalSampleCount())
            break;
    }

}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

// Fetch processed samples for inference
at::Tensor Dataset::GetSamples(std::vector<mlperf::QuerySampleIndex>& samples){
    // if (samples.size() == 1) return Dataset::data_in_memory_[samples[0]];

    std::vector<at::Tensor> tensors;
    for(auto& sample : samples){
        tensors.push_back( Dataset::data_in_memory_[sample] );
    }
    at::Tensor ret = at::cat(tensors, 0);
    return ret;
}

void Dataset::GetSamplesTensor(std::vector<mlperf::QuerySampleIndex>& samples, at::Tensor& tensor){
    if (samples.size() == 1) tensor=Dataset::data_in_memory_[samples[0]];

    int num_samples = samples.size();
    int8_t* tensor_pointer = tensor.data_ptr<int8_t>();
    for (int ID_C = 0; ID_C < num_samples; ID_C ++) {
        int8_t* data_ptr = Dataset::data_in_memory_[samples[ID_C]].data_ptr<int8_t>();
        memcpy(tensor_pointer+ID_C*3*224*224, data_ptr, 3*224*224*sizeof(int8_t));
    }
}

void Dataset::GetSamplesPointer(std::vector<mlperf::QuerySampleIndex>& samples, int8_t* tensor_pointer){
    int num_samples = samples.size();
    // at::Tensor input_zeros = (torch::ones({9,3,224,224})*10).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    for (int ID_C = 0; ID_C < num_samples; ID_C ++) {
        int8_t* data_ptr = Dataset::data_in_memory_[samples[ID_C]].data_ptr<int8_t>();
        // int8_t* data_ptr = input_zeros.data_ptr<int8_t>();
        memcpy(tensor_pointer+ID_C*3*224*224, data_ptr, 3*224*224*sizeof(int8_t));
    }
}

void Dataset::GetPointerToSamples(std::vector<mlperf::QuerySampleIndex>& samples, int64_t* input_pointers){
    int num_samples = samples.size();
    // at::Tensor input_zeros = (torch::zeros({1, 3, 224, 224})).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    for (int ID_C = 0; ID_C < num_samples; ID_C ++) {
        input_pointers[ID_C] = reinterpret_cast<int64_t>(Dataset::data_in_memory_[samples[ID_C]].data_ptr<int8_t>());
        // input_pointers[ID_C] = reinterpret_cast<int64_t>(input_zeros.data_ptr<int8_t>());
        // printf("Data pointer value %ld\n", input_pointers[ID_C]);
        // int8_t* data_ptr = Dataset::data_in_memory_[samples[ID_C]].data_ptr<int8_t>();
        // memcpy(input_pointers[ID_C], data_ptr, 3*224*224*sizeof(int8_t));
    }
}

// Fetch image file at given index
std::string Dataset::GetFileAtIndex(size_t index){
    return Dataset::image_list_[index];
}

// Load samples into memory
void Dataset::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples){
    for(auto &sample : samples){
        std::string filename = Dataset::GetFileAtIndex(sample);

        at::Tensor tensor = PreProcessSample( filename, RESIZE_HEIGHT, RESIZE_WIDTH, CROP_HEIGHT, CROP_WIDTH);
        Dataset::data_in_memory_[sample] = tensor;

    }

}

// Set QSL name 
void Dataset::SetName(std::string new_name){
    Dataset::name_ = new_name;
}

// Unload samples from memory
void Dataset::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples){
    Dataset::data_in_memory_.clear();
}

// Performance sample count
size_t Dataset::PerformanceSampleCount(){
    return Dataset::perf_count_;
}

// Total sample count
size_t Dataset::TotalSampleCount(){
    return total_sample_count_;
}

// Get QSL name
const std::string& Dataset::Name()  {
    return name_;
}


ImageNet::ImageNet(const std::string data_path, size_t total_sample_count) : Dataset(data_path, total_sample_count){




}
std::vector<mlperf::QuerySampleResponse> ImageNet::PostProcess( Item qitem, torch::jit::IValue&  output){

   
    torch::Tensor results = output.toTensor();
    long unsigned int actual_size = qitem.response_ids_.size()-qitem.number_dummies;
    std::vector <mlperf::QuerySampleResponse> responses(actual_size);

    torch::Tensor arg_max = results.argmax(1);
    for (int i=0; i < actual_size; i++){
        responses[i].id = qitem.response_ids_[i];
        responses[i].data = reinterpret_cast<uintptr_t>(arg_max[i].data_ptr());
        responses[i].size = arg_max[i].nbytes();
    }


    return responses;
}
