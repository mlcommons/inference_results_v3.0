#ifndef BACKEND_H_
#define BACKEND_H_

#include <vector>
#include <string>
#include <queue>
#include <map>

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include "ATen/autocast_mode.h"
#include <kernel_rn50/rn50_backbone.hpp>
#include <kernel_rn50/shape.hpp>

#include "dnnl.hpp"
#include "dnnl_types.h"

const int Start_Out_C = 64;
const int Start_Out_H = 112;
const int Start_Out_W = 112;
const int Batch_Size = 8;
const int Start_W_H = 7;
const int Start_W_W = 7;
const int Start_I_H = 224;
const int Start_I_W = 224;
const int Start_In_C = 3;
const int Num_Classes = 1000;

const int End_Out_C = 2048;
const int End_Out_H = 7;
const int End_Out_W = 7;

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

class Backend {
public:

    // Requires 3 input models: entire rn50 model, first part (up to maxpool), last part (from avgpool)
    Backend( std::string, std::string, std::string, int);

    // Loads relevant models and initiate kernels
    void load();

    // Runs 'forwards'. Second argument is the backbone output tensor to be modified by kernel
    torch::jit::IValue predict(at::Tensor&, at::Tensor&);

    void prepareOneDNN();

    int8_t* getInputPtr() {return input_ptr_;}
    int64_t* getBackbonePtr() {return backbone_in_ptr_;}

    int64_t* input_vector_;
private:
    // Model components model_start_ -> custom_backbone -> model_end_;
    torch::jit::script::Module model_start_, model_end_;

    torch::TensorOptions start_option_ = torch::TensorOptions().dtype(torch::kInt8);
    at::Tensor start_out_ = torch::zeros({Batch_Size, Start_Out_C, Start_Out_H/2, Start_Out_W/2}, start_option_);

    // Path to component model
    std::string rn50_start_path_, rn50_end_path_, full_model_path_;

    // Map layer weights
    std::unordered_map<std::string, at::Tensor> weight_map_;

    int batch_size_;
    torch::Tensor outputs = torch::zeros({Batch_Size,1000});
    // Create weight maps
    void setWeightMap();

    // oneDNN primitives
    dnnl::engine eng_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream s_ = dnnl::stream(eng_);

    std::vector<dnnl::primitive> end_net_;
    std::vector<std::unordered_map<int, dnnl::memory>> end_net_args_;

    dnnl::memory::dims conv_src_tz = {this->batch_size_, Start_In_C, Start_I_H, Start_I_W};
    dnnl::memory conv_src_memory_ = dnnl::memory({{conv_src_tz}, dt::s8, tag::nhwc}, eng_);

    dnnl::memory::dims pool_dst_tz = {this->batch_size_, Start_Out_C, Start_Out_H/2, Start_Out_W/2};
    dnnl::memory pool_dst_memory_ = dnnl::memory({{pool_dst_tz}, dt::s8, tag::nhwc}, eng_);

    dnnl::memory::dims avg_pool_src_tz = {this->batch_size_, 2048, 7, 7};
    dnnl::memory::dims avg_pool_dst_tz = {this->batch_size_, 2048, 1, 1};
    dnnl::memory avg_pool_src_memory_ = dnnl::memory({{avg_pool_src_tz}, dt::s8, tag::nhwc}, eng_);
    dnnl::memory avg_pool_dst_memory_ = dnnl::memory({{avg_pool_dst_tz}, dt::s8, tag::nhwc}, eng_);

    // Post Backbone

    dnnl::memory::dims fc_src_tz = {this->batch_size_, 2048,1,1};
    dnnl::memory::dims fc_dst_tz = {this->batch_size_, 1000};
    dnnl::memory fc_src_memory_ = dnnl::memory({{fc_src_tz}, dt::s8, tag::nhwc}, eng_);
    dnnl::memory fc_dst_memory_ = dnnl::memory({{fc_dst_tz}, dt::f32, tag::nc}, eng_);
    

    int8_t* input_ptr_ = static_cast<int8_t *>(conv_src_memory_.get_data_handle());
    int64_t* backbone_in_ptr_ = static_cast<int64_t *>(pool_dst_memory_.get_data_handle());
    
   
    torch::Tensor temp_out = torch::randn({this->batch_size_,1000});
    float* fc_out = temp_out.data_ptr<float>();
    
    void (*rn50_backbone_)(int8_t*, int64_t*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*);
    void (*init_backbone_)(float*, float*, float*, float*);

public:
    // Backbone function
    void runStart();
    void runEnd();
    void runBackboneKernel(int8_t* output, int64_t* input,float* final_out){
       this->rn50_backbone_(output, input_vector_,final_out, weight_map_["layer1.0.downsample.0.weight"].data_ptr<float>(),weight_map_["layer1.0.downsample.0.bias"].data_ptr<float>(),weight_map_["layer1.0.conv1.weight"].data_ptr<float>(),weight_map_["layer1.0.conv1.bias"].data_ptr<float>(),weight_map_["layer1.0.conv2.weight"].data_ptr<float>(),weight_map_["layer1.0.conv2.bias"].data_ptr<float>(),weight_map_["layer1.0.conv3.weight"].data_ptr<float>(),weight_map_["layer1.0.conv3.bias"].data_ptr<float>(),weight_map_["layer1.1.conv1.weight"].data_ptr<float>(),weight_map_["layer1.1.conv1.bias"].data_ptr<float>(),weight_map_["layer1.1.conv2.weight"].data_ptr<float>(),weight_map_["layer1.1.conv2.bias"].data_ptr<float>(),weight_map_["layer1.1.conv3.weight"].data_ptr<float>(),weight_map_["layer1.1.conv3.bias"].data_ptr<float>(),weight_map_["layer1.2.conv1.weight"].data_ptr<float>(),weight_map_["layer1.2.conv1.bias"].data_ptr<float>(),weight_map_["layer1.2.conv2.weight"].data_ptr<float>(),weight_map_["layer1.2.conv2.bias"].data_ptr<float>(),weight_map_["layer1.2.conv3.weight"].data_ptr<float>(),weight_map_["layer1.2.conv3.bias"].data_ptr<float>(),weight_map_["layer2.0.downsample.0.weight"].data_ptr<float>(),weight_map_["layer2.0.downsample.0.bias"].data_ptr<float>(),weight_map_["layer2.0.conv1.weight"].data_ptr<float>(),weight_map_["layer2.0.conv1.bias"].data_ptr<float>(),weight_map_["layer2.0.conv2.weight"].data_ptr<float>(),weight_map_["layer2.0.conv2.bias"].data_ptr<float>(),weight_map_["layer2.0.conv3.weight"].data_ptr<float>(),weight_map_["layer2.0.conv3.bias"].data_ptr<float>(),weight_map_["layer2.1.conv1.weight"].data_ptr<float>(),weight_map_["layer2.1.conv1.bias"].data_ptr<float>(),weight_map_["layer2.1.conv2.weight"].data_ptr<float>(),weight_map_["layer2.1.conv2.bias"].data_ptr<float>(),weight_map_["layer2.1.conv3.weight"].data_ptr<float>(),weight_map_["layer2.1.conv3.bias"].data_ptr<float>(),weight_map_["layer2.2.conv1.weight"].data_ptr<float>(),weight_map_["layer2.2.conv1.bias"].data_ptr<float>(),weight_map_["layer2.2.conv2.weight"].data_ptr<float>(),weight_map_["layer2.2.conv2.bias"].data_ptr<float>(),weight_map_["layer2.2.conv3.weight"].data_ptr<float>(),weight_map_["layer2.2.conv3.bias"].data_ptr<float>(),weight_map_["layer2.3.conv1.weight"].data_ptr<float>(),weight_map_["layer2.3.conv1.bias"].data_ptr<float>(),weight_map_["layer2.3.conv2.weight"].data_ptr<float>(),weight_map_["layer2.3.conv2.bias"].data_ptr<float>(),weight_map_["layer2.3.conv3.weight"].data_ptr<float>(),weight_map_["layer2.3.conv3.bias"].data_ptr<float>(),weight_map_["layer3.0.downsample.0.weight"].data_ptr<float>(),weight_map_["layer3.0.downsample.0.bias"].data_ptr<float>(),weight_map_["layer3.0.conv1.weight"].data_ptr<float>(),weight_map_["layer3.0.conv1.bias"].data_ptr<float>(),weight_map_["layer3.0.conv2.weight"].data_ptr<float>(),weight_map_["layer3.0.conv2.bias"].data_ptr<float>(),weight_map_["layer3.0.conv3.weight"].data_ptr<float>(),weight_map_["layer3.0.conv3.bias"].data_ptr<float>(),weight_map_["layer3.1.conv1.weight"].data_ptr<float>(),weight_map_["layer3.1.conv1.bias"].data_ptr<float>(),weight_map_["layer3.1.conv2.weight"].data_ptr<float>(),weight_map_["layer3.1.conv2.bias"].data_ptr<float>(),weight_map_["layer3.1.conv3.weight"].data_ptr<float>(),weight_map_["layer3.1.conv3.bias"].data_ptr<float>(),weight_map_["layer3.2.conv1.weight"].data_ptr<float>(),weight_map_["layer3.2.conv1.bias"].data_ptr<float>(),weight_map_["layer3.2.conv2.weight"].data_ptr<float>(),weight_map_["layer3.2.conv2.bias"].data_ptr<float>(),weight_map_["layer3.2.conv3.weight"].data_ptr<float>(),weight_map_["layer3.2.conv3.bias"].data_ptr<float>(),weight_map_["layer3.3.conv1.weight"].data_ptr<float>(),weight_map_["layer3.3.conv1.bias"].data_ptr<float>(),weight_map_["layer3.3.conv2.weight"].data_ptr<float>(),weight_map_["layer3.3.conv2.bias"].data_ptr<float>(),weight_map_["layer3.3.conv3.weight"].data_ptr<float>(),weight_map_["layer3.3.conv3.bias"].data_ptr<float>(),weight_map_["layer3.4.conv1.weight"].data_ptr<float>(),weight_map_["layer3.4.conv1.bias"].data_ptr<float>(),weight_map_["layer3.4.conv2.weight"].data_ptr<float>(),weight_map_["layer3.4.conv2.bias"].data_ptr<float>(),weight_map_["layer3.4.conv3.weight"].data_ptr<float>(),weight_map_["layer3.4.conv3.bias"].data_ptr<float>(),weight_map_["layer3.5.conv1.weight"].data_ptr<float>(),weight_map_["layer3.5.conv1.bias"].data_ptr<float>(),weight_map_["layer3.5.conv2.weight"].data_ptr<float>(),weight_map_["layer3.5.conv2.bias"].data_ptr<float>(),weight_map_["layer3.5.conv3.weight"].data_ptr<float>(),weight_map_["layer3.5.conv3.bias"].data_ptr<float>(),weight_map_["layer4.0.downsample.0.weight"].data_ptr<float>(),weight_map_["layer4.0.downsample.0.bias"].data_ptr<float>(),weight_map_["layer4.0.conv1.weight"].data_ptr<float>(),weight_map_["layer4.0.conv1.bias"].data_ptr<float>(),weight_map_["layer4.0.conv2.weight"].data_ptr<float>(),weight_map_["layer4.0.conv2.bias"].data_ptr<float>(),weight_map_["layer4.0.conv3.weight"].data_ptr<float>(),weight_map_["layer4.0.conv3.bias"].data_ptr<float>(),weight_map_["layer4.1.conv1.weight"].data_ptr<float>(),weight_map_["layer4.1.conv1.bias"].data_ptr<float>(),weight_map_["layer4.1.conv2.weight"].data_ptr<float>(),weight_map_["layer4.1.conv2.bias"].data_ptr<float>(),weight_map_["layer4.1.conv3.weight"].data_ptr<float>(),weight_map_["layer4.1.conv3.bias"].data_ptr<float>(),weight_map_["layer4.2.conv1.weight"].data_ptr<float>(),weight_map_["layer4.2.conv1.bias"].data_ptr<float>(),weight_map_["layer4.2.conv2.weight"].data_ptr<float>(),weight_map_["layer4.2.conv2.bias"].data_ptr<float>(),weight_map_["layer4.2.conv3.weight"].data_ptr<float>(),weight_map_["layer4.2.conv3.bias"].data_ptr<float>());
    }
}; // BACKEND_H_

#endif