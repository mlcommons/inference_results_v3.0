#include <chrono>
#include <vector>
#include <string>
#include "omp.h"


using namespace std::chrono;
#include "backend.hpp"

Backend::Backend( std::string rn50_part1, std::string rn50_part3, std::string full_model, int batch_size) :
        rn50_start_path_(rn50_part1), rn50_end_path_(rn50_part3), full_model_path_(full_model), batch_size_(batch_size){
    input_vector_ = (int64_t*) malloc(batch_size*sizeof(int64_t));

    if(batch_size_==256)
    {
    this->rn50_backbone_ = &rn50_backbone_wrapper_bs256;
    this->init_backbone_ = &sc_init_rn50_backbone_wrapper_bs256;
    }
    else if(batch_size_==8)
    {
    this->rn50_backbone_ = &rn50_backbone_wrapper_bs8;
    this->init_backbone_ = &sc_init_rn50_backbone_wrapper_bs8;
    }
    else if(batch_size_==4)
    {
    this->rn50_backbone_ = &rn50_backbone_bs4;
    this->init_backbone_ = &sc_init_rn50_backbone_bs4;     
    }
    
}


void Backend::load(){

    // Load first model
    model_start_ = torch::jit::load(rn50_start_path_);
    model_start_.eval();

    //Load final model
    model_end_ = torch::jit::load(rn50_end_path_);
    model_end_.eval();

    // Load fp weights of "backbone"
    setWeightMap();

    // Initialize custom kernels
    init_backbone_(weight_map_["conv1.weight"].data_ptr<float>(), weight_map_["conv1.bias"].data_ptr<float>(), weight_map_["fc.weight"].data_ptr<float>(), weight_map_["fc.bias"].data_ptr<float>());
    
}

torch::jit::IValue Backend::predict(torch::Tensor& input_tensor, torch::Tensor& backbone_output){

    int8_t* backbone_out_ptr = backbone_output.data_ptr<int8_t>();

    runBackboneKernel(backbone_out_ptr, backbone_in_ptr_,this->fc_out);

    return this->temp_out;

}

void Backend::setWeightMap(){

    torch::jit::script::Module traced_fp_model = torch::jit::load(full_model_path_);
    for (const auto& pair : traced_fp_model.named_parameters()) {
        std::string pname = pair.name;
        torch::Tensor value = pair.value;
        weight_map_.insert(std::pair<std::string, at::Tensor>{pname, value});
    }
}