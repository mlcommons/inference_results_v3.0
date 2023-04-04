#include <iostream>
#include "utils.hpp"

//using torch::indexing::Slice;


DefaultBoxes::DefaultBoxes(const at::Tensor& anchor_sizes,
                           const at::Tensor& aspect_ratios,
                           std::vector<at::Tensor>& feature_grid_sizes){
    this->anchor_sizes_ = anchor_sizes;
    this->aspect_ratios_ = aspect_ratios;
    this->feature_grid_sizes_ = feature_grid_sizes;
    this->num_scales_ = aspect_ratios.size(1); // aspect ratio is [num_grids x num_scales]
    this->num_anchors_ = anchor_sizes.size(1);

    for (size_t i=0; i < anchor_sizes.size(0); i++){
        this->cell_anchors_.push_back( GenerateAnchors(anchor_sizes[i], aspect_ratios[i].reshape({1,3})));
    }

    int num_feat_grid = feature_grid_sizes.size();
    std::vector<int> anchors_per_loc = DefaultBoxes::NumAnchorsPerLocations();

    this->split_num_anchors_per_level_.resize(num_feat_grid);
    at::Tensor grid;
    for(int i=0; i < num_feat_grid; i++){
        grid = feature_grid_sizes[i];
        this->split_num_anchors_per_level_[i] = anchors_per_loc[i] * grid[0].item<int>() * grid[1].item<int>();
    }

}

std::vector<int> DefaultBoxes::NumAnchorsPerLocations(){
    std::vector<int> out;
    for (size_t i=0; i < this->anchor_sizes_.size(0); i++){
        out.push_back(static_cast<int>(this->num_scales_ * this->num_anchors_));
    }
    
    return out;
}

at::Tensor DefaultBoxes::GenerateAnchors( const at::Tensor& anchor_size,
                                          const at::Tensor& aspect_ratios){
    at::Tensor scales = anchor_size.to(at::kFloat).reshape({1,3});
    
    at::Tensor h_ratios = at::sqrt(aspect_ratios).reshape({3,1});
    at::Tensor w_ratios = h_ratios.pow(-1);

    at::Tensor ws = w_ratios.matmul(scales).view(-1);
    at::Tensor hs = h_ratios.matmul(scales).view(-1);
    
    at::Tensor base_anchors = at::stack({-ws, -hs, ws, hs}, 1);
    return base_anchors.divide(2).round();
}


std::vector<at::Tensor> DefaultBoxes::GridAnchors(std::vector<at::Tensor>& grid_sizes, std::vector<at::Tensor>& strides){
    std::vector<at::Tensor> anchors, shift_grids;
    at::Tensor size, stride, base_anchors, shift_x, shift_y, shifts_x, shifts_y, shifts, centered_shifts;
    for (size_t i=0; i < grid_sizes.size(); i++){
        size = grid_sizes[i];
        stride = strides[i];
        base_anchors = this->cell_anchors_[i];

        shifts_y = at::arange(size[0].item<int>()) * stride[0].item<int>();
        shifts_x = at::arange(size[1].item<int>()) * stride[1].item<int>();

        shift_grids = at::meshgrid({shifts_y, shifts_x});
        shift_y = shift_grids[0].reshape(-1), shift_x = shift_grids[1].reshape(-1);
        shifts = at::stack({shift_x, shift_y, shift_x, shift_y}, 1);
        
        centered_shifts = shifts.view({-1,1,4}) + base_anchors.view({1,-1,4});
        anchors.push_back( centered_shifts.reshape({-1,4}) );
    }

    return anchors;
}


at::Tensor DefaultBoxes::GetFeatureAnchors(){
    std::vector<at::Tensor> strides, anchors_in_image;
    at::Tensor grid;
    for (size_t i=0; i<this->feature_grid_sizes_.size(); i++){
        grid = this->feature_grid_sizes_[i];
        strides.push_back(torch::tensor({this->image_size_[0] / grid[0].item<int>(), 
                                         this->image_size_[0] / grid[1].item<int>()}));
    }

    anchors_in_image = DefaultBoxes::GridAnchors(this->feature_grid_sizes_, strides);
    return at::cat(anchors_in_image);
}

std::vector<at::Tensor> DefaultBoxes::GetXYWH(){
    at::Tensor bboxes = DefaultBoxes::GetFeatureAnchors();
    at::Tensor widths = bboxes.index({"...",2}) - bboxes.index({"...",0});
    at::Tensor heights = bboxes.index({"...",3}) - bboxes.index({"...",1});

    at::Tensor ctr_x = bboxes.index({"...",0}) + 0.5*widths;
    at::Tensor ctr_y = bboxes.index({"...",1}) + 0.5*heights;

    return {ctr_x, ctr_y, widths, heights};//xywh;

}


std::vector<at::Tensor> DefaultBoxes::GetXYWH(bool split_tensors){

    std::vector<at::Tensor> xywh = DefaultBoxes::GetXYWH();
    if (!split_tensors)
        return xywh;

    auto arr_ref = torch::ArrayRef<long int>(this->split_num_anchors_per_level_);
    at::Tensor anchors_stack = at::stack(xywh).permute({1,0}); // should now be: num_anchors x 4
    std::vector<at::Tensor> out = anchors_stack.split_with_sizes(arr_ref,0);

    return out;
}


std::vector<std::vector<at::Tensor>> DefaultBoxes::SplitAnchors(){
    std::vector<at::Tensor> xywh = DefaultBoxes::GetXYWH();
    auto arr_ref = torch::ArrayRef<long int>(this->split_num_anchors_per_level_);
    at::Tensor x = xywh[0];
    at::Tensor y = xywh[1];
    at::Tensor w = xywh[2];
    at::Tensor h = xywh[3];

    /* Need to format to (cx,cy,width,height) */
    std::vector<at::Tensor> widths = w.split_with_sizes(arr_ref);
    std::vector<at::Tensor> heights = h.split_with_sizes(arr_ref);
    std::vector<at::Tensor> ctr_x = x.split_with_sizes(arr_ref);
    std::vector<at::Tensor> ctr_y = y.split_with_sizes(arr_ref);

    return {ctr_x, ctr_y, widths, heights};
}


std::vector<at::Tensor> DefaultBoxes::SplitFeatureAnchors(){
    at::Tensor anchors = DefaultBoxes::GetFeatureAnchors();
    auto arr_ref = torch::ArrayRef<long int>(this->split_num_anchors_per_level_);
    return anchors.split_with_sizes(arr_ref, 0);
}


