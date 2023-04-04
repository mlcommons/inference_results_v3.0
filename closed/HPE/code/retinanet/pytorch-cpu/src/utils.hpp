#ifndef RETINANET_PROCESSOR_
#define RETINANET_PROCESSOR_

//#include <torch/torch.h>
#include <vector>
#include <list>
#include <math.h>

#include <torch/torch.h>
#include <ATen/ATen.h>

#include "dataset.hpp"

using torch::indexing::Slice;

class DefaultBoxes {

public:

    DefaultBoxes(const at::Tensor& anchor_sizes, const at::Tensor& aspect_ratios, std::vector<at::Tensor>& feature_grid_sizes);
    
    ~DefaultBoxes(){};

    std::vector<int> NumAnchorsPerLocations();

    at::Tensor GenerateAnchors( std::tuple<int,int,int>anchor_size,
                                std::tuple<float,float,float> aspect_ratio);

    at::Tensor GenerateAnchors(const at::Tensor& anchor_size, const at::Tensor& aspect_ratios);

    std::vector<at::Tensor> GridAnchors(std::vector<at::Tensor>& grid_sizes, std::vector<at::Tensor>& strides);

    at::Tensor GetFeatureAnchors();
    
    std::vector<at::Tensor> SplitFeatureAnchors();

    std::vector<std::vector<at::Tensor>> SplitAnchors();

    std::vector<at::Tensor> GetXYWH(); // Returns a vector of cx, cy, w, h

    std::vector<at::Tensor> GetXYWH(bool split_anchors); // Split anchor boxes by levels

private:
    
    at::Tensor anchor_sizes_; // 5 x 3
    at::Tensor aspect_ratios_; // 5 X 3
    int num_scales_ = 3; // [0.5, 1.0, 2.0]
    int num_anchors_ = 3;
    std::vector<at::Tensor> cell_anchors_;
    std::vector<int> image_size_ = {800,800};

    std::vector<at::Tensor> feature_grid_sizes_;
    std::vector<long int> split_num_anchors_per_level_; 

}; // DefaultBoxes


// TODO: Move this to 'dataset' header, and rename 'dataset.hpp' to base_dataset.hpp
class RetinaNetDataset : public Dataset {
public:
    RetinaNetDataset(const std::string data_path, size_t total_sample_count);

    /* Post-process output of query item */
    void PostProcess( Item qitem, torch::jit::IValue& output);

    /* Decode single image predictions: Threshold scores and scale boxes */
    std::vector<at::Tensor> DecodeSingle(std::vector<at::Tensor>& bboxes, std::vector<at::Tensor>& logits);

    // Scale the selected bounding boxes
    at::Tensor ScaleFeatureBboxes(at::Tensor& level_bboxes, at::Tensor& anchor_idxs, int level_id);

private:
    std::vector<std::vector<at::Tensor>> split_anchors_; // split default anchors - 4 entries: x,y,w,h
    std::vector<at::Tensor> dboxes_xywh_;
    std::vector<at::Tensor> cx_, cy_, sw_, sh_; // center_x, center_y, shift_width, shift_height
    at::IntArrayRef arr_ref_;
    std::vector<long int> vec_arr_;

    // Post-processing parameters
    float score_thresh_ = -2.944; //0.05; Used when checking x > -ln(18.99) vs sigmoid(x) > 0.05
    float nms_thresh_ = 0.5;
    float fg_iou_thresh_ = 0.5;
    float bg_iou_thresh_ = 0.4;
    float bbox_xform_clip_ = std::log(1000. / 16);

    int detections_per_img_ = 1000;
    long int topk_candidates_ = 1000;
    long int num_classes_ = 264;
    int image_width_ = 800;
    int image_height_ = 800;

};

#endif

