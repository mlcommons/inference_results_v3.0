#include <chrono>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include <csrc/aten/cpu/Nms.h>
#include "nms.h"
#include "dataset.hpp"
#include "utils.hpp"

#ifndef CHANNELS
#define CHANNELS 3
#endif

#ifndef IM_WIDTH
#define IM_WIDTH 800
#endif

#ifndef IM_HEIGHT
#define IM_HEIGHT 800
#endif

#ifndef IMG_BUF_SIZE
#define IMG_BUF_SIZE 1920000
#endif

std::tuple< std::vector<int>, std::vector<float> > ThresholdScores(const float* scores, const float threshold, size_t num_scores, int topk){
    std::vector<int> vec;
    std::vector<float> threshed_scores;
#pragma omp parallel
{
    std::vector<int> vec_private;
    std::vector<float> score_private;
    #pragma omp for nowait
    for (int i = 0; i < num_scores; ++i) {
        if (scores[i] > threshold){
            vec_private.push_back(i);
        }
    }

    #pragma omp critical
    vec.insert(vec.end(), vec_private.begin(), vec_private.end());
    // threshed_scores.insert(threshed_scores.end(), score_private.begin(), score_private.end());
}

    // TODO: Get topk scores
    std::stable_sort(vec.begin(), vec.end(), 
        [&scores](size_t i, size_t j){return scores[i] > scores[j];}
        );

    if (vec.size() > topk){ /* If there's more elements than required topk */
        vec.resize(topk);
    }

    // Now copy the scores
    threshed_scores.resize(vec.size());
#pragma omp parallel
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        threshed_scores[i] = scores[vec[i]];
    }
}
    return std::make_pair(vec, threshed_scores);
}

// Preprocess image file
const at::Tensor PreProcessSample(std::string filename, int height, int width){
    cv::Mat image = cv::imread(filename);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Resize after scaling
    cv::resize(image, image, cv::Size(height, width));

    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    // Normalize
    
    cv::Mat mean(height, width, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(height, width, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::subtract(image, mean, image);
    cv::divide(image, std, image);
    
    auto options = torch::TensorOptions().dtype(at::kFloat);
    at::Tensor tensor = torch::from_blob(image.data, {1, height, width, 3}).clone();
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(at::kFloat);
    tensor = tensor.to(torch::MemoryFormat::ChannelsLast);

    // Quantize
    tensor = torch::quantize_per_tensor(tensor, 0.02065122127532959, 0, at::kQInt8).int_repr();


    return tensor;
}


// Instantiate dataset and prefetches the image lists as well as ids, and sizes
Dataset::Dataset(const std::string data_path, size_t total_sample_count){

    this->total_sample_count_ = total_sample_count;
    std::cout << " Reading dataset \n";
    std::string annotations_file = data_path + "/annotations/openimages-mlperf.json";
    std::ifstream ifs(annotations_file);
    rapidjson::IStreamWrapper isw(ifs);
    
    rapidjson::Document annot_doc;
    try {

        annot_doc.ParseStream(isw);
    }
    catch (int e) {
        throw;
    }

    assert(annot_doc.IsObject());
    assert(annot_doc.HasMember("annotations"));
    assert(annot_doc.HasMember("images"));

    const rapidjson::Value& images = annot_doc["images"];
    assert(images.IsArray());

    for(size_t i=0; i < images.Size(); i++){
        std::string filename = data_path + "/validation/data/" + images[i]["file_name"].GetString();
        int id = images[i]["id"].GetInt();
        std::tuple<int, int> dims (images[i]["height"].GetInt(), images[i]["width"].GetInt());
        
        Dataset::image_list_.push_back( filename );
        Dataset::image_dims_.push_back( dims );
        Dataset::ids_.push_back( id );

        if (i == Dataset::TotalSampleCount()){
            break;
        }
    }
    std::cout << " Dataset read\n";

}


// Fetch processed samples for inference
at::Tensor Dataset::GetSamples(std::vector<mlperf::QuerySampleIndex>& samples){
    if (samples.size() == 1) return Dataset::data_in_memory_[samples[0]];
    std::vector<at::Tensor> tensors;
    for(auto& sample : samples){
        tensors.push_back( Dataset::data_in_memory_[sample] );
    }
    at::Tensor ret = at::cat(tensors, 0);
    return ret;
}

void Dataset::GetSamplesTensor(std::vector<mlperf::QuerySampleIndex>& samples, at::Tensor& tensor){
    int8_t* tensor_pointer = tensor.data_ptr<int8_t>();
    #pragma omp parallel// shared(num_samples, tensor_pointer)
    {
        for (size_t i = 0; i < samples.size(); i++) {
            int8_t* data_ptr = Dataset::data_in_memory_[samples[i]].data_ptr<int8_t>();
            memcpy(tensor_pointer+i*3*800*800, data_ptr, 3*800*800*sizeof(int8_t));
        }
    }
}

void Dataset::GetSamplesCopyTensor(std::vector<mlperf::QuerySampleIndex>& samples, at::Tensor& input){
    //std::cout << " : samples to batch " << samples << std::endl;
    int8_t* input_data_ptr = input.data_ptr<int8_t>();
    input_data_ptr = Dataset::data_in_memory_[samples[0]].data_ptr<int8_t>();
    //printf("Copying remainder tensor\n");
    for (size_t i = 1; i < samples.size(); i++) {
        //printf("Sample %ld\n", samples[i]);
        int8_t* data_ptr = Dataset::data_in_memory_[samples[i]].data_ptr<int8_t>();
        //printf("Actual copy\n");
        memcpy(input_data_ptr+i*3*800*800, data_ptr, 3*800*800*sizeof(int8_t));
    }
}

void Dataset::GetSamplesPtr(std::vector<mlperf::QuerySampleIndex>& samples, int8_t* input_data_ptr){
    std::cout << " : samples to batch " << samples << std::endl;
    input_data_ptr = Dataset::data_in_memory_[samples[0]].data_ptr<int8_t>();
    printf("Copying remainder tensor\n");
    for (size_t i = 1; i < samples.size(); i++) {
        printf("Sample %ld\n", samples[i]);
        int8_t* data_ptr = Dataset::data_in_memory_[samples[i]].data_ptr<int8_t>();
        printf("Actual copy\n");
        memcpy(input_data_ptr+i*3*800*800, data_ptr, 3*800*800*sizeof(int8_t));
    }
}

// Fetch image file at given index
std::string Dataset::GetFileAtIndex(size_t index){
    return Dataset::image_list_[index];
}

// Load samples into memory
void Dataset::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples){
    size_t batch_size = 2;
    for(auto &sample : samples){
        std::string filename = Dataset::GetFileAtIndex(sample);
        at::Tensor tensor = PreProcessSample( filename, Dataset::height_, Dataset::width_ ).contiguous();
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
const std::string& Dataset::Name() {
    return name_;
}


RetinaNetDataset::RetinaNetDataset(const std::string data_path, size_t total_sample_count) : Dataset(data_path, total_sample_count){
    auto options = torch::TensorOptions().dtype(at::kInt);
    torch::Tensor gs_tensor = torch::tensor({32, 64, 128, 256, 512}).view({1,5});
    torch::Tensor ar_tensor = torch::tensor({0.5, 1.0, 2.0}).view({1,3});
    torch::Tensor anchor_sizes = at::cat({gs_tensor, gs_tensor * std::cbrt(2), gs_tensor * std::pow(2,2.0/3)});

    anchor_sizes = anchor_sizes.to(at::kInt);
    anchor_sizes = anchor_sizes.transpose(1,0);

    torch::Tensor aspect_ratios = at::repeat_interleave(ar_tensor, 5, 0, 5);

    std::vector<at::Tensor> feature_sz = {torch::tensor({100,100}),
                                          torch::tensor({50,50}),
                                          torch::tensor({25,25}),
                                          torch::tensor({13,13}),
                                          torch::tensor({7,7})};

    DefaultBoxes dboxes(anchor_sizes, aspect_ratios, feature_sz);
    this->dboxes_xywh_ = dboxes.GetXYWH(true);
    split_anchors_ = dboxes.SplitAnchors();//split default anchors according by feature level into(xc,yc,w,h)
    cx_ = split_anchors_[0];
    cy_ = split_anchors_[1];
    sw_ = split_anchors_[2];
    sh_ = split_anchors_[3];

    for (auto& level_anchors : split_anchors_[0]){
        vec_arr_.push_back(level_anchors.size(0));
    }
    this->arr_ref_ = torch::ArrayRef<long int>(vec_arr_);
}

void RetinaNetDataset::PostProcess( Item qitem, torch::jit::IValue& output){

    auto t0 = std::chrono::high_resolution_clock::now();

    // Get output scores as vector of level-batched scores
    //printf(" --Retrieving logits\n");
    auto logits = output.toTuple()->elements()[0].toTuple()->elements()[0].toTuple()->elements();
    std::vector<at::Tensor> fpn_batch_scores = {logits[0].toTensor(),
                                                logits[1].toTensor(),
                                                logits[2].toTensor(),
                                                logits[3].toTensor(),
                                                logits[4].toTensor()};

    //printf(" --Retrieving boxes\n");
    // Get output boxes as vector of level-batched boxes
    auto boxes = output.toTuple()->elements()[0].toTuple()->elements()[1].toTuple()->elements();
    std::vector<at::Tensor> fpn_batch_boxes = {boxes[0].toTensor(),
                                                boxes[1].toTensor(),
                                                boxes[2].toTensor(),
                                                boxes[3].toTensor(),
                                                boxes[4].toTensor()};

    // Scale regression boxes according to anchor boxes
    at::Tensor labels, bboxes;
    std::vector<at::Tensor> image_result;
    std::vector<mlperf::QuerySampleResponse> batch_responses;

    std::vector<at::Tensor> scores;
    std::vector<at::Tensor> rel_boxes;
    scores.resize(fpn_batch_scores.size());
    rel_boxes.resize(fpn_batch_boxes.size());

    for(long int i=0; i < fpn_batch_scores[0].size(0) - qitem.number_dummies; i++){
        for (int level=0; level < scores.size(); level++){
            scores[level] = fpn_batch_scores[level][i].squeeze(0);
            rel_boxes[level] = fpn_batch_boxes[level][i].squeeze(0);
        }

        // Decode the image
        image_result = RetinaNetDataset::DecodeSingle(rel_boxes, scores);


        if (image_result.size()==0){ // No detections in image
            std::vector<float> processed_results;
            mlperf::QuerySampleResponse response{ qitem.response_ids_[i],
              reinterpret_cast<std::uintptr_t>(&processed_results), 0};
            mlperf::QuerySamplesComplete(&response, 1);
            continue;
        }

        at::Tensor boxes = image_result[0];//.unsqueeze(0);
        at::Tensor scores = image_result[1];
        at::Tensor labels = image_result[2];


        // batch_score_nms returns bboxes for single image
        auto index = qitem.sample_idxs_[i];
        std::vector<float> processed_results = torch_ipex::batch_score_nms(boxes, scores, labels, this->image_height_, this->image_width_, this->nms_thresh_, this->detections_per_img_, {index});

        mlperf::QuerySampleResponse response{ qitem.response_ids_[i],
          reinterpret_cast<std::uintptr_t>(&processed_results[0]),
          (sizeof(float) * processed_results.size())};
        
        mlperf::QuerySamplesComplete(&response, 1);
    }
}


/* Decode single image predictions: Threshold scores and scale boxes */
std::vector<at::Tensor> RetinaNetDataset::DecodeSingle(std::vector<at::Tensor>& bboxes, std::vector<at::Tensor>& logits){
    std::tuple<at::Tensor, at::Tensor> out;
    std::vector<at::Tensor> image_boxes, image_scores, image_labels;

    auto options_int = torch::TensorOptions().dtype(at::kInt);
    auto options_float = torch::TensorOptions().dtype(at::kFloat);
    std::vector<int> indices;
    std::vector<float> threshed_scores;

    // Iterate through each feature levels
    at::Tensor level_boxes, level_anchors, level_logits, scaled_bboxes, scores_per_level, topk_idxs, keep_idxs, idxs, labels_per_level, topk_boxes;
    long int num_topk;
    for (size_t i=0; i < bboxes.size(); i++){ /* per_level_boxes */
        level_anchors = this->dboxes_xywh_[i]; // Get anchors at this feature level
        level_boxes = bboxes[i]; // BBoxes at this feature level
        scores_per_level = logits[i]; // For Logits at this feature level

        float* scores_ptr = scores_per_level.data_ptr<float>();

        size_t num_scores = scores_per_level.size(0);
        std::tie(indices, threshed_scores) = ThresholdScores(scores_ptr, this->score_thresh_, num_scores, this->topk_candidates_); // This should return topk scores and indices

        if (indices.size() < 1) continue; // No detection meets score threshold

        at::Tensor threshed_score_tensor = torch::from_blob(threshed_scores.data(), {threshed_scores.size()}, options_float).clone();
        at::Tensor indices_tensor = torch::from_blob(indices.data(), {indices.size()}, options_int);

        // num_topk = std::min(indices_tensor.size(0),this->topk_candidates_); // limit no. of topk per lev

        // // Compute topk
        // // TODO: Use custom topk implementation - topk is added to thresholding but accuracy is low
        // auto topk_out = threshed_score_tensor.topk(num_topk);

        // threshed_score_tensor = std::get<0>(topk_out); // Topk scores
        // idxs = std::get<1>(topk_out);
        // if (idxs.size(0) < 1) continue;

        // indices_tensor = indices_tensor.index_select(0,idxs); // Select the topk indexes

        // Map selected topk indicesto their corresponding anchors boxes.
        // An anchor box may be repeated (typical) if multiple class scoresfor a particular box passes the 'score_thres_'
        //TODO: Select the unique anchor boxes - repeatitions across different class_labels should be removed
        auto anchor_idxs = at::div(indices_tensor, this->num_classes_,"floor");
        labels_per_level = indices_tensor.remainder(this->num_classes_); // Map selected indices to their correct classes

        topk_boxes = level_boxes.index_select(0,anchor_idxs);

        /* Scale the topk predicted boxes at this feature level */
        scaled_bboxes = RetinaNetDataset::ScaleFeatureBboxes(topk_boxes, anchor_idxs, i);
        image_boxes.push_back(scaled_bboxes);
        image_scores.push_back(threshed_score_tensor);
        image_labels.push_back(labels_per_level);
    }

    if (image_boxes.size() < 1){
        return image_boxes; // TODO: Return a proper return type - i.e a vector of 3 tensors
    }

    at::Tensor cat_image_boxes = at::cat(image_boxes);//.unsqueeze(0);
    at::Tensor cat_image_scores = at::cat(image_scores);//.unsqueeze(0);
    at::Tensor cat_image_labels = at::cat(image_labels);//.unsqueeze(0);
    return {cat_image_boxes, cat_image_scores, cat_image_labels};
}



// Scale the selected bounding boxes
at::Tensor RetinaNetDataset::ScaleFeatureBboxes(at::Tensor& level_bboxes, at::Tensor& anchor_idxs, int level_id){
    /*
        level_boxes->Tensor: Predicted bboxes (for a feature level) that have been thresholded
        anchor_idxs->Tensor: Prediction indexes to select from anchor boxes for this level
        level_id -> int: The id of the feature level we're processing
    */

    // Select cx,cy,w,h at the 'level_id' feature level, and at the 'topk' anchor_idxs
    at::Tensor widths = this->sw_[level_id].index_select(0,anchor_idxs);
    at::Tensor heights = this->sh_[level_id].index_select(0,anchor_idxs);

    at::Tensor ctr_x = this->cx_[level_id].index_select(0,anchor_idxs);
    at::Tensor ctr_y = this->cy_[level_id].index_select(0,anchor_idxs);

    // Clamp the predicted boxes before taking exp
    at::Tensor dx = level_bboxes.index({"...",0});
    at::Tensor dy = level_bboxes.index({"...",1});
    at::Tensor dw = level_bboxes.index({"...",2}).clamp_max(this->bbox_xform_clip_);
    at::Tensor dh = level_bboxes.index({"...",3}).clamp_max(this->bbox_xform_clip_);

    at::Tensor pred_ctr_x = dx.mul(widths).add(ctr_x);
    at::Tensor pred_ctr_y = dy.mul(heights).add(ctr_y);

    at::Tensor pred_w = at::exp(dw).mul(widths);
    at::Tensor pred_h = at::exp(dh).mul(heights);

    at::Tensor c_to_c_h = 0.5 * pred_h;
    at::Tensor c_to_c_w = 0.5 * pred_w;

    // Shift and clamp box predicted coordinates
    at::Tensor pred_boxes1 = pred_ctr_x.subtract(c_to_c_w);
    at::Tensor pred_boxes2 = pred_ctr_y.subtract(c_to_c_h);
    at::Tensor pred_boxes3 = pred_ctr_x.add(c_to_c_w);
    at::Tensor pred_boxes4 = pred_ctr_y.add(c_to_c_h);
    
    at::Tensor pred_boxes = at::stack({pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4},1);

    // Clamp prediction boxes
    pred_boxes = pred_boxes.clamp(0,this->image_height_); // Should clamp per w/h if h != w

    return pred_boxes;

}


