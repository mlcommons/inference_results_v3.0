#ifndef IMAGENET_PROCESSOR_
#define IMAGENET_PROCESSOR_

//#include <torch/torch.h>
#include <vector>
#include <list>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include "dataset.hpp"

using torch::indexing::Slice;



// TODO: Move this to 'dataset' header, and rename 'dataset.hpp' to base_dataset.hpp
class ImageNet : public Dataset {
public:
    ImageNet(const std::string data_path, size_t total_sample_count);
    std::vector<mlperf::QuerySampleResponse> PostProcess( Item qitem, torch::jit::IValue& output);
    void resizeWithAspectratio(cv::Mat* image, cv::Mat* resized_image,int out_height, int out_width, int interpol ) ;
    void centerCrop(cv::Mat* image, int out_height, int out_width, cv::Mat* cropped_image) ;
  

private:

    int image_width_ = 224;
    int image_height_ = 224;

};

#endif

