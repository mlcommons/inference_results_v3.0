#ifndef BACKEND_H_
#define BACKEND_H_

#include <vector>
#include <string>
#include <queue>

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

class Backend {
public:

    Backend( std::string pth_file) {
        model_filename_ = pth_file;
    }

    void load(){
        model_ = torch::jit::load(model_filename_);
        model_.eval();
    }

    torch::jit::IValue predict(at::Tensor& input_tensor){
        auto out = model_({input_tensor});
        return out;
    }

    /*
    std::vector<at::Tensor> predict(at::Tensor input_tensor){
        std::cout << " ********************** In inference step *******************\n";
        auto out = model_({input_tensor});
        return {out.toTuple()->elements()[0].toTensor(), out.toTuple()->elements()[1].toTensor()};
    }
    */

private:

    torch::jit::script::Module model_;
    std::string model_filename_;

}; // BACKEND_H_

#endif
