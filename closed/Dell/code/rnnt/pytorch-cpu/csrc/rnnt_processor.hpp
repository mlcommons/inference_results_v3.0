#pragma once

#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>
#include <torch/script.h>

#include <string>

namespace rnnt {
namespace models {
//
// Functionality:
//   1. Model load&adjust
//
class AudioProcessor {
public:
  AudioProcessor(const std::string filename)
      : model_(torch::jit::load(filename)) {
    model_.eval();
    socket_model_[0] = model_;
    socket_model_[1] = model_.clone();
  }

  AudioProcessor();

  void load(const std::string filename) { model_ = torch::jit::load(filename); }

  template <typename... Args>
  std::tuple<at::Tensor, at::Tensor> forward(
      at::Tensor& wav, at::Tensor& wav_lens, bool pad_batch_size) {
    auto res =
        model_.forward({wav, wav_lens, pad_batch_size}).toTuple()->elements();
    auto fea = res[0].toTensor();
    auto fea_lens = res[1].toTensor();
    return {fea, fea_lens};
  }

  template <typename... Args>
  std::tuple<at::Tensor, at::Tensor> forward(
      int socket, at::Tensor& wav, at::Tensor& wav_lens, bool pad_batch_size) {
    auto res = socket_model_[socket]
                   .forward({wav, wav_lens, pad_batch_size})
                   .toTuple()
                   ->elements();
    auto fea = res[0].toTensor();
    auto fea_lens = res[1].toTensor();
    return {fea, fea_lens};
  }

private:
  torch::jit::script::Module model_;
  torch::jit::script::Module socket_model_[2];
};

}  // namespace models
}  // namespace rnnt
