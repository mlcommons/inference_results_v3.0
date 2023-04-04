#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm(
    const at::Tensor& x,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights);

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_1dnn(
    const at::Tensor& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& b_ih,
    const at::Tensor& b_hh);

std::tuple<at::Tensor, at::Tensor> prepack_lstm_weights (
    const at::Tensor& w_ih,
    const at::Tensor& w_hh);

}
