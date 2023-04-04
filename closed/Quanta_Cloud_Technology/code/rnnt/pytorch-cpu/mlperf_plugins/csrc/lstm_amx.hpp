#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm_amx_int8(
    const at::Tensor &x, const std::vector<at::Tensor> &hx,
    const std::vector<at::Tensor> &cx,
    const std::vector<std::vector<at::Tensor>> all_weights, const at::Tensor &rb_scale,
    const at::Tensor &i_scale, const at::Tensor &o_scale, const bool skip_quant_y);

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_amx_int8(
    const at::Tensor &x, const at::Tensor &hx, const at::Tensor &cx,
    const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &b_ih,
    const at::Tensor &b_hh, const c10::optional<at::Scalar> &rb_scale,
    const c10::optional<at::Scalar> &i_scale, const c10::optional<at::Scalar> &o_scale,
    const bool skip_quant_y);

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm_amx_bf16(
    const at::Tensor &x, const std::vector<at::Tensor> &hx,
    const std::vector<at::Tensor> &cx,
    const std::vector<std::vector<at::Tensor>> all_weights);

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_amx_bf16(
    const at::Tensor &x, const at::Tensor &hx, const at::Tensor &cx,
    const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &b_ih,
    const at::Tensor &b_hh);

}  // namespace intel_mlperf
