#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

bool greedy_decode_update(
    const at::Tensor &symbols,
    at::Tensor &symbols_added,
    at::Tensor &res,
    at::Tensor &res_idx,
    const at::Tensor &f,
    const at::Tensor &f_lens,
    at::Tensor &time_idx,
    at::Tensor &fi,
    at::Tensor &pre_g,
    const std::vector<at::Tensor> &pre_hg,
    const std::vector<at::Tensor> &pre_cg,
    const std::vector<at::Tensor> &hg,
    const std::vector<at::Tensor> &cg);

}
