#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor frame_splicing(
    const at::Tensor &input, const at::Tensor &length, const at::Scalar &factor);

}
