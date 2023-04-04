#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor power_spectrum(const at::Tensor &input, const at::Tensor &length);

}
