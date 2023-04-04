#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor amx_linear(
  const at::Tensor& input,
  const at::Tensor& weight,
  const at::Tensor& bias,
  const at::Scalar& scale, 
  const bool post_op,
  const at::Scalar& o_scale
);

}