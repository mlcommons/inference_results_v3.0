#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor amx_linear(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Scalar& scale, const bool post_op, const at::Scalar& o_scale);

at::Tensor amx_linear_i8o32(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Scalar& scale);

at::Tensor amx_linear_bf16_accum_relu(
    const at::Tensor& x0, const at::Tensor& w0, const at::Tensor& x1,
    const at::Tensor& w1, const at::Tensor& bias);

at::Tensor amx_linear_i16o32(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias);

}  // namespace intel_mlperf
