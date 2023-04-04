#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Scalar>& scale,
    const c10::optional<at::Scalar>& zero);

at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Scalar>& M,
    const c10::optional<at::Scalar>& scale,
    const c10::optional<at::Scalar>& zero);

at::Tensor prepack_linear_weight(
    const at::Tensor& weight);

at::Tensor baddbmm_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha);

at::Tensor matmul_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const c10::optional<at::Scalar>& oscale,
    const c10::optional<at::Scalar>& zero);

at::Tensor reorder_test(const at::Tensor& weight);
}
