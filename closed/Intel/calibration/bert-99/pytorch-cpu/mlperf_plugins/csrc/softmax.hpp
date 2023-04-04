#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_softmax(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale);

at::Tensor i_softmax_u(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale);

at::Tensor i_softmax_(
    at::Tensor& self,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale);
}
