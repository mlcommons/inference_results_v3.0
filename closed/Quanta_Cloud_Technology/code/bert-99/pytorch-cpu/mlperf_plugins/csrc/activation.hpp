#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor i_gelu (
    const at::Tensor& input,
    const at::Scalar& M,
    const at::Scalar& oscale,
    const at::Scalar& o_off);

// Possible problem
at::Tensor i_gelu_ (
    at::Tensor& input,
    const at::Scalar& M,
    const at::Scalar& oscale,
    const at::Scalar& o_off);

at::Tensor i_identity (
    const at::Tensor& input,
    const c10::optional<at::Scalar>& M,
    const at::Scalar& oscale,
    const at::Scalar& o_off);

at::Tensor i_identity_cin (
    const at::Tensor& input,
    const at::Scalar& oscale);

at::Tensor i_identity_ (
    at::Tensor& self,
    const c10::optional<at::Scalar>& M,
    const at::Scalar& oscale,
    const at::Scalar& o_off);

}
