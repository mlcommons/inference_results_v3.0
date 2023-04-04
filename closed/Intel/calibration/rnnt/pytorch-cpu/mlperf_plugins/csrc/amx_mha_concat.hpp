#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor amx_mha_concat(
    const at::Tensor& qkv,
    const at::Tensor& att_mask,
    const at::Tensor& length_ids,
    const at::Scalar& m1,
    const at::Scalar& oscale,
    const at::Scalar& m2
);

}
