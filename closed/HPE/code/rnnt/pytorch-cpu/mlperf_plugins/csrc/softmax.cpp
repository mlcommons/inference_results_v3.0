#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>

#include "i_softmax_tpp.hpp"

namespace intel_mlperf {
at::Tensor i_softmax(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale) {

  auto in_sz = input.sizes();

  i_softmax_tpp<16> compute(in_sz[0], in_sz[1], in_sz[2], in_sz[3]);

  // We only have ref yet.
  auto output = at::empty(in_sz,
      at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  auto att_sz = att_mask.sizes();

  if (att_sz[1] == 1) { /* all broadcasting */
    auto* patt = reinterpret_cast<int32_t *>(att_mask.data_ptr());
    compute.ref(
        output.data_ptr(), input.data_ptr(),
        patt, M.toFloat(), oscale.toFloat());
  } else {
    auto* patt = reinterpret_cast<float *>(att_mask.data_ptr());
    compute.ref(
        output.data_ptr(), input.data_ptr(),
        patt, M.toFloat(), oscale.toFloat());
  }

  return output;
}

at::Tensor i_softmax_u(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale) {

  auto in_sz = input.sizes();

  i_softmax_tpp<16> compute(in_sz[0], in_sz[1], in_sz[2], in_sz[3]);

  // We only have ref yet.
  auto output = at::empty(in_sz,
      at::TensorOptions().dtype<uint8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  auto att_sz = att_mask.sizes();

  if (att_sz[1] == 1) { /* all broadcasting */
    auto* patt = reinterpret_cast<int32_t *>(att_mask.data_ptr());
    compute.ref(
        output.data_ptr(), input.data_ptr(),
        patt, M.toFloat(), oscale.toFloat());
  } else {              /* line broadcasting */
    auto* patt = reinterpret_cast<float *>(att_mask.data_ptr());
    compute.ref(
        output.data_ptr(), input.data_ptr(),
        patt, M.toFloat(), oscale.toFloat());
  }

  return output;
}

at::Tensor i_softmax_(
    at::Tensor& self,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale) {

  auto in_sz = self.sizes();

  i_softmax_tpp<16> compute(in_sz[0], in_sz[1], in_sz[2], in_sz[3]);

  compute.ref(
      self.data_ptr(), self.data_ptr(),
      (int32_t *)att_mask.data_ptr(), M.toFloat(), oscale.toFloat());

  return self;
}
}
