#include "preemphasis.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "tpps/preemphasis_tpp.hpp"
#include "tpps/reflection_copy_tpp.hpp"

namespace intel_mlperf {
at::Tensor preemphasis(
    const at::Tensor &input, const at::Tensor &length,
    const at::optional<at::Scalar> &coeff, const at::optional<at::Scalar> &pad_size) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto seq_len = in_sz[1];
  auto pad_size_ = pad_size.value_or(0).toInt();
  auto output = at::empty(
      {batch, seq_len + 2 * pad_size_},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto src = input.accessor<float, 2>();
  auto dst = output.accessor<float, 2>();
  auto len = length.accessor<int32_t, 1>();
  auto coeff_ = coeff.value_or(0.97f).toFloat();

  if (coeff_ == 0.0f) {
    output = input;
  } else {
#pragma omp parallel for
    for (auto i = 0; i < batch; i++) {
      preemphasis_tpp::ref(&dst[i][pad_size_], &src[i][0], len[i], coeff_);
      if (pad_size_ > 0) {
        reflection_copy_tpp::ref(&dst[i][0], &dst[i][pad_size_ + 1], pad_size_);
        reflection_copy_tpp::ref(
            &dst[i][pad_size_ + len[i]], &dst[i][len[i] - 1], pad_size_);
      }
      memset(&dst[i][len[i] + 2 * pad_size_], 0, sizeof(float) * (seq_len - len[i]));
    }
  }

  return output;
}

}  // namespace intel_mlperf
