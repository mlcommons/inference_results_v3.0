#include "frame_splicing.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <stdexcept>

#include "tpps/frame_splicing_tpp.hpp"

namespace intel_mlperf {
at::Tensor frame_splicing(
    const at::Tensor &input, const at::Tensor &length, const at::Scalar &factor) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  // N * C * T -> N * (C*factor) * (T//factor)
  auto batch = in_sz[0];
  auto feat = in_sz[1];
  auto time = in_sz[2];
  auto factor_ = factor.toInt();
  auto padded_time = ((time + factor_ - 1) / factor_);
  auto output = at::empty(
      {batch, feat * factor_, padded_time},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto in = input.accessor<float, 3>();
  auto out = output.accessor<float, 3>();
  auto len = length.accessor<int32_t, 1>();

  if (factor_ == 3) {
#pragma omp parallel for
    for (auto i = 0; i < batch; i++) {
      for (int32_t j = 0; j < feat; j++) {
        frame_splicing_tpp<3>::ref(
            &out[i][j][0], &in[i][j][0], feat, len[i], padded_time);
      }
    }
  } else {
    throw std::runtime_error("Unsupported frame splicing factor.");
  }

  return output;
}

}  // namespace intel_mlperf
