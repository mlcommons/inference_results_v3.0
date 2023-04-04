#include "stack_time.hpp"
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <stdexcept>

namespace intel_mlperf {
/*
 * 1. Mask pre-padded
 * 2. Convert x from {T, N, C} to {⌈T/factor⌉, N, C*factor}
 * 3. Calculate x_lens from [ti]*N to [⌈ti/factor⌉]*N
 * e.g.
 * input: {T=3, N=2, C=4}, input_lens: [3, 2] (# represents padded value)
 * [| 0  | 1  | 2  | 3  |
 *  | 4  | 5  | 6  | 7  |]
 * [| 8  | 9  | 10 | 11 |
 *  | 12 | 13 | 14 | 15 |]
 * [| 16 | 17 | 18 | 19 |
 *  | #  | #  | #  | #  |]
 * output: {2, 2, 8}, output_lens: [2, 1] (* represents zero)
 * [| 0  | 1  | 2  | 3  | 8  | 9  | 10 | 11 |
 *  | 4  | 5  | 6  | 7  | 12 | 13 | 14 | 15 |],
 * [| 16 | 17 | 18 | 19 | *  | *  | *  | *  |
 *  | *  | *  | *  | *  | *  | *  | *  | *  |]
 * Notice: for int8 only
*/
at::Tensor stack_time(
    const at::Tensor &input,
    const at::Tensor &input_lens,
    const at::Scalar &factor) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  std::vector<int32_t> input_lens_(input_lens.data_ptr<int32_t>(), input_lens.data_ptr<int32_t>() + input_lens.numel());
  auto factor_ = factor.toInt();
  auto seq_len = in_sz[0];
  auto batch_size = in_sz[1];
  auto hidden_size = in_sz[2];
  auto padded_len = (seq_len + factor_ - 1) / factor_;
  auto output = at::empty({padded_len, batch_size, hidden_size * factor_}, input.dtype());

  auto in = input.accessor<int8_t, 3>();
  auto out = output.accessor<int8_t, 3>();

#pragma omp parallel for collapse(1)
  for (int32_t ni = 0; ni < batch_size; ++ni) {
    for (int32_t ti = 0; ti < input_lens_[ni]; ++ti) {
      int32_t c_offset = ti % factor_ * hidden_size;
      memcpy(&out[ti / factor_][ni][c_offset], &in[ti][ni][0], hidden_size);
    }
    for (int32_t ti = input_lens_[ni]; ti < padded_len * factor_; ++ti) {
      int32_t c_offset = ti % factor_ * hidden_size;
      memset(&out[ti / factor_][ni][c_offset], 0, hidden_size);
    }
  }

  return output;
}

} // namespace intel_mlperf
