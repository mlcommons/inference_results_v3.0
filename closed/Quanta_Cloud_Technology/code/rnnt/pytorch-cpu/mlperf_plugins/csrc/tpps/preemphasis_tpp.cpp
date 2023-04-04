#include "preemphasis_tpp.hpp"
#include <cstdlib>
#include <stdexcept>

namespace intel_mlperf {

void preemphasis_tpp::ref_head(float *pout, float *pin, int64_t rl,
                               float coeff) {
  if (rl > 16)
    throw std::runtime_error("length should not be larger than 16.");

  auto vcoeff = _mm512_set1_ps(coeff);
  // Head & Tail
  {
    auto rem = rl;
    __mmask16 k_a = (1 << rem) - 1;
    __mmask16 k_b = k_a << 1;
    auto zeros = _mm512_setzero_ps();
    auto a = _mm512_mask_loadu_ps(zeros, k_a, &pin[0]);
    auto b = _mm512_mask_loadu_ps(zeros, k_b, &pin[-1]);
    auto o = a - vcoeff * b;
    _mm512_mask_storeu_ps(&pout[0], k_a, o);
  }
}

void preemphasis_tpp::ref(float *pout, float *pin, int64_t rl, float coeff) {
  int64_t d = 0;
  auto vcoeff = _mm512_set1_ps(coeff);

  // Head
  if (rl <= 16) {
    return ref_head(pout, pin, rl, coeff);
  } else {
    ref_head(pout, pin, 16, coeff);
  }
  // Body
  for (d = 16; d < rl / 16 * 16; d += 16) {
    auto a = _mm512_loadu_ps(&pin[d]);
    auto b = _mm512_loadu_ps(&pin[d - 1]);
    auto o = a - vcoeff * b;
    _mm512_mask_storeu_ps(&pout[d], 0xffff, o);
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm512_setzero_ps();
    auto a = _mm512_mask_loadu_ps(zeros, k, &pin[d]);
    auto b = _mm512_mask_loadu_ps(zeros, k, &pin[d - 1]);
    auto o = a - vcoeff * b;
    _mm512_mask_storeu_ps(&pout[d], k, o);
  }
}

} // namespace intel_mlperf
