#include "power_spectrum_tpp.hpp"

#include <cstdlib>
#include <stdexcept>

#include "el_common_intrin.hpp"

namespace intel_mlperf {

inline __m512 helper(__m512 a, __m512 b) {
  auto c0 = _mm512_set1_ps(0.0);
  // a, 0, b, 0, c, 0, d, 0
  auto p0 = _mm512_mask_add_ps(c0, 0x5555, a, b);
  // a, b, 0, 0, c, d, 0, 0
  auto p1 = _mm512_permute_ps(p0, _MM_SHUFFLE(3, 1, 2, 0));
  // 0, 0, a, b, 0, 0, c, d
  auto p2 = _mm512_permute_ps(p1, _MM_SHUFFLE(1, 0, 3, 2));
  // 0, 0, c, d, 0, 0, a, b
  auto p3 = _mm512_shuffle_f32x4(p2, p2, _MM_SHUFFLE(2, 3, 0, 1));
  // a, b, c, d, 0, 0, 0, 0
  auto p4 = _mm512_mask_add_ps(c0, 0x0f0f, p1, p3);
  return p4;
}

void power_spectrum_tpp::ref(float *pout, float *pin, int64_t rl) {
  int64_t d = 0;
  auto c0 = _mm512_set1_ps(0.0);
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto a0 = _mm512_loadu_ps(&pin[d * 2]);
    auto a0_2 = a0 * a0;
    auto b0_2 = _mm512_permute_ps(a0_2, _MM_SHUFFLE(2, 3, 0, 1));
    auto p0 = helper(a0_2, b0_2);
    // a, b, c, d, e, f, g, i, 0, ..., 0
    auto o0 = _mm512_shuffle_f32x4(p0, p0, _MM_SHUFFLE(3, 1, 2, 0));

    auto a1 = _mm512_loadu_ps(&pin[d * 2 + 16]);
    auto a1_2 = a1 * a1;
    auto b1_2 = _mm512_permute_ps(a1_2, _MM_SHUFFLE(2, 3, 0, 1));
    auto p1 = helper(a1_2, b1_2);
    // 0, ..., 0, a, b, c, d, e, f, g, i
    auto o1 = _mm512_shuffle_f32x4(p1, p1, _MM_SHUFFLE(2, 0, 3, 1));

    _mm512_storeu_ps(&pout[d], o0 + o1);
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;

    if (rem < 8) {
      __mmask16 k0 = (1 << (rem * 2)) - 1;
      auto a0 = _mm512_mask_loadu_ps(c0, k0, &pin[d * 2]);
      auto a0_2 = a0 * a0;
      auto b0_2 = _mm512_permute_ps(a0_2, _MM_SHUFFLE(2, 3, 0, 1));
      auto p0 = helper(a0_2, b0_2);
      // a, b, c, d, e, f, g, i, 0, ..., 0
      auto o0 = _mm512_shuffle_f32x4(p0, p0, _MM_SHUFFLE(3, 1, 2, 0));
      _mm512_mask_storeu_ps(&pout[d], k, o0);
    } else {
      auto a0 = _mm512_loadu_ps(&pin[d * 2]);
      auto a0_2 = a0 * a0;
      auto b0_2 = _mm512_permute_ps(a0_2, _MM_SHUFFLE(2, 3, 0, 1));
      auto p0 = helper(a0_2, b0_2);
      // a, b, c, d, e, f, g, i, 0, ..., 0
      auto o0 = _mm512_shuffle_f32x4(p0, p0, _MM_SHUFFLE(3, 1, 2, 0));

      __mmask16 k1 = (1 << ((rem - 8) * 2)) - 1;
      auto a1 = _mm512_mask_loadu_ps(c0, k1, &pin[d * 2 + 16]);
      auto a1_2 = a1 * a1;
      auto b1_2 = _mm512_permute_ps(a1_2, _MM_SHUFFLE(2, 3, 0, 1));
      auto p1 = helper(a1_2, b1_2);
      // 0, ..., 0, a, b, c, d, e, f, g, i
      auto o1 = _mm512_shuffle_f32x4(p1, p1, _MM_SHUFFLE(2, 0, 3, 1));
      _mm512_mask_storeu_ps(&pout[d], k, o0 + o1);
    }
  }
}

}  // namespace intel_mlperf
