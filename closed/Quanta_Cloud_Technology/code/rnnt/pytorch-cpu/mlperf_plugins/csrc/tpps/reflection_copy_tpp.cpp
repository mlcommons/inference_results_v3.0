#include "reflection_copy_tpp.hpp"

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

void reflection_copy_tpp::ref(float *pout, float *pin, int64_t rl) {
  int64_t d = 0;
  auto c0 = _mm512_set1_ps(0.0);
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto a = _mm512_loadu_ps(&pin[rl - 16 - d]);
    auto p = _mm512_permute_ps(a, _MM_SHUFFLE(0, 1, 2, 3));
    auto o = _mm512_shuffle_f32x4(p, p, _MM_SHUFFLE(0, 1, 2, 3));
    _mm512_storeu_ps(&pout[d], o);
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto a = _mm512_mask_loadu_ps(c0, k << (16 - rem), &pin[rem - 16]);
    auto p = _mm512_permute_ps(a, _MM_SHUFFLE(0, 1, 2, 3));
    auto o = _mm512_shuffle_f32x4(p, p, _MM_SHUFFLE(0, 1, 2, 3));
    _mm512_mask_storeu_ps(&pout[d], k, o);
  }
}

}  // namespace intel_mlperf
