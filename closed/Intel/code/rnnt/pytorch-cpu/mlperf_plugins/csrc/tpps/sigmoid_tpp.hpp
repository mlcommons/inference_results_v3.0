#pragma once
#include <immintrin.h>

#include <cstdlib>

#include "el_common_intrin.hpp"

namespace intel_mlperf {

template <int vec_length, int unroll = 4>
class sigmoid_tpp {
public:
  static void ref(void *out, void *in, int64_t line);
  static void ref_f32(void *out, void *in, int64_t line);
};

template <int vec_l, int N>
struct sigmoid_fp32 {
  inline static void run(float *out, float *in);
};

template <int N>
struct sigmoid_fp32<16, N> {
  static constexpr int64_t batch = 16 * N;
  inline static void run(float *out, float *in) {
#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      auto x = _mm512_loadu_ps(&in[i * 16]);
      auto o = _mm512_sigmoid_ps(x);
      _mm512_store_ps(&out[i * 16], o);
    }
  }
};

template <int vec_l, int N>
struct sigmoid_fp16 {
  inline static void run(_Float16 *out, float *in);
};

template <int N>
struct sigmoid_fp16<32, N> {
  static constexpr int64_t batch = 32 * N;

  inline static void run(_Float16 *out, float *in) {
#pragma unroll(N)
    for (int i = 0, j = 0; i < N; ++i, j = j + 2) {
      // in:fp32 out:fp16
      auto x_1 = _mm512_loadu_ps(&in[j * 16]);
      auto x_2 = _mm512_loadu_ps(&in[(j + 1) * 16]);
      auto x = _mm512_concat_cvteps_ph(x_1, x_2);
      auto o = _mm512_sigmoid_ph(x);
      _mm512_store_ph(&out[i * 32], o);

      // in:fp16 out:fp32
      // auto x = _mm512_loadu_ph(&in[i*32]);
      // auto o = helper::_mm512_sigmoid_ph(x);
      // _mm512_store_ph(&out[i*32],o);
      // auto z = _mm512_castph_ps(o);
      // auto y_1 = _mm512_extractf32x8_ps(z,0);
      // auto y_2 = _mm512_extractf32x8_ps(z,1);
      // auto o_1 = _mm512_cvtxph_ps(_mm256_castps_ph(y_1));
      // auto o_2 = _mm512_cvtxph_ps(_mm256_castps_ph(y_2));
      // _mm512_store_ps(&out[j*16],o_1);
      // _mm512_store_ps(&out[(j+1)*16],o_2);
    }
  }
};

}  // namespace intel_mlperf
