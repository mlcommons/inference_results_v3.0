#pragma once
#include <immintrin.h>

#include <cstdlib>

#include "el_common_intrin.hpp"

namespace intel_mlperf {
template <int vec_length, int unroll = 4>
class tanh_tpp {
public:
  static void ref(void *out, void *in, int64_t line);
  static void ref_(void *out, void *in, int64_t line);
  static void ref_f32(void *out, void *in, int64_t line);
};

template <int vec_l, int N>
struct tanh_fp16 {
  inline static void run(_Float16 *out, float *in);
  inline static void run(_Float16 *out, _Float16 *in);
};

template <int vec_l, int N>
struct tanh_fp32 {
  inline static void run_f32(float *out, float *in);
};

template <int N>
struct tanh_fp32<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run_f32(float *out, float *in) {
#pragma unroll(N)
    for (int i = 0; i < N; i++) {
      auto x = _mm512_loadu_ps(&in[i * 16]);
      auto o = _mm512_tanh_ps(x);
      _mm512_store_ps(&out[i * 16], o);
    }
  }
};

template <int N>
struct tanh_fp16<32, N> {
  static constexpr int64_t batch = 32 * N;

  inline static void run(_Float16 *out, float *in) {
#pragma unroll(N)
    for (int i = 0, j = 0; i < N; ++i, j = j + 2) {
      auto x_1 = _mm512_loadu_ps(&in[j * 16]);
      auto x_2 = _mm512_loadu_ps(&in[(j + 1) * 16]);
      auto x = _mm512_concat_cvteps_ph(x_1, x_2);
      auto o = _mm512_tanh_ph(x);
      _mm512_store_ph(&out[i * 32], o);
    }
  }

  inline static void run(_Float16 *out, _Float16 *in) {
#pragma unroll(N)
    for (int i = 0, j = 0; i < N; ++i, j = j + 2) {
      auto x = _mm512_loadu_ph(&in[i * 32]);
      auto o = _mm512_tanh_ph(x);
      _mm512_store_ph(&out[i * 32], o);
    }
  }
};

}  // namespace intel_mlperf
