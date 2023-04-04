#include "i_softmax_tpp.hpp"
#include "el_common_intrin.hpp"
#include <cstdint>
#include <immintrin.h>

namespace intel_mlperf {

template <int vec_l>
inline float scale_add_and_max(float *out, float scale, int32_t *in,
                               float *att_mask, int64_t ld);

// Integer softmax reference, int32 in, int8 out
template <>
inline float scale_add_and_max<8>(float *out, float scale, int32_t *in,
                                  float *att_mask, int64_t ld) {
  auto pin = reinterpret_cast<int32_t(*)>(in);
  auto pout = reinterpret_cast<float(*)>(out);
  auto full = _mm256_set1_epi32(-5000);
  auto full_ps = _mm256_set1_ps(-5000.0f);

  int64_t d2 = 0;
  auto vmax = _mm256_set1_ps(0.0f);

  for (; d2 < (ld / 8 * 8); d2 += 8) {
    auto l = _mm256_lddqu_si256((__m256i *)&pin[d2]);
    auto m = _mm256_loadu_ps(&att_mask[d2]);
    auto f = _mm256_cvtepi32_ps(l) * _mm256_set1_ps(scale) + m;
    vmax = _mm256_max_ps(f, vmax);
    _mm256_storeu_ps(&pout[d2], f);
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask8 mask = (1 << rem) - 1;

    auto l = _mm256_mask_loadu_epi32(full, mask, &pin[d2]);
    auto m = _mm256_mask_loadu_ps(full_ps, mask, &att_mask[d2]);
    auto f = _mm256_cvtepi32_ps(l) * _mm256_set1_ps(scale) + m;
    vmax = _mm256_max_ps(f, vmax);
    _mm256_mask_storeu_ps(&pout[d2], mask, f);
  }

  return _mm256_reduce_max_ps(vmax);
}

template <>
inline float scale_add_and_max<16>(float *out, float scale, int32_t *in,
                                   float *att_mask, int64_t ld) {
  auto pin = reinterpret_cast<int32_t(*)>(in);
  auto pout = reinterpret_cast<float(*)>(out);
  auto full = _mm512_set1_epi32(-5000);
  auto full_ps = _mm512_set1_ps(-5000.0f);

  int64_t d2 = 0;
  auto vmax = _mm512_set1_ps(0.0f);

  for (; d2 < (ld / 16 * 16); d2 += 16) {
    auto l = _mm512_loadu_si512(&pin[d2]);
    auto m = _mm512_loadu_ps(&att_mask[d2]);
    auto f = _mm512_cvtepi32_ps(l) * _mm512_set1_ps(scale) + m;
    vmax = _mm512_max_ps(f, vmax);
    _mm512_storeu_ps(&pout[d2], f);
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask16 mask = (1 << rem) - 1;

    auto l = _mm512_mask_loadu_epi32(full, mask, &pin[d2]);
    auto m = _mm512_mask_loadu_ps(full_ps, mask, &att_mask[d2]);
    auto f = _mm512_cvtepi32_ps(l) * _mm512_set1_ps(scale) + m;
    vmax = _mm512_max_ps(f, vmax);
    _mm512_mask_storeu_ps(&pout[d2], mask, f);
  }

  return _mm512_reduce_max_ps(vmax);
}

template <int vec_l>
inline float subtract_and_exp_reduce(float *out, float max, float *in,
                                     int64_t ld);

template <>
inline float subtract_and_exp_reduce<8>(float *out, float max, float *in,
                                        int64_t ld) {
  int64_t d2 = 0;
  auto vsum = _mm256_setzero_ps();
  auto full = _mm256_set1_ps(-5000.0f); // XXX: Denorm clear?
  auto vmax = _mm256_set1_ps(max);

  for (; d2 < (ld / 8 * 8); d2 += 8) {
    auto f = _mm256_loadu_ps(&in[d2]);
    auto d = f - vmax;
    auto e = exp_ps_0_1(d);
    _mm256_storeu_ps(&out[d2], e);
    vsum += e;
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask8 mask = (1 << rem) - 1;
    auto f = _mm256_mask_loadu_ps(full, mask, &in[d2]);
    auto d = f - vmax;
    auto e = exp_ps_0_1(d);
    _mm256_mask_storeu_ps(&out[d2], mask, e);
    vsum += e;
  }

  return _mm256_reduce_add_ps(vsum);
}

template <>
inline float subtract_and_exp_reduce<16>(float *out, float max, float *in,
                                         int64_t ld) {
  int64_t d2 = 0;
  auto vsum = _mm512_setzero_ps();
  auto full = _mm512_set1_ps(-5000.0f); // XXX: Denorm clear?
  auto vmax = _mm512_set1_ps(max);

  for (; d2 < (ld / 16 * 16); d2 += 16) {
    auto f = _mm512_loadu_ps(&in[d2]);
    auto d = f - vmax;
    auto e = exp_ps_0_1(d);
    _mm512_storeu_ps(&out[d2], e);
    vsum += e;
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask16 mask = (1 << rem) - 1;
    auto f = _mm512_mask_loadu_ps(full, mask, &in[d2]);
    auto d = f - vmax;
    auto e = exp_ps_0_1(d);
    _mm512_mask_storeu_ps(&out[d2], mask, e);
    vsum += e;
  }

  return _mm512_reduce_add_ps(vsum);
}

template <int vec_l>
static inline void scale_quant_out(int8_t *out, float sum, float scale,
                                   float *in, int64_t ld);

template <>
inline void scale_quant_out<8>(int8_t *out, float sum, float scale, float *in,
                               int64_t ld) {
  int64_t d2 = 0;
  auto vsum = _mm256_set1_ps(sum);
  auto s = _mm256_set1_ps(scale) / vsum;
  auto full = _mm256_set1_ps(0.0);

  for (; d2 < ld / 8 * 8; d2 += 8) {
    auto l = _mm256_loadu_ps(&in[d2]);
    auto m =
        _mm256_round_ps(l * s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto i_4 = _mm256_cvtps_epi32(m);
    _mm256_mask_cvtepi32_storeu_epi8(&out[d2], 0xff, i_4);
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask8 mask = (1 << rem) - 1;
    auto l = _mm256_mask_loadu_ps(full, mask, &in[d2]);
    auto m =
        _mm256_round_ps(l * s, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto i_4 = _mm256_cvtps_epi32(m);
    _mm256_mask_cvtepi32_storeu_epi8(&out[d2], mask, i_4);
  }
}

template <>
inline void scale_quant_out<16>(int8_t *out, float sum, float scale, float *in,
                                int64_t ld) {
  int64_t d2 = 0;
  auto vsum = _mm512_set1_ps(sum);
  auto s = _mm512_set1_ps(scale) / vsum;
  auto full = _mm512_setzero_ps();

  for (; d2 < ld / 16 * 16; d2 += 16) {
    auto l = _mm512_loadu_ps(&in[d2]);
    auto m = _mm512_mul_round_ps(l, s,
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto i_4 = _mm512_cvtps_epi32(m);
    _mm512_mask_cvtepi32_storeu_epi8(&out[d2], 0xffff, i_4);
  }

  if (d2 < ld) {
    int rem = ld - d2;
    __mmask16 mask = (1 << rem) - 1;
    auto l = _mm512_mask_loadu_ps(full, mask, &in[d2]);
    auto m = _mm512_mul_round_ps(l, s,
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto i_4 = _mm512_cvtps_epi32(m);
    _mm512_mask_cvtepi32_storeu_epi8(&out[d2], mask, i_4);
  }
}

template <int vec_length, int N> struct i32_scale_softmax_scale_i8 {
  inline static void run(void *out, void *in, void *att_mask, float M,
                         float oscale, int64_t ld);
};

template <int N> struct i32_scale_softmax_scale_i8<16, N> {
  inline static void run(void *out, void *in, float *att_mask, float M,
                         float oscale, int64_t ld) {
    auto pin = reinterpret_cast<int32_t(*)[ld]>(in);
    auto ld_16 = (ld + 15) / 16 * 16;

    // Scratch for max subtraction
    alignas(64) float dout[N][ld_16];

    auto zero = _mm512_setzero_epi32();
    auto full_ps = _mm512_set1_ps(-5000.f);
    auto vscale = _mm512_set1_ps(M);

    __m512 vmax[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_setzero_ps();
    }

    int64_t d2;
    for (d2 = 0; d2 < ld / 16 * 16; d2 += 16) {
      auto m = _mm512_loadu_ps(&att_mask[d2]);

#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_si512(&pin[i][d2]);
        auto f = _mm512_cvtepi32_ps(l) * vscale + m;
        vmax[i] = _mm512_max_ps(f, vmax[i]);
        _mm512_storeu_ps(&dout[i][d2], f);
      }
    }

    if (d2 < ld) {
      int rem = ld - d2;
      __mmask16 mask = (1 << rem) - 1;
      auto m = _mm512_mask_loadu_ps(full_ps, mask, &att_mask[d2]);

#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_mask_loadu_epi32(zero, mask, &pin[i][d2]);
        auto f = _mm512_cvtepi32_ps(l) * vscale + m;
        vmax[i] = _mm512_max_ps(f, vmax[i]);
        _mm512_storeu_ps(&dout[i][d2], f);
      }
    }

    __m512 vsum[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_max_reduce_ps(vmax[i]);
      vsum[i] = _mm512_setzero_ps();
    }

    for (d2 = 0; d2 < ld_16; d2 += 16) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto f = _mm512_loadu_ps(&dout[i][d2]);
        auto d = f - vmax[i];
        auto e = exp_ps_0_1(d);
        _mm512_storeu_ps(&dout[i][d2], e);
        vsum[i] += e;
      }
    }

    auto voscale = _mm512_set1_ps(oscale);

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
#ifdef usercp
      vsum[i] = voscale * _mm512_rcp14_ps(_mm512_add_reduce_ps(vsum[i]));
#else
      vsum[i] = voscale / _mm512_add_reduce_ps(vsum[i]);
#endif
    }

    auto pout = reinterpret_cast<int8_t(*)[ld]>(out);
    for (d2 = 0; d2 < ld / 16 * 16; d2 += 16) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_ps(&dout[i][d2]);
        auto i_4 = _mm512_scale_minmax_i8_ps(l, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(&pout[i][d2], 0xffff, i_4);
      }
    }

    if (d2 < ld) {
      int rem = ld - d2;
      __mmask16 mask = (1 << rem) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_ps(&dout[i][d2]);
        auto i_4 = _mm512_scale_minmax_i8_ps(l, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(&pout[i][d2], mask, i_4);
      }
    }
  }
};

inline void f_i32_scale_softmax_scale_i8(int8_t *out, int32_t *in,
                                         float *att_mask, float M, float oscale,
                                         int64_t ld, int l) {
  switch (l) {
  case 1:
    return i32_scale_softmax_scale_i8<16, 1>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 2:
    return i32_scale_softmax_scale_i8<16, 2>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 3:
    return i32_scale_softmax_scale_i8<16, 3>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 4:
    return i32_scale_softmax_scale_i8<16, 4>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 5:
    return i32_scale_softmax_scale_i8<16, 5>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 6:
    return i32_scale_softmax_scale_i8<16, 6>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 7:
    return i32_scale_softmax_scale_i8<16, 7>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 8:
    return i32_scale_softmax_scale_i8<16, 8>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 9:
    return i32_scale_softmax_scale_i8<16, 9>::run(out, in, att_mask, M, oscale,
                                                  ld);
  case 10:
    return i32_scale_softmax_scale_i8<16, 10>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 11:
    return i32_scale_softmax_scale_i8<16, 11>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 12:
    return i32_scale_softmax_scale_i8<16, 12>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 13:
    return i32_scale_softmax_scale_i8<16, 13>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 14:
    return i32_scale_softmax_scale_i8<16, 14>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 15:
    return i32_scale_softmax_scale_i8<16, 15>::run(out, in, att_mask, M, oscale,
                                                   ld);
  case 16:
    return i32_scale_softmax_scale_i8<16, 16>::run(out, in, att_mask, M, oscale,
                                                   ld);
  }

  auto l1 = l / 2;
  auto l2 = l - l1;

  auto pin = reinterpret_cast<int32_t(*)[ld]>(in);
  auto pout = reinterpret_cast<int8_t(*)[ld]>(out);

  f_i32_scale_softmax_scale_i8(pout[0], pin[0], att_mask, M, oscale, ld, l1);
  f_i32_scale_softmax_scale_i8(pout[l1], pin[l1], att_mask, M, oscale, ld, l2);
}

inline void f_i32_scale_softmax_scale_i8(int8_t *out, int32_t *in,
                                         int32_t att_len, float M, float oscale,
                                         int64_t ld, int l) {
  switch (l) {
  case 1:
    return i32_scale_attlen_softmax_scale_i8<16, 1>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 2:
    return i32_scale_attlen_softmax_scale_i8<16, 2>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 3:
    return i32_scale_attlen_softmax_scale_i8<16, 3>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 4:
    return i32_scale_attlen_softmax_scale_i8<16, 4>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 5:
    return i32_scale_attlen_softmax_scale_i8<16, 5>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 6:
    return i32_scale_attlen_softmax_scale_i8<16, 6>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 7:
    return i32_scale_attlen_softmax_scale_i8<16, 7>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 8:
    return i32_scale_attlen_softmax_scale_i8<16, 8>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 9:
    return i32_scale_attlen_softmax_scale_i8<16, 9>::run(out, in, att_len, M,
                                                         oscale, ld);
  case 10:
    return i32_scale_attlen_softmax_scale_i8<16, 10>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 11:
    return i32_scale_attlen_softmax_scale_i8<16, 11>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 12:
    return i32_scale_attlen_softmax_scale_i8<16, 12>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 13:
    return i32_scale_attlen_softmax_scale_i8<16, 13>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 14:
    return i32_scale_attlen_softmax_scale_i8<16, 14>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 15:
    return i32_scale_attlen_softmax_scale_i8<16, 15>::run(out, in, att_len, M,
                                                          oscale, ld);
  case 16:
    return i32_scale_attlen_softmax_scale_i8<16, 16>::run(out, in, att_len, M,
                                                          oscale, ld);
  }

  auto l1 = l / 2;
  auto l2 = l - l1;

  auto pin = reinterpret_cast<int32_t(*)[ld]>(in);
  auto pout = reinterpret_cast<int8_t(*)[ld]>(out);

  f_i32_scale_softmax_scale_i8(pout[0], pin[0], att_len, M, oscale, ld, l1);
  f_i32_scale_softmax_scale_i8(pout[l1], pin[l1], att_len, M, oscale, ld, l2);
}

template <int vec_length>
void i_softmax_tpp<vec_length>::ref(void *out, void *in, float *att_mask,
                                    float M, float oscale) {
#pragma omp parallel for collapse(2)
  for (auto d0 = 0; d0 < dim0; ++d0) {
    for (auto d1 = 0; d1 < dim1; ++d1) {

      auto *p_att_mask = reinterpret_cast<float(*)[1 * 1 * dim3]>(att_mask);
      auto *p_in = reinterpret_cast<int32_t(*)[dim1][dim2 * dim3]>(in);
      auto *p_out = reinterpret_cast<int8_t(*)[dim1][dim2 * dim3]>(out);

      f_i32_scale_softmax_scale_i8(p_out[d0][d1], p_in[d0][d1], p_att_mask[d0],
                                   M, oscale, dim3, dim2);
    }
  }
}

// Accept attention mask as attention length
template <int vec_length>
void i_softmax_tpp<vec_length>::ref(void *out, void *in, int32_t *att_lens,
                                    float M, float oscale) {
#pragma omp parallel for collapse(2)
  for (auto d0 = 0; d0 < dim0; ++d0) {
    for (auto d1 = 0; d1 < dim1; ++d1) {

      auto *p_in = reinterpret_cast<int32_t(*)[dim1][dim2 * dim3]>(in);
      auto *p_out = reinterpret_cast<int8_t(*)[dim1][dim2 * dim3]>(out);

      f_i32_scale_softmax_scale_i8(p_out[d0][d1], p_in[d0][d1], att_lens[d0], M,
                                   oscale, dim3, dim2);
    }
  }
}

template void i_softmax_tpp<8>::ref(void *, void *, float *, float, float);
template void i_softmax_tpp<16>::ref(void *, void *, float *, float, float);
template void i_softmax_tpp<8>::ref(void *, void *, int32_t *, float, float);
template void i_softmax_tpp<16>::ref(void *, void *, int32_t *, float, float);

} // namespace intel_mlperf
