#include <cstdlib>
#include <float.h>

#include "el_common_intrin.hpp"
#include "i_layernorm_tpp.hpp"

namespace intel_mlperf {

template <>
void i_layernorm_tpp<16>::ref(float *out, float *in, float *weight, float *bias,
                              int64_t rl, float eps, bool unbiased) {
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();

  auto *pin = in;

  // Pass 1, statistics
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_ps(&pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm512_setzero_ps();
    auto f = _mm512_mask_loadu_ps(zeros, k, &pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ps(eps);
  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;
  if (unbiased) {
    auto factor = _mm512_set1_ps(1. * rl / (rl - 1));
    vvar2 = vvar2 * factor;
  }

#ifdef usercp
  auto r_vvar = _mm512_rsqrt14_ps(vvar2);
#else
  auto r_vvar = 1. / _mm512_sqrt_ps(vvar2);
#endif
  auto *pout = reinterpret_cast<float *>(out);
  // pass 2 adjusts
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_ps(&pin[d]);
    auto w = _mm512_loadu_ps(&weight[d]);
    auto b = _mm512_loadu_ps(&bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    // auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    //_mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], 0xffff, o, vo_off);
    _mm512_mask_storeu_ps(&pout[d], 0xffff, o);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ps();
    auto f = _mm512_mask_loadu_ps(zero, k, &pin[d]);
    auto w = _mm512_mask_loadu_ps(zero, k, &weight[d]);
    auto b = _mm512_mask_loadu_ps(zero, k, &bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    _mm512_mask_storeu_ps(&pout[d], k, o);
  }
}

template <>
void i_layernorm_tpp<16>::ref(int8_t *out, float *in, float *weight,
                              float *bias, float oscale, int64_t rl, float eps,
                              int8_t o_off) {
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();

  auto *pin = in;

  // Pass 1, statistics
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_ps(&pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm512_setzero_ps();
    auto f = _mm512_mask_loadu_ps(zeros, k, &pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ps(eps);
  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;

#ifdef usercp
  auto r_vvar = _mm512_rsqrt14_ps(vvar2);
#else
  auto r_vvar = 1. / _mm512_sqrt_ps(vvar2);
#endif
  auto voscale = _mm512_set1_ps(oscale);
  auto *pout = reinterpret_cast<int8_t *>(out);
  auto vo_off = _mm_set1_epi8(o_off);
  // pass 2 adjusts
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_ps(&pin[d]);
    auto w = _mm512_loadu_ps(&weight[d]);
    auto b = _mm512_loadu_ps(&bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], 0xffff, i, vo_off);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ps();
    auto f = _mm512_mask_loadu_ps(zero, k, &pin[d]);
    auto w = _mm512_mask_loadu_ps(zero, k, &weight[d]);
    auto b = _mm512_mask_loadu_ps(zero, k, &bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], k, i, vo_off);
  }
}

template <>
void i_layernorm_tpp<16>::ref(int8_t *out, int8_t *in, float *weight,
                              float *bias, float oscale, int64_t rl, float eps,
                              int8_t o_off) {
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();

  auto pin = in;
  auto rl_16 = (rl + 15) / 16 * 16;
  alignas(64) float f_saved[rl_16];
  // Pass 1
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_i8_to_fp32(&pin[d]);
    auto s = f;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm_setzero_si128();
    auto f = _mm512_mask_loadu_i8_to_fp32(zeros, k, &pin[d]);
    auto s = f;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ps(eps);

  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;

  auto r_vvar = _mm512_rsqrt14_ps(vvar2);
  auto voscale = _mm512_set1_ps(oscale);
  auto *pout = reinterpret_cast<int8_t *>(out);
  auto vo_off = _mm_set1_epi8(o_off);
  // pass 2
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_loadu_ps(&weight[d]);
    auto b = _mm512_loadu_ps(&bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], 0xffff, i, vo_off);
  }
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ps();
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_mask_loadu_ps(zero, k, &weight[d]);
    auto b = _mm512_mask_loadu_ps(zero, k, &bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], k, i, vo_off);
  }
}

template <>
void i_residual_layernorm_tpp<16>::ref(int8_t *out, int8_t *src1, int8_t *src2,
                                       float *weight, float *bias, float s1,
                                       float s2, float oscale, int64_t rl,
                                       float eps, int8_t o_off) {
  auto *pin1 = reinterpret_cast<int8_t *>(src1);
  auto *pin2 = reinterpret_cast<int8_t *>(src2);

  int64_t d;

  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();
  auto vS1 = _mm512_set1_ps(s1);
  auto vS2 = _mm512_set1_ps(s2);

  auto rl_16 = (rl + 15) / 16 * 16;

  alignas(64) float f_saved[rl_16];

  // Pass 1
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f1 = _mm512_loadu_i8_to_fp32(&pin1[d]);
    auto f2 = _mm512_loadu_i8_to_fp32(&pin2[d]);
    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm_setzero_si128();
    auto f1 = _mm512_mask_loadu_i8_to_fp32(zeros, k, &pin1[d]);
    auto f2 = _mm512_mask_loadu_i8_to_fp32(zeros, k, &pin2[d]);

    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ps(eps);

  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;

  auto r_vvar = _mm512_rsqrt14_ps(vvar2);
  auto voscale = _mm512_set1_ps(oscale);
  auto vo_off = _mm_set1_epi8(o_off);
  auto *pout = reinterpret_cast<int8_t *>(out);
  // pass 2
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_loadu_ps(&weight[d]);
    auto b = _mm512_loadu_ps(&bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], 0xffff, i, vo_off);
  }

  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ps();
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_mask_loadu_ps(zero, k, &weight[d]);
    auto b = _mm512_mask_loadu_ps(zero, k, &bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], k, i, vo_off);
  }
}

template <>
void i_residual_layernorm_tpp<32>::ref_fp16(int8_t *out, int8_t *src1,
                                            int8_t *src2, _Float16 *weight,
                                            _Float16 *bias, float s1, float s2,
                                            float oscale, int64_t rl, float eps,
                                            int8_t o_off) {
  auto *pin1 = reinterpret_cast<int8_t *>(src1);
  auto *pin2 = reinterpret_cast<int8_t *>(src2);

  int64_t d;

  auto vsum = _mm512_setzero_ph();
  auto vsum2 = _mm512_setzero_ph();
  auto vS1 = _mm512_set1_ph(s1);
  auto vS2 = _mm512_set1_ph(s2);

  auto rl_32 = (rl + 31) / 32 * 32;

  alignas(64) _Float16 f_saved[rl_32];

  // Pass 1
  for (d = 0; d < rl / 32 * 32; d += 32) {
    auto f1 = _mm512_loadu_i8_to_fp16(&pin1[d]);
    auto f2 = _mm512_loadu_i8_to_fp16(&pin2[d]);

    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ph(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask32 k = (1 << rem) - 1;
    auto zeros = _mm256_setzero_si256();
    auto f1 = _mm512_mask_loadu_i8_to_fp16(zeros, k, &pin1[d]);
    auto f2 = _mm512_mask_loadu_i8_to_fp16(zeros, k, &pin2[d]);

    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ph(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ph(eps);

  auto vmean = _mm512_mean_reduce_ph(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ph(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;

  auto r_vvar = _mm512_rsqrt_ph(vvar2);
  auto voscale = _mm512_set1_ph(oscale);
  auto vo_off = _mm256_set1_epi8(o_off);
  auto *pout = reinterpret_cast<int8_t *>(out);
  // pass 2
  for (d = 0; d < rl / 32 * 32; d += 32) {
    auto f = _mm512_load_ph(&f_saved[d]);
    // TODO: weight should be fp16?
    auto w = _mm512_loadu_ph(&weight[d]);
    auto b = _mm512_loadu_ph(&bias[d]);

    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ph(o, voscale);
    // _mm512_mask_cvtepi16_storeu_epi8_compensate(&pout[d], 0xffffffff, i,
    // vo_off);
    _mm512_mask_cvtepi16_storeu_epi8(&pout[d], 0xffffffff, i);
  }

  if (d < rl) {
    auto rem = rl - d;
    __mmask32 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ph();
    auto f = _mm512_load_ph(&f_saved[d]);
    // TODO: weight should be fp16?
    auto w = _mm512_mask_loadu_epi16(_mm512_castph_si512(zero), k, &weight[d]);
    auto b = _mm512_mask_loadu_epi16(_mm512_castph_si512(zero), k, &bias[d]);
    
    auto gamma = _mm512_castsi512_ph(w) * r_vvar; 
    auto o = (f - vmean) * gamma + _mm512_castsi512_ph(b);
    auto i = _mm512_scale_minmax_i8_ph(o, voscale);
    // _mm512_mask_cvtepi16_storeu_epi8_compensate(&pout[d], k, i, vo_off);
    _mm512_mask_cvtepi16_storeu_epi8(&pout[d], 0xffffffff, i);
  }
}

template <>
void i_residual_layernorm_tpp<16>::ref_cin(int8_t *out, int8_t *src1,
                                           int8_t *src2, float *weight,
                                           float *bias, float s1, float s2,
                                           float oscale, int64_t rl, float eps,
                                           int8_t o_off) {
  auto *pin1 = reinterpret_cast<int8_t *>(src1);
  auto *pin2 = reinterpret_cast<int8_t *>(src2);

  int64_t d;

  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();
  auto vS1 = _mm512_set1_ps(s1);
  auto vS2 = _mm512_set1_ps(s2);

  auto rl_16 = (rl + 15) / 16 * 16;

  alignas(64) float f_saved[rl_16];

  // Pass 1
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f1 = _mm512_loadu_i8_to_fp32(&pin1[d]);
    auto f2 = _mm512_loadu_c8_to_fp32(&pin2[d]);
    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zeros = _mm_setzero_si128();
    auto f1 = _mm512_mask_loadu_i8_to_fp32(zeros, k, &pin1[d]);
    auto f2 = _mm512_mask_loadu_c8_to_fp32(zeros, k, &pin2[d]);

    auto s = vS1 * f1 + vS2 * f2;
    auto ss = s * s;
    _mm512_store_ps(&f_saved[d], s);

    vsum += s;
    vsum2 += ss;
  }

  auto veps = _mm512_set1_ps(eps);

  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 = vmean2 - vmean * vmean + veps;

  auto r_vvar = _mm512_rsqrt14_ps(vvar2);
  auto voscale = _mm512_set1_ps(oscale);
  auto vo_off = _mm_set1_epi8(o_off);
  auto *pout = reinterpret_cast<int8_t *>(out);
  // pass 2
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_loadu_ps(&weight[d]);
    auto b = _mm512_loadu_ps(&bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], 0xffff, i, vo_off);
  }

  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto zero = _mm512_setzero_ps();
    auto f = _mm512_load_ps(&f_saved[d]);
    auto w = _mm512_mask_loadu_ps(zero, k, &weight[d]);
    auto b = _mm512_mask_loadu_ps(zero, k, &bias[d]);
    auto gamma = w * r_vvar;
    auto o = (f - vmean) * gamma + b;
    auto i = _mm512_scale_minmax_i8_ps(o, voscale);
    _mm512_mask_cvtepi32_storeu_epi8_compensate(&pout[d], k, i, vo_off);
  }
}

} // namespace intel_mlperf
