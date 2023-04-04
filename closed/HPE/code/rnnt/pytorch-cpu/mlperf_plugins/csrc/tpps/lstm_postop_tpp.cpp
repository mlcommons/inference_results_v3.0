#include "lstm_postop_tpp.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <iostream>

#include "sigmoid_tpp.hpp"
#include "tanh_tpp.hpp"

namespace intel_mlperf {
void lstm_postop_tpp::ref(
    void *out_yt_bf16, void *out_yt_q, void *out_ht_q, void *it, void *ft, void *gt,
    void *ot, void *ct, float in_scale, float out_scale, size_t line,
    bool last_layer_flag) {
  // compute four post-op
  auto pin_it = reinterpret_cast<float *>(it);
  auto pin_ft = reinterpret_cast<float *>(ft);
  auto pin_gt = reinterpret_cast<float *>(gt);
  auto pin_ot = reinterpret_cast<float *>(ot);

  alignas(64) _Float16 it_out[line];
  alignas(64) _Float16 ft_out[line];
  alignas(64) _Float16 gt_out[line];
  alignas(64) _Float16 ot_out[line];
  if (line % (32 * 4) != 0) {
    sigmoid_tpp<32, 1>::ref(it_out, pin_it, line);
    sigmoid_tpp<32, 1>::ref(ft_out, pin_ft, line);
    tanh_tpp<32, 1>::ref(gt_out, pin_gt, line);
    sigmoid_tpp<32, 1>::ref(ot_out, pin_ot, line);
  } else {
    sigmoid_tpp<32>::ref(it_out, pin_it, line);
    sigmoid_tpp<32>::ref(ft_out, pin_ft, line);
    tanh_tpp<32>::ref(gt_out, pin_gt, line);
    sigmoid_tpp<32>::ref(ot_out, pin_ot, line);
  }

  // comput ct
  size_t half_len = 32;
  auto n_batch = line / half_len;
  auto pin_ct = reinterpret_cast<_Float16 *>(ct);
#pragma unroll(32)
  for (int j = 0; j < n_batch; j++) {
    auto i = _mm512_loadu_ph((&it_out[j * 32]));
    auto f = _mm512_loadu_ph((&ft_out[j * 32]));
    auto g = _mm512_loadu_ph((&gt_out[j * 32]));
    auto c = _mm512_loadu_ph((&pin_ct[j * 32]));
    auto a = _mm512_mul_ph(f, c);
    auto o = _mm512_fmadd_ph(i, g, a);
    _mm512_store_ph(&pin_ct[j * 32], o);
  }

  // inplace it_out and ft_out;ft_out = ct_tanh
  auto pout_yt_bf16 = reinterpret_cast<__bfloat16 *>(out_yt_bf16);
  if (line % (32 * 4) != 0) {
    tanh_tpp<32, 1>::ref_(ft_out, pin_ct, line);
  } else {
    tanh_tpp<32>::ref_(ft_out, pin_ct, line);
  }
#pragma unroll(32)
  for (int j = 0; j < n_batch; j++) {
    auto a = _mm512_loadu_ph(&ot_out[j * 32]);
    auto b = _mm512_loadu_ph(&ft_out[j * 32]);
    auto o_ht = _mm512_mul_ph(a, b);
    _mm512_store_ph(&it_out[j * 32], o_ht);
    if (last_layer_flag) {
      // ht:fp16->fp32->bf16
      auto y_1 = _mm512_extractf32x8_ps(o_ht, 0);
      auto y_2 = _mm512_extractf32x8_ps(o_ht, 1);
      auto o_1 = _mm512_cvtph_ps(_mm256_castps_ph(y_1));
      auto o_2 = _mm512_cvtph_ps(_mm256_castps_ph(y_2));
      auto o_bf = _mm512_cvtne2ps_pbh(o_2, o_1);
      _mm512_storeu_epi16(&pout_yt_bf16[j * 32], o_bf);
    }
  }

  // quant
  auto pout_yt_q = reinterpret_cast<int8_t *>(out_yt_q);

  if (!last_layer_flag) {
#pragma unroll(32)
    for (int j = 0; j < n_batch; j++) {
      auto vout_scale = _mm512_set1_ph(out_scale);
      auto ht_ph = _mm512_loadu_ph(&it_out[j * 32]);
      auto yt_quant = _mm512_scale_min128max_i8_ph(ht_ph, vout_scale);
      _mm512_mask_cvtepi16_storeu_epi8(&pout_yt_q[j * 32], 0xffffffff, yt_quant);
    }
  }

  auto pout_ht_q = reinterpret_cast<int8_t *>(out_ht_q);
#pragma unroll(32)
  for (int j = 0; j < n_batch; j++) {
    auto vin_scale = _mm512_set1_ph(in_scale);
    auto ht_ph = _mm512_loadu_ph(&it_out[j * 32]);
    auto ht_quant = _mm512_scale_min128max_i8_ph(ht_ph, vin_scale);
    _mm512_mask_cvtepi16_storeu_epi8(&pout_ht_q[j * 32], 0xffffffff, ht_quant);
  }
}

void lstm_postop_tpp::ref_bf16(
    void *out_yt, void *out_ht, void *out_ct, void *it, void *ft, void *gt, void *ot,
    void *ct, size_t line) {
  // compute four post-op
  auto pin_it = reinterpret_cast<float *>(it);
  auto pin_ft = reinterpret_cast<float *>(ft);
  auto pin_gt = reinterpret_cast<float *>(gt);
  auto pin_ot = reinterpret_cast<float *>(ot);

  alignas(64) float it_out[line];
  alignas(64) float ft_out[line];
  alignas(64) float gt_out[line];
  alignas(64) float ot_out[line];
  if (line % (16 * 4) != 0) {
    sigmoid_tpp<16, 1>::ref_f32(it_out, pin_it, line);
    sigmoid_tpp<16, 1>::ref_f32(ft_out, pin_ft, line);
    tanh_tpp<16, 1>::ref_f32(gt_out, pin_gt, line);
    sigmoid_tpp<16, 1>::ref_f32(ot_out, pin_ot, line);
  } else {
    sigmoid_tpp<16>::ref_f32(it_out, pin_it, line);
    sigmoid_tpp<16>::ref_f32(ft_out, pin_ft, line);
    tanh_tpp<16>::ref_f32(gt_out, pin_gt, line);
    sigmoid_tpp<16>::ref_f32(ot_out, pin_ot, line);
  }

  // comput ct
  size_t half_len = 16;
  auto n_batch = line / half_len;
  auto pin_ct = reinterpret_cast<float *>(ct);
  auto pout_ct = reinterpret_cast<float *>(out_ct);
#pragma unroll(16)
  for (int j = 0; j < n_batch; j++) {
    auto i = _mm512_loadu_ps((&it_out[j * half_len]));
    auto f = _mm512_loadu_ps((&ft_out[j * half_len]));
    auto g = _mm512_loadu_ps((&gt_out[j * half_len]));
    auto c = _mm512_loadu_ps((&pin_ct[j * half_len]));
    auto a = _mm512_mul_ps(f, c);
    auto o = _mm512_fmadd_ps(i, g, a);
    _mm512_store_ps(&pout_ct[j * half_len], o);
  }

  // inplace it_out and ft_out;ft_out = ct_tanh
  auto pout_yt = reinterpret_cast<__bfloat16 *>(out_yt);
  auto pout_ht = reinterpret_cast<__bfloat16 *>(out_ht);
  // TODO: remove redundant load&store
  if (line % (16 * 4) != 0) {
    tanh_tpp<16, 1>::ref_f32(ft_out, pout_ct, line);
  } else {
    tanh_tpp<16>::ref_f32(ft_out, pout_ct, line);
  }
#pragma unroll(8)
  for (int j = 0; j < n_batch / 2; j++) {
    auto a0 = _mm512_loadu_ps(&ot_out[j * half_len * 2]);
    auto b0 = _mm512_loadu_ps(&ft_out[j * half_len * 2]);
    auto o_ht0 = _mm512_mul_ps(a0, b0);
    auto a1 = _mm512_loadu_ps(&ot_out[j * half_len * 2 + half_len]);
    auto b1 = _mm512_loadu_ps(&ft_out[j * half_len * 2 + half_len]);
    auto o_ht1 = _mm512_mul_ps(a1, b1);
    // ht:fp32->bf16
    auto o_bf = _mm512_cvtne2ps_pbh(o_ht1, o_ht0);
    _mm512_storeu_epi16(&pout_ht[j * half_len * 2], o_bf);
    _mm512_storeu_epi16(&pout_yt[j * half_len * 2], o_bf);
  }
}

}  // namespace intel_mlperf
