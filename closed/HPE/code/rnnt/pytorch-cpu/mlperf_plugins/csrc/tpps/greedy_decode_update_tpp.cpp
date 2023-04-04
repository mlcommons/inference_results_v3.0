#include "greedy_decode_update_tpp.hpp"

#include <immintrin.h>

#include "el_common_intrin.hpp"

namespace intel_mlperf {

void greedy_decode_update_tpp::update_mask(
    int64_t* symbols, int32_t* symbols_added, int32_t* res, int32_t* res_idx,
    int32_t* time_idx, int32_t* f_lens, int32_t* pre_g, size_t seq_len,
    size_t batch, unsigned short& update_g, unsigned short& finish) {
  __mmask16 mask_batch = (1 << batch) - 1;
  const auto k1 = _mm512_set1_epi32(1);
  const auto k0 = _mm512_set1_epi32(0);
  auto symbols1_ = _mm512_mask_loadu_epi64(k0, mask_batch, symbols);
  auto symbols2_ = _mm512_mask_loadu_epi64(k0, mask_batch >> 8, &symbols[8]);
  auto symbols_ = _mm512_concat_cvtepi64_epi32(symbols1_, symbols2_);
  auto symbols_added_ = _mm512_mask_loadu_epi32(k0, mask_batch, symbols_added);
  auto res_idx_ = _mm512_mask_loadu_epi32(k0, mask_batch, res_idx);
  auto f_lens_ = _mm512_mask_loadu_epi32(k0, mask_batch, f_lens);
  auto finish_ = _mm512_cmpeq_epi32_mask(f_lens_, k0);

  auto mask_no_blank = _mm512_mask_cmpneq_epi32_mask(
      mask_batch, symbols_, _mm512_set1_epi32(kBlank));
  auto mask_no_max_symbols = _mm512_mask_cmpneq_epi32_mask(
      mask_batch, symbols_added_, _mm512_set1_epi32(kMaxSymbolsPerStep));
  __mmask16 mask_update_g = mask_no_blank & mask_no_max_symbols & ~finish_;
  __mmask16 mask_update_f = ~mask_update_g & mask_batch;
  const auto seq_len_ = _mm512_set1_epi32(seq_len);
  auto idx_base =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

  res_idx_ = _mm512_mask_add_epi32(res_idx_, mask_update_g, res_idx_, k1);
  idx_base = _mm512_add_epi32(_mm512_mullo_epi32(seq_len_, idx_base), res_idx_);
  _mm512_mask_i32scatter_epi32(res, mask_update_g, idx_base, symbols_, 4);
  _mm512_mask_storeu_epi32(res_idx, mask_batch, res_idx_);

  symbols_added_ =
      _mm512_mask_add_epi32(symbols_added_, mask_update_g, symbols_added_, k1);
  symbols_added_ = _mm512_mask_set1_epi32(symbols_added_, mask_update_f, 0);
  _mm512_mask_storeu_epi32(symbols_added, mask_batch, symbols_added_);
  _mm512_mask_storeu_epi32(pre_g, mask_update_g, symbols_);

  auto time_idx_ = _mm512_mask_loadu_epi32(k0, mask_batch, time_idx);
  time_idx_ = _mm512_mask_add_epi32(time_idx_, mask_update_f, time_idx_, k1);
  _mm512_mask_storeu_epi32(time_idx, mask_batch, time_idx_);

  finish = _mm512_mask_cmpge_epi32_mask(mask_batch, time_idx_, f_lens_);
  update_g = mask_update_g;
}

}  // namespace intel_mlperf
