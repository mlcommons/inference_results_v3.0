#pragma once
#include <cstdlib>

namespace intel_mlperf {

class greedy_decode_update_tpp {
public:
  static const int kBlank = 28;
  static const int kMaxSymbolsPerStep = 30;

  static void update_mask(
      int64_t* symbols, int32_t* symbols_added, int32_t* res, int32_t* res_idx,
      int32_t* time_idx, int32_t* f_lens, int32_t* pre_g, size_t seq_len,
      size_t batch, unsigned short& update_g, unsigned short& finish);
};

}  // namespace intel_mlperf
