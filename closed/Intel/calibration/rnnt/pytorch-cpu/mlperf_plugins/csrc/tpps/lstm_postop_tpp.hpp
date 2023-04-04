#pragma once

#include <cstdlib>

#include "el_common_intrin.hpp"

namespace intel_mlperf {

class lstm_postop_tpp {
public:
  static void ref(
      void *out_yt_bf16, void *out_yt_q, void *out_ht_q, void *it, void *ft, void *gt,
      void *ot, void *ct, float in_scale, float out_scale, size_t line,
      bool last_layer_flag);

  static void ref_bf16(
      void *out_yt, void *out_ht, void *out_ct, void *it, void *ft, void *gt, void *ot,
      void *ct, size_t line);
};

}  // namespace intel_mlperf
