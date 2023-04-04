#pragma once
#include <cstdlib>
#include <immintrin.h>
#include "el_common_intrin.hpp"

namespace intel_mlperf {

class quant_tpp{
public:
  static void quant_ps_epi8(void *out, void *in, float scale, int64_t num);
};

}