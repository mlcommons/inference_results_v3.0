#pragma once
#include <immintrin.h>

namespace intel_mlperf {

template <int factor>
class frame_splicing_tpp {
public:
  static void ref(float *pout, float *pin, int64_t fl, int64_t tl, int64_t out_tl);
};

}  // namespace intel_mlperf
