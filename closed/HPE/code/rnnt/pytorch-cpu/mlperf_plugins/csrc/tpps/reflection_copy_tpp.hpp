#pragma once
#include <immintrin.h>

namespace intel_mlperf {

class reflection_copy_tpp {
public:
  static void ref(float *pout, float *pin, int64_t rl);
};

}  // namespace intel_mlperf

