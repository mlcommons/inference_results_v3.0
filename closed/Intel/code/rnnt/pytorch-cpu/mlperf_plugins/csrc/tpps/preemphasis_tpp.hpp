#pragma once
#include <immintrin.h>

namespace intel_mlperf {

class preemphasis_tpp {
public:
  static void ref(float *pout, float *pin, int64_t rl, float coeff = 0.97);

private:
  static void ref_head(float *pout, float *pin, int64_t rl, float coeff = 0.97);
};

} // namespace intel_mlperf
