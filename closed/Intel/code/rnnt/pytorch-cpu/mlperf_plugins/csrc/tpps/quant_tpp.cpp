#include "quant_tpp.hpp"
#include <cstdlib>
#include <iostream>

namespace intel_mlperf {
void quant_tpp::quant_ps_epi8(void *out, void *in, float scale, int64_t num){
  auto n_batch = num / 16;
  auto pin = reinterpret_cast<float (*)>(in);
  auto pout = reinterpret_cast<int8_t (*)>(out);
  #pragma omp parallel for
  for(int z = 0; z < n_batch; z++){
    auto vin_scale = _mm512_set1_ps(scale);
    auto in_ps = _mm512_loadu_ps(&pin[z*16]);
    auto out_quant = _mm512_scale_min128max_i8_ps(in_ps, vin_scale);
    _mm512_mask_cvtepi32_storeu_epi8(&pout[z*16], 0xffff, out_quant);
  }
}
}