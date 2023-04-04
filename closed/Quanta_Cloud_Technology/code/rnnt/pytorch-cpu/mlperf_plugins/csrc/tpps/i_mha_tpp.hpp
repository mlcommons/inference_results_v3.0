#pragma once

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

namespace intel_mlperf {

void amx_per_head(const void *qkv_ptr, size_t ldqkv, void *a_ptr, size_t ldatt, size_t sl,
                      float M, float oscale, int32_t att_mask, float M2);

} // namespace intel_mlperf
