#pragma once
#include <immintrin.h>

namespace intel_mlperf {
template <int vec_length> class i_residual_layernorm_tpp {
public:
  static void ref(int8_t *out, int8_t *src1, int8_t *src2, float *weight,
                  float *bias, float s1, float s2, float oscale, int64_t rl,
                  float eps = 1e-12, int8_t o_off = 0);

  // first embedding layer
  static void ref(int8_t *out, int8_t *src1, int8_t *src2, int8_t *src3,
                  float *weight, float *bias, float s1, float s2, float oscale,
                  int64_t rl, float eps = 1e-12, int8_t o_off = 0);

  static void ref_cin(int8_t *out, int8_t *src1, int8_t *src2, float *weight,
                      float *bias, float s1, float s2, float oscale, int64_t rl,
                      float eps = 1e-12, int8_t o_off = 0);

  static void ref_fp16(int8_t *out, int8_t *src1, int8_t *src2,
                       _Float16 *weight, _Float16 *bias, float s1, float s2,
                       float oscale, int64_t rl, float eps = 1e-12,
                       int8_t o_off = 0);
};

template <int vec_length> class i_layernorm_tpp {
public:
  static void ref(float *out, float *in, float *weight, float *bias, int64_t rl,
                  float eps = 1e-12, bool unbiased = false);

  static void ref(int8_t *out, float *in, float *weight, float *bias,
                  float oscale, int64_t rl, float eps = 1e-12,
                  int8_t o_off = 0);

  static void ref(int8_t *out, int8_t *in, float *weight, float *bias,
                  float oscale, int64_t rl, float eps = 1e-12,
                  int8_t o_off = 0);
};
} // namespace intel_mlperf
