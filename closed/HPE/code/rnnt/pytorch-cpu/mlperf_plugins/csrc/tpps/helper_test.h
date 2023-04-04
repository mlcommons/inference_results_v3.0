#pragma once

#include <cmath>
#include <iostream>
#include <chrono>

using Time = std::chrono::high_resolution_clock;
namespace intel_mlperf {

inline float gelu_func(const float x) {
  float rsqrt_2 = 0.70710678;
  auto y = std::erf(x * rsqrt_2) + 1;
  return x * y * 0.5;
}

void set_data_act(void *a, size_t sl, size_t hidden_length = 1024);

void set_data_wei(void *w, void* b, size_t ic = 1024, size_t oc = 256);

void send_input(void* input, void* ninput, const size_t dim0, const size_t dim1);

void send_weight(void* weight, void* nweight, const size_t dim1, const size_t dim2);

void send_weight_2(void* weight, void* nweight, const size_t dim1, const size_t dim2);
 
void naive_linear(void* a, const size_t lda, void* b, const size_t ldb, void* c, const size_t ldc,
                  void* bias, float scale, int sl, bool with_op = false, float scale2 = 1.0);

}
