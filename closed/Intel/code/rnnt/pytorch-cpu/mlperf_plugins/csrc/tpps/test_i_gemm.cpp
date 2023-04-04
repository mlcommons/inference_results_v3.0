#include "../amx_init.hpp"
#include "amx_config.hpp"
#include "helper.hpp"
#include "helper_test.h"
#include "i_gemm_tpp.hpp"
#include "i_linear_tpp.hpp"

using namespace intel_mlperf;

void performance_gemm(const int sl, const int ic, const int oc) {
  void* weight;
  posix_memalign(&weight, 4096, ic * oc);

  size_t row_chunks = ic / 256;
  size_t col_chunks = oc / 64;
  auto weight_ = reinterpret_cast<int8_t (*)[row_chunks][64][256]>(weight);

  alignas(64) float bias[col_chunks][64];
  alignas(64) int8_t input[sl][row_chunks][256];

  set_data_act(input, sl, ic);
  set_data_wei(weight, bias, ic, oc);
  // memset(weight, 1, ic * oc);

  alignas(64) int8_t output[sl][col_chunks][64];
  alignas(64) int acc_pad[4096];
  memset(acc_pad, 0, 4096 * 4);

  float scale = 0.018;
  bool post_op = false;
  float o_scale = 1.5;
  int count = 4000000;

  auto start = Time::now();
  for (int k = 0; k < count; ++k) {
    for (int i = 0; i < col_chunks; ++i) {
      for (int j = 0; j < row_chunks; ++j) {
        _tile_gemm_64x256::compute(input[0][j], ic, weight_[i][j], acc_pad);
      }
      _tile_gemm_64x256::quant_out(output[0][i], oc, acc_pad, bias[i], scale, post_op, o_scale);
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  printf("%d x %d gemm time : %f ns\n", ic, oc, (float)during / count);
}

void accuracy_gemm(const int sl, const int ic, const int oc) {
  size_t row_chunks = ic / 256;
  size_t col_chunks = oc / 64;

  void* weight = nullptr;
  posix_memalign(&weight, 4096, ic * oc);
  auto weight_ = reinterpret_cast<int8_t (*)[row_chunks][64][256]>(weight);

  alignas(64) float bias[col_chunks][64];
  alignas(64) int8_t input[sl][row_chunks][256];

  set_data_act(input, sl, ic);
  set_data_wei(weight, bias, ic, oc);

  alignas(64) int8_t output[sl][col_chunks][64];
  auto acc_pad = new int[4096];
  _tile_gemm_64x256::zero_pad(acc_pad);

  float scale = 0.018;
  bool post_op = false;
  float o_scale = 1.5;
  int count = 1000000;

  for (int i = 0; i < col_chunks; ++i) {
    for (int j = 0; j < row_chunks; ++j) {
      _tile_gemm_64x256::compute(input[0][j], ic, weight_[i][j], acc_pad);
    }
    _tile_gemm_64x256::quant_out(output[0][i], oc, acc_pad, bias[i], scale, post_op, o_scale);
  }

  auto ninput = new int[sl * ic];
  auto nweight = new int[ic * oc];
  auto noutput = new int[sl * oc];

  send_input(input, ninput, sl, ic);
  send_weight_2(weight, nweight, ic, oc);

  naive_linear(ninput, ic, nweight, oc, noutput, oc, bias, scale, sl);

  compare_naive_output(noutput, (int8_t*)output, sl, oc, oc, oc);
}

int main(int argc, char* argv[]) {
  performance_gemm(64, 1024, 64);
  return 0;
}
