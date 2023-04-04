#include "amx_config.hpp"
#include "../amx_init.hpp"
#include "i_linear_tpp.hpp"
#include "helper_test.h"
#include "helper.hpp"

using namespace intel_mlperf;

void performance_linear(const int sl, const int ic, const int oc) {
  void* weight;
  posix_memalign(&weight, 4096, ic * oc);

  size_t col_tile = ic / 64;
  size_t col_chunks = oc / 64;
  auto weight_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(weight);

  alignas(64) float bias[col_chunks][64];
  alignas(64) int8_t input[sl][col_tile][64];

  set_data_act(input, sl, ic);
  set_data_wei(weight, bias, ic, oc);
  // memset(weight, 1, ic * oc);

  alignas(64) int8_t output[sl][col_chunks][64];

  float scale = 0.018;
  bool post_op = false;
  float o_scale = 1.5;
  int count = 100000;
  
  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int row_tile = sl / 32;

  auto start = Time::now();
  for (int k = 0; k < count; ++k) {
    for (int i = 0; i < col_chunks; ++i) {
      for (int j = 0; j < row_tile; ++j) {
        _tile_dot_product_16x256<2, 16, plain_io>::compute(output[32*j][i], oc, input[32*j][i], weight_[i], bias[i], scale);
      }
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  printf("%d x %d linear time : %f ns\n", ic, oc, (float)during / count);
}

void performance_linear_i8o32(const int sl, const int ic, const int oc) {
  void* weight;
  posix_memalign(&weight, 4096, ic * oc);

  size_t col_tile = ic / 64;
  size_t col_chunks = oc / 64;
  auto weight_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(weight);

  alignas(64) float bias[col_chunks][64];
  alignas(64) int8_t input[sl][col_tile][64];

  set_data_act(input, sl, ic);
  set_data_wei(weight, bias, ic, oc);
  // memset(weight, 1, ic * oc);

  alignas(64) float output[sl][col_chunks][64];

  float scale = 0.018;
  bool post_op = false;
  float o_scale = 0.0;
  int count = 100000;

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int row_tile = sl / 32;

  auto start = Time::now();
  for (int k = 0; k < count; ++k) {
    for (int i = 0; i < col_chunks; ++i) {
      for (int j = 0; j < row_tile; ++j) {
        _tile_dot_product_16x256<2, 16, plain_io>::compute(output[32*j][i], oc, input[32*j][i], weight_[i], bias[i], scale, post_op, o_scale);
      }
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  printf("%d x %d linear_i8o32 time : %f ns\n", ic, oc, (float)during / count);
}

int main(int argc, char* argv[]) {
  amx_init::amx_init();
  intel_mlperf::Tilecfg().set_config();

  //intel_mlperf::performance_linear(64, 1024, 4096);
  //intel_mlperf::performance_linear(32, 2048, 4096);
  //intel_mlperf::performance_linear(32, 256, 4096);
  performance_linear_i8o32(32, 1024, 4096);
  //intel_mlperf::performance_linear_i8o32(32, 2048, 4096);
  //intel_mlperf::performance_linear_i8o32(32, 256, 4096);
  //intel_mlperf::performance_gemm(64, 1024, 64);
  return 0;
}

