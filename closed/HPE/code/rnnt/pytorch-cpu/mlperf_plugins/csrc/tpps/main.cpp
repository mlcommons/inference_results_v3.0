#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <math.h>
#include <omp.h>

#include "i_softmax_tpp.hpp"
#include "i_linear_tpp.hpp"
#include "lstm_postop_tpp.hpp"
#include "amx_config.hpp"
#include "../amx_init.hpp"
#include "helper.hpp"
#include "helper_test.h"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"

float gelu_func(float x) {
  float rsqrt_2 = 0.70710678;
  auto y = std::erf(x * rsqrt_2) + 1;
  return x * y * 0.5;
}

void send_data2naive(void* a, void* b, void* na, void* nb, int row_tile, bool is_block, size_t lda = 1024, size_t ldb = 64) {
  if (is_block) {
    auto a_ = reinterpret_cast<int8_t (*)[16][16][64]>(a);
    auto na_ = reinterpret_cast<int (*)[lda]>(na);
    for (int i = 0; i < row_tile; i++) {
      for (int j = 0; j < 16; j++) {
        for (int k = 0; k < 16; k++) {
          for (int m = 0; m < 64; m++) {
            na_[i * 16 + k][j * 64 + m] = static_cast<int>(a_[i][j][k][m]);
          }
        }
      }
    }
  } else {
    auto a_ = reinterpret_cast<int8_t (*)[lda]>(a);
    auto na_ = reinterpret_cast<int (*)[lda]>(na);
    for (int i = 0; i < row_tile * 16; i++) {
      for (int j = 0; j < lda; j++) {
        na_[i][j] = static_cast<int>(a_[i][j]);
      }
    }
  }
  
  auto b_ = reinterpret_cast<int8_t (*)[16][16][64]>(b);
  auto nb_ = reinterpret_cast<int (*)[ldb]>(nb);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
        for (int m = 0; m < 64; m++) {
          auto row_offset = m % 4;
          auto col_offset = m / 4;
          nb_[j * 64 + k * 4 + row_offset][i * 16 + col_offset] = static_cast<int>(b_[i][j][k][m]);
        }
      }
    }
  }
}

void performance_test_256x256(int row_tile, void* C, size_t ldc, void* A, void* B, float* bias, float scale, 
                              size_t counter, size_t core_num, bool post_op = false, float o_scale = 1.0) {

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int loop_num = row_tile / 2;
  int remaining = row_tile % 2;
  size_t single_loop = counter / core_num;

# pragma omp parallel for collapse(1) num_threads (core_num)
  for (size_t i = 0; i < counter; i++) {
#   pragma unroll
    for (int j = 0; j < loop_num; j++) {
      auto core_out = reinterpret_cast<int8_t (*)[row_tile * 16][ldc]>(C);
      auto act = reinterpret_cast<int8_t (*)[row_tile * 16][ldc]>(A);
      auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(B);

      // intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(core_out[i / single_loop][j * 2 * 16], ldc, act[i / single_loop][j * 2], wei400m_[i % counter], bias, scale);
      intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(core_out[i / single_loop][j * 2 * 16], ldc, act[i / single_loop][j * 2], B, bias, scale, post_op, o_scale);
    }
    // if (remaining) {
    //   intel_mlperf::_tile_dot_product_16x256<1, 16, plain_io>::compute(core_out[i / single_loop][loop_num * 2 * 16], ldc, act[i / single_loop][loop_num * 2], wei400m_[i % counter], bias, scale);
    // }
  }
}

void accuracy_test_256x256(int row_tile, void* C, size_t ldc, void* A, void* B, float* bias, float scale, bool post_op = false, float o_scale = 1.0) {

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int loop_num = row_tile / 2;
  int remaining = row_tile % 2;

  auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
  auto act = reinterpret_cast<int8_t (*)[ldc]>(A);
  
  for (int j = 0; j < row_tile - 1; j += 2) {
    intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(C_[j * 16], ldc, act[j * 16], B, bias, scale, post_op, o_scale);
  }
  if (remaining) {
    intel_mlperf::_tile_dot_product_16x256<1, 16, plain_io>::compute(C_[loop_num * 2 * 16], ldc, act[loop_num * 2 * 16], B, bias, scale, post_op, o_scale);
  }
}

void test_tile_16x256(int row_tile) {
  alignas(64) int8_t act[row_tile][16][1024];
  alignas(64) int8_t wei[256 * 256];
  alignas(64) int nact[row_tile * 16 * 1024];
  alignas(64) int nwei[1024 * 64];
  
  alignas(64) float bias[64];

  size_t ldc = 1024;
  
  alignas(64) int nout[row_tile * 16 * ldc];
  float scale = 0.0018;

  intel_mlperf::set_data_act(act, row_tile * 16);
  intel_mlperf::set_data_wei(wei, bias);

  using tile_io  = intel_mlperf::io_policy<16, intel_mlperf::i_format::tile>;

  // first : do tile block
  send_data2naive(act, wei, nact, nwei, row_tile, true);

  // intel_mlperf::amx_init();
  // intel_mlperf::Tilecfg().set_config();

  intel_mlperf::naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile * 16);
  printf("****************** start block test row_tile = %d... *********************\n", row_tile);
  printf("****************** accuracy...*********************\n");
  alignas(64) int8_t out[row_tile][16][64];

  // switch (row_tile) {
  // case (2):
  //   intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (3):
  //   intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (4):
  //   intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (5):
  //   intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (6):
  //   intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (7):
  //   intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (8):
  //   intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (9):
  //   intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (10):
  //   intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (11):
  //   intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (12):
  //   intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // }

  auto out_ = reinterpret_cast<int8_t (*)[16][64]>(out);
  auto nout_ = reinterpret_cast<int (*)[ldc]>(nout);
  // for (int i = 0; i < row_tile; i++) {
  //   intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)out_[i], 16, 64, ldc, 64);
  // } 

  printf("************************ start performance test... **************************\n");
  size_t count = 64000;
  // auto wei400m = new int8_t[6400 * 256 * 256];
  // auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);
  auto lstart = Time::now();
  for (size_t i = 0; i < count; i++) {
    // switch (row_tile) {
    // case (2):
    //   intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (3):
    //   intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (4):
    //   intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (5):
    //   intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (6):
    //   intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (7):
    //   intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (8):
    //   intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (9):
    //   intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (10):
    //   intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (11):
    //   intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (12):
    //   intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // }
  }
  auto lduring =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart)
          .count();
  std::cout << count  << " times block tile linear time : "
            << (float)lduring / 1000 / 1000 << " ms " << std::endl;
  std::cout << "single linear time : " << (float)lduring / count << " ns" << std::endl;

  // getchar();
  printf("****************** start plain test... *********************\n");

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;

  bool post_op = false;
  float o_scale = 1.5;

  send_data2naive(act, wei, nact, nwei, row_tile, false);
  intel_mlperf::naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile * 16, post_op, o_scale);
  int8_t p_out[row_tile * 16][ldc];
  printf("****************** accuracy...*********************\n");
  
  accuracy_test_256x256(row_tile, p_out, ldc, act, wei, bias, scale, post_op, o_scale);

  auto p_out_ = reinterpret_cast<int8_t (*)[16][ldc]>(p_out);
  nout_ = reinterpret_cast<decltype(nout_)>(nout);
  // for (int i = 0; i < row_tile; i++) {
  //   intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)p_out_[i], 16, 64, ldc, ldc);
  // } 

  printf("************************ start performance test... **************************\n");
  
  int core_num = 1;
  size_t block_num = core_num * 128;
  int counter = core_num * 200;
  int single_loop = block_num / core_num;
  int row_total_num = 50;

  // auto wei400m = new int8_t[block_num * 256 * 256];
  void* wei400m = nullptr;
  posix_memalign(&wei400m, 4096, block_num * 256 * 256);
  auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);

// # pragma omp parallel for
//   for (int i = 0; i < block_num; i++) {
//     set_data_wei(wei400m_[i], bias);
//   }
  memset(wei400m, 1, block_num * 256 * 256);

  std::vector<float> durings;
  durings.emplace_back(0);
 
  for (int i = 2; i < row_total_num; i += 2) {
    int row_tile_ = i;

    void* core_act_ = nullptr;
    posix_memalign(&core_act_, 4096, core_num * row_tile_ * 16 * ldc);
    memset(core_act_, 1, core_num * row_tile_ * 16 * ldc);

    void* core_out_ = nullptr;
    posix_memalign(&core_out_, 4096, core_num * row_tile_ * 16 * ldc);
    lstart = Time::now();
    for (int j = 0; j < counter; j++) {
      performance_test_256x256(row_tile_, core_out_, ldc, core_act_, wei, bias, scale, block_num, core_num);
    }
    lduring = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart).count();
    std::cout << row_tile_ << " linear time : " << (float)lduring / counter / single_loop << " ns durings : " << 
      (float)lduring / counter / single_loop - durings.back() << " ns" << std::endl;
    durings.emplace_back((float)lduring / counter / single_loop);
  }
  
  // for (int i = 0; i < durings.size() - 1; i += 2) {
  //   std::cout << durings[i + 1] - durings[i] << std::endl;
  // }

  // delete[] wei400m;
  free(wei400m);
}

void test_block_gemm_cache_hit(const size_t batch_size, const size_t input_f, const size_t output_f) {
  bool has_bias = true;
  bool post_op = false;
  float o_scale = 0.0;

  size_t row_tile = (batch_size + 15) / 16;
  size_t col_step = output_f / 64;
  size_t col_tile = input_f / 64;

  size_t block_row = input_f / 4;
  size_t block_num = 1;
  size_t hidden = output_f / 4;
  const size_t seq_len = 2;

  auto gemm_ = intel_mlperf::i_linear_i8o32(batch_size, input_f, output_f, has_bias, post_op);
  auto gemm_hh_ = intel_mlperf::i_linear_i8o32(batch_size, hidden, output_f, has_bias, post_op);

  void* weight_ih = nullptr;
  posix_memalign(&weight_ih, 4096, block_num * input_f * output_f);
  void* weight_hh = nullptr;
  posix_memalign(&weight_hh, 4096, block_num * hidden * output_f);

  float bias[output_f];
  float bias_hh[output_f];
  float scale = 0.0018;
  float in_scale = 0.0018;
  float out_scale = 0.0018;

  for (int i = 0; i < block_num; i++) {
    auto w = reinterpret_cast<int8_t (*)[input_f][output_f]>(weight_ih);
    intel_mlperf::set_data_wei(w[i], bias, input_f, output_f);
    auto w_h = reinterpret_cast<int8_t (*)[hidden][output_f]>(weight_ih);
    intel_mlperf::set_data_wei(w_h[i], bias_hh, input_f, output_f);
  }

  printf("**************************** start test performance **********************\n");
  int loop_num = 10000;
  auto start = Time::now();
  for (int i = 0; i < loop_num; i++) {
    alignas(64) int8_t input[seq_len][batch_size][input_f];
    alignas(64) float output[seq_len][batch_size][output_f];
    for (int j = 0; j < seq_len; j++) {
      # pragma omp parallel
      {
        auto total_core_num = omp_get_num_threads();
        auto core_id = omp_get_thread_num();
        gemm_.tile_dot_product_16x256_shortage(output[j], input[j], weight_ih, bias, scale, o_scale, batch_size, core_id, total_core_num);
      }
    }
    alignas(64) int8_t hx[batch_size][hidden];
    alignas(64) float output_hh[seq_len][batch_size][output_f];
    for (int j = 0; j < seq_len; j++) {
      # pragma omp parallel
      {
        auto total_core_num = omp_get_num_threads();
        auto core_id = omp_get_thread_num();
        gemm_hh_.tile_dot_product_16x256_shortage(output_hh[j], hx, weight_hh, bias_hh, scale, o_scale, batch_size, core_id, total_core_num);
      }

      #pragma omp parallel for collapse(2)
      for (int k1 = 0; k1 < batch_size; k1++) {
        for (int k2 = 0; k2 < output_f; k2++) {
          output[j][k1][k2] += output_hh[j][k1][k2];
        }
      }

      alignas(64) _Float16 pin_ct[batch_size][hidden];
      alignas(64) float pout_1[batch_size][hidden];
      alignas(64) int8_t pout_1_q[batch_size][hidden];
      alignas(64) int8_t pout_2[batch_size][hidden];
      alignas(64) _Float16 pout_3[batch_size][hidden];
      #pragma omp parallel for
      for (auto b = 0; b < batch_size; ++b) {
        intel_mlperf::lstm_postop_tpp::ref(pout_1[b],pout_1_q[b],&output[j][b][0],&output[j][b][hidden],&output[j][b][hidden*2],&output[j][b][hidden*3],pin_ct[b],in_scale,out_scale,hidden,false);
      }
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  std::cout << batch_size << " x " << input_f << " x " << output_f << " : " << (float)during / loop_num << " ns " << std::endl;
  delete[] weight_hh;
  delete[] weight_ih;
}

void test_block_gemm_perf(const size_t batch_size, const size_t input_f, const size_t output_f, const size_t core_num) {
  printf("test_block_gemm_perf %dc: %d * %d\n", core_num, input_f, output_f);
  bool has_bias = true;
  bool post_op = false;
  float o_scale = 0.0;

  size_t row_tile = (batch_size + 15) / 16;
  size_t col_step = output_f / 64;
  size_t col_tile = input_f / 64;

  size_t block_row = input_f / 4;
  size_t block_num = 1;

  auto gemm_ = intel_mlperf::i_linear_i8o32(batch_size, input_f, output_f, has_bias, post_op);

  void* weight_ih = nullptr;
  posix_memalign(&weight_ih, 4096, block_num * input_f * output_f);

  float bias[output_f];
  float scale = 0.0018;

  for (int i = 0; i < block_num; i++) {
    auto w = reinterpret_cast<int8_t (*)[input_f][output_f]>(weight_ih);
    intel_mlperf::set_data_wei(w[i], bias, input_f, output_f);
  }

  printf("**************************** start test performance **********************\n");
  int warmup_num = 100;
  int loop_num = 500000;
  alignas(64) int8_t input[batch_size][input_f];
  alignas(64) float output[batch_size][output_f];
  for (int i = 0; i < warmup_num; i++) {
    # pragma omp parallel
    {
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      size_t start_ = batch_size * core_id / total_core_num;
      size_t chunk_sl_ = (batch_size * core_id + batch_size) / total_core_num - start_;
      gemm_.tile_dot_product_16x256_shortage(output, input, weight_ih, bias, scale, o_scale, batch_size, core_id, total_core_num);
      //gemm_.tile_dot_product_16x256(output[start_], input[start_], weight_ih, bias, scale, o_scale, chunk_sl_, core_id, total_core_num);
    }
  }
  auto start = Time::now();
  for (int i = 0; i < loop_num; i++) {
    # pragma omp parallel
    {
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      size_t start_ = batch_size * core_id / total_core_num;
      size_t chunk_sl_ = (batch_size * core_id + batch_size) / total_core_num - start_;
      //gemm_.tile_dot_product_16x256_shortage(output, input, weight_ih, bias, scale, o_scale, batch_size, core_id, total_core_num);
      gemm_.tile_dot_product_16x256(output[start_], input[start_], weight_ih, bias, scale, o_scale, chunk_sl_, core_id, total_core_num);
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  std::cout << batch_size << " x " << input_f << " x " << output_f << " : " << (float)during / loop_num << " ns " << std::endl;
  delete[] weight_ih;
}

int main(int argc, char* argv[]) {
  int row_tile = 2;
  int is_block = 1;
  int num_cores = 56;
  int input_f = 1024;
  if (argc >= 2) {
    num_cores = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    input_f = std::atoi(argv[2]);
  }

  amx_init::amx_init();
  intel_mlperf::Tilecfg().set_config();

  //test_block_gemm_cache_hit(32, 2048, 4096, 4);
  test_block_gemm_perf(32 * num_cores, input_f, 1024, num_cores);
  //test_block_gemm_perf(32, input_f, 1024 * num_cores, num_cores);

  return 0;
}
