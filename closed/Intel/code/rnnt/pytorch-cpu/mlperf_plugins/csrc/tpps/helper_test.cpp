#include <iostream>
#include <random>
#include <chrono>

#include "helper_test.h"
#include "helper.hpp"

namespace intel_mlperf {

static constexpr int qmax = 127.0;
static constexpr int qmin = -127.0;

void set_data_act(void *a, size_t sl, size_t hidden_length) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(19941101);
  std::normal_distribution<double> dis(0, 1);
  auto a_ = reinterpret_cast<int8_t (*)>(a);
  size_t elenum = sl * hidden_length;
  for (int i = 0; i < elenum; i++) {
    a_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // a_[i] = 0;
  }
}

void set_data_wei(void *w, void* b, size_t ic, size_t oc) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(19941101);
  std::normal_distribution<double> dis(0, 1);
  auto w_ = reinterpret_cast<int8_t (*)>(w);
  size_t elenum = ic * oc;
  for (int i = 0; i < elenum; i++) {
    w_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // w_[i] = 0;
  }
  auto b_ = reinterpret_cast<float (*)>(b);
  for (int i = 0; i < oc; i++) {
    b_[i] = static_cast<float>(dis(gen) * 0xf);
    // b_[i] = 0;
  }
}

void send_input(void* input, void* ninput, const size_t dim0, const size_t dim1) {
  auto input_ = reinterpret_cast<int8_t (*)[dim1]>(input);
  auto ninput_ = reinterpret_cast<int (*)[dim1]>(ninput);

  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      ninput_[i][j] = static_cast<int>(input_[i][j]);
    }
  }
}

void send_weight(void* weight, void* nweight, const size_t dim1, const size_t dim2) {
  size_t col_step = dim2 / 64;
  size_t col_tile = dim1 / 64;

  auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight);
  auto nweight_ = reinterpret_cast<int (*)[dim2]>(nweight);
  for (size_t i = 0; i < col_step; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t m = 0; m < col_tile; m++) {
        for (size_t k = 0; k < 16; k++) {
          for (size_t n = 0; n < 64; n++) {
            auto row_offset = n % 4;
            auto col_offset = n / 4;
            nweight_[m * 64 + k * 4 + row_offset][i * 64 + j * 16 + col_offset] = static_cast<int>(weight_[i][j][m][k][n]);
          }
        }
      }
    }
  }
}

void send_weight_2(void* weight, void* nweight, const size_t ic, const size_t oc) {
  size_t row_chunks = ic / 256;
  size_t col_chunks = oc / 64;

  auto weight_ = reinterpret_cast<int8_t (*)[row_chunks][4][4][16][64]>(weight);
  auto nweight_ = reinterpret_cast<int (*)[oc]>(nweight);
  for (size_t i = 0; i < col_chunks; ++i) {
    for (size_t m = 0; m < row_chunks; ++m) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t l = 0; l < 4; ++l) {
          for (size_t k = 0; k < 16; ++k) {
            for (size_t n = 0; n < 64; ++n) {
              auto row_offset = n % 4;
              auto col_offset = n / 4;
              nweight_[m * 256 + l * 64 + k * 4 + row_offset][i * 64 + j * 16 + col_offset] = static_cast<int>(weight_[i][m][j][l][k][n]);
            }
          }
        }
      }      
    }
  }
}

void naive_linear(void* a, const size_t lda, void* b, const size_t ldb, void* c, const size_t ldc,
                  void* bias, float scale, int sl, bool with_op, float scale2) {
  // this function is to test _tile_gemm output 64 of ldc
  auto a_ = reinterpret_cast<int (*)[lda]>(a);
  auto b_ = reinterpret_cast<int (*)[ldb]>(b);
  auto c_ = reinterpret_cast<int (*)[ldc]>(c);
  auto bias_ = reinterpret_cast<float (*)>(bias);

  for (int i = 0; i < sl; i++) {
    for (int j = 0; j < ldc; j++) {
      c_[i][j] = 0;
    }
  }

  for (int i = 0; i < sl; i++) {
    for (int j = 0; j < ldc; j++) {
      for (int k = 0; k < lda; k++) {
        c_[i][j] += a_[i][k] * b_[k][j];
        // std::cout << i << " " << k << " " << j << " : " << a_[i][k] << " , " << b_[k][j] << " : " << c_[i][j] << std::endl;
      }
      // getchar();
    }
  }

  for (int i = 0; i < sl; i++) {
    for (int j = 0; j < ldc; j++) {
      float tem = static_cast<float>(c_[i][j]);
      tem += bias_[j];
      tem *= scale;
      if (with_op) {
        tem = gelu_func(tem);
        tem = tem * scale2;
      }
      c_[i][j] = static_cast<int>(round(tem));
      c_[i][j] = c_[i][j] < qmax ? c_[i][j] : qmax;
      c_[i][j] = c_[i][j] > qmin ? c_[i][j] : qmin;
    }
  }
}

}
