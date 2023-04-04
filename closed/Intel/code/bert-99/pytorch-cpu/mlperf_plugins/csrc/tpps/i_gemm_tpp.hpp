#pragma once

#include <cstdlib>
#include <iostream>
#include <immintrin.h>
#include <assert.h>
#include "amx_tdpbssd.hpp"
#include "amx_loadd.hpp"
#include "amx_config.hpp"
#include "el_common_intrin.hpp"

#include "helper.hpp"

namespace intel_mlperf {

struct _tile_gemm_64x256 {

  inline static void zero_pad(void* acc_pad) {
    // the acc_pad assumed to be [4][4][16][16]
    auto acc_pad_ = reinterpret_cast<int (*)[16][16]>(acc_pad);
    _tile_zero(TMM0);
    
    #pragma unroll (16)
    for (int i = 0; i < 16; ++i) {
      _tile_stored(TMM0, acc_pad_[i], 64);
    }
  }

  inline static void load_input(void* A, const size_t lda, void* B) {
    auto A_ = reinterpret_cast<int8_t (*)[16][lda]>(A);
    _tile_loadd(TMM4, A_[0], lda);
    _tile_loadd(TMM5, A_[1], lda);

    auto B_ = reinterpret_cast<int8_t (*)[4][16][64]>(B);
    _tile_loadd(TMM6, B_[0], 64);
    _tile_loadd(TMM7, B_[1], 64);
  }

  inline static void dot_pro(void* A, const size_t lda, void* B) {
    load_input(A, lda, B);

    _tile_dpbssd(TMM0, TMM4, TMM6);
    _tile_dpbssd(TMM2, TMM5, TMM6);
    _tile_dpbssd(TMM1, TMM4, TMM7);
    _tile_dpbssd(TMM3, TMM5, TMM7);
  }

  // use C as scratch pad, not exact output
  inline static void load_acc(void* C) {
    auto C_ = reinterpret_cast<int (*)[4][16][16]>(C);

    _tile_loadd(TMM0, C_[0][0], 64);
    _tile_loadd(TMM1, C_[0][1], 64);
    _tile_loadd(TMM2, C_[1][0], 64);
    _tile_loadd(TMM3, C_[1][1], 64);
  }

  inline static void store(void* C) {
    auto C_ = reinterpret_cast<int (*)[4][16][16]>(C);

    _tile_stored(TMM0, C_[0][0], 64);
    _tile_stored(TMM1, C_[0][1], 64);
    _tile_stored(TMM2, C_[1][0], 64);
    _tile_stored(TMM3, C_[1][1], 64);
  }

  inline static void compute(void* A, const size_t lda, void* B, void* acc_pad) {
    auto A_ = reinterpret_cast<int8_t (*)[16][lda]>(A);
    auto B_ = reinterpret_cast<int8_t (*)[4][16][64]>(B);

    //acc_pad has shape 4x4x16x16
    auto acc_pad_ = reinterpret_cast<int (*)[4][16][16]>(acc_pad);
    
    // acc_pad must be all 0s
    load_acc(acc_pad_[0][0]);
    #pragma unroll (4)
    for (int i = 0; i < 4; ++i) {
      dot_pro(&A_[0][0][64*i], lda, B_[0][i]);
    }
    store(acc_pad_[0][0]);

    load_acc(acc_pad_[2][0]);
    #pragma unroll (4)
    for (int i = 0; i < 4; ++i) {
      dot_pro(&A_[2][0][64*i], lda, B_[0][i]);
    }
    store(acc_pad_[2][0]);

    load_acc(acc_pad_[0][2]);
    #pragma unroll (4)
    for (int i = 0; i < 4; ++i) {
      dot_pro(&A_[0][0][64*i], lda, B_[2][i]);
    }
    store(acc_pad_[0][2]);

    load_acc(acc_pad_[2][2]);
    #pragma unroll (4)
    for (int i = 0; i < 4; ++i) {
      dot_pro(&A_[2][0][64*i], lda, B_[2][i]);
    }
    store(acc_pad_[2][2]);
  }

  inline static void quant_out(void* C, const size_t ldc, void* acc_pad, 
                               void* bias, float scale, bool post_op, float o_scale) {
    auto scale_ = _mm512_set1_ps(scale);
    __m512 o_scale_;
    if (post_op) {
      o_scale_ = _mm512_set1_ps(o_scale);
    }

    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16][ldc]>(C);
    auto acc_pad_ = reinterpret_cast<int (*)[4][16][16]>(acc_pad);

    #pragma unroll (4)
    for (int t = 0; t < 4; ++t) {

      #pragma unroll (16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_loadu_epi32(acc_pad_[t][0][i]);
        auto i1 = _mm512_loadu_epi32(acc_pad_[t][1][i]);
        auto i2 = _mm512_loadu_epi32(acc_pad_[t][2][i]);
        auto i3 = _mm512_loadu_epi32(acc_pad_[t][3][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        __m512i o0, o1, o2, o3;

        if (post_op) {
          o0 = _mm512_scale_minmax_gelu_i8_ps(f0, scale_, o_scale_);
          o1 = _mm512_scale_minmax_gelu_i8_ps(f1, scale_, o_scale_);
          o2 = _mm512_scale_minmax_gelu_i8_ps(f2, scale_, o_scale_);
          o3 = _mm512_scale_minmax_gelu_i8_ps(f3, scale_, o_scale_);
        } else {
          o0 = _mm512_scale_minmax_i8_ps(f0, scale_);
          o1 = _mm512_scale_minmax_i8_ps(f1, scale_);
          o2 = _mm512_scale_minmax_i8_ps(f2, scale_);
          o3 = _mm512_scale_minmax_i8_ps(f3, scale_);
        }

        _mm512_mask_cvtepi32_storeu_epi8(C_[t][i] + 0, 0xffff, o0);
        _mm512_mask_cvtepi32_storeu_epi8(C_[t][i] + 16, 0xffff, o1);
        _mm512_mask_cvtepi32_storeu_epi8(C_[t][i] + 32, 0xffff, o2);
        _mm512_mask_cvtepi32_storeu_epi8(C_[t][i] + 48, 0xffff, o3);
      }
    }
    zero_pad(acc_pad);
  }
};
}