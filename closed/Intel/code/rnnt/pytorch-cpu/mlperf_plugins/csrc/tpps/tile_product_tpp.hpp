#pragma once

#include <assert.h>
#include <immintrin.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "amx_config.hpp"
#include "amx_loadd.hpp"
#include "amx_tdpbf16ps.hpp"
#include "amx_tdpbssd.hpp"
#include "el_common_intrin.hpp"

namespace intel_mlperf {
enum struct i_format { plain, tile };

template <int col_tile, i_format i_fmt>
class io_policy;

template <int col_tile>
class io_policy<col_tile, i_format::tile> {
public:
  typedef char (*tile_array)[16 * 64];
  typedef char (*block_pointer)[col_tile][16][64];

  template <int tile_num>
  inline static void tile_load(void *A, int idx) {
    auto A_ = reinterpret_cast<block_pointer>(A);
    __tile_loadd<tile_num>(A_[idx], 64);
  }

  inline static void _mm512_coalescing_store_epi8(
      void *C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t(*)[4][16]>(C);

    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][0], 0xffff, o0);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][1], 0xffff, o1);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][2], 0xffff, o2);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][3], 0xffff, o3);
  }
};

template <int col_tile>
class io_policy<col_tile, i_format::plain> {
public:
  typedef char (*tile_array)[64];
  typedef char (*block_pointer)[16][col_tile][64];

  template <int tile_num>
  inline static void tile_load(void *A, int idx) {
    auto A_ = reinterpret_cast<block_pointer>(A);
    __tile_loadd<tile_num>(A_[idx], col_tile * 64);
  }

  inline static void _mm512_coalescing_store_epi8(
      void *C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t(*)[ldc]>(C);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 0, 0xffff, o0);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 16, 0xffff, o1);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 32, 0xffff, o2);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 48, 0xffff, o3);
  }

  inline static void _mm512_coalescing_store_ps(
      void *C, size_t ldc, int idx, __m512 o0, __m512 o1, __m512 o2, __m512 o3) {
    auto C_ = reinterpret_cast<float(*)[ldc]>(C);
    _mm512_storeu_ps(C_[idx] + 0, o0);
    _mm512_storeu_ps(C_[idx] + 16, o1);
    _mm512_storeu_ps(C_[idx] + 32, o2);
    _mm512_storeu_ps(C_[idx] + 48, o3);
  }

  inline static void _mm512_coalescing_store_epi32(
      void *C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int32_t(*)[ldc]>(C);
    _mm512_storeu_epi32(C_[idx] + 0, o0);
    _mm512_storeu_epi32(C_[idx] + 16, o1);
    _mm512_storeu_epi32(C_[idx] + 32, o2);
    _mm512_storeu_epi32(C_[idx] + 48, o3);
  }

  inline static void _mm512_coalescing_packs_epi32_store_epi8(
      void *C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t(*)[ldc]>(C);

    auto _m512_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    auto t0 = _mm512_packs_epi32(o0, o1);
    auto t1 = _mm512_packs_epi32(o2, o3);
    auto p0 = _mm512_permutexvar_epi64(_m512_idx, t0);
    auto p1 = _mm512_permutexvar_epi64(_m512_idx, t1);
    auto t2 = _mm512_packs_epi16(p0, p1);
    auto p2 = _mm512_permutexvar_epi64(_m512_idx, t2);

    _mm512_storeu_epi8(C_[idx], p2);
  }

  inline static void _mm512_coalescing_packs_epi16_store_epi8(
      void *C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t(*)[ldc]>(C);

    auto _m512_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    auto t0 = _mm512_packs_epi16(o0, o1);
    auto t1 = _mm512_packs_epi16(o2, o3);

    auto p0 = _mm512_permutexvar_epi64(_m512_idx, t0);
    auto p1 = _mm512_permutexvar_epi64(_m512_idx, t1);

    _mm512_storeu_epi8(C_[idx], p0);
    _mm512_storeu_epi8(C_[idx + 1], p1);
  }

  inline static void _mm512_coalescing_packs_ps_store_pbh(
      void *C, __m512 o0, __m512 o1) {
    auto o_bf = _mm512_cvtne2ps_pbh(o1, o0);
    _mm512_storeu_epi16(C, o_bf);
  }
};

template <int row_tile, int col_tile, typename io_policy>
class _tile_dot_product_16x256;

template <int row_tile, int col_tile, typename io_policy>
class _tile_dot_product_16x256_base {
public:
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  friend class _tile_dot_product_16x256<row_tile, col_tile, io_policy>;

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale, bool post_op,
      float o_scale) {
    auto s_0_ = reinterpret_cast<int(*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    __m512 o_scale_;
    if (post_op) {
      o_scale_ = _mm512_set1_ps(o_scale);
    }
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[t][0][i]);
        auto i1 = _mm512_load_epi32(s_0_[t][1][i]);
        auto i2 = _mm512_load_epi32(s_1_[t][0][i]);
        auto i3 = _mm512_load_epi32(s_1_[t][1][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        if (post_op) {
          auto o0 = _mm512_scale_minmax_gelu_i8_ps(f0, scale_, o_scale_);
          auto o1 = _mm512_scale_minmax_gelu_i8_ps(f1, scale_, o_scale_);
          auto o2 = _mm512_scale_minmax_gelu_i8_ps(f2, scale_, o_scale_);
          auto o3 = _mm512_scale_minmax_gelu_i8_ps(f3, scale_, o_scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_epi32_store_epi8(
              C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ps(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ps(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ps(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ps(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_epi32_store_epi8(
              C_[t], ldc, i, o0, o1, o2, o3);
        }
      }
    }
  }

  inline static void quant_out_fp16(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale, bool post_op,
      float o_scale) {
    auto s_0_ = reinterpret_cast<int(*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ph(scale);
    __m512h o_scale_;
    if (post_op) {
      o_scale_ = _mm512_set1_ph(o_scale);
    }
    // TODO: wait for model bias to fp16
    auto bias_ = reinterpret_cast<_Float16(*)[32]>(bias);

    // TODO: when model bias is fp16, this step could be optimized
    auto bias32_0 = _mm512_loadu_ph(bias_[0]);
    auto bias32_1 = _mm512_loadu_ph(bias_[1]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(8)
      for (int i = 0; i < 16; i += 2) {
        // TODO: bias and another scale
        auto i00 = _mm512_load_epi32(s_0_[t][0][i]);
        auto i10 = _mm512_load_epi32(s_0_[t][1][i]);
        auto f0 = _mm512_concat_cvtepi32_ph(i00, i10) + bias32_0;

        auto i20 = _mm512_load_epi32(s_1_[t][0][i]);
        auto i30 = _mm512_load_epi32(s_1_[t][1][i]);
        auto f1 = _mm512_concat_cvtepi32_ph(i20, i30) + bias32_1;

        auto i01 = _mm512_load_epi32(s_0_[t][0][i + 1]);
        auto i11 = _mm512_load_epi32(s_0_[t][1][i + 1]);
        auto f2 = _mm512_concat_cvtepi32_ph(i01, i11) + bias32_0;

        auto i21 = _mm512_load_epi32(s_1_[t][0][i + 1]);
        auto i31 = _mm512_load_epi32(s_1_[t][1][i + 1]);
        auto f3 = _mm512_concat_cvtepi32_ph(i21, i31) + bias32_1;

        if (post_op) {
          auto o0 = _mm512_scale_minmax_gelu_i8_ph(f0, scale_, o_scale_);
          auto o1 = _mm512_scale_minmax_gelu_i8_ph(f1, scale_, o_scale_);
          auto o2 = _mm512_scale_minmax_gelu_i8_ph(f2, scale_, o_scale_);
          auto o3 = _mm512_scale_minmax_gelu_i8_ph(f3, scale_, o_scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_epi16_store_epi8(
              C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ph(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ph(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ph(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ph(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_epi16_store_epi8(
              C_[t], ldc, i, o0, o1, o2, o3);
        }
      }
    }
  }

  inline static void dequant_float_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<float(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[t][0][i]);
        auto i1 = _mm512_load_epi32(s_0_[t][1][i]);
        auto i2 = _mm512_load_epi32(s_1_[t][0][i]);
        auto i3 = _mm512_load_epi32(s_1_[t][1][i]);

        auto o0 = _mm512_cvtepi32_ps(i0) + b0;
        auto o1 = _mm512_cvtepi32_ps(i1) + b1;
        auto o2 = _mm512_cvtepi32_ps(i2) + b2;
        auto o3 = _mm512_cvtepi32_ps(i3) + b3;

        // every 16 got output
        io_policy::_mm512_coalescing_store_ps(
            C_[t], ldc, i, o0 * scale_, o1 * scale_, o2 * scale_, o3 * scale_);
      }
    }
  }

  inline static void int32_out_no_bias(void *C, size_t ldc, void *s_0, void *s_1) {
    auto s_0_ = reinterpret_cast<int32_t(*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int32_t(*)[2][16][16]>(s_1);

    auto C_ = reinterpret_cast<int32_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto o0 = _mm512_load_epi32(s_0_[t][0][i]);
        auto o1 = _mm512_load_epi32(s_0_[t][1][i]);
        auto o2 = _mm512_load_epi32(s_1_[t][0][i]);
        auto o3 = _mm512_load_epi32(s_1_[t][1][i]);

        // every 16 got output
        io_policy::_mm512_coalescing_store_epi32(C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  inline static void dequant_float_out_accum(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int32_t(*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int32_t(*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<float(*)[16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[t][0][i]);
        auto i1 = _mm512_load_epi32(s_0_[t][1][i]);
        auto i2 = _mm512_load_epi32(s_1_[t][0][i]);
        auto i3 = _mm512_load_epi32(s_1_[t][1][i]);

        auto a0 = _mm512_load_epi32(C_[t][i]);
        auto a1 = _mm512_load_epi32(C_[t][i] + 16);
        auto a2 = _mm512_load_epi32(C_[t][i] + 32);
        auto a3 = _mm512_load_epi32(C_[t][i] + 48);

        auto o0 = _mm512_cvtepi32_ps(i0 + a0) + b0;
        auto o1 = _mm512_cvtepi32_ps(i1 + a1) + b1;
        auto o2 = _mm512_cvtepi32_ps(i2 + a2) + b2;
        auto o3 = _mm512_cvtepi32_ps(i3 + a3) + b3;
        // every 16 got output
        io_policy::_mm512_coalescing_store_ps(
            C_[t], ldc, i, o0 * scale_, o1 * scale_, o2 * scale_, o3 * scale_);
      }
    }
  }

  inline static void float_out(void *C, size_t ldc, void *s, void *bias) {
    auto s_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(s);

    auto bias_ = *reinterpret_cast<float(*)[2][16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);

    auto C_ = *reinterpret_cast<float(*)[row_tile][16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_ps(s_[t][0][i]);
        auto i1 = _mm512_load_ps(s_[t][1][i]);

        _mm512_storeu_ps(&C_[t][i][0], i0 + b0);
        _mm512_storeu_ps(&C_[t][i][16], i1 + b1);
      }
    }
  }

  inline static void float_out_no_bias(void *C, size_t ldc, void *s) {
    auto s_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(s);

    auto C_ = *reinterpret_cast<float(*)[row_tile][16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto o0 = _mm512_load_ps(s_[t][0][i]);
        _mm512_storeu_ps(&C_[t][i][0], o0);
        auto o1 = _mm512_load_ps(s_[t][1][i]);
        _mm512_storeu_ps(&C_[t][i][16], o1);
      }
    }
  }

  inline static void float_out_accum(void *C, size_t ldc, void *s, void *bias) {
    auto s_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(s);

    auto bias_ = *reinterpret_cast<float(*)[2][16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);

    auto C_ = *reinterpret_cast<float(*)[row_tile][16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_ps(s_[t][0][i]);
        auto i1 = _mm512_load_ps(s_[t][1][i]);

        auto a0 = _mm512_load_ps(&C_[t][i][0]);
        auto a1 = _mm512_load_ps(&C_[t][i][16]);

        _mm512_storeu_ps(&C_[t][i][0], i0 + a0 + b0);
        _mm512_storeu_ps(&C_[t][i][16], i1 + a1 + b1);
      }
    }
  }

  inline static void bf16_out_no_bias(void *C, size_t ldc, void *s) {
    auto s_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(s);

    auto C_ = *reinterpret_cast<__bfloat16(*)[row_tile][16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_ps(s_[t][0][i]);
        auto i1 = _mm512_load_ps(s_[t][1][i]);
        io_policy::_mm512_coalescing_packs_ps_store_pbh(C_[t][i], i0, i1);
      }
    }
  }

  inline static void bf16_out_accum_relu(void *C, size_t ldc, void *s, void *bias) {
    auto s_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(s);

    auto bias_ = *reinterpret_cast<float(*)[2][16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto zeros = _mm512_set1_ps(0.0);

    auto C_ = *reinterpret_cast<__bfloat16(*)[row_tile][16][ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_ps(s_[t][0][i]);
        auto i1 = _mm512_load_ps(s_[t][1][i]);
        auto a_bf = _mm512_loadu_epi16(C_[t][i]);
        auto a0 = _mm512_cvtpbh_ps(_mm512_extractf32x8_ps(a_bf, 0));
        auto a1 = _mm512_cvtpbh_ps(_mm512_extractf32x8_ps(a_bf, 1));
        auto o0 = _mm512_max_ps(i0 + a0 + b0, zeros);
        auto o1 = _mm512_max_ps(i1 + a1 + b1, zeros);
        io_policy::_mm512_coalescing_packs_ps_store_pbh(C_[t][i], o0, o1);
      }
    }
  }

  inline static void compute_i8o8b32(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[2][row_tile][2][16][16];
    _compute_impl(scratch, A, B);
    quant_out(C, ldc, scratch[0], scratch[1], bias, scale, post_op, o_scale);
  }

  constexpr static auto compute = compute_i8o8b32;

  inline static void compute_i8o8b16(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[2][row_tile][2][16][16];
    _compute_impl(scratch, A, B);
    quant_out_fp16(C, ldc, scratch[0], scratch[1], bias, scale, post_op, o_scale);
  }

  constexpr static auto compute_fp16 = compute_i8o8b16;

  inline static void compute_i8o32b32(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[2][row_tile][2][16][16];
    _compute_impl(scratch, A, B);
    dequant_float_out(C, ldc, scratch[0], scratch[1], bias, scale);
  }

  inline static void compute_i8o32b0(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[2][row_tile][2][16][16];
    _compute_impl(scratch, A, B);
    int32_out_no_bias(C, ldc, scratch[0], scratch[1]);
  }

  inline static void compute_i8o32b32_accum(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[2][row_tile][2][16][16];
    _compute_impl(scratch, A, B);
    dequant_float_out_accum(C, ldc, scratch[0], scratch[1], bias, scale);
  }

  inline static void compute_i16o32b32(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[row_tile][2][16][16];
    _compute_impl_bf16(scratch, A, B);
    float_out(C, ldc, scratch, bias);
  }

  inline static void compute_i16o32b0(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) float scratch[row_tile][2][16][16];
    _compute_impl_bf16(scratch, A, B);
    float_out_no_bias(C, ldc, scratch);
  }

  inline static void compute_i16o32b32_accum(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) float scratch[row_tile][2][16][16];
    _compute_impl_bf16(scratch, A, B);
    float_out_accum(C, ldc, scratch, bias);
  }

  inline static void compute_i16o16b0(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[row_tile][2][16][16];
    _compute_impl_bf16(scratch, A, B);
    bf16_out_no_bias(C, ldc, scratch);
  }

  inline static void compute_i16o16b32_accum_relu(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale,
      bool post_op = false, float o_scale = 1.0) {
    alignas(64) int scratch[row_tile][2][16][16];
    _compute_impl_bf16(scratch, A, B);
    bf16_out_accum_relu(C, ldc, scratch, bias);
  }

  inline static void _compute_impl(void *scratch, void *A, void *B) {
    // compute two scratch once to get 64=2*16*register_column(2)*1byte(int8) results
    // per row_tile(16 rows per row_tile), and to match cache line size 64.
    auto scratch_ = *reinterpret_cast<int(*)[2][row_tile][2][16][16]>(scratch);

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = *reinterpret_cast<int8_t(*)[4][col_tile][16][64]>(B);

    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::zero_accum();

#pragma unroll(col_tile)
    for (int i = 0; i < col_tile; ++i) {
      _tile_dot_product_16x256<row_tile, col_tile, io_policy>::dot_prod(
          A_[i], B_[0][i]);
    }

    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::store(scratch_[0]);
    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::zero_accum();

#pragma unroll(col_tile)
    for (int i = 0; i < col_tile; ++i) {
      _tile_dot_product_16x256<row_tile, col_tile, io_policy>::dot_prod(
          A_[i], B_[2][i]);
    }

    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::store(scratch_[1]);
  }

  inline static void _compute_impl_bf16(void *scratch, void *A, void *B) {
    // only need to compute one scratch once to get
    // 64=1*16*register_column(2)*2bytes(bf16) results per row_tile(16 rows
    // per row_tile), and to match cache line size 64.
    auto scratch_ = *reinterpret_cast<float(*)[row_tile][2][16][16]>(scratch);

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = *reinterpret_cast<__bfloat16(*)[2][col_tile][16][32]>(B);

    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::zero_accum();

#pragma unroll(col_tile)
    for (int i = 0; i < col_tile; ++i) {
      _tile_dot_product_16x256<row_tile, col_tile, io_policy>::dot_prod_bf16(
          A_[i], B_[0][i]);
    }

    _tile_dot_product_16x256<row_tile, col_tile, io_policy>::store(scratch_);
  }
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<1, col_tile, io_policy>
    : public _tile_dot_product_16x256_base<1, col_tile, io_policy> {
public:
  constexpr static int row_tile = 1;
  inline static void dot_prod(void *A, void *B) {
    auto B_ = *reinterpret_cast<int8_t(*)[row_tile * 2][col_tile][16][64]>(B);

    _tile_loadd(TMM6, B_[0], 64);

    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbssd<TMM0, TMM4, TMM6>();

    _tile_loadd(TMM7, B_[1], 64);

    __tile_dpbssd<TMM1, TMM4, TMM7>();
  }

  inline static void dot_prod_bf16(void *A, void *B) {
    auto B_ = *reinterpret_cast<__bfloat16(*)[row_tile * 2][col_tile][16][32]>(B);

    _tile_loadd(TMM6, B_[0], 64);

    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbf16ps<TMM0, TMM4, TMM6>();

    _tile_loadd(TMM7, B_[1], 64);

    __tile_dpbf16ps<TMM1, TMM4, TMM7>();
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<char32_t(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
  }

  using _tile_dot_product_16x256_base<1, col_tile, io_policy>::compute;
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<2, col_tile, io_policy>
    : public _tile_dot_product_16x256_base<2, col_tile, io_policy> {
public:
  constexpr static int row_tile = 2;
  inline static void dot_prod(void *A, void *B) {
    auto B_ = *reinterpret_cast<int8_t(*)[row_tile * 2][col_tile][16][64]>(B);

    _tile_loadd(TMM6, B_[0], 64);

    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbssd<TMM0, TMM4, TMM6>();

    io_policy::template tile_load<TMM5>(A, 1);
    __tile_dpbssd<TMM2, TMM5, TMM6>();

    _tile_loadd(TMM7, B_[1], 64);
    __tile_dpbssd<TMM3, TMM5, TMM7>();
    __tile_dpbssd<TMM1, TMM4, TMM7>();
  }

  inline static void dot_prod_bf16(void *A, void *B) {
    auto B_ = *reinterpret_cast<__bfloat16(*)[row_tile * 2][col_tile][16][32]>(B);

    _tile_loadd(TMM6, B_[0], 64);

    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbf16ps<TMM0, TMM4, TMM6>();

    io_policy::template tile_load<TMM5>(A, 1);
    __tile_dpbf16ps<TMM2, TMM5, TMM6>();

    _tile_loadd(TMM7, B_[1], 64);
    __tile_dpbf16ps<TMM3, TMM5, TMM7>();
    __tile_dpbf16ps<TMM1, TMM4, TMM7>();
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<char32_t(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
    _tile_stored(TMM3, S_[3], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<3, col_tile, io_policy> {
public:
  static constexpr size_t row_tile = 3;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  template <int tile_reg>
  inline static void dot_prod(void *A, void *B) {
    __tile_loadd<tile_reg>(B, 64);
    auto B_ = reinterpret_cast<int8_t(*)[16][64]>(B);

    io_policy::template tile_load<TMM3>(A, 0);
    __tile_dpbssd<TMM0, TMM3, tile_reg>();
    _mm_prefetch(B_[1][0], _MM_HINT_T0);
    _mm_prefetch(B_[1][1], _MM_HINT_T0);
    _mm_prefetch(B_[1][2], _MM_HINT_T0);
    _mm_prefetch(B_[1][3], _MM_HINT_T0);
    _mm_prefetch(B_[1][4], _MM_HINT_T0);
    io_policy::template tile_load<TMM4>(A, 1);
    __tile_dpbssd<TMM1, TMM4, tile_reg>();
    _mm_prefetch(B_[1][5], _MM_HINT_T0);
    _mm_prefetch(B_[1][6], _MM_HINT_T0);
    _mm_prefetch(B_[1][7], _MM_HINT_T0);
    _mm_prefetch(B_[1][8], _MM_HINT_T0);
    _mm_prefetch(B_[1][9], _MM_HINT_T0);
    io_policy::template tile_load<TMM5>(A, 2);
    __tile_dpbssd<TMM2, TMM5, tile_reg>();
    _mm_prefetch(B_[1][10], _MM_HINT_T0);
    _mm_prefetch(B_[1][11], _MM_HINT_T0);
    _mm_prefetch(B_[1][12], _MM_HINT_T0);
    _mm_prefetch(B_[1][13], _MM_HINT_T0);
    _mm_prefetch(B_[1][14], _MM_HINT_T0);
    _mm_prefetch(B_[1][15], _MM_HINT_T0);
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
  }

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[0][t][i]);
        auto i1 = _mm512_load_epi32(s_0_[1][t][i]);
        auto i2 = _mm512_load_epi32(s_1_[0][t][i]);
        auto i3 = _mm512_load_epi32(s_1_[1][t][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _mm512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _mm512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _mm512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _mm512_scale_minmax_i8_ps(scale_, f3);

        io_policy::_mm512_coalescing_packs_epi32_store_epi8(
            C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  inline static void compute(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
    alignas(64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t(*)[col_tile][16][64]>(B);

#pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#pragma unroll(col_tile / 2)
      for (int i = 0; i < col_tile / 2; i++) {
        dot_prod<TMM7>(A_[2 * i], B_[j][2 * i]);
        dot_prod<TMM6>(A_[2 * i + 1], B_[j][2 * i + 1]);
      }

      if (col_tile % 2 == 1) {
        dot_prod<TMM7>(A_[col_tile - 1], B_[j][col_tile - 1]);
      }

      store(scratch[j]);
    }

    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<4, col_tile, io_policy> {
public:
  static constexpr size_t row_tile = 4;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  template <int tile_reg>
  inline static void dot_prod(void *A, void *B) {
    __tile_loadd<tile_reg>(B, 64);
    auto B_ = reinterpret_cast<int8_t(*)[16][64]>(B);

    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbssd<TMM0, TMM4, tile_reg>();
    _mm_prefetch(B_[1][0], _MM_HINT_T0);
    _mm_prefetch(B_[1][1], _MM_HINT_T0);
    _mm_prefetch(B_[1][2], _MM_HINT_T0);
    _mm_prefetch(B_[1][3], _MM_HINT_T0);
    io_policy::template tile_load<TMM5>(A, 1);
    __tile_dpbssd<TMM1, TMM5, tile_reg>();
    _mm_prefetch(B_[1][4], _MM_HINT_T0);
    _mm_prefetch(B_[1][5], _MM_HINT_T0);
    _mm_prefetch(B_[1][6], _MM_HINT_T0);
    _mm_prefetch(B_[1][7], _MM_HINT_T0);
    io_policy::template tile_load<TMM4>(A, 2);
    __tile_dpbssd<TMM2, TMM4, tile_reg>();
    _mm_prefetch(B_[1][8], _MM_HINT_T0);
    _mm_prefetch(B_[1][9], _MM_HINT_T0);
    _mm_prefetch(B_[1][10], _MM_HINT_T0);
    _mm_prefetch(B_[1][11], _MM_HINT_T0);
    io_policy::template tile_load<TMM5>(A, 3);
    __tile_dpbssd<TMM3, TMM5, tile_reg>();
    _mm_prefetch(B_[1][12], _MM_HINT_T0);
    _mm_prefetch(B_[1][13], _MM_HINT_T0);
    _mm_prefetch(B_[1][14], _MM_HINT_T0);
    _mm_prefetch(B_[1][15], _MM_HINT_T0);
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
    _tile_stored(TMM3, S_[3], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[0][t][i]);
        auto i1 = _mm512_load_epi32(s_0_[1][t][i]);
        auto i2 = _mm512_load_epi32(s_1_[0][t][i]);
        auto i3 = _mm512_load_epi32(s_1_[1][t][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _mm512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _mm512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _mm512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _mm512_scale_minmax_i8_ps(scale_, f3);

        io_policy::_mm512_coalescing_packs_epi32_store_epi8(
            C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  inline static void compute(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
    alignas(64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t(*)[col_tile][16][64]>(B);

#pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#pragma unroll(col_tile / 2)
      for (int i = 0; i < col_tile / 2; i++) {
        dot_prod<TMM7>(A_[2 * i], B_[j][2 * i]);
        dot_prod<TMM6>(A_[2 * i + 1], B_[j][2 * i + 1]);
      }

      if (col_tile % 2 == 1) {
        dot_prod<TMM7>(A_[col_tile - 1], B_[j][col_tile - 1]);
      }

      store(scratch[j]);
    }

    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<5, col_tile, io_policy> {
public:
  static constexpr size_t row_tile = 5;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B) {
    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t(*)[16][64]>(B);

    io_policy::template tile_load<TMM5>(A, 0);
    __tile_dpbssd<TMM0, TMM5, TMM7>();
    _mm_prefetch(B_[1][0], _MM_HINT_T0);
    _mm_prefetch(B_[1][1], _MM_HINT_T0);
    _mm_prefetch(B_[1][2], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 1);
    __tile_dpbssd<TMM1, TMM6, TMM7>();
    _mm_prefetch(B_[1][3], _MM_HINT_T0);
    _mm_prefetch(B_[1][4], _MM_HINT_T0);
    _mm_prefetch(B_[1][5], _MM_HINT_T0);
    io_policy::template tile_load<TMM5>(A, 2);
    __tile_dpbssd<TMM2, TMM5, TMM7>();
    _mm_prefetch(B_[1][6], _MM_HINT_T0);
    _mm_prefetch(B_[1][7], _MM_HINT_T0);
    _mm_prefetch(B_[1][8], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 3);
    __tile_dpbssd<TMM3, TMM6, TMM7>();
    _mm_prefetch(B_[1][9], _MM_HINT_T0);
    _mm_prefetch(B_[1][10], _MM_HINT_T0);
    _mm_prefetch(B_[1][11], _MM_HINT_T0);
    io_policy::template tile_load<TMM5>(A, 4);
    __tile_dpbssd<TMM4, TMM5, TMM7>();
    _mm_prefetch(B_[1][12], _MM_HINT_T0);
    _mm_prefetch(B_[1][13], _MM_HINT_T0);
    _mm_prefetch(B_[1][14], _MM_HINT_T0);
    _mm_prefetch(B_[1][15], _MM_HINT_T0);
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
    _tile_stored(TMM3, S_[3], 64);
    _tile_stored(TMM4, S_[4], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
    _tile_zero(TMM4);
  }

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[0][t][i]);
        auto i1 = _mm512_load_epi32(s_0_[1][t][i]);
        auto i2 = _mm512_load_epi32(s_1_[0][t][i]);
        auto i3 = _mm512_load_epi32(s_1_[1][t][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _mm512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _mm512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _mm512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _mm512_scale_minmax_i8_ps(scale_, f3);

        io_policy::_mm512_coalescing_packs_epi32_store_epi8(
            C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  // Pure tile format
  inline static void compute(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
    alignas(64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t(*)[col_tile][16][64]>(B);

#pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#pragma unroll(col_tile)
      for (int i = 0; i < col_tile; i++) {
        dot_prod(A_[i], B_[j][i]);
      }
      store(scratch[j]);
    }

    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

template <int col_tile, typename io_policy>
class _tile_dot_product_16x256<6, col_tile, io_policy> {
public:
  static constexpr size_t row_tile = 6;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B) {
    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t(*)[16][64]>(B);

    io_policy::template tile_load<TMM6>(A, 0);
    __tile_dpbssd<TMM0, TMM6, TMM7>();
    _mm_prefetch(B_[1][0], _MM_HINT_T0);
    _mm_prefetch(B_[1][1], _MM_HINT_T0);
    _mm_prefetch(B_[1][2], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 1);
    __tile_dpbssd<TMM1, TMM6, TMM7>();
    _mm_prefetch(B_[1][3], _MM_HINT_T0);
    _mm_prefetch(B_[1][4], _MM_HINT_T0);
    _mm_prefetch(B_[1][5], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 2);
    __tile_dpbssd<TMM2, TMM6, TMM7>();
    _mm_prefetch(B_[1][6], _MM_HINT_T0);
    _mm_prefetch(B_[1][7], _MM_HINT_T0);
    _mm_prefetch(B_[1][8], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 3);
    __tile_dpbssd<TMM3, TMM6, TMM7>();
    _mm_prefetch(B_[1][9], _MM_HINT_T0);
    _mm_prefetch(B_[1][10], _MM_HINT_T0);
    _mm_prefetch(B_[1][11], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 4);
    __tile_dpbssd<TMM4, TMM6, TMM7>();
    _mm_prefetch(B_[1][12], _MM_HINT_T0);
    _mm_prefetch(B_[1][13], _MM_HINT_T0);
    _mm_prefetch(B_[1][14], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, 5);
    __tile_dpbssd<TMM5, TMM6, TMM7>();
    _mm_prefetch(B_[1][15], _MM_HINT_T0);
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int(*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
    _tile_stored(TMM3, S_[3], 64);
    _tile_stored(TMM4, S_[4], 64);
    _tile_stored(TMM5, S_[5], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
    _tile_zero(TMM4);
    _tile_zero(TMM5);
  }

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[0][t][i]);
        auto i1 = _mm512_load_epi32(s_0_[1][t][i]);
        auto i2 = _mm512_load_epi32(s_1_[0][t][i]);
        auto i3 = _mm512_load_epi32(s_1_[1][t][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _mm512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _mm512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _mm512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _mm512_scale_minmax_i8_ps(scale_, f3);

        io_policy::_mm512_coalescing_packs_epi32_store_epi8(
            C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  // Pure tile format
  inline static void compute(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
    alignas(64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t(*)[col_tile][16][64]>(B);

#pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#pragma unroll(col_tile)
      for (int i = 0; i < col_tile; i++) {
        dot_prod(A_[i], B_[j][i]);
      }
      store(scratch[j]);
    }
    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

// for row_tile >= 7
template <int row_tile, int col_tile, typename io_policy>
class _tile_dot_product_16x256 {
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B, void *scrach_) {
    auto scratch_pad = reinterpret_cast<int(*)[16][16]>(scrach_);

    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t(*)[16][64]>(B);

    // TMM0 is the tile used twice
    for (int i = 0; i < (row_tile - 5); i++) {
      io_policy::template tile_load<TMM6>(A, i);
      _tile_loadd(TMM0, scratch_pad[i], 64);
      __tile_dpbssd<TMM0, TMM6, TMM7>();
      _tile_stored(TMM0, scratch_pad[i], 64);
    }
    _mm_prefetch(B_[1][1], _MM_HINT_T0);
    _mm_prefetch(B_[1][2], _MM_HINT_T0);
    _mm_prefetch(B_[1][3], _MM_HINT_T0);

    io_policy::template tile_load<TMM6>(A, row_tile - 5);
    __tile_dpbssd<TMM1, TMM6, TMM7>();
    _mm_prefetch(B_[1][4], _MM_HINT_T0);
    _mm_prefetch(B_[1][5], _MM_HINT_T0);
    _mm_prefetch(B_[1][6], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, row_tile - 4);
    __tile_dpbssd<TMM2, TMM6, TMM7>();
    _mm_prefetch(B_[1][7], _MM_HINT_T0);
    _mm_prefetch(B_[1][8], _MM_HINT_T0);
    _mm_prefetch(B_[1][9], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, row_tile - 3);
    __tile_dpbssd<TMM3, TMM6, TMM7>();
    _mm_prefetch(B_[1][10], _MM_HINT_T0);
    _mm_prefetch(B_[1][11], _MM_HINT_T0);
    _mm_prefetch(B_[1][12], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, row_tile - 2);
    __tile_dpbssd<TMM4, TMM6, TMM7>();
    _mm_prefetch(B_[1][13], _MM_HINT_T0);
    _mm_prefetch(B_[1][14], _MM_HINT_T0);
    _mm_prefetch(B_[1][15], _MM_HINT_T0);
    io_policy::template tile_load<TMM6>(A, row_tile - 1);
    __tile_dpbssd<TMM5, TMM6, TMM7>();
    _mm_prefetch(B_[1][16], _MM_HINT_T0);
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int(*)[16][16]>(S);
    _tile_stored(TMM1, S_[row_tile - 5], 64);
    _tile_stored(TMM2, S_[row_tile - 4], 64);
    _tile_stored(TMM3, S_[row_tile - 3], 64);
    _tile_stored(TMM4, S_[row_tile - 2], 64);
    _tile_stored(TMM5, S_[row_tile - 1], 64);
  }

  inline static void zero_accum(void *scrach_) {
    // _tile_zero(TMM0);
    auto scratch_pad = reinterpret_cast<int(*)[16][16]>(scrach_);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
    _tile_zero(TMM4);
    _tile_zero(TMM5);

    for (int i = 0; i < row_tile - 5; i++) {
      _tile_stored(TMM1, scratch_pad[i], 64);
    }
  }

  inline static void quant_out(
      void *C, size_t ldc, void *s_0, void *s_1, void *bias, float scale) {
    auto s_0_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int(*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float(*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t(*)[16 * ldc]>(C);
#pragma unroll(row_tile)
    for (int t = 0; t < row_tile; ++t) {
#pragma unroll(16)
      for (int i = 0; i < 16; ++i) {
        auto i0 = _mm512_load_epi32(s_0_[0][t][i]);
        auto i1 = _mm512_load_epi32(s_0_[1][t][i]);
        auto i2 = _mm512_load_epi32(s_1_[0][t][i]);
        auto i3 = _mm512_load_epi32(s_1_[1][t][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _mm512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _mm512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _mm512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _mm512_scale_minmax_i8_ps(scale_, f3);

        io_policy::_mm512_coalescing_packs_store(C_[t], ldc, i, o0, o1, o2, o3);
      }
    }
  }

  // Pure tile format
  inline static void compute(
      void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
    alignas(64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t(*)[col_tile][16][64]>(B);

#pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
      zero_accum(scratch[j]);
#pragma unroll(col_tile)
      for (int i = 0; i < col_tile; i++) {
        dot_prod(A_[i], B_[j][i], scratch[j]);
      }
      store(scratch[j]);
    }
    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

}  // namespace intel_mlperf
