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

enum struct i_format {plain, tile};

template <int col_tile, i_format i_fmt> struct io_policy;

template <int col_tile>
struct io_policy<col_tile, i_format::tile> {
  typedef int8_t (* tile_array)[16 * 64];
  typedef int8_t (* block_pointer)[col_tile][16][64];

  template <int tile_num> 
  inline static void tile_load(void* A, int idx) {
    auto A_ = reinterpret_cast<block_pointer>(A);
    __tile_loadd<tile_num>(A_[idx], 64);
  }

  inline static void _mm512_coalescing_store(void* C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t (*)[4][16]>(C);

    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][0], 0xffff, o0);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][1], 0xffff, o1);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][2], 0xffff, o2);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx][3], 0xffff, o3);

  }
};

template <int col_tile>
struct io_policy<col_tile, i_format::plain> {
  typedef int8_t (* tile_array)[64];
  typedef int8_t (* block_pointer)[16][col_tile][64];

  template <int tile_num> 
  inline static void tile_load(void* A, int idx) {
    auto A_ = reinterpret_cast<block_pointer>(A);
    __tile_loadd<tile_num>(A_[idx], col_tile * 64);
  }

  inline static void _mm512_coalescing_store(void* C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 0 , 0xffff, o0);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 16, 0xffff, o1);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 32, 0xffff, o2);
    _mm512_mask_cvtepi32_storeu_epi8(C_[idx] + 48, 0xffff, o3);
  }

  inline static void _mm512_coalescing_packs_store(void* C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
    
    auto _m512_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    auto t0 = _mm512_packs_epi32(o0, o1);
    auto t1 = _mm512_packs_epi32(o2, o3);
    auto p0 = _mm512_permutexvar_epi64(_m512_idx, t0);
    auto p1 = _mm512_permutexvar_epi64(_m512_idx, t1);
    auto t2 = _mm512_packs_epi16(p0, p1);
    auto p2 = _mm512_permutexvar_epi64(_m512_idx, t2);

    _mm512_storeu_epi8(C_[idx], p2);
  }

  inline static void _mm512_coalescing_packs_store_epi16(void* C, size_t ldc, int idx, __m512i o0, __m512i o1, __m512i o2, __m512i o3) {
    auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
    
    auto _m512_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    auto t0 = _mm512_packs_epi16(o0, o1);
    auto t1 = _mm512_packs_epi16(o2, o3);

    auto p0 = _mm512_permutexvar_epi64(_m512_idx, t0);
    auto p1 = _mm512_permutexvar_epi64(_m512_idx, t1);

    _mm512_storeu_epi8(C_[idx], p0);
    _mm512_storeu_epi8(C_[idx + 1], p1);
  }
};

template <int row_tile, int col_tile, typename io_policy>
struct _tile_dot_product_16x256;

template <int col_tile, typename io_policy>
struct _tile_dot_product_16x256<1, col_tile, io_policy> {
  static constexpr size_t row_tile = 1;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  inline static void dot_prod(void *A, void *B) {
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);
    
    _tile_loadd(TMM6, B_[0], 64);
    
    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbssd<TMM0, TMM4, TMM6>();
    
    _tile_loadd(TMM7, B_[1], 64);

    __tile_dpbssd<TMM1, TMM4, TMM7>();
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
  }

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale,
                               bool post_op, float o_scale) {
    auto s_0_ = reinterpret_cast<int (*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[2][16][16]>(s_1);

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

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
    #pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {

      #pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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
          io_policy::_mm512_coalescing_packs_store(C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ps(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ps(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ps(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ps(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_store(C_[t], ldc, i, o0, o1, o2, o3);
        }
        
      }
    }
  }

  inline static void quant_out_fp16(void *C, size_t ldc, void *s_0, void *s_1, _Float16 *bias, float scale, 
                                    bool post_op, float o_scale) {
    auto s_0_ = reinterpret_cast<int (*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ph(scale);
    __m512h o_scale_;
    if (post_op) {
      o_scale_ = _mm512_set1_ph(o_scale);
    }
    // TODO: wait for model bias to fp16
    auto bias_ = reinterpret_cast<_Float16 (*)[32]>(bias);

    // TODO: when model bias is fp16, this step could be optimized
    auto bias32_0 = _mm512_loadu_ph(bias_[0]);
    auto bias32_1 = _mm512_loadu_ph(bias_[1]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
    #pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {

      #pragma unroll (8)
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
          io_policy::_mm512_coalescing_packs_store_epi16(C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ph(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ph(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ph(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ph(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_store_epi16(C_[t], ldc, i, o0, o1, o2, o3);
        }
        
      }
    }
  }

  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale, 
                             bool post_op = false, float o_scale = 1.0) {
    alignas (64) int scratch_0[row_tile][2][16][16];
    alignas (64) int scratch_1[row_tile][2][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[0][i]);
    }

    store(scratch_0);
    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[2][i]);
    }

    store(scratch_1);
    quant_out(C, ldc, scratch_0, scratch_1, bias, scale, post_op, o_scale);
  }

  inline static void compute_fp16(void *C, size_t ldc, void *A, void *B, _Float16 *bias, 
                                  float scale, bool post_op = false, float o_scale = 1.0) {
    alignas (64) int scratch_0[row_tile][2][16][16];
    alignas (64) int scratch_1[row_tile][2][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[0][i]);
    }

    store(scratch_0);
    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[2][i]);
    }

    store(scratch_1);
    quant_out_fp16(C, ldc, scratch_0, scratch_1, bias, scale, post_op, o_scale);
  }
};

template <int col_tile, typename io_policy>
struct _tile_dot_product_16x256<2, col_tile, io_policy> {
  static constexpr size_t row_tile = 2;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  inline static void dot_prod(void *A, void *B) {
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);
    auto A_ = reinterpret_cast<int8_t (*)[16][lda]>(A);

    _tile_loadd(TMM6, B_[0], 64);
    
    io_policy::template tile_load<TMM4>(A, 0);
    __tile_dpbssd<TMM0, TMM4, TMM6>();
    
    io_policy::template tile_load<TMM5>(A, 1);
    __tile_dpbssd<TMM2, TMM5, TMM6>();

    _tile_loadd(TMM7, B_[1], 64);
    __tile_dpbssd<TMM1, TMM4, TMM7>();
    __tile_dpbssd<TMM3, TMM5, TMM7>();
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
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

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale, 
                               bool post_op, float o_scale) {
    auto s_0_ = reinterpret_cast<int (*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[2][16][16]>(s_1);

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

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
    #pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {

      #pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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
          io_policy::_mm512_coalescing_packs_store(C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ps(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ps(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ps(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ps(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_store(C_[t], ldc, i, o0, o1, o2, o3);
        }
        
      }
    }
  }

  inline static void quant_out_fp16(void *C, size_t ldc, void *s_0, void *s_1, _Float16 *bias, float scale, 
                                    bool post_op, float o_scale) {
    auto s_0_ = reinterpret_cast<int (*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ph(scale);
    __m512h o_scale_;
    if (post_op) {
      o_scale_ = _mm512_set1_ph(o_scale);
    }
    // TODO: wait for model bias to fp16
    auto bias_ = reinterpret_cast<_Float16 (*)[32]>(bias);

    // TODO: when model bias is fp16, this step could be optimized
    auto bias32_0 = _mm512_loadu_ph(bias_[0]);
    auto bias32_1 = _mm512_loadu_ph(bias_[1]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
    #pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {

      #pragma unroll (8)
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
          io_policy::_mm512_coalescing_packs_store_epi16(C_[t], ldc, i, o0, o1, o2, o3);
        } else {
          auto o0 = _mm512_scale_minmax_i8_ph(f0, scale_);
          auto o1 = _mm512_scale_minmax_i8_ph(f1, scale_);
          auto o2 = _mm512_scale_minmax_i8_ph(f2, scale_);
          auto o3 = _mm512_scale_minmax_i8_ph(f3, scale_);

          // every 16 got output
          io_policy::_mm512_coalescing_packs_store_epi16(C_[t], ldc, i, o0, o1, o2, o3);
        }
        
      }
    }
  }

  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, 
                             float scale, bool post_op = false, float o_scale = 1.0) {
    alignas (64) int scratch_0[row_tile][2][16][16];
    alignas (64) int scratch_1[row_tile][2][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[0][i]);
    }

    store(scratch_0);
    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[2][i]);
    }

    store(scratch_1);
    quant_out(C, ldc, scratch_0, scratch_1, bias, scale, post_op, o_scale);
  }

  inline static void compute_fp16(void *C, size_t ldc, void *A, void *B, _Float16 *bias, 
                                  float scale, bool post_op = false, float o_scale = 1.0) {
    alignas (64) int scratch_0[row_tile][2][16][16];
    alignas (64) int scratch_1[row_tile][2][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[0][i]);
    }

    store(scratch_0);
    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      dot_prod(A_[i], B_[2][i]);
    }

    store(scratch_1);
    quant_out_fp16(C, ldc, scratch_0, scratch_1, bias, scale, post_op, o_scale);
  }
};

template <int col_tile, typename io_policy>
struct _tile_dot_product_16x256 <3, col_tile, io_policy> {
  static constexpr size_t row_tile = 3;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  template <int tile_reg>
  inline static void dot_prod(void *A, void *B) {
    __tile_loadd<tile_reg>(B, 64);
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);

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
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    _tile_stored(TMM2, S_[2], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
  }

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale) {
    auto s_0_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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

  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale) {
    alignas (64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

#   pragma unroll (4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#     pragma unroll (col_tile / 2)
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
struct _tile_dot_product_16x256 <4, col_tile, io_policy> {
  static constexpr size_t row_tile = 4;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  template <int tile_reg>
  inline static void dot_prod(void *A, void *B) {
    __tile_loadd<tile_reg>(B, 64);
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);

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
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
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

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale) {
    auto s_0_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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

  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale) {
    alignas (64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

#   pragma unroll (4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#     pragma unroll (col_tile/2)
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
struct _tile_dot_product_16x256 <5, col_tile, io_policy> {
  static constexpr size_t row_tile = 5;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B) {
    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);

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
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
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

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale) {
    auto s_0_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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
  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale) {
    alignas (64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

#   pragma unroll (4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#     pragma unroll (col_tile)
      for (int i = 0; i < col_tile; i++) {
        dot_prod(A_[i], B_[j][i]);
      }
      store(scratch[j]);
    }

    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

template <int col_tile, typename io_policy>
struct _tile_dot_product_16x256 <6, col_tile, io_policy> {
  static constexpr size_t row_tile = 6;
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B) {
    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);

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
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
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

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale) {
    auto s_0_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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
  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale) {
    alignas (64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

#   pragma unroll (4)
    for (int j = 0; j < 4; ++j) {
      zero_accum();
#     pragma unroll (col_tile)
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
struct _tile_dot_product_16x256 {
  static constexpr size_t A_footprint = row_tile * col_tile * 16 * 64;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;
  static constexpr size_t lda = col_tile * 64;

  // This version is with A/B tile interleaving
  inline static void dot_prod(void *A, void *B, void* scrach_) {
    auto scratch_pad = reinterpret_cast<int (*)[16][16]>(scrach_);

    _tile_loadd(TMM7, B, 64);
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);

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
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
    _tile_stored(TMM1, S_[row_tile - 5], 64);
    _tile_stored(TMM2, S_[row_tile - 4], 64);
    _tile_stored(TMM3, S_[row_tile - 3], 64);
    _tile_stored(TMM4, S_[row_tile - 2], 64);
    _tile_stored(TMM5, S_[row_tile - 1], 64);
  }

  inline static void zero_accum(void* scrach_) {
    // _tile_zero(TMM0);
    auto scratch_pad = reinterpret_cast<int (*)[16][16]>(scrach_);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
    _tile_zero(TMM4);
    _tile_zero(TMM5);

    for (int i = 0; i < row_tile - 5; i++) {
      _tile_stored(TMM1, scratch_pad[i], 64);
    }
  }

  inline static void quant_out(void *C, size_t ldc, void *s_0, void *s_1, float *bias, float scale) {
    auto s_0_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[row_tile][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

    auto C_ = reinterpret_cast<int8_t (*)[16 * ldc]>(C);
#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
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
  inline static void compute(void *C, size_t ldc, void *A, void *B, float *bias, float scale) {
    alignas (64) int scratch[4][row_tile][16][16];

    auto A_ = reinterpret_cast<typename io_policy::tile_array>(A);
    auto B_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(B);

#   pragma unroll (4)
    for (int j = 0; j < 4; ++j) {
      zero_accum(scratch[j]);
#     pragma unroll (col_tile)
      for (int i = 0; i < col_tile; i++) {
        dot_prod(A_[i], B_[j][i], scratch[j]);
      }
      store(scratch[j]);
    }
    quant_out(C, ldc, scratch[0], scratch[2], bias, scale);
  }
};

// Use linear interface instead of iGEMM
// A compute block just compute slx256, delete total work
class i_linear {
public:
  i_linear(size_t sequence_length, size_t input_feature, size_t output_feature, bool bias, bool post_op)
      : sl_(sequence_length), ic_(input_feature), oc_(output_feature), has_bias_(bias), post_op_(post_op) {
        cols_in_tile_ = input_feature / 64;
  }

  template <int row_tile, int col_tile>
  void compute_block(void* C, size_t ldc, void* A, void* B, float* bias, float scale, bool post_op = false, float o_scale = 1.0);
  void tile_dot_product_16x256(void *C, void *A, void *B, float *bias, float scale, float o_scale, 
                               const size_t sl, const size_t col_step, size_t cur_id=0, size_t total_chunks=1);

  void tile_dot_product_16x256_shortage(void *C, void *A, void *B, float *bias, float scale, float o_scale, 
                                        const size_t sl, const size_t col_step, size_t cur_id=0, size_t total_chunks=1);
  
  void tile_linear(const int row_tile, size_t roll_back, const int col_tile, 
                   void *C, void *A, void *B, float *bias, float scale, float o_scale);

  typedef void (i_linear::* compute_block_t) (
      void*, size_t, void*, void*, float*, float, bool, float);
  static const compute_block_t compute_block_tbl_ [3][2];
protected:
  
  // using compute_block_t = void (*)(void*, size_t, void*, void*, float*, float);

  size_t sl_;
  size_t ic_;
  size_t oc_;
  bool has_bias_;
  bool post_op_;

  // output division
  size_t cols_in_tile_;
};

class i_linear_fp16 {
public:
  i_linear_fp16(size_t sequence_length, size_t input_feature, size_t output_feature, bool bias, bool post_op)
      : sl_(sequence_length), ic_(input_feature), oc_(output_feature), has_bias_(bias), post_op_(post_op) {
        cols_in_tile_ = input_feature / 64;
  }

  template <int row_tile, int col_tile>
  void compute_block(void* C, size_t ldc, void* A, void* B, _Float16* bias, float scale, bool post_op = false, float o_scale = 1.0);
  void tile_dot_product_16x256(void *C, void *A, void *B, _Float16 *bias, float scale, float o_scale, 
                               const size_t sl, const size_t col_step, size_t cur_id=0, size_t total_chunks=1);

  void tile_dot_product_16x256_shortage(void *C, void *A, void *B, _Float16 *bias, float scale, float o_scale, 
                                        const size_t sl, const size_t col_step, size_t cur_id=0, size_t total_chunks=1);
  
  typedef void (i_linear_fp16::* compute_block_t) (
      void*, size_t, void*, void*, _Float16*, float, bool, float);
  static const compute_block_t compute_block_tbl_ [3][2];
protected:
  
  // using compute_block_t = void (*)(void*, size_t, void*, void*, float*, float);

  size_t sl_;
  size_t ic_;
  size_t oc_;
  bool has_bias_;
  bool post_op_;

  // output division
  size_t cols_in_tile_;
};

}
