#pragma once

#include <cstdlib>
#include <type_traits>

#include "tile_product_tpp.hpp"

namespace intel_mlperf {

// Use linear interface instead of iGEMM
// A compute block just compute slx256, delete total work
class i_linear {
public:
  i_linear(
      size_t sequence_length, size_t input_feature, size_t output_feature, bool bias,
      bool post_op)
      : sl_(sequence_length),
        ic_(input_feature),
        oc_(output_feature),
        has_bias_(bias),
        post_op_(post_op) {}

  constexpr inline void set_compute_blk_cfg(int el_per_tile_row) {
    cols_in_tile_ = ic_ / el_per_tile_row;
    cols_step_ = oc_ / el_per_tile_row;
    if (cols_in_tile_ == 16) {
      compute_blk_idx_ = 0;
    } else if (cols_in_tile_ == 32) {
      compute_blk_idx_ = 1;
    } else if (cols_in_tile_ == 64) {
      compute_blk_idx_ = 2;
    } else if (cols_in_tile_ == 10) {
      compute_blk_idx_ = 3;
    } else if (cols_in_tile_ == 42) {
      compute_blk_idx_ = 4;
    } else {
      compute_blk_idx_ = compute_blk_tbl_col - 1;
    }
  }

  enum Version {
    i8o8b32,
    i8o8b16,
    i8o32b32,
    i8o32b0,
    i8o32b32_accum,
    i16o32b32,
    i16o32b0,
    i16o32b32_accum,
    i16o16b0,
    i16o16b32_accum_relu,
  };

  template <typename input_type, typename output_type, typename bias_type, Version ver>
  void tile_dot_product_16x256(
      void* C, void* A, void* B, void* bias, float scale, float o_scale,
      const size_t chunk_sl, size_t cur_id = 0, size_t total_chunks = 1);

  template <typename input_type, typename output_type, typename bias_type, Version ver>
  void tile_dot_product_16x256_shortage(
      void* C, void* A, void* B, void* bias, float scale, float o_scale,
      const size_t chunk_sl, size_t cur_id = 0, size_t total_chunks = 1);

protected:
  constexpr static int compute_blk_tbl_row = 3;
  constexpr static int compute_blk_tbl_col = 6;

  typedef void (i_linear::*compute_block_t)(
      void*, size_t, void*, void*, void*, float, bool, float);

  template <Version ver>
  static const compute_block_t compute_block_tbl_[compute_blk_tbl_row]
                                                 [compute_blk_tbl_col];

#define FOREACH_COMPUTE_IMPL(cb) \
  cb(i8o8b16);                   \
  cb(i8o32b32);                  \
  cb(i8o32b0);                   \
  cb(i8o32b32_accum);            \
  cb(i16o32b32);                 \
  cb(i16o32b0);                  \
  cb(i16o32b32_accum);           \
  cb(i16o16b0);                  \
  cb(i16o16b32_accum_relu);      \
  cb(i8o8b32);

#define DEF_COMPUTE_BLK(ver)                                                        \
  template <int row_tile, int col_tile, Version version>                            \
  inline typename std::enable_if<version == ver, void>::type compute_block(         \
      void* C, size_t ldc, void* A, void* B, void* bias, float scale, bool post_op, \
      float o_scale) {                                                              \
    _tile_dot_product_16x256<                                                       \
        row_tile, col_tile, io_policy<col_tile, i_format::plain>>::                 \
        compute_##ver(C, ldc, A, B, bias, scale, post_op, o_scale);                 \
  }

  FOREACH_COMPUTE_IMPL(DEF_COMPUTE_BLK);

  size_t sl_;
  size_t ic_;
  size_t oc_;
  bool has_bias_;
  bool post_op_;

  // weight division
  size_t cols_in_tile_;
  size_t cols_step_;
  size_t compute_blk_idx_;
};

}  // namespace intel_mlperf
