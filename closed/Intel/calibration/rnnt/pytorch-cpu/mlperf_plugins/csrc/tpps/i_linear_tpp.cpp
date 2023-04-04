#include "i_linear_tpp.hpp"

#include <chrono>
#include <iostream>

#include "amx_config.hpp"

using Time = std::chrono::high_resolution_clock;
using Ver = intel_mlperf::i_linear::Version;

namespace intel_mlperf {

template <typename input_type, typename output_type, typename bias_type, Ver ver>
void i_linear::tile_dot_product_16x256(
    void* C, void* A, void* B, void* bias, float scale, float o_scale,
    const size_t chunk_sl, size_t cur_id, size_t total_chunks) {
  constexpr int el_per_tile_row = std::is_same<input_type, int8_t>::value ? 64 : 32;
  // int8 need 4 tiles output to pack 64 results for cache line
  constexpr int tiles_per_compute_row = std::is_same<input_type, int8_t>::value ? 4 : 2;
  set_compute_blk_cfg(el_per_tile_row);

  if (16 > chunk_sl) {
    throw std::runtime_error(
        "per core chunk_sl should be more than 16 in input split i_linear.");
  }
  auto C_ = *reinterpret_cast<output_type(*)[chunk_sl][cols_step_][el_per_tile_row]>(C);
  auto A_ = *reinterpret_cast<input_type(*)[chunk_sl][ic_]>(A);
  auto B_ = *reinterpret_cast<input_type(*)[cols_step_][tiles_per_compute_row]
                                           [cols_in_tile_][16][el_per_tile_row]>(B);
  auto bias_ = *reinterpret_cast<bias_type(*)[cols_step_][el_per_tile_row]>(bias);

  compute_block_t computer_2 = compute_block_tbl_<ver>[2][compute_blk_idx_];
  compute_block_t computer_1 = compute_block_tbl_<ver>[1][compute_blk_idx_];

  size_t row_tile = (chunk_sl + 15) / 16;
  size_t roll_back = row_tile * 16 - chunk_sl;

  bool odd_tile = row_tile % 2;
  size_t row_step = row_tile / 2;

  size_t col_start = 0;
  // col sharding to avoid bank conflict
  if (cols_step_ >= total_chunks) {
    col_start = cols_step_ * cur_id / total_chunks;
  }
  if (odd_tile) {
    for (size_t k = 0; k < cols_step_; k++) {
      auto col_pos = col_start + k;
      col_pos = col_pos - (int)(col_pos >= cols_step_) * cols_step_;
      size_t row_pos = 0, j = 0;
      for (j = 0; j < row_step; j++) {
        row_pos = j * 32;
        (this->*computer_2)(
            C_[row_pos][col_pos], oc_, A_[row_pos], B_[col_pos], bias_[col_pos], scale,
            post_op_, o_scale);
      }
      row_pos = j * 32 - roll_back;
      (this->*computer_1)(
          C_[row_pos][col_pos], oc_, A_[row_pos], B_[col_pos], bias_[col_pos], scale,
          post_op_, o_scale);
    }
  } else {
    for (size_t k = 0; k < cols_step_; k++) {
      auto col_pos = col_start + k;
      col_pos = col_pos - (int)(col_pos >= cols_step_) * cols_step_;
      for (size_t j = 0; j < row_step; j++) {
        size_t rollback_ = (j == row_step - 1) * roll_back;
        auto row_pos = j * 32 - rollback_;
        (this->*computer_2)(
            C_[row_pos][col_pos], oc_, A_[row_pos], B_[col_pos], bias_[col_pos], scale,
            post_op_, o_scale);
      }
    }
  }
}

template <typename input_type, typename output_type, typename bias_type, Ver ver>
void i_linear::tile_dot_product_16x256_shortage(
    void* C, void* A, void* B, void* bias, float scale, float o_scale,
    const size_t chunk_sl, size_t cur_id, size_t total_chunks) {
  constexpr int el_per_tile_row = std::is_same<input_type, int8_t>::value ? 64 : 32;
  // int8 need 4 tiles output to pack 64 results for cache line
  constexpr int tiles_per_compute_row = std::is_same<input_type, int8_t>::value ? 4 : 2;
  set_compute_blk_cfg(el_per_tile_row);

  auto C_ = *reinterpret_cast<output_type(*)[chunk_sl][cols_step_][el_per_tile_row]>(C);
  auto A_ = *reinterpret_cast<input_type(*)[chunk_sl][ic_]>(A);
  auto B_ = *reinterpret_cast<input_type(*)[cols_step_][tiles_per_compute_row]
                                           [cols_in_tile_][16][el_per_tile_row]>(B);
  auto bias_ = *reinterpret_cast<bias_type(*)[cols_step_][el_per_tile_row]>(bias);

  compute_block_t computer_2 = compute_block_tbl_<ver>[2][compute_blk_idx_];
  compute_block_t computer_1 = compute_block_tbl_<ver>[1][compute_blk_idx_];

  size_t row_tile = (chunk_sl + 15) / 16;
  size_t roll_back = row_tile * 16 - chunk_sl;

  bool odd_tile = row_tile % 2;
  size_t row_step = row_tile / 2;

  // must divided total_chunks up
  size_t chunk_step = cols_step_ / total_chunks;
  if (cols_step_ % total_chunks != 0) {
    throw std::runtime_error(
        "output_feature must be divisible by core_num * 64(for int8) or 32(for bf16) "
        "in weight split i_linear.");
  }
  if (chunk_sl < 16) {
    throw std::runtime_error("chunk_sl less than 16 is not supported in i_linear.");
  }
  size_t wei_start = cur_id * chunk_step;
  size_t wei_end = (cur_id + 1) * chunk_step;
  size_t row_start = 0;
  // row sharding to avoid bank conflict
  if (chunk_sl >= total_chunks * 32) {
    row_start = chunk_sl * cur_id / total_chunks / 32 * 32;
  }
  if (chunk_sl < 32) {
    size_t k = 0;
    for (k = wei_start; k < wei_end; k++) {
      size_t row_pos = 0;
      (this->*computer_1)(
          C_[row_pos][k], oc_, A_[row_pos], B_[k], bias_[k], scale, post_op_, o_scale);
      row_pos += 16 - roll_back;
      (this->*computer_1)(
          C_[row_pos][k], oc_, A_[row_pos], B_[k], bias_[k], scale, post_op_, o_scale);
    }
  } else if (odd_tile) {
    size_t k = 0;
    for (k = wei_start; k < wei_end; k++) {
      size_t row_pos = 0;
      for (size_t j = 0; j < row_step; j++) {
        row_pos = j * 32;
        (this->*computer_2)(
            C_[row_pos][k], oc_, A_[row_pos], B_[k], bias_[k], scale, post_op_,
            o_scale);
      }
      row_pos += 32 - roll_back;
      (this->*computer_1)(
          C_[row_pos][k], oc_, A_[row_pos], B_[k], bias_[k], scale, post_op_, o_scale);
    }
  } else {
    for (size_t k = wei_start; k < wei_end; k++) {
      for (size_t j = 0; j < row_step; j++) {
        size_t rollback_ = (j == row_step - 1) * roll_back;
        auto row_pos = row_start + j * 32 - rollback_;
        row_pos = row_pos - int(row_pos >= chunk_sl) * chunk_sl;
        (this->*computer_2)(
            C_[row_pos][k], oc_, A_[row_pos], B_[k], bias_[k], scale, post_op_,
            o_scale);
      }
    }
  }
}

#define DEF_COMPUTE_BLK_TBL(ver)                                   \
  template <>                                                      \
  const i_linear::compute_block_t i_linear::compute_block_tbl_<    \
      i_linear::ver>[compute_blk_tbl_row][compute_blk_tbl_col] = { \
      {nullptr, nullptr, nullptr, nullptr},                        \
      {&i_linear::compute_block<1, 16, i_linear::ver>,             \
       &i_linear::compute_block<1, 32, i_linear::ver>,             \
       &i_linear::compute_block<1, 64, i_linear::ver>,             \
       &i_linear::compute_block<1, 10, i_linear::ver>,             \
       &i_linear::compute_block<1, 42, i_linear::ver>,             \
       &i_linear::compute_block<1, 4, i_linear::ver>},             \
      {&i_linear::compute_block<2, 16, i_linear::ver>,             \
       &i_linear::compute_block<2, 32, i_linear::ver>,             \
       &i_linear::compute_block<2, 64, i_linear::ver>,             \
       &i_linear::compute_block<2, 10, i_linear::ver>,             \
       &i_linear::compute_block<2, 42, i_linear::ver>,             \
       &i_linear::compute_block<2, 4, i_linear::ver>},             \
  }

FOREACH_COMPUTE_IMPL(DEF_COMPUTE_BLK_TBL);

#define DEF_TEMPLATE_SPECIALIZATION_TILE_PRODUCT(                        \
    input_type, output_type, bias_type, ver)                             \
  template void i_linear::tile_dot_product_16x256<                       \
      input_type, output_type, bias_type, i_linear::ver>(                \
      void* C, void* A, void* B, void* bias, float scale, float o_scale, \
      const size_t chunk_sl, size_t cur_id, size_t total_chunks);        \
  template void i_linear::tile_dot_product_16x256_shortage<              \
      input_type, output_type, bias_type, i_linear::ver>(                \
      void* C, void* A, void* B, void* bias, float scale, float o_scale, \
      const size_t chunk_sl, size_t cur_id, size_t total_chunks)

#define FOREACH_TILE_PRODUCT_VER(cb)                       \
  cb(int8_t, int8_t, _Float16, i8o8b16);                   \
  cb(int8_t, float, float, i8o32b32);                      \
  cb(int8_t, float, float, i8o32b0);                       \
  cb(int8_t, float, float, i8o32b32_accum);                \
  cb(__bfloat16, float, float, i16o32b32);                 \
  cb(__bfloat16, float, float, i16o32b0);                  \
  cb(__bfloat16, float, float, i16o32b32_accum);           \
  cb(__bfloat16, __bfloat16, float, i16o16b0);             \
  cb(__bfloat16, __bfloat16, float, i16o16b32_accum_relu); \
  cb(int8_t, int8_t, float, i8o8b32);

FOREACH_TILE_PRODUCT_VER(DEF_TEMPLATE_SPECIALIZATION_TILE_PRODUCT)

}  // namespace intel_mlperf
