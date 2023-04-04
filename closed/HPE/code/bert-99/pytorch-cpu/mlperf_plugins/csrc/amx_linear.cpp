#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>

#include "amx_linear.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

namespace intel_mlperf {

at::Tensor amx_linear(
  const at::Tensor& input,
  const at::Tensor& weight,
  const at::Tensor& bias,
  const at::Scalar& scale,
  const bool post_op,
  const at::Scalar& o_scale
) {
  // input shape: [bs, sl, hidden_size]
  auto ishape = input.sizes();
  auto bs = ishape[0];
  auto sl = ishape[1];
  auto hidden_size = ishape[2];

  // weight shape: [col_step, 4, col_tile, 16, 64]
  auto wshape = weight.sizes();
  auto col_step = wshape[0];
  auto col_tile = wshape[2];

  // output shape: [bs, sl, col_step * 64]
  auto output = at::empty({bs, sl, col_step * 64},
                          at::TensorOptions().dtype<int8_t>().memory_format(
                          c10::MemoryFormat::Contiguous));


  auto scale_ = scale.toFloat();
  float o_scale_ = post_op ? o_scale.toFloat() : 1.0;

  auto total_sl = bs * sl;

  size_t os_ = col_step * 64;
  auto block_computer = i_linear(sl, hidden_size, os_, true, post_op);
  auto block_computer_fp16 = i_linear_fp16(sl, hidden_size, os_, true, post_op);

  auto input_data_ptr = input.data_ptr();
  auto weight_data_ptr = weight.data_ptr();
  auto output_data_ptr = output.data_ptr();
  auto bias_data_ptr = bias.data_ptr();

  int bias_dtype = bias.options().dtype().name() == "c10::Half" ? 1 : 0;
  
  // 4 loop
  // or only omp parallel 
  switch (bias_dtype)
  {
  case (1):
    # pragma omp parallel 
    {
      auto input_ = reinterpret_cast<int8_t (*)[hidden_size]>(input_data_ptr);
      auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight_data_ptr);
      auto output_ = reinterpret_cast<int8_t (*)[col_step][64]>(output_data_ptr);
      auto bias_ = reinterpret_cast<_Float16 (*)>(bias_data_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      // printf("------------ core_id : %d / %d\n", core_id, total_core_num);
      size_t start_ = total_sl * core_id / total_core_num;
      size_t chunk_sl_ = (total_sl * core_id + total_sl) / total_core_num - start_;
      size_t minimum_sl = 32 * total_core_num;

      // block_computer.tile_dot_product_16x256_shortage(output_, input_, weight_, bias_, scale_, o_scale_, total_sl, col_step, core_id, total_core_num);  
      if (total_sl < minimum_sl) {
        block_computer_fp16.tile_dot_product_16x256_shortage(output_, input_, weight_, bias_, scale_, o_scale_, total_sl, col_step, core_id, total_core_num);  
      }
      else {
        block_computer_fp16.tile_dot_product_16x256(output_[start_], input_[start_], weight_, bias_, scale_, o_scale_, chunk_sl_, col_step, core_id, total_core_num);
      }
      Tilecfg().release_config();
    }
    break;
  default:
    # pragma omp parallel 
    {
      auto input_ = reinterpret_cast<int8_t (*)[hidden_size]>(input_data_ptr);
      auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight_data_ptr);
      auto output_ = reinterpret_cast<int8_t (*)[col_step][64]>(output_data_ptr);
      auto bias_ = reinterpret_cast<float (*)>(bias_data_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      // printf("------------ core_id : %d / %d\n", core_id, total_core_num);
      size_t start_ = total_sl * core_id / total_core_num;
      size_t chunk_sl_ = (total_sl * core_id + total_sl) / total_core_num - start_;
      size_t minimum_sl = 32 * total_core_num;

      // block_computer.tile_dot_product_16x256_shortage(output_, input_, weight_, bias_, scale_, o_scale_, total_sl, col_step, core_id, total_core_num);  
      if (total_sl < minimum_sl) {
        block_computer.tile_dot_product_16x256_shortage(output_, input_, weight_, bias_, scale_, o_scale_, total_sl, col_step, core_id, total_core_num);  
      }
      else {
        block_computer.tile_dot_product_16x256(output_[start_], input_[start_], weight_, bias_, scale_, o_scale_, chunk_sl_, col_step, core_id, total_core_num);
      }
      Tilecfg().release_config();
    }
    break;
  }
  return output;
}

}

