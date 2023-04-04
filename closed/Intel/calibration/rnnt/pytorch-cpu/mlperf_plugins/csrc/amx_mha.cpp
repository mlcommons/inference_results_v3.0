#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>

#include "amx_mha.hpp"
#include "i_mha_tpp.hpp"
#include "amx_config.hpp"
#include "amx_init.hpp"

namespace intel_mlperf {

at::Tensor amx_mha(const at::Tensor &qkv, const at::Tensor &att_mask,
                   const at::Scalar &m1, const at::Scalar &oscale,
                   const at::Scalar &m2) {
  auto qkv_sizes = qkv.sizes();
  assert(qkv_sizes.size() == 3);
  auto bs = qkv_sizes[0];
  auto sl = qkv_sizes[1];
  auto stride = qkv_sizes[2];

  auto qkv_block = stride / 3;
  int head_size = 64;
  int head_num = qkv_block / head_size;

  auto attention = at::empty({bs, sl, qkv_block},
                             at::TensorOptions().dtype<int8_t>().memory_format(
                                 c10::MemoryFormat::Contiguous));

  // TODO: release amx_init
  // TODO: amx_init moved to SUT
  // TODO: amx tile config and release need added here; context manager

  // create attention tensor
  auto att_data_ptr = attention.data_ptr();
  auto qkv_data_ptr = qkv.data_ptr();
  auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());

  auto m1_ = m1.toFloat();
  auto m2_ = m2.toFloat();
  auto oscale_ = oscale.toFloat();
  amx_init::amx_init();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < bs; i++) {         // batch size
    for (int j = 0; j < head_num; j++) { // head num
      auto att_ptr = reinterpret_cast<int8_t(*)[sl][head_num][head_size]>(att_data_ptr);
      auto origin_ptr = reinterpret_cast<int8_t(*)[sl][3][head_num][head_size]>(qkv_data_ptr);
      auto cur_q_ptr = origin_ptr[i][0][0][j];
      auto cur_a_ptr = att_ptr[i][0][j];

      // auto total_core_num = omp_get_num_threads();
      // auto core_id = omp_get_thread_num();
      // printf("------------ core_id : %d / %d\n", core_id, total_core_num);

      // if (core_id == 1) {
      //   printf("%d - %ld   %d - %d\n", i, bs, j, head_num);
      // }

      amx_per_head(cur_q_ptr, stride, cur_a_ptr, qkv_block, sl, m1_,
                   oscale_, att_mask_p[i], m2_);
    }
  }
  return attention;
}

} // namespace intel_mlperf
