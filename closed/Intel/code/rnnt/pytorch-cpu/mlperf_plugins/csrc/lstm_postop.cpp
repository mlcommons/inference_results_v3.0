#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <vector>
#include "lstm_postop_tpp.hpp"

namespace intel_mlperf {
typedef unsigned short __bfloat16;
std::vector<at::Tensor> lstm_postop (
  const at::Tensor& it,
  const at::Tensor& ft,
  const at::Tensor& gt,
  const at::Tensor& ot,
  const at::Tensor& ct,
  const c10::optional<at::Scalar>& i_scale,
  const c10::optional<at::Scalar>& o_scale,
  const bool& skip_quant_y) {
    std::vector<at::Tensor> output = {};
    auto sizes = it.sizes();
    auto batch = sizes[0];
    auto line = sizes[1];
    auto stride = it.strides();
    auto lda = stride[0];
    auto in_scale = i_scale->toFloat();
    auto out_scale = o_scale->toFloat();

    auto out_yt_q = at::empty(sizes,
    at::TensorOptions().dtype<int8_t>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto out_ht_q = at::empty(sizes,
    at::TensorOptions().dtype<int8_t>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto out_yt_bf16 = at::empty(sizes,
    at::TensorOptions().dtype<at::BFloat16>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto *in_it = it.data_ptr();
    auto *in_ft = ft.data_ptr();
    auto *in_gt = gt.data_ptr();
    auto *in_ot = ot.data_ptr();
    auto *in_ct = ct.data_ptr();
    auto *out_yt_q_ptr = out_yt_q.data_ptr();
    auto *out_ht_q_ptr = out_ht_q.data_ptr();
    auto *out_yt_bf16_ptr = out_yt_bf16.data_ptr();

    #pragma omp parallel for
    for (auto b=0; b < batch;++b) {
      auto pin_it = reinterpret_cast<float (*)[line]>(in_it);
      auto pin_ft = reinterpret_cast<float (*)[line]>(in_ft);
      auto pin_gt = reinterpret_cast<float (*)[line]>(in_gt);
      auto pin_ot = reinterpret_cast<float (*)[line]>(in_ot);
      auto pin_ct = reinterpret_cast<_Float16 (*)[line]>(in_ct);

      auto pout_yt_q = reinterpret_cast<int8_t (*)[line]>(out_yt_q_ptr);
      auto pout_ht_q = reinterpret_cast<int8_t (*)[line]>(out_ht_q_ptr);
      auto pout_yt_bf16 = reinterpret_cast<__bfloat16 (*)[line]>(out_yt_bf16_ptr);

      if(lda==line)
        lstm_postop_tpp::ref(pout_yt_bf16[b], pout_yt_q[b],pout_ht_q[b],pin_it[b],pin_ft[b],pin_gt[b],pin_ot[b],pin_ct[b],in_scale,out_scale,line,skip_quant_y);
      else
        lstm_postop_tpp::ref(pout_yt_bf16[b], pout_yt_q[b],pout_ht_q[b],pin_it[4*b],pin_ft[4*b],pin_gt[4*b],pin_ot[4*b],pin_ct[b],in_scale,out_scale,line,skip_quant_y);
    }

    return {out_yt_bf16, out_yt_q, out_ht_q, ct};
}

}
