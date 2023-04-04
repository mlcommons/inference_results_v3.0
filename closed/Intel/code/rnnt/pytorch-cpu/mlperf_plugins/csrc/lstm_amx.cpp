#include "lstm_amx.hpp"

#include <ATen/Functions.h>
#include <ATen/ops/add_ops.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "amx_config.hpp"
#include "amx_init.hpp"
#include "lstm_postop.hpp"
#include "quant_tpp.hpp"
#include "tpps/i_linear_tpp.hpp"
#include "tpps/lstm_postop_tpp.hpp"

namespace intel_mlperf {

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm_amx_int8(
    const at::Tensor& x, const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights, const at::Tensor& rb_scale,
    const at::Tensor& i_scale, const at::Tensor& o_scale, const bool skip_quant_y) {
  auto ishape = x.sizes();
  auto num_layers = all_weights.size();
  auto hx_ = hx;
  auto cx_ = cx;
  at::Tensor x_;
  if (x.dtype() == at::kFloat) {
    x_ = at::empty(
        ishape, at::TensorOptions().dtype<int8_t>().memory_format(
                    c10::MemoryFormat::Contiguous));
    auto pin = reinterpret_cast<float(*)>(x.data_ptr());
    auto x_ptr = reinterpret_cast<int8_t(*)>(x_.data_ptr());
    quant_tpp::quant_ps_epi8(x_ptr, pin, i_scale[0].item().toFloat(), x.numel());
  } else {
    x_ = x;
  }

  for (int i = 0; i < num_layers; i++) {
    auto weights_layer = all_weights[i];
    auto skip_quant = (i == (num_layers - 1)) & skip_quant_y;
    auto remainder = x_.size(2) % 64;
    if (remainder != 0)
      x_ = at::pad(x_, {0, 64 - remainder}, "constant", 0);
    std::tie(x_, hx_[i], cx_[i]) = lstm_layer_amx_int8(
        x_, hx_[i], cx_[i], weights_layer[0], weights_layer[1], weights_layer[2],
        weights_layer[3], rb_scale[i].item(), i_scale[i].item(), o_scale[i].item(),
        skip_quant);
  }
  return {x_, hx_, cx_};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_amx_int8(
    const at::Tensor& x, const at::Tensor& hx, const at::Tensor& cx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& b_ih,
    const at::Tensor& b_hh, const c10::optional<at::Scalar>& rb_scale,
    const c10::optional<at::Scalar>& i_scale, const c10::optional<at::Scalar>& o_scale,
    const bool skip_quant_y) {
  auto ishape = x.sizes();
  auto seq_len = ishape[0];
  auto bs = ishape[1];
  auto input_f = ishape[2];
  auto hidden = hx.size(1);
  auto output_f = b_ih.size(0);

  float rb_scale_ = rb_scale.value_or(at::Scalar(1.f)).toFloat();
  float i_scale_ = i_scale.value_or(at::Scalar(1.f)).toFloat();
  float o_scale_ = o_scale.value_or(at::Scalar(1.f)).toFloat();

  // output shape: [sl, batch_size, output_f]
  auto y = at::empty(
      {seq_len, bs, output_f},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto y_q = at::empty(
      {seq_len, bs, hidden},
      at::TensorOptions().dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));
  auto y_bf16 = at::empty(
      {seq_len, bs, hidden}, at::TensorOptions().dtype<at::BFloat16>().memory_format(
                                 c10::MemoryFormat::Contiguous));
  auto linear_ih = i_linear(bs, input_f, output_f, true, false);

  auto x_ptr = x.data_ptr();
  auto w_ih_ptr = w_ih.data_ptr();
  auto y_ptr = y.data_ptr();
  auto y_q_ptr = y_q.data_ptr();
  auto b_ih_ptr = b_ih.data_ptr();
  auto y_bf16_ptr = y_bf16.data_ptr();

  amx_init::amx_init();
  for (int i = 0; i < seq_len; i++) {
    // linear for input
#pragma omp parallel
    {
      auto input_ = reinterpret_cast<int8_t(*)[bs][input_f]>(x_ptr);
      auto weight_ = reinterpret_cast<int8_t*>(w_ih_ptr);
      auto output_ = reinterpret_cast<float(*)[bs][output_f]>(y_ptr);
      auto bias_ = reinterpret_cast<float*>(b_ih_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_ih
          .tile_dot_product_16x256_shortage<int8_t, float, float, i_linear::i8o32b0>(
              output_[i], input_[i], weight_, bias_, rb_scale_, 0.0, bs, core_id,
              total_core_num);
      Tilecfg().release_config();
    }
  }

  auto linear_hh = i_linear(bs, hidden, output_f, true, false);

  auto hx_ptr = hx.data_ptr();
  auto w_hh_ptr = w_hh.data_ptr();
  auto b_hh_ptr = b_hh.data_ptr();
  auto cx_ptr = cx.data_ptr();

  for (int i = 0; i < seq_len; i++) {
    // linear for hidden state
#pragma omp parallel
    {
      auto input_ = reinterpret_cast<int8_t*>(hx_ptr);
      auto weight_ = reinterpret_cast<int8_t*>(w_hh_ptr);
      auto output_ = reinterpret_cast<float(*)[bs][output_f]>(y_ptr);
      auto bias_ = reinterpret_cast<float*>(b_hh_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_hh.tile_dot_product_16x256_shortage<
          int8_t, float, float, i_linear::i8o32b32_accum>(
          output_[i], input_, weight_, bias_, rb_scale_, 0.0, bs, core_id,
          total_core_num);
      Tilecfg().release_config();
    }
    // post op
#pragma omp parallel for
    for (auto b = 0; b < bs; ++b) {
      auto y_ = reinterpret_cast<float(*)[bs][output_f]>(y_ptr);
      auto y_q_ = reinterpret_cast<int8_t(*)[bs][hidden]>(y_q_ptr);
      auto y_bf16_ = reinterpret_cast<__bfloat16(*)[bs][hidden]>(y_bf16_ptr);
      auto hx_ = reinterpret_cast<int8_t(*)[hidden]>(hx_ptr);
      auto cx_ = reinterpret_cast<_Float16(*)[hidden]>(cx_ptr);
      lstm_postop_tpp::ref(
          y_bf16_[i][b], y_q_[i][b], hx_[b], &y_[i][b][0], &y_[i][b][hidden],
          &y_[i][b][hidden * 2], &y_[i][b][hidden * 3], cx_[b], i_scale_, o_scale_,
          hidden, skip_quant_y);
    }
  }
  if (skip_quant_y) {
    return {y_bf16, hx, cx};
  } else {
    return {y_q, hx, cx};
  }
}

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm_amx_bf16(
    const at::Tensor& x, const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights) {
  auto num_layers = all_weights.size();
  std::vector<at::Tensor> hy, cy;
  hy.reserve(num_layers);
  cy.reserve(num_layers);
  at::Tensor hy_layer, cy_layer;

  auto x_ = x;
  for (int i = 0; i < num_layers; i++) {
    auto weights_layer = all_weights[i];
    // auto remainder = x_.size(2) % 32;
    // if (remainder != 0)
    //   x_ = at::pad(x_, {0, 32 - remainder}, "constant", 0);
    std::tie(x_, hy_layer, cy_layer) = lstm_layer_amx_bf16(
        x_, hx[i], cx[i], weights_layer[0], weights_layer[1], weights_layer[2],
        weights_layer[3]);
    hy.emplace_back(hy_layer);
    cy.emplace_back(cy_layer);
  }
  return {x_, hy, cy};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_amx_bf16(
    const at::Tensor& x, const at::Tensor& hx, const at::Tensor& cx,
    const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& b_ih,
    const at::Tensor& b_hh) {
  auto ishape = x.sizes();
  auto seq_len = ishape[0];
  auto bs = ishape[1];
  auto input_f = ishape[2];
  auto hidden = hx.size(1);
  auto output_f = b_ih.size(0);

  // output shape: [sl, batch_size, output_f]
  auto y = at::empty(
      {seq_len, bs, output_f},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto y_bf16 = at::empty(
      {seq_len, bs, hidden}, at::TensorOptions().dtype<at::BFloat16>().memory_format(
                                 c10::MemoryFormat::Contiguous));
  auto hy = at::empty(
      {bs, hidden}, at::TensorOptions().dtype<at::BFloat16>().memory_format(
                        c10::MemoryFormat::Contiguous));
  auto cy = at::empty(
      {bs, hidden},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto linear_ih = i_linear(bs, input_f, output_f, true, false);

  auto x_ = x.accessor<at::BFloat16, 3>();
  auto w_ih_ = w_ih.accessor<at::BFloat16, 5>();
  auto y_ = y.accessor<float, 3>();
  auto y_bf16_ = y_bf16.accessor<at::BFloat16, 3>();
  auto b_ih_ = b_ih.accessor<float, 1>();

  amx_init::amx_init();
  for (int i = 0; i < seq_len; i++) {
    // linear for input
#pragma omp parallel
    {
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_ih.tile_dot_product_16x256_shortage<
          __bfloat16, float, float, i_linear::i16o32b0>(
          y_[i].data(), x_[i].data(), w_ih_.data(), b_ih_.data(), 0.0, 0.0, bs, core_id,
          total_core_num);
      Tilecfg().release_config();
    }
  }

  auto linear_hh = i_linear(bs, hidden, output_f, true, false);

  auto w_hh_ = w_hh.accessor<at::BFloat16, 5>();
  auto b_hh_ = b_hh.accessor<float, 1>();
  auto hx_ = hx.accessor<at::BFloat16, 2>();
  auto cx_ = cx.accessor<float, 2>();
  auto hy_ = hy.accessor<at::BFloat16, 2>();
  auto cy_ = cy.accessor<float, 2>();

  for (int i = 0; i < seq_len; i++) {
    // linear for hidden state
#pragma omp parallel
    {
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_hh.tile_dot_product_16x256_shortage<
          __bfloat16, float, float, i_linear::i16o32b32_accum>(
          y_[i].data(), hx_.data(), w_hh_.data(), b_hh_.data(), 0.0, 0.0, bs, core_id,
          total_core_num);
      Tilecfg().release_config();
    }
    // post op
#pragma omp parallel for
    for (auto b = 0; b < bs; ++b) {
      lstm_postop_tpp::ref_bf16(
          y_bf16_[i][b].data(), hy_[b].data(), cy_[b].data(), &y_[i][b][0],
          &y_[i][b][hidden], &y_[i][b][hidden * 2], &y_[i][b][hidden * 3],
          cx_[b].data(), hidden);
    }
    hx_ = hy_;
  }
  return {y_bf16, hy, cy};
}

}  // namespace intel_mlperf
