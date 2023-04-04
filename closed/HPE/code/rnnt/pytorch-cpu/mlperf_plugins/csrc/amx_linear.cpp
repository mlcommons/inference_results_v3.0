#include "amx_linear.hpp"

#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>

#include <chrono>
#include <iostream>

#include "amx_init.hpp"
#include "tpps/amx_config.hpp"
#include "tpps/i_linear_tpp.hpp"

namespace intel_mlperf {

template <typename itype, typename otype, typename btype, i_linear::Version ver>
at::Tensor amx_linear_impl(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const float scale, const bool post_op, const float o_scale) {
  using itype_pt = typename std::conditional<
      std::is_same<itype, __bfloat16>::value, at::BFloat16, itype>::type;
  using otype_pt = typename std::conditional<
      std::is_same<otype, __bfloat16>::value, at::BFloat16, otype>::type;
  using btype_pt = typename std::conditional<
      std::is_same<btype, _Float16>::value, at::Half, btype>::type;
  constexpr int el_per_tile_row = std::is_same<itype, int8_t>::value ? 64 : 32;
  // weight shape: [col_step, 2, col_tile, 16, 32] for bf16
  // weight shape: [col_step, 4, col_tile, 16, 64] for int8
  auto wshape = weight.sizes();
  auto col_step = wshape[0];
  auto output_f = col_step * el_per_tile_row;

  // input shape: [bs, sl, input_feature]
  auto ishape = input.sizes();
  auto oshape = ishape.vec();
  long bs, sl, input_f, total_sl;
  if (ishape.size() == 3) {
    bs = ishape[0];
    sl = ishape[1];
    input_f = ishape[2];
    total_sl = bs * sl;
    oshape[2] = output_f;
  } else {
    bs = ishape[0];
    input_f = ishape[1];
    total_sl = bs;
    oshape[1] = output_f;
  }

  // output shape: [bs, sl, output_feature]
  auto output = at::empty(
      oshape, at::TensorOptions().dtype<otype_pt>().memory_format(
                  c10::MemoryFormat::Contiguous));

  auto block_computer = i_linear(total_sl, input_f, output_f, true, post_op);

  auto input_2d = input;
  auto output_2d = output;
  if (ishape.size() == 3) {
    input_2d = input.view({-1, input_f});
    output_2d = output.view({-1, output_f});
  }
  auto input_ = input_2d.template accessor<itype_pt, 2>();
  auto weight_ = weight.template accessor<itype_pt, 5>();
  auto output_ = output_2d.template accessor<otype_pt, 2>();
  auto bias_ = bias.template accessor<btype_pt, 1>();

  amx_init::amx_init();
#pragma omp parallel
  {
    Tilecfg().set_config();
    auto total_core_num = omp_get_num_threads();
    auto core_id = omp_get_thread_num();
    size_t start_ = total_sl * core_id / total_core_num;
    size_t chunk_sl_ = (total_sl * core_id + total_sl) / total_core_num - start_;
    size_t minimum_sl = 32 * total_core_num;

    if (total_sl < minimum_sl && output_f > el_per_tile_row) {
      block_computer.tile_dot_product_16x256_shortage<itype, otype, btype, ver>(
          output_.data(), input_.data(), weight_.data(), bias_.data(), scale, o_scale,
          total_sl, core_id, total_core_num);
    } else {
      block_computer.tile_dot_product_16x256<itype, otype, btype, ver>(
          output_[start_].data(), input_[start_].data(), weight_.data(), bias_.data(),
          scale, o_scale, chunk_sl_, core_id, total_core_num);
    }
    Tilecfg().release_config();
  }
  return output;
}

at::Tensor amx_linear(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Scalar& scale, const bool post_op, const at::Scalar& o_scale) {
  if (bias.options().dtype() == at::kHalf) {
    return amx_linear_impl<int8_t, int8_t, _Float16, i_linear::i8o8b16>(
        input, weight, bias, scale.toFloat(), post_op, o_scale.toFloat());
  } else {
    return amx_linear_impl<int8_t, int8_t, float, i_linear::i8o8b32>(
        input, weight, bias, scale.toFloat(), post_op, o_scale.toFloat());
  }
}

at::Tensor amx_linear_i8o32(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Scalar& scale) {
  return amx_linear_impl<int8_t, float, float, i_linear::i8o32b32>(
      input, weight, bias, scale.toFloat(), false, 0.0);
}

at::Tensor amx_linear_i16o32(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias) {
  return amx_linear_impl<__bfloat16, float, float, i_linear::i16o32b32>(
      input, weight, bias, 0.0, false, 0.0);
}

at::Tensor amx_linear_bf16_accum_relu(
    const at::Tensor& x0, const at::Tensor& w0, const at::Tensor& x1,
    const at::Tensor& w1, const at::Tensor& bias) {
  auto bs = x0.size(0);
  auto input0_f = x0.size(1);
  auto input1_f = x1.size(1);
  auto output_f = bias.size(0);

  auto y = at::empty(
      {bs, output_f},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto y_bf16 = at::empty(
      {bs, output_f}, at::TensorOptions().dtype<at::BFloat16>().memory_format(
                          c10::MemoryFormat::Contiguous));
  auto linear0 = i_linear(bs, input0_f, output_f, true, false);

  auto x0_ = x0.accessor<at::BFloat16, 2>();
  auto w0_ = w0.accessor<at::BFloat16, 5>();
  auto b_ = bias.accessor<float, 1>();
  auto y_bf16_ = y_bf16.accessor<at::BFloat16, 2>();

  amx_init::amx_init();
#pragma omp parallel
  {
    Tilecfg().set_config();
    auto total_core_num = omp_get_num_threads();
    auto core_id = omp_get_thread_num();
    linear0.tile_dot_product_16x256_shortage<
        __bfloat16, __bfloat16, float, i_linear::i16o16b0>(
        y_bf16_.data(), x0_.data(), w0_.data(), b_.data(), 0.0, 0.0, bs, core_id,
        total_core_num);
    Tilecfg().release_config();
  }

  auto linear1 = i_linear(bs, input1_f, output_f, true, false);

  auto x1_ = x1.accessor<at::BFloat16, 2>();
  auto w1_ = w1.accessor<at::BFloat16, 5>();

#pragma omp parallel
  {
    Tilecfg().set_config();
    auto total_core_num = omp_get_num_threads();
    auto core_id = omp_get_thread_num();
    linear1.tile_dot_product_16x256_shortage<
        __bfloat16, __bfloat16, float, i_linear::i16o16b32_accum_relu>(
        y_bf16_.data(), x1_.data(), w1_.data(), b_.data(), 0.0, 0.0, bs, core_id,
        total_core_num);
    Tilecfg().release_config();
  }
  return y_bf16;
}

}  // namespace intel_mlperf
