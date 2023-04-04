#include "normalization.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "i_layernorm_tpp.hpp"

namespace intel_mlperf {
std::tuple<at::Tensor, at::Tensor> i_layernorm_pad(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Tensor &length, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &unbiased,
    const c10::optional<at::Tensor> &output_shape) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto output_length = length;
  auto in_sz = input.sizes();
  auto actual_batch = in_sz[0];
  int64_t padded_batch, out_feat;
  if (output_shape) {
    auto output_shape_ = output_shape->accessor<int32_t, 1>();
    padded_batch = output_shape_[0];
    if (padded_batch == 1) {
      padded_batch = actual_batch;
    } else {
      output_length = at::pad(length, {0, padded_batch - actual_batch}, "constant", 0);
    }
    out_feat = output_shape_[1];
  } else {
    padded_batch = actual_batch;
    out_feat = in_sz[1];
  }
  auto inner = in_sz[1];
  auto max_len = *at::_ops::max::call(length).data_ptr<int32_t>();
  at::Tensor output = at::empty(
      {padded_batch, out_feat, max_len},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto in = input.accessor<float, 3>();
  auto w = weight.accessor<float, 3>();
  auto b = bias.accessor<float, 3>();
  auto out = output.accessor<float, 3>();
  auto len = length.accessor<int32_t, 1>();
  auto data_type = input.scalar_type();

  if (data_type == c10::ScalarType::Float) {
#pragma omp parallel for
    for (auto i = 0; i < padded_batch; ++i) {
      if (i < actual_batch) {
        for (auto j = 0; j < inner; ++j) {
          i_layernorm_tpp<16>::ref(
              &out[i][j][0], &in[i][j][0], &w[0][j][0], &b[0][j][0], len[i],
              eps.value_or(1e-12).toFloat(), unbiased.value_or(false).toBool());
        }
      } else {
        memset(&out[i][0][0], 0, sizeof(float) * out_feat * max_len);
      }
    }
  }  // throw here

  return {output, output_length};
}

at::Tensor i_layernorm(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
    const at::Scalar &oscale, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &o_off) {
  auto in_sz = input.sizes();
  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];
  auto output = at::empty(
      in_sz,
      at::TensorOptions().dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));

  auto *in = input.data_ptr();
  auto *w = weight.data_ptr();
  auto *b = bias.data_ptr();
  auto *out = output.data_ptr();
  auto _o_off = o_off.value_or(0).toChar();
  auto data_type = input.scalar_type();

  if (data_type == c10::ScalarType::Char) {
#pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto *pin = reinterpret_cast<int8_t(*)[reduce_l]>(in);
      auto *pw = reinterpret_cast<float *>(w);
      auto *pb = reinterpret_cast<float *>(b);
      auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

      i_layernorm_tpp<16>::ref(
          pout[i], pin[i], pw, pb, oscale.toFloat(), reduce_l,
          eps.value_or(1e-12).toFloat(), _o_off);
    }
  } else if (data_type == c10::ScalarType::Float) {
#pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto *pin = reinterpret_cast<float(*)[reduce_l]>(in);
      auto *pw = reinterpret_cast<float *>(w);
      auto *pb = reinterpret_cast<float *>(b);
      auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

      i_layernorm_tpp<16>::ref(
          pout[i], pin[i], pw, pb, oscale.toFloat(), reduce_l,
          eps.value_or(1e-12).toFloat(), _o_off);
    }
  }  // throw here

  return output;
}

at::Tensor i_residual_layernorm(
    const at::Tensor &input1, const at::Tensor &input2, const at::Tensor &weight,
    const at::Tensor &bias, const at::Scalar &scale_1, const at::Scalar &scale_2,
    const at::Scalar &oscale, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &o_off) {
  auto in_sz = input1.sizes();

  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];

  auto output = at::empty(
      in_sz,
      at::TensorOptions().dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));

  auto *src1 = input1.data_ptr();
  auto *src2 = input2.data_ptr();
  auto *w = weight.data_ptr();
  auto *b = bias.data_ptr();
  auto *out = output.data_ptr();
  auto _o_off = o_off.value_or(0).toChar();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto *psrc1 = reinterpret_cast<int8_t(*)[reduce_l]>(src1);
    auto *psrc2 = reinterpret_cast<int8_t(*)[reduce_l]>(src2);
    auto *pw = reinterpret_cast<float *>(w);
    auto *pb = reinterpret_cast<float *>(b);
    auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

    i_residual_layernorm_tpp<16>::ref(
        pout[i], psrc1[i], psrc2[i], pw, pb, scale_1.toFloat(), scale_2.toFloat(),
        oscale.toFloat(), reduce_l, eps.value_or(1e-12).toFloat(), _o_off);
  }

  return output;
}

at::Tensor i_residual_layernorm_fp32_(
    at::Tensor &input1, const at::Tensor &input2, const at::Tensor &weight,
    const at::Tensor &bias, const at::Scalar &scale_1, const at::Scalar &scale_2,
    const at::Scalar &oscale, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &o_off) {
  auto in_sz = input1.sizes();

  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];

  auto *src1 = input1.data_ptr();
  auto *src2 = input2.data_ptr();
  auto *w = weight.data_ptr();
  auto *b = bias.data_ptr();
  auto *out = src1;
  auto _o_off = o_off.value_or(0).toChar();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto *psrc1 = reinterpret_cast<int8_t(*)[reduce_l]>(src1);
    auto *psrc2 = reinterpret_cast<int8_t(*)[reduce_l]>(src2);
    // TODO: add switch float or _Float16
    auto *pw = reinterpret_cast<float *>(w);
    auto *pb = reinterpret_cast<float *>(b);
    auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

    i_residual_layernorm_tpp<16>::ref(
        pout[i], psrc1[i], psrc2[i], pw, pb, scale_1.toFloat(), scale_2.toFloat(),
        oscale.toFloat(), reduce_l, eps.value_or(1e-12).toFloat(), _o_off);
  }

  return input1;
}

at::Tensor i_residual_layernorm_(
    at::Tensor &input1, const at::Tensor &input2, const at::Tensor &weight,
    const at::Tensor &bias, const at::Scalar &scale_1, const at::Scalar &scale_2,
    const at::Scalar &oscale, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &o_off) {
  auto in_sz = input1.sizes();

  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];

  auto *src1 = input1.data_ptr();
  auto *src2 = input2.data_ptr();
  auto *w = weight.data_ptr();
  auto *b = bias.data_ptr();
  auto *out = src1;
  auto _o_off = o_off.value_or(0).toChar();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto *psrc1 = reinterpret_cast<int8_t(*)[reduce_l]>(src1);
    auto *psrc2 = reinterpret_cast<int8_t(*)[reduce_l]>(src2);
    // TODO: add switch float or _Float16
    auto *pw = reinterpret_cast<_Float16 *>(w);
    auto *pb = reinterpret_cast<_Float16 *>(b);
    auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

    i_residual_layernorm_tpp<32>::ref_fp16(
        pout[i], psrc1[i], psrc2[i], pw, pb, scale_1.toFloat(), scale_2.toFloat(),
        oscale.toFloat(), reduce_l, eps.value_or(1e-12).toFloat(), _o_off);
  }

  return input1;
}

at::Tensor i_residual_layernorm_cin_(
    at::Tensor &input1, const at::Tensor &input2, const at::Tensor &weight,
    const at::Tensor &bias, const at::Scalar &scale_1, const at::Scalar &scale_2,
    const at::Scalar &oscale, const c10::optional<at::Scalar> &eps,
    const c10::optional<at::Scalar> &o_off) {
  auto in_sz = input1.sizes();

  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];

  auto *src1 = input1.data_ptr();
  auto *src2 = input2.data_ptr();
  auto *w = weight.data_ptr();
  auto *b = bias.data_ptr();
  auto *out = src1;
  auto _o_off = o_off.value_or(0).toChar();

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto *psrc1 = reinterpret_cast<int8_t(*)[reduce_l]>(src1);
    auto *psrc2 = reinterpret_cast<int8_t(*)[reduce_l]>(src2);
    auto *pw = reinterpret_cast<float *>(w);
    auto *pb = reinterpret_cast<float *>(b);
    auto *pout = reinterpret_cast<int8_t(*)[reduce_l]>(out);

    i_residual_layernorm_tpp<16>::ref_cin(
        pout[i], psrc1[i], psrc2[i], pw, pb, scale_1.toFloat(), scale_2.toFloat(),
        oscale.toFloat(), reduce_l, eps.value_or(1e-12).toFloat(), _o_off);
  }

  return input1;
}

}  // namespace intel_mlperf
