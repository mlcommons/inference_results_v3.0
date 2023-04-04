#include <ATen/Functions.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <dnnl.hpp>
#include <numeric>
#include <immintrin.h>
#include <unordered_map>
#include <fstream>
#include "cpu.hpp"
#include "linear.hpp"
#include "lru_cache.hpp"
#include "dnnl_ext.hpp"

#if defined(USE_MKL)
#include <mkl_cblas.h>
#endif

namespace std {

template <> struct hash<dnnl::memory::dims> {
  size_t operator()(dnnl::memory::dims const& vec) const {
    size_t seed = vec.size();
    for(auto& i : vec) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}

namespace intel_mlperf {

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

static int cache_capacity = 512;

memory::data_type cast(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::Char:
      return dt::s8;
    case at::ScalarType::Float:
      return dt::f32;
    case at::ScalarType::Int:
      return dt::s32;
    case at::ScalarType::Byte:
      return dt::u8;
    default:
      return dt::undef;
  }
}

at::ScalarType cast(memory::data_type type) {
  switch (type) {
    case dt::s8:
      return at::ScalarType::Char;
    case dt::f32:
      return at::ScalarType::Float;
    case dt::s32:
      return at::ScalarType::Int;
    default:
      return at::ScalarType::Undefined;
  }
}

enum class behavior {
  query, infer, plain, blocking
};

// jit_brgemm_inner_product_utils.cpp "get_desired_weights_tag"
tag match_prepacked_weight_tag(c10::ArrayRef<int64_t> sizes) {
  tag fit_tag = tag::undef;

  if (sizes.size() == 5
      && sizes[2] == 4
      && sizes[4] == 4) {
    switch (sizes[3]) {
    case 64:
      fit_tag = tag::AB4b64a4b;
    case 32:
      fit_tag = tag::AB4b32a4b;
    case 16:
      fit_tag = tag::AB4b16a4b;
    }
  } else if (sizes.size() == 5 && sizes[2] == 16
      && sizes[4] == 4) {
    switch (sizes[3]) {
    case 64:
      fit_tag = tag::OI16i64o4i;
    case 32:
      fit_tag = tag::OI16i32o4i;
    case 16:
      fit_tag = tag::OI16i16o4i;
    }
  }

  return fit_tag;
}

memory::dims dims_from(c10::ArrayRef<int64_t> sizes,
    behavior b = behavior::plain) {
  if (b == behavior::infer) {
    if (match_prepacked_weight_tag(sizes) != tag::undef) {
      size_t A = 0, B = 1, a = 3, b[2] = {2,4};
      auto dim0 = sizes[A] * sizes[a];
      auto dim1 = sizes[B] * sizes[b[0]] * sizes[b[1]];
      return {dim0, dim1};
    } else
      return sizes.vec();
  }
  return sizes.vec();
}

template <typename T>
T concat(const T& t1, at::ScalarType d) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back((int64_t)d);

  return t;
}

template <typename T>
T concat(const T& t1, bool b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, int b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, const T& t2) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.insert(t.end(), t2.begin(), t2.end());

  return t;
}

template <typename T1, typename T2, typename ...Ts>
T1 concat(const T1& t1, const T2& t2, const Ts&... ts) {
  return concat(concat(t1, t2), ts...);
}

memory::desc md_from(const at::Tensor& tensor, behavior b = behavior::plain) {
  auto m_sz = dims_from(tensor.sizes());
  auto m_strides = dims_from(tensor.strides());
  auto data_type = cast(tensor.scalar_type());

  if (b == behavior::query) {
    return memory::desc(m_sz, data_type, tag::any);
  } else {
    // Warning: Encoding conflict possible!
    if (match_prepacked_weight_tag(m_sz) != tag::undef) {
      size_t A = 0, B = 1, a = 3, b[2] = {2,4};
      auto dim0 = m_sz[A] * m_sz[a];
      auto dim1 = m_sz[B] * m_sz[b[0]] * m_sz[b[1]];
      return memory::desc({dim0, dim1}, data_type, tag::any/* TODO:specify detail information */);
    } else {
      return memory::desc(m_sz, data_type, m_strides);
    }
  }

  throw std::exception();
}

memory memory_from(const at::Tensor& tensor) {
  auto md = md_from(tensor);
  return memory (md, g_cpu(), tensor.data_ptr());
}

at::Tensor scratch_tensor_from(const memory::desc* md) {
  int64_t scratch_sz = md->get_size();
  return at::empty({scratch_sz}, at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));
}

// weak reference to something inside compute
const memory::desc* get_scratch_md(primitive compute) {
  auto *scratch_cmd = dnnl_primitive_desc_query_md(compute.get_primitive_desc(),
    dnnl_query_t::dnnl_query_scratchpad_md, 0);

  return reinterpret_cast<const memory::desc *>(scratch_cmd);
}

memory::dims block_to_plain(memory::desc& desc) {
  auto basic_dims = desc.dims();
  auto &b_desc = desc.data.format_desc.blocking;
  for (int i = 0; i < b_desc.inner_nblks; ++i) {
    auto logic_dim = b_desc.inner_idxs[i];
    auto newd = b_desc.inner_blks[i];
    basic_dims[logic_dim] /= newd;
    basic_dims.push_back(newd);
  }

  return basic_dims;
}

// TODO: Support unsigned int8
at::Tensor prepack_linear_weight (
    const at::Tensor& weight) {
  auto m_sz = weight.sizes();
  if (match_prepacked_weight_tag(m_sz) != tag::undef)
    return weight;

  auto w_sz = dims_from(weight.sizes());
  // for (auto& id : w_sz)
  // {
  //   std::cout << id << std::endl;
  // }
  memory::dims synthetic_input_sz {128, w_sz[1]};
  memory::dims synthetic_output_sz {128, w_sz[0]};

  memory::desc synthetic_src_md (synthetic_input_sz, dt::s8, tag::any);
  memory::desc synthetic_dst_md (synthetic_output_sz, dt::s8, tag::any);
  auto weight_md = md_from(weight, behavior::query);

  auto desc = inner_product_forward::desc(
      prop_kind::forward_inference,
      synthetic_src_md,
      weight_md,
      synthetic_dst_md);

  primitive_attr attr;
  attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
  auto pd = inner_product_forward::primitive_desc(desc, attr, g_cpu());
  auto prepacked_weight_md = pd.weights_desc();

  auto new_size = block_to_plain(prepacked_weight_md);
  if (new_size == m_sz)
    return weight;

  int64_t nbytes = pd.weights_desc().get_size();
  int itemsize = sizeof(int8_t); // for now

  // occupy memory space
  auto output = at::empty({nbytes / itemsize},
      at::TensorOptions().dtype(cast(prepacked_weight_md.data_type()))
      .memory_format(c10::MemoryFormat::Contiguous));

  // Rely on torch not to shrink storage
  output.resize_(new_size);

  TORCH_CHECK(output.storage().nbytes() >= pd.weights_desc().get_size(),
      "Output Tensor Storage must preserve more space than "
      "just hold elements, extra data are filled at the end of tensor");

  memory prepacked_weight(pd.weights_desc(), g_cpu(), output.data_ptr());
  auto m_weight = memory_from(weight);
  stream s(g_cpu());
  reorder(m_weight, prepacked_weight).execute(s, m_weight, prepacked_weight);

  return output;
}

at::Tensor reorder_test(const at::Tensor& weight) {
  auto dims = dims_from(weight.sizes(), behavior::plain);

  // Start build test
  memory::desc test_case(
      dims, cast(weight.scalar_type()), tag::AB4b64a4b);
  test_case.data.extra.flags = 0
    | dnnl_memory_extra_flags_t::dnnl_memory_extra_flag_compensation_conv_s8s8
    | dnnl_memory_extra_flags_t::dnnl_memory_extra_flag_scale_adjust;
  test_case.data.extra.compensation_mask = (1<<0);
  test_case.data.extra.scale_adjust = 1.0f;

  auto m_weight = memory_from(weight);

  auto numel = test_case.get_size() / sizeof(int8_t);
  auto output = at::empty(numel,
      at::TensorOptions().dtype(cast(test_case.data_type()))
      .memory_format(c10::MemoryFormat::Contiguous));

  auto new_size = block_to_plain(test_case);
  output.resize_(new_size);

  memory m_output(test_case, g_cpu(), output.data_ptr());
  stream s(g_cpu());
  reorder(m_weight, m_output).execute(s, m_weight, m_output);
  return output;
}

// Fast key and search
using primitive_cache = lru_cache<memory::dims, primitive>;

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Scalar>& scale,
    const c10::optional<at::Scalar>& zero) {

  static thread_local primitive_cache cached(cache_capacity);

  _mm_setcsr(0x9fc0);
  auto src_sz = dims_from(input.sizes());
  at::Tensor _input;

  // Squeeze front
  if (src_sz.size() > 2) {
    auto _1d = std::accumulate(src_sz.begin(), src_sz.end() - 1, 1,
        std::multiplies<int64_t>());
    memory::dims src_2d_sz {_1d, *(src_sz.end() -1)};
    _input = input.view(src_2d_sz);
  } else {
    _input = input;
  }
  auto _src_sz = dims_from(_input.sizes());
  auto weight_sz = dims_from(weight.sizes(), behavior::infer);
  auto bias_sz = bias ? dims_from(bias.value().sizes()) : memory::dims();
  memory::dims _dst_sz {_src_sz[0], weight_sz[0]};

  primitive compute;

  auto key = concat(
      _src_sz, weight_sz,
      (bool)bias,
      bias ? bias->scalar_type() : at::ScalarType::Undefined,
      (bool)scale, (bool)zero
  );

  auto i_compute = cached.find(key);

  if (i_compute == cached.end()) {
    auto src_md = md_from(_input);

    /* TODO: adjust weight md according to input */
    auto weight_md = md_from(weight);
    auto bias_md = bias ? md_from(bias.value()) : memory::desc();
    memory::desc dst_md(_dst_sz, scale ? dt::s8 : dt::s32, tag::ab);

    // Singnaling skip compensation in BRGEMM
    prop_kind prop = zero ? prop_kind::forward_training
      : prop_kind::forward_inference;

    auto desc = bias ?
      inner_product_forward::desc(
        prop, src_md, weight_md, bias_md, dst_md) :
      inner_product_forward::desc(
        prop, src_md, weight_md, dst_md);

    primitive_attr attr;
    if (scale) {
      attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
    }
    // slow
    auto pd = inner_product_forward::primitive_desc(desc, attr, g_cpu());
    compute = inner_product_forward(pd);
    // end of slow

    // save what we have done
    cached.insert(std::make_pair(key, compute));
  } else {
    compute = i_compute->second;
  }

  primitive_ext compute_ext(compute);
  auto m_input = memory(*compute_ext.src_desc(), g_cpu(), input.data_ptr());
  auto m_weight = memory(*compute_ext.weights_desc(), g_cpu(), weight.data_ptr());

  // Use runtime output scale
  float _scale = scale.value_or(at::Scalar(1.f)).toFloat();
  memory m_oscale({{1}, dt::f32, {1}}, g_cpu(), &_scale);

  memory::dims dst_sz;
  dst_sz.reserve(input.dim());
  dst_sz.insert(dst_sz.end(), src_sz.begin(), src_sz.end() -1);
  dst_sz.push_back(weight_sz[0]);

  auto output = at::empty(
      dst_sz,
      at::TensorOptions()
        .dtype(cast(scale ? dt::s8 : dt::s32))
        .memory_format(c10::MemoryFormat::Contiguous)
  );

  auto m_dst = memory(*compute_ext.dst_desc(), g_cpu(), output.data_ptr());

  // What happens when zero
  auto scratch_md = get_scratch_md(compute);
  auto scratch_tensor =  scratch_tensor_from(compute_ext.scratchpad_desc());
  memory m_scratch(*scratch_md, g_cpu(), scratch_tensor.data_ptr());

  stream s(g_cpu());

  if (bias) {
    auto m_bias = memory(
        *compute_ext.weights_desc(1), g_cpu(), bias.value().data_ptr());

    compute.execute(s, {
      {DNNL_ARG_SRC, m_input},
      {DNNL_ARG_WEIGHTS, m_weight},
      {DNNL_ARG_BIAS, m_bias},
      {DNNL_ARG_DST, m_dst},
      {DNNL_ARG_ATTR_OUTPUT_SCALES, m_oscale},
      {DNNL_ARG_SCRATCHPAD, m_scratch}
    });
  } else {
    compute.execute(s, {
      {DNNL_ARG_SRC, m_input},
      {DNNL_ARG_WEIGHTS, m_weight},
      {DNNL_ARG_DST, m_dst},
      {DNNL_ARG_ATTR_OUTPUT_SCALES, m_oscale},
      {DNNL_ARG_SCRATCHPAD, m_scratch}
    });
  }
  return output;  

}

//
// M is for middle output
// scale is for final output
//
at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Scalar>& M,
    const c10::optional<at::Scalar>& scale,
    const c10::optional<at::Scalar>& zero) {

  static thread_local primitive_cache cached(cache_capacity);

  _mm_setcsr(0x9fc0);
  auto src_sz = dims_from(input.sizes());
  at::Tensor _input;

  // Squeeze front
  if (src_sz.size() > 2) {
    auto _1d = std::accumulate(src_sz.begin(), src_sz.end() - 1, 1,
        std::multiplies<int64_t>());
    memory::dims src_2d_sz {_1d, *(src_sz.end() -1)};
    _input = input.view(src_2d_sz);
  } else {
    _input = input;
  }
  auto _src_sz = dims_from(_input.sizes());
  auto weight_sz = dims_from(weight.sizes(), behavior::infer);
  auto bias_sz = bias ? dims_from(bias.value().sizes()) : memory::dims();
  memory::dims _dst_sz {_src_sz[0], weight_sz[0]};

  primitive compute;

  auto key = concat(
      _src_sz, weight_sz, (bool)bias,
      bias ? bias->scalar_type() : at::ScalarType::Undefined,
      (bool)M, (bool)scale, (bool)zero
  );

  auto i_compute = cached.find(key);

  if (i_compute == cached.end()) {
    auto src_md = md_from(_input);
    auto weight_md = md_from(weight);
    auto bias_md = bias ? md_from(bias.value()) : memory::desc();
    memory::desc dst_md(_dst_sz, scale ? dt::s8 : dt::f32, tag::ab);
    // Singnaling skip compensation in BRGEMM
    prop_kind prop = zero ? prop_kind::forward_training
      : prop_kind::forward_inference;

    auto desc = bias ?
      inner_product_forward::desc(
        prop, src_md, weight_md, bias_md, dst_md) :
      inner_product_forward::desc(
        prop, src_md, weight_md, dst_md);

    post_ops po;
    // TODO: 1.f will disable runtime scaling
    // Use scale selectively open/close this
    po.append_eltwise(
        2.2f, algorithm::eltwise_gelu_erf_2dts, 0.f, 0.f);

    primitive_attr attr;
    attr.set_post_ops(po);

    if (M) {
      attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});
    }

    // slow
    auto pd = inner_product_forward::primitive_desc(desc, attr, g_cpu());
    compute = inner_product_forward(pd);
    // end of slow

    // save what we have done
    cached.insert(std::make_pair(key, compute));
  } else {
    compute = i_compute->second;
  }

  primitive_ext compute_ext(compute);
  auto m_input = memory(*compute_ext.src_desc(), g_cpu(), input.data_ptr());
  auto m_weight = memory(*compute_ext.weights_desc(), g_cpu(), weight.data_ptr());

  // Use runtime output scale
  float _M = M.value_or(1.).toFloat();
  memory m_oscale({{1}, dt::f32, {1}}, g_cpu(), &_M);

  memory::dims dst_sz;
  dst_sz.reserve(input.dim());
  dst_sz.insert(dst_sz.end(), src_sz.begin(), src_sz.end() -1);
  dst_sz.push_back(weight_sz[0]);

  auto output = at::empty(
      dst_sz,
      at::TensorOptions()
        .dtype(cast(scale ? dt::s8 : dt::f32))
        .memory_format(c10::MemoryFormat::Contiguous)
  );

  auto m_dst = memory(*compute_ext.dst_desc(), g_cpu(), output.data_ptr());

  // What happens when zero
  auto scratch_md = get_scratch_md(compute);
  auto scratch_tensor = scratch_tensor_from(compute_ext.scratchpad_desc());
  memory m_scratch(*scratch_md, g_cpu(), scratch_tensor.data_ptr());

  stream s(g_cpu());

  std::unordered_map<int, memory> args {
    {DNNL_ARG_SRC, m_input},
    {DNNL_ARG_WEIGHTS, m_weight},
    {DNNL_ARG_DST, m_dst},
    {DNNL_ARG_ATTR_OUTPUT_SCALES, m_oscale},
    {DNNL_ARG_SCRATCHPAD, m_scratch}
  };

  if (bias) {
    auto m_bias = memory(
        *compute_ext.weights_desc(1), g_cpu(), bias.value().data_ptr());
    args.emplace(std::make_pair(DNNL_ARG_BIAS, m_bias));
  }

  auto _scale = scale.value_or(1.f).toFloat();
  auto m_scale = memory(
     {{1}, dt::f32, {1}}, g_cpu(), &_scale);
  args.emplace(std::make_pair(DNNL_ARG_SCALE, m_scale));

  compute.execute(s, args);

  return output;
}


at::Tensor baddbmm_out_onednn(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha);

template <int N> struct parallel_gemm {
  template <typename T>
  using accessor_t = at::TensorAccessor<T, N>;
  template <typename F>
  parallel_gemm(c10::ArrayRef<int64_t> sizes,
      accessor_t<int8_t> a_acc,
      accessor_t<int8_t> b_acc,
      accessor_t<int32_t> c_acc, F gemm) = delete;
};

template <> struct parallel_gemm<4> {
  template <typename T>
  using accessor_t = at::TensorAccessor<T, 4>;
  template <typename F>
  parallel_gemm(c10::ArrayRef<int64_t> sizes,
      accessor_t<int8_t> a_acc,
      accessor_t<int8_t> b_acc,
      accessor_t<int32_t> c_acc, F gemm) {
#   pragma omp parallel for
    for (int b0 = 0; b0 < sizes[0]; ++ b0) {
      for (int b1 = 0; b1 < sizes[1]; ++ b1) {
        auto *A = a_acc[b0][b1].data();
        auto *B = b_acc[b0][b1].data();
        auto *C = c_acc[b0][b1].data();
        gemm(A, B, C);
      }
    }
  }
};

template <> struct parallel_gemm<3> {
  template <typename T>
  using accessor_t = at::TensorAccessor<T, 3>;
  template <typename F>
  parallel_gemm(c10::ArrayRef<int64_t> sizes,
      accessor_t<int8_t> a_acc,
      accessor_t<int8_t> b_acc,
      accessor_t<int32_t> c_acc, F gemm) {
#   pragma omp parallel for
    for (int b0 = 0; b0 < sizes[0]; ++ b0) {
      auto *A = a_acc[b0].data();
      auto *B = b_acc[b0].data();
      auto *C = c_acc[b0].data();
      gemm(A, B, C);
    }
  }
};


//
// inline status gemm_s8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
//         dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
//         dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
//         float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co);
//
// NOTE: column major interface, support upto 5-D dimension
//
template <int dim>
at::Tensor baddbmm_out_gemm_s8s8s32_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    float beta,
    float alpha) {
  // matricis are 2 dimension, others are constructs
  auto start_dim = dim - 2;

  const auto b1_sizes = batch1.sizes();
  const auto b2_sizes = batch2.sizes();

  const auto b1_strides = batch1.strides();
  const auto b2_strides = batch2.strides();

  const auto sf_strides = self.strides();

  // From pytorch
  auto is_transposed = [](
      const c10::IntArrayRef strides,
      const c10::IntArrayRef sizes,
      int start_dim) {
    return strides[start_dim] == 1
      && strides[start_dim + 1] >= sizes[start_dim + 1];
  };

  //
  // column major requires rework of arithmetic
  // What we want to the result: C.T = B.T * A.T
  // Since already transposed then we don't need it to be transposed
  //
  char transa = is_transposed(b1_strides, b1_sizes, start_dim) ? 't' : 'n';
  char transb = is_transposed(b2_strides, b2_sizes, start_dim) ? 't' : 'n';

  const auto M = b1_sizes[start_dim];
  const auto N = b2_sizes[start_dim + 1];
  const auto K = b1_sizes[start_dim + 1];

  const auto lda = transa == 't' ? b1_strides[start_dim + 1] : b1_strides[start_dim];
  const auto ldb = transb == 't' ? b2_strides[start_dim + 1] : b2_strides[start_dim];
  const auto ldc = sf_strides[start_dim];

  auto a_acc = batch1.accessor<int8_t, dim>();
  auto b_acc = batch2.accessor<int8_t, dim>();
  auto c_acc = self.accessor<int32_t, dim>();
  int32_t offset_c = 0;

  parallel_gemm<dim> (
      b1_sizes, a_acc, b_acc, c_acc,
      [&](int8_t *A, int8_t *B, int32_t *C) {
    // Row major or Col major???
    gemm_s8s8s32(
        transa,
        transb,
        'f',
        M, N, K,
        alpha,
        A, lda, 0, B, ldb, 0,
        beta, C, ldc, &offset_c);
  });

  return self;
}


// We have muliple strategy choice
//   1. oneDNN MatMul (heavy interface, cache needed)
//   2. parallel oneDNN gemm_s8s8s32 (Could avoid transpose by setting ldc)
//   3. parallel cblas_gemm_s8u8s32
//
at::Tensor baddbmm_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  switch (batch1.dim()) {
  case 4:
    return baddbmm_out_gemm_s8s8s32_<4>(
        self, batch1, batch2, beta.toFloat(), alpha.toFloat());
  case 3:
    return baddbmm_out_gemm_s8s8s32_<3>(
        self, batch1, batch2, beta.toFloat(), alpha.toFloat());
  }
  throw std::exception();
}

at::Tensor matmul_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const c10::optional<at::Scalar>& oscale,
    const c10::optional<at::Scalar>& zero) {

  auto self_dims = dims_from(self.sizes());
  auto b1_dims = dims_from(batch1.sizes());
  auto b2_dims = dims_from(batch2.sizes());

  auto key = concat(
      self_dims, b1_dims, b2_dims, (bool)oscale, (bool)zero,
      (int)cast(batch1.scalar_type()));

  static thread_local primitive_cache cached(cache_capacity);
  primitive compute;

  auto i_comp = cached.find(key);
  if (i_comp == cached.end()) {
    auto self_md = md_from(self);
    auto b1_md = md_from(batch1);
    auto b2_md = md_from(batch2);

    primitive_attr attr;
    if (oscale)
      attr.set_output_scales(0, {DNNL_RUNTIME_F32_VAL});

    matmul::desc matmul_d(b1_md, b2_md, self_md);

    matmul_d.data.prop_kind =
      zero ? dnnl_prop_kind_t::dnnl_forward_training:
      dnnl_prop_kind_t::dnnl_forward_inference;

    matmul::primitive_desc matmul_pd(matmul_d, attr, g_cpu());
    compute = matmul::primitive(matmul_pd);
    cached.insert(std::make_pair(key, compute));
  } else {
    compute = i_comp->second;
  }

  primitive_ext ext_compute(compute);
  memory self_m(*ext_compute.dst_desc(), g_cpu(), self.data_ptr());
  memory b1_m(*ext_compute.src_desc(), g_cpu(), batch1.data_ptr());
  memory b2_m(*ext_compute.weights_desc(), g_cpu(), batch2.data_ptr());

  float scale = oscale.value_or(1.0).toFloat();
  memory m_oscale({{1}, dt::f32, {1}}, g_cpu(), &scale);

  auto scratch = scratch_tensor_from(ext_compute.scratchpad_desc());
  memory m_scratch(*ext_compute.scratchpad_desc(), g_cpu(), scratch.data_ptr());
  stream s(g_cpu());

  std::unordered_map<int, memory> args {
    {DNNL_ARG_SRC, b1_m},
    {DNNL_ARG_WEIGHTS, b2_m},
    {DNNL_ARG_ATTR_OUTPUT_SCALES, m_oscale},
    {DNNL_ARG_DST, self_m},
    {DNNL_ARG_SCRATCHPAD, m_scratch}
  };

  compute.execute(s, args);
  return self;
}

}
