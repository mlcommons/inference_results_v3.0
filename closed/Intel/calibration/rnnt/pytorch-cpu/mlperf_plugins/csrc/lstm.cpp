#include <dnnl.hpp>
#include "cpu.hpp"
#include "dnnl_ext.hpp"
#include "lru_cache.hpp"
#include "lstm.hpp"

namespace intel_mlperf {

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
// Fast key and search
using primitive_cache = lru_cache<memory::dims, primitive>;

static int cache_capacity = 512;

enum class behavior {
  query, infer, plain, blocking
};

memory memory_from(const at::Tensor& tensor);

at::ScalarType cast(memory::data_type type);

memory::data_type cast(at::ScalarType type);

at::Tensor scratch_tensor_from(const memory::desc* md);

memory::dims block_to_plain(memory::desc& desc);

struct LSTMParams {
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  int64_t num_gates;
  memory::dims x_sz;
  memory::dims hx_sz;
  memory::dims cx_sz;
  memory::dims w_ih_sz;
  memory::dims w_hh_sz;
  memory::dims bias_sz;
  memory::dims y_sz;
  memory::dims hy_sz;
  memory::dims cy_sz;

  LSTMParams(
      const int64_t seq_length_,
      const int64_t mini_batch_,
      const int64_t input_size_,
      const int64_t hidden_size_,
      const int64_t num_layers_=1,
      const int64_t num_directions_=1,
      const int64_t num_gates_=4) {
    seq_length = seq_length_;  // T
    mini_batch = mini_batch_;  // N
    input_size = input_size_;  // IC = SC = SLC, if L > 1, = DLC
    hidden_size = hidden_size_;  // OC = DC = DLC = DHC = DIC, if T > 1, = SIC
    num_layers = num_layers_;  // L
    num_directions = num_directions_;  // D
    num_gates = num_gates_;  // G
    x_sz = {seq_length, mini_batch, input_size};  // {T, N, SLC}};
    hx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, SIC}
    cx_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
    w_ih_sz = {num_layers, num_directions, input_size, num_gates, hidden_size};  // {L, D, SLC, G, DHC}
    w_hh_sz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};  // {L, D, SIC, G, DHC}
    bias_sz = {num_layers, num_directions, num_gates, hidden_size};  // {L, D, G, DHC}
    y_sz = {seq_length, mini_batch, hidden_size};  // {T, N, DLC}
    hy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DIC}
    cy_sz = {num_layers, num_directions, mini_batch, hidden_size};  // {L, D, N, DHC}
  }
};

tag match_prepacked_lstm_weights(c10::ArrayRef<int64_t> sizes) {
  tag fit_tag = tag::undef;
  if (sizes.size() == 7)
    fit_tag = tag::abdEC32e4c;  // int8 weight
  if (sizes.size() == 6)
    fit_tag = tag::abdEc32e;  // fp32 weight
  if (sizes.size() == 5)
    fit_tag = tag::abcde;  // bf16 weight
  if (sizes.size() == 4)
    fit_tag = tag::abcd;  // bias
  return fit_tag;
}

std::tuple<at::Tensor, at::Tensor> prepack_lstm_weights (
    const at::Tensor& w_ih, const at::Tensor& w_hh) {
  if (match_prepacked_lstm_weights(w_ih.sizes()) != tag::undef
    && match_prepacked_lstm_weights(w_hh.sizes()) != tag::undef)
    return {w_ih, w_hh};

  LSTMParams lstm(1, 32, w_ih.size(1), w_hh.size(1));

  // Shuffle weights
  // w_ih: {G*OC, IC} -> {L, D, G, OC, IC} -> {L, D, SLC(IC), G, DHC(OC)}
  auto w_ih_ = w_ih.reshape(
      {lstm.num_layers, lstm.num_directions, lstm.num_gates, lstm.hidden_size, lstm.input_size})
      .permute({0, 1, 4, 2, 3}).contiguous();
  // w_hh: {G*OC, OC} -> {L, D, G, OC, OC} -> {L, D, SIC(OC), G, DHC(OC)}
  auto w_hh_ = w_hh.reshape(
      {lstm.num_layers, lstm.num_directions, lstm.num_gates, lstm.hidden_size, lstm.hidden_size})
      .permute({0, 1, 4, 2, 3}).contiguous();

  // Create memory descriptor
  auto w_dt = cast(w_ih.scalar_type());
  memory::desc x_md (lstm.x_sz, w_dt, tag::tnc);
  memory::desc hx_md (lstm.hx_sz, w_dt, tag::ldnc);
  memory::desc cx_md (lstm.cx_sz, w_dt, tag::ldnc);
  memory::desc w_ih_md (lstm.w_ih_sz, w_dt, tag::any);
  memory::desc w_hh_md (lstm.w_hh_sz, w_dt, tag::any);
  memory::desc bias_md (lstm.bias_sz, dt::f32, tag::any);
  memory::desc y_md (lstm.y_sz, w_dt, tag::tnc);
  memory::desc hy_md (lstm.hy_sz, w_dt, tag::ldnc);
  memory::desc cy_md (lstm.cy_sz, w_dt, tag::ldnc);

  // Create operation descriptor
  lstm_forward::desc lstm_desc (
      prop_kind::forward_inference,
      rnn_direction::unidirectional_left2right,
      x_md, hx_md, cx_md,
      w_ih_md, w_hh_md, bias_md,
      y_md, hy_md, cy_md);

  // Create primitive descriptor
  auto lstm_pd = lstm_forward::primitive_desc(lstm_desc, g_cpu());
  auto prepacked_w_ih_md = lstm_pd.weights_layer_desc();
  auto prepacked_w_hh_md = lstm_pd.weights_iter_desc();

  auto prepacked_w_ih_sz = block_to_plain(prepacked_w_ih_md);
  auto prepacked_w_hh_sz = block_to_plain(prepacked_w_hh_md);
  if (prepacked_w_ih_sz == w_ih.sizes()
      && prepacked_w_hh_sz == w_hh.sizes())
    return {w_ih, w_hh};

  int prepacked_w_ih_nbytes = prepacked_w_ih_md.get_size();
  int prepacked_w_hh_nbytes = prepacked_w_hh_md.get_size();
  int itemsize;
  switch (w_dt) {
    case dt::u8:
      itemsize = 1;
    case dt::s8:
      itemsize = 1;
    case dt::bf16:
      itemsize = 2;
    case dt::f16:
      itemsize = 2;
    case dt::f32:
      itemsize = 4;
    case dt::s32:
      itemsize = 4;
    default:
      itemsize = 1;
  }

  at::Tensor prepacked_w_ih = at::empty(
      {prepacked_w_ih_nbytes / itemsize},
      at::TensorOptions().dtype(cast(prepacked_w_ih_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));
  prepacked_w_ih.resize_(prepacked_w_ih_sz);
  TORCH_CHECK(prepacked_w_ih.storage().nbytes() >= prepacked_w_ih_md.get_size(),
      "prepacked_w_ih storage must preserve more space than "
      "just hold elements, extra data are filled at the end of tensor");

  at::Tensor prepacked_w_hh = at::empty(
      {prepacked_w_hh_nbytes / itemsize},
      at::TensorOptions().dtype(cast(prepacked_w_hh_md.data_type()))
        .memory_format(c10::MemoryFormat::Contiguous));
  prepacked_w_hh.resize_(prepacked_w_hh_sz);
  TORCH_CHECK(prepacked_w_hh.storage().nbytes() >= prepacked_w_hh_md.get_size(),
      "prepacked_w_hh storage must preserve more space than "
      "just hold elements, extra data are filled at the end of tensor");

  stream s(g_cpu());
  auto m_w_ih = memory_from(w_ih_);
  memory m_prepacked_w_ih(lstm_pd.weights_layer_desc(), g_cpu(), prepacked_w_ih.data_ptr());
  reorder(m_w_ih, m_prepacked_w_ih).execute(s, m_w_ih, m_prepacked_w_ih);
  auto m_w_hh = memory_from(w_hh_);
  memory m_prepacked_w_hh(lstm_pd.weights_iter_desc(), g_cpu(), prepacked_w_hh.data_ptr());
  reorder(m_w_hh, m_prepacked_w_hh).execute(s, m_w_hh, m_prepacked_w_hh);

  return {prepacked_w_ih, prepacked_w_hh};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_1dnn(
    const at::Tensor& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& b_ih,
    const at::Tensor& b_hh) {

  auto hidden_size = b_ih.size(0) / 4;
  LSTMParams lstm(x.size(0), x.size(1), x.size(2), hidden_size);
  
  auto input_dt = cast(x.scalar_type());

  // Shuffle hx, cx & bias
  // hx: {D*L, N, OC} -> {L, D, N, SIC}
  auto hx_ = at::reshape(hx, lstm.hx_sz);
  // cx: {D*L, N, OC} -> {L, D, N, DHC}
  auto cx_ = at::reshape(cx, lstm.cx_sz);
  // bias = b_ih + b_hh: {G*OC} -> {L, D, G, DHC}
  auto bias = (b_ih + b_hh).resize_(lstm.bias_sz);

  // Prepack weights
  at::Tensor w_ih_, w_hh_;
  std::tie(w_ih_, w_hh_) = prepack_lstm_weights(w_ih, w_hh);

  static thread_local primitive_cache cached(cache_capacity);

  // Create stream
  stream s(g_cpu());

  auto key = concat(
    lstm.x_sz, lstm.hx_sz, lstm.cx_sz,
    lstm.w_ih_sz, lstm.w_hh_sz, lstm.bias_sz
  );

  auto i_compute = cached.find(key);

  primitive compute;
  if (i_compute == cached.end()) {
    // TODO: check params dtype
    // Create memory descriptor
    memory::desc x_md (lstm.x_sz, input_dt, tag::tnc);
    memory::desc hx_md (lstm.hx_sz, input_dt, tag::ldnc);
    memory::desc cx_md (lstm.cx_sz, cast(cx_.scalar_type()), tag::ldnc);  // check dt
    memory::desc w_ih_md (lstm.w_ih_sz, input_dt, tag::any);
    memory::desc w_hh_md (lstm.w_hh_sz, input_dt, tag::any);
    memory::desc bias_md (lstm.bias_sz, dt::f32, tag::any);  // check dt
    memory::desc y_md (lstm.y_sz, input_dt, tag::tnc);
    memory::desc hy_md (lstm.hy_sz, input_dt, tag::ldnc);
    memory::desc cy_md (lstm.cy_sz, cast(cx_.scalar_type()), tag::ldnc);  // check dt

    // Create operation descriptor
    lstm_forward::desc lstm_desc (
      prop_kind::forward_inference,
      rnn_direction::unidirectional_left2right,
      x_md, hx_md, cx_md,
      w_ih_md, w_hh_md, bias_md,
      y_md, hy_md, cy_md);

    // Create primitive desctiptor
    lstm_forward::primitive_desc lstm_pd (lstm_desc, g_cpu());
    
    // Create primitive
    compute = lstm_forward(lstm_pd);
  
    // Save key::primitive
    cached.insert(std::make_pair(key, compute));
  } else {
    compute = i_compute->second;
  }

  // Create memory object
  primitive_ext ext_compute(compute);
  memory m_x (*ext_compute.src_desc(), g_cpu(), x.data_ptr());
  memory m_hx (*ext_compute.src_desc(1), g_cpu(), hx_.data_ptr());
  memory m_cx (*ext_compute.src_desc(2), g_cpu(), cx_.data_ptr());

  memory m_w_ih (*ext_compute.weights_desc(), g_cpu(), w_ih_.data_ptr());
  memory m_w_hh (*ext_compute.weights_desc(1), g_cpu(), w_hh_.data_ptr());
  memory m_bias (*ext_compute.weights_desc(2), g_cpu(), bias.data_ptr());

  auto y = at::empty(lstm.y_sz, at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous));

  auto hy = at::empty(lstm.hy_sz, at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous));

  auto cy = at::empty(lstm.cy_sz, at::TensorOptions().dtype(cast(input_dt))
      .memory_format(c10::MemoryFormat::Contiguous));

  memory m_y (*ext_compute.dst_desc(), g_cpu(), y.data_ptr());
  memory m_hy (*ext_compute.dst_desc(1), g_cpu(), hy.data_ptr());
  memory m_cy (*ext_compute.dst_desc(2), g_cpu(), cy.data_ptr());

  auto scratch = scratch_tensor_from(ext_compute.scratchpad_desc());
  memory m_scratch(*ext_compute.scratchpad_desc(), g_cpu(), scratch.data_ptr());

  // Create primitive arguments
  std::unordered_map<int, memory> lstm_args = {
    {DNNL_ARG_SRC_LAYER, m_x},
    {DNNL_ARG_SRC_ITER, m_hx},
    {DNNL_ARG_SRC_ITER_C, m_cx},
    {DNNL_ARG_WEIGHTS_LAYER, m_w_ih},
    {DNNL_ARG_WEIGHTS_ITER, m_w_hh},
    {DNNL_ARG_BIAS, m_bias},
    {DNNL_ARG_DST_LAYER, m_y},
    {DNNL_ARG_DST_ITER, m_hy},
    {DNNL_ARG_DST_ITER_C, m_cy},
    {DNNL_ARG_SCRATCHPAD, m_scratch}
  };

  // Execute primitive
  compute.execute(s, lstm_args);

  // {L=1, D=1, N, C} -> {N, C}
  hy.resize_({lstm.mini_batch, lstm.hidden_size});
  cy.resize_({lstm.mini_batch, lstm.hidden_size});

  return {y, hy, cy};
}

std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm(
    const at::Tensor& x,
    const std::vector<at::Tensor>& hx,
    const std::vector<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights) {
  auto x_layer = x.contiguous();
  at::Tensor hy_layer, cy_layer;
  std::vector<at::Tensor> hy, cy;
  auto num_layers = all_weights.size();
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    auto weights_layer = all_weights[layer];
    std::tie(x_layer, hy_layer, cy_layer) = lstm_layer_1dnn(
        x_layer, hx[layer], cx[layer],
        weights_layer[0], weights_layer[1],
        weights_layer[2], weights_layer[3]);
    hy.emplace_back(hy_layer);
    cy.emplace_back(cy_layer);
  }
  return {x_layer, hy, cy};
}

}