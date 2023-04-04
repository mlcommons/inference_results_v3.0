#include <chrono>
#include <kernel_mlp/mlp.hpp>
#include <kernel_mlp/pack.hpp>
#include <kernel_mlp/shape.hpp>
#include <stdint.h>
#include <stdio.h>
#include <vector>

template <typename T, size_t N> size_t product(const std::array<T, N> &v) {
  size_t ret = 1;
  for (auto a : v) {
    ret *= a;
  }
  return ret;
}

int main() {
  // plain inputs
  std::vector<uint16_t> plain_relu_out0(
      product(mlp_fwd_4k_shape::relu_out0())); // 128 * 512
  std::vector<uint16_t> plain_relu_out1(
      product(mlp_fwd_4k_shape::relu_out1())); // 128 * 256
  std::vector<uint16_t> plain_relu_out2(
      product(mlp_fwd_4k_shape::relu_out2())); // 128 * 256
  std::vector<uint16_t> plain_input(product(mlp_fwd_4k_shape::input()));
  std::vector<uint16_t> plain_weight0(product(mlp_fwd_4k_shape::weight0()));
  std::vector<uint16_t> plain_bias0(product(mlp_fwd_4k_shape::bias0()));
  std::vector<uint16_t> plain_weight1(product(mlp_fwd_4k_shape::weight1()));
  std::vector<uint16_t> plain_bias1(product(mlp_fwd_4k_shape::bias1()));
  std::vector<uint16_t> plain_weight2(product(mlp_fwd_4k_shape::weight2()));
  std::vector<uint16_t> plain_bias2(product(mlp_fwd_4k_shape::bias2()));

  // pack data
  std::vector<uint16_t> relu_out0(
      product(mlp_fwd_4k_shape::relu_out0())); // 128 * 512
  mlp_fwd_4k_shape::pack_relu_out0(relu_out0.data(), plain_relu_out0.data());
  std::vector<uint16_t> relu_out1(
      product(mlp_fwd_4k_shape::relu_out1())); // 128 * 256
  mlp_fwd_4k_shape::pack_relu_out1(relu_out1.data(), plain_relu_out1.data());
  std::vector<uint16_t> weight0(product(mlp_fwd_4k_shape::weight0()));
  mlp_fwd_4k_shape::pack_weight0(weight0.data(), plain_weight0.data());
  std::vector<uint16_t> bias0(product(mlp_fwd_4k_shape::bias0()));
  mlp_fwd_4k_shape::pack_bias0(bias0.data(), plain_bias0.data());
  std::vector<uint16_t> weight1(product(mlp_fwd_4k_shape::weight1()));
  mlp_fwd_4k_shape::pack_weight1(weight1.data(), plain_weight1.data());
  std::vector<uint16_t> bias1(product(mlp_fwd_4k_shape::bias1()));
  mlp_fwd_4k_shape::pack_bias1(bias1.data(), plain_bias1.data());
  std::vector<uint16_t> weight2(product(mlp_fwd_4k_shape::weight2()));
  mlp_fwd_4k_shape::pack_weight2(weight2.data(), plain_weight2.data());
  std::vector<uint16_t> bias2(product(mlp_fwd_4k_shape::bias2()));
  mlp_fwd_4k_shape::pack_bias2(bias2.data(), plain_bias2.data());

  // grad outputs
  std::vector<uint16_t> g_weight0(
      product(mlp_bwd_4k_shape::out_grad_weight0()));
  std::vector<uint16_t> g_bias0(product(mlp_bwd_4k_shape::out_grad_bias0()));
  std::vector<uint16_t> g_weight1(
      product(mlp_bwd_4k_shape::out_grad_weight1()));
  std::vector<uint16_t> g_bias1(product(mlp_bwd_4k_shape::out_grad_bias1()));
  std::vector<uint16_t> g_weight2(
      product(mlp_bwd_4k_shape::out_grad_weight2()));
  std::vector<uint16_t> g_bias2(product(mlp_bwd_4k_shape::out_grad_bias2()));
  std::vector<uint16_t> g_input(product(mlp_bwd_4k_shape::out_grad_input0()));
  std::vector<uint16_t> gradient(product(mlp_bwd_4k_shape::gradient()));

  // init kernel
  sc_init_mlp_training_backward_4k();
  sc_init_mlp_training_forward_4k();

  using namespace std::chrono;
  auto start_time = steady_clock::now();
  const int times = 100;
  for (int i = 0; i < times; i++) {
    mlp_training_forward_4k(relu_out0.data(), relu_out1.data(),
                            plain_relu_out2.data(), plain_input.data(),
                            weight0.data(), bias0.data(), weight1.data(),
                            bias1.data(), weight2.data(), bias2.data());

    mlp_training_backward_4k(
        g_bias2.data(), g_weight2.data(), g_bias1.data(), g_weight1.data(),
        g_bias0.data(), g_weight0.data(), g_input.data(), gradient.data(),
        plain_relu_out2.data(), relu_out1.data(), weight2.data(),
        relu_out0.data(), weight1.data(), plain_input.data(), weight0.data());
  }
  printf("Done FWD+BWD %d times in %ld us\n", times,
         duration_cast<microseconds>(steady_clock::now() - start_time).count());

  // unpack data
  std::vector<uint16_t> plain_g_weight0(
      product(mlp_bwd_4k_shape::out_grad_weight0()));
  mlp_bwd_4k_shape::unpack_out_grad_weight0(plain_g_weight0.data(),
                                            g_weight0.data());
  std::vector<uint16_t> plain_g_bias0(
      product(mlp_bwd_4k_shape::out_grad_bias0()));
  mlp_bwd_4k_shape::unpack_out_grad_bias0(plain_g_bias0.data(), g_bias0.data());
  std::vector<uint16_t> plain_g_weight1(
      product(mlp_bwd_4k_shape::out_grad_weight1()));
  mlp_bwd_4k_shape::unpack_out_grad_weight1(plain_g_weight1.data(),
                                            g_weight1.data());
  std::vector<uint16_t> plain_g_bias1(
      product(mlp_bwd_4k_shape::out_grad_bias1()));
  mlp_bwd_4k_shape::unpack_out_grad_bias1(plain_g_bias1.data(), g_bias1.data());
  std::vector<uint16_t> plain_g_weight2(
      product(mlp_bwd_4k_shape::out_grad_weight2()));
  mlp_bwd_4k_shape::unpack_out_grad_weight2(plain_g_weight2.data(),
                                            g_weight2.data());
  std::vector<uint16_t> plain_g_bias2(
      product(mlp_bwd_4k_shape::out_grad_bias2()));
  mlp_bwd_4k_shape::unpack_out_grad_bias2(plain_g_bias2.data(), g_bias2.data());
  std::vector<uint16_t> plain_g_input(
      product(mlp_bwd_4k_shape::out_grad_input0()));
  mlp_bwd_4k_shape::unpack_out_grad_input0(plain_g_input.data(),
                                           g_input.data());
  std::vector<uint16_t> plain_gradient(product(mlp_bwd_4k_shape::gradient()));
  mlp_bwd_4k_shape::unpack_gradient(plain_gradient.data(), gradient.data());

  return 0;
}