#include "utils.hpp"

#define BACKBONE_SHAPE backbone_4_shape
#define SC_INIT_RN50_BACKBONE sc_init_rn50_backbone_bs4
#define RN50_BACKBONE rn50_backbone_bs4

int main() {
#define I8TENSOR(NAME)                                                         \
  aligned_vector<int8_t> NAME(product(BACKBONE_SHAPE::NAME()), 0)
#define U8TENSOR(NAME)                                                         \
  aligned_vector<uint8_t> NAME(product(BACKBONE_SHAPE::NAME()), 0)
#define TENSOR(NAME)                                                           \
  aligned_vector<float> NAME(product(BACKBONE_SHAPE::NAME()), 0.f)
  // cat ../src/kernel_rn50/shape.hpp | grep "inline const" | awk -F"&" '{print
  // $2}' | awk -F"(" '{print "TENSOR("$1 ");"}'
  // plain inputs
  I8TENSOR(backbone_input);
  I8TENSOR(backbone_output);
  TENSOR(res2a_bias_0);
  TENSOR(res2a_bias_1);
  TENSOR(res2a_bias_2);
  TENSOR(res2a_bias_b);
  TENSOR(res2a_weight_0);
  TENSOR(res2a_weight_1);
  TENSOR(res2a_weight_2);
  TENSOR(res2a_weight_b);
  TENSOR(res2b_bias_0);
  TENSOR(res2b_bias_1);
  TENSOR(res2b_bias_2);
  TENSOR(res2b_weight_0);
  TENSOR(res2b_weight_1);
  TENSOR(res2b_weight_2);
  TENSOR(res2c_bias_0);
  TENSOR(res2c_bias_1);
  TENSOR(res2c_bias_2);
  TENSOR(res2c_weight_0);
  TENSOR(res2c_weight_1);
  TENSOR(res2c_weight_2);
  TENSOR(res3a_bias_0);
  TENSOR(res3a_bias_1);
  TENSOR(res3a_bias_2);
  TENSOR(res3a_bias_b);
  TENSOR(res3a_weight_0);
  TENSOR(res3a_weight_1);
  TENSOR(res3a_weight_2);
  TENSOR(res3a_weight_b);
  TENSOR(res3b_bias_0);
  TENSOR(res3b_bias_1);
  TENSOR(res3b_bias_2);
  TENSOR(res3b_weight_0);
  TENSOR(res3b_weight_1);
  TENSOR(res3b_weight_2);
  TENSOR(res3c_bias_0);
  TENSOR(res3c_bias_1);
  TENSOR(res3c_bias_2);
  TENSOR(res3c_weight_0);
  TENSOR(res3c_weight_1);
  TENSOR(res3c_weight_2);
  TENSOR(res3d_bias_0);
  TENSOR(res3d_bias_1);
  TENSOR(res3d_bias_2);
  TENSOR(res3d_weight_0);
  TENSOR(res3d_weight_1);
  TENSOR(res3d_weight_2);
  TENSOR(res4a_bias_0);
  TENSOR(res4a_bias_1);
  TENSOR(res4a_bias_2);
  TENSOR(res4a_bias_b);
  TENSOR(res4a_weight_0);
  TENSOR(res4a_weight_1);
  TENSOR(res4a_weight_2);
  TENSOR(res4a_weight_b);
  TENSOR(res4b_bias_0);
  TENSOR(res4b_bias_1);
  TENSOR(res4b_bias_2);
  TENSOR(res4b_weight_0);
  TENSOR(res4b_weight_1);
  TENSOR(res4b_weight_2);
  TENSOR(res4c_bias_0);
  TENSOR(res4c_bias_1);
  TENSOR(res4c_bias_2);
  TENSOR(res4c_weight_0);
  TENSOR(res4c_weight_1);
  TENSOR(res4c_weight_2);
  TENSOR(res4d_bias_0);
  TENSOR(res4d_bias_1);
  TENSOR(res4d_bias_2);
  TENSOR(res4d_weight_0);
  TENSOR(res4d_weight_1);
  TENSOR(res4d_weight_2);
  TENSOR(res4e_bias_0);
  TENSOR(res4e_bias_1);
  TENSOR(res4e_bias_2);
  TENSOR(res4e_weight_0);
  TENSOR(res4e_weight_1);
  TENSOR(res4e_weight_2);
  TENSOR(res4f_bias_0);
  TENSOR(res4f_bias_1);
  TENSOR(res4f_bias_2);
  TENSOR(res4f_weight_0);
  TENSOR(res4f_weight_1);
  TENSOR(res4f_weight_2);
  TENSOR(res5a_bias_0);
  TENSOR(res5a_bias_1);
  TENSOR(res5a_bias_2);
  TENSOR(res5a_bias_b);
  TENSOR(res5a_weight_0);
  TENSOR(res5a_weight_1);
  TENSOR(res5a_weight_2);
  TENSOR(res5a_weight_b);
  TENSOR(res5b_bias_0);
  TENSOR(res5b_bias_1);
  TENSOR(res5b_bias_2);
  TENSOR(res5b_weight_0);
  TENSOR(res5b_weight_1);
  TENSOR(res5b_weight_2);
  TENSOR(res5c_bias_0);
  TENSOR(res5c_bias_1);
  TENSOR(res5c_bias_2);
  TENSOR(res5c_weight_0);
  TENSOR(res5c_weight_1);
  TENSOR(res5c_weight_2);

  // init kernel
  SC_INIT_RN50_BACKBONE();

  using namespace std::chrono;
  auto start_time = steady_clock::now();
  const int times = 100;
  for (int i = 0; i < times; i++) {
    RN50_BACKBONE(
        backbone_output.data(), backbone_input.data(), res2a_weight_b.data(),
        res2a_bias_b.data(), res2a_weight_0.data(), res2a_bias_0.data(),
        res2a_weight_1.data(), res2a_bias_1.data(), res2a_weight_2.data(),
        res2a_bias_2.data(), res2b_weight_0.data(), res2b_bias_0.data(),
        res2b_weight_1.data(), res2b_bias_1.data(), res2b_weight_2.data(),
        res2b_bias_2.data(), res2c_weight_0.data(), res2c_bias_0.data(),
        res2c_weight_1.data(), res2c_bias_1.data(), res2c_weight_2.data(),
        res2c_bias_2.data(), res3a_weight_b.data(), res3a_bias_b.data(),
        res3a_weight_0.data(), res3a_bias_0.data(), res3a_weight_1.data(),
        res3a_bias_1.data(), res3a_weight_2.data(), res3a_bias_2.data(),
        res3b_weight_0.data(), res3b_bias_0.data(), res3b_weight_1.data(),
        res3b_bias_1.data(), res3b_weight_2.data(), res3b_bias_2.data(),
        res3c_weight_0.data(), res3c_bias_0.data(), res3c_weight_1.data(),
        res3c_bias_1.data(), res3c_weight_2.data(), res3c_bias_2.data(),
        res3d_weight_0.data(), res3d_bias_0.data(), res3d_weight_1.data(),
        res3d_bias_1.data(), res3d_weight_2.data(), res3d_bias_2.data(),
        res4a_weight_b.data(), res4a_bias_b.data(), res4a_weight_0.data(),
        res4a_bias_0.data(), res4a_weight_1.data(), res4a_bias_1.data(),
        res4a_weight_2.data(), res4a_bias_2.data(), res4b_weight_0.data(),
        res4b_bias_0.data(), res4b_weight_1.data(), res4b_bias_1.data(),
        res4b_weight_2.data(), res4b_bias_2.data(), res4c_weight_0.data(),
        res4c_bias_0.data(), res4c_weight_1.data(), res4c_bias_1.data(),
        res4c_weight_2.data(), res4c_bias_2.data(), res4d_weight_0.data(),
        res4d_bias_0.data(), res4d_weight_1.data(), res4d_bias_1.data(),
        res4d_weight_2.data(), res4d_bias_2.data(), res4e_weight_0.data(),
        res4e_bias_0.data(), res4e_weight_1.data(), res4e_bias_1.data(),
        res4e_weight_2.data(), res4e_bias_2.data(), res4f_weight_0.data(),
        res4f_bias_0.data(), res4f_weight_1.data(), res4f_bias_1.data(),
        res4f_weight_2.data(), res4f_bias_2.data(), res5a_weight_b.data(),
        res5a_bias_b.data(), res5a_weight_0.data(), res5a_bias_0.data(),
        res5a_weight_1.data(), res5a_bias_1.data(), res5a_weight_2.data(),
        res5a_bias_2.data(), res5b_weight_0.data(), res5b_bias_0.data(),
        res5b_weight_1.data(), res5b_bias_1.data(), res5b_weight_2.data(),
        res5b_bias_2.data(), res5c_weight_0.data(), res5c_bias_0.data(),
        res5c_weight_1.data(), res5c_bias_1.data(), res5c_weight_2.data(),
        res5c_bias_2.data());
  }
  printf("Done rn50 backbone_bs4 in %.3fms\n",
         1.f *
             duration_cast<microseconds>(steady_clock::now() - start_time)
                 .count() /
             (1000 * times));

  return 0;
}
