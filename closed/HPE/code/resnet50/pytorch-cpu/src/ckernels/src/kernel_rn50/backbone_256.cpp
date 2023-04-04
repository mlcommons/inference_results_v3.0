
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t rn50_backbone_bs256_data[218688];
static constexpr int8_t* __module_data = rn50_backbone_bs256_data;
alignas(64) static int8_t __uninitialized_data[23657544UL];

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106)));
static bool batchwise_256_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)));
static bool batchwise_128_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59)));
static bool res5a_conv_b_cast_mul_add_cast__683(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_0_cast_mul_add_cast_relu_reorder__682(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_1_cast_mul_add_cast_relu_reorder__681(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_2_cast_mul_add_cast_add_relu__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5b_conv_0_cast_mul_add_cast_relu__679(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_1_cast_mul_add_cast_relu_reorder__678(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_2_cast_mul_add_cast_add_relu__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5c_conv_0_cast_mul_add_cast_relu_reorder__676(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_1_cast_mul_add_cast_relu__675(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2a_conv_b_cast_mul_add_cast__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_0_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_1_cast_mul_add_cast_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_2_cast_mul_add_cast_add_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3b_conv_0_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_1_cast_mul_add_cast_relu__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_2_cast_mul_add_cast_add_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3c_conv_0_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_1_cast_mul_add_cast_relu__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_2_cast_mul_add_cast_add_relu__80(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3d_conv_0_cast_mul_add_cast_relu__84(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_1_cast_mul_add_cast_relu__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_2_cast_mul_add_cast_add_relu__93(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
extern "C" void* memset(void* ptr, int32_t v, uint64_t len) noexcept;
static bool res4a_conv_b_cast_mul_add_cast__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4d_conv_0_cast_mul_add_cast_relu__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_1_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_2_cast_mul_add_cast_add_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4e_conv_0_cast_mul_add_cast_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_1_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_2_cast_mul_add_cast_add_relu__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4f_conv_0_cast_mul_add_cast_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_1_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_2_cast_mul_add_cast_add_relu__77(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__419(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__429(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__461(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__464(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__497(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__500(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__503(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__506(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__509(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__512(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__515(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__537(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__540(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__543(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__546(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__549(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__552(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__555(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__558(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__430(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__468(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__471(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__480(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__483(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__516(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__519(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__525(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__528(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__465(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__474(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__477(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__522(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__531(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__534(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__673(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__110(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__111(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__421(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__107(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__108(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__418(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__116(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__117(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__125(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__126(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__428(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__134(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__135(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__119(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__120(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__128(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__129(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__431(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__140(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__141(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__113(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__114(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__122(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__123(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__427(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__131(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__132(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__432(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__146(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__147(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__155(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__156(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__164(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__165(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__463(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__173(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__174(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__472(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__149(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__150(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__158(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__159(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__167(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__168(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__466(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__137(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__138(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__179(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__180(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__478(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__143(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__144(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__152(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__153(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__161(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__162(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__170(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__171(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__469(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__185(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__186(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__194(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__195(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__203(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__204(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__502(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__212(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__213(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__511(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__221(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__222(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__520(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__230(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__231(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__529(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__188(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__189(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__197(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__198(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__206(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__207(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__505(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__215(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__216(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__514(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__224(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__225(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__523(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__176(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__177(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__475(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__236(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__237(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__535(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__182(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__183(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__481(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__191(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__192(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__200(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__201(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__499(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__209(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__210(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__508(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__218(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__219(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__517(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__227(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__228(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__526(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__242(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__243(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__539(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__251(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__252(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__548(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__260(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__261(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__557(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__245(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__246(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__542(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__254(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__255(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__551(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__233(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__234(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__532(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__239(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__240(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__536(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__248(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__249(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__545(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__257(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__258(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__554(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));


extern "C" void rn50_backbone_bs256(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[216064UL];
  float* folded_const_162 = (float*)&__uninitialized_data[1536UL];
  float* folded_const_214 = (float*)&__uninitialized_data[107520UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[211968UL];
  float* folded_const_163 = (float*)&__uninitialized_data[2560UL];
  float* folded_const_156 = (float*)&__uninitialized_data[0UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[347136UL];
  float* folded_const_164 = (float*)&__uninitialized_data[2816UL];
  float* folded_const_157 = (float*)&__uninitialized_data[256UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[232448UL];
  float* folded_const_165 = (float*)&__uninitialized_data[3072UL];
  float* folded_const_215 = (float*)&__uninitialized_data[108544UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[281600UL];
  float* folded_const_166 = (float*)&__uninitialized_data[4096UL];
  float* folded_const_158 = (float*)&__uninitialized_data[512UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[384000UL];
  float* folded_const_167 = (float*)&__uninitialized_data[4352UL];
  float* folded_const_159 = (float*)&__uninitialized_data[768UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[248832UL];
  float* folded_const_168 = (float*)&__uninitialized_data[4608UL];
  float* folded_const_216 = (float*)&__uninitialized_data[109568UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[297984UL];
  float* folded_const_169 = (float*)&__uninitialized_data[5632UL];
  float* folded_const_160 = (float*)&__uninitialized_data[1024UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[420864UL];
  float* folded_const_170 = (float*)&__uninitialized_data[5888UL];
  float* folded_const_161 = (float*)&__uninitialized_data[1280UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[265216UL];
  float* folded_const_171 = (float*)&__uninitialized_data[6144UL];
  float* folded_const_217 = (float*)&__uninitialized_data[110592UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[916480UL];
  float* folded_const_172 = (float*)&__uninitialized_data[7168UL];
  float* folded_const_238 = (float*)&__uninitialized_data[128000UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[314368UL];
  float* folded_const_173 = (float*)&__uninitialized_data[9216UL];
  float* folded_const_218 = (float*)&__uninitialized_data[111616UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[1178624UL];
  float* folded_const_174 = (float*)&__uninitialized_data[9728UL];
  float* folded_const_219 = (float*)&__uninitialized_data[112128UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[457728UL];
  float* folded_const_175 = (float*)&__uninitialized_data[10240UL];
  float* folded_const_239 = (float*)&__uninitialized_data[130048UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[719872UL];
  float* folded_const_176 = (float*)&__uninitialized_data[12288UL];
  float* folded_const_220 = (float*)&__uninitialized_data[112640UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[1326080UL];
  float* folded_const_177 = (float*)&__uninitialized_data[12800UL];
  float* folded_const_221 = (float*)&__uninitialized_data[113152UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[523264UL];
  float* folded_const_178 = (float*)&__uninitialized_data[13312UL];
  float* folded_const_240 = (float*)&__uninitialized_data[132096UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[785408UL];
  float* folded_const_179 = (float*)&__uninitialized_data[15360UL];
  float* folded_const_222 = (float*)&__uninitialized_data[113664UL];
  int8_t* folded_const_282 = (int8_t*)&__uninitialized_data[1473536UL];
  float* folded_const_180 = (float*)&__uninitialized_data[15872UL];
  float* folded_const_223 = (float*)&__uninitialized_data[114176UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[588800UL];
  float* folded_const_181 = (float*)&__uninitialized_data[16384UL];
  float* folded_const_241 = (float*)&__uninitialized_data[134144UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[850944UL];
  float* folded_const_182 = (float*)&__uninitialized_data[18432UL];
  float* folded_const_224 = (float*)&__uninitialized_data[114688UL];
  int8_t* folded_const_283 = (int8_t*)&__uninitialized_data[1620992UL];
  float* folded_const_183 = (float*)&__uninitialized_data[18944UL];
  float* folded_const_225 = (float*)&__uninitialized_data[115200UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[654336UL];
  float* folded_const_184 = (float*)&__uninitialized_data[19456UL];
  float* folded_const_242 = (float*)&__uninitialized_data[136192UL];
  int8_t* folded_const_295 = (int8_t*)&__uninitialized_data[4652032UL];
  float* folded_const_185 = (float*)&__uninitialized_data[21504UL];
  float* folded_const_249 = (float*)&__uninitialized_data[150528UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[1047552UL];
  float* folded_const_186 = (float*)&__uninitialized_data[25600UL];
  float* folded_const_226 = (float*)&__uninitialized_data[115712UL];
  int8_t* folded_const_297 = (int8_t*)&__uninitialized_data[5700608UL];
  float* folded_const_187 = (float*)&__uninitialized_data[26624UL];
  float* folded_const_227 = (float*)&__uninitialized_data[116736UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[1768448UL];
  float* folded_const_188 = (float*)&__uninitialized_data[27648UL];
  float* folded_const_250 = (float*)&__uninitialized_data[154624UL];
  int8_t* folded_const_290 = (int8_t*)&__uninitialized_data[3341312UL];
  float* folded_const_189 = (float*)&__uninitialized_data[31744UL];
  float* folded_const_228 = (float*)&__uninitialized_data[117760UL];
  int8_t* folded_const_298 = (int8_t*)&__uninitialized_data[6290432UL];
  float* folded_const_190 = (float*)&__uninitialized_data[32768UL];
  float* folded_const_229 = (float*)&__uninitialized_data[118784UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[2030592UL];
  float* folded_const_191 = (float*)&__uninitialized_data[33792UL];
  float* folded_const_251 = (float*)&__uninitialized_data[158720UL];
  int8_t* folded_const_291 = (int8_t*)&__uninitialized_data[3603456UL];
  float* folded_const_192 = (float*)&__uninitialized_data[37888UL];
  float* folded_const_230 = (float*)&__uninitialized_data[119808UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[6880256UL];
  float* folded_const_193 = (float*)&__uninitialized_data[38912UL];
  float* folded_const_231 = (float*)&__uninitialized_data[120832UL];
  int8_t* folded_const_286 = (int8_t*)&__uninitialized_data[2292736UL];
  float* folded_const_194 = (float*)&__uninitialized_data[39936UL];
  float* folded_const_252 = (float*)&__uninitialized_data[162816UL];
  int8_t* folded_const_292 = (int8_t*)&__uninitialized_data[3865600UL];
  float* folded_const_195 = (float*)&__uninitialized_data[44032UL];
  float* folded_const_232 = (float*)&__uninitialized_data[121856UL];
  int8_t* folded_const_300 = (int8_t*)&__uninitialized_data[7470080UL];
  float* folded_const_196 = (float*)&__uninitialized_data[45056UL];
  float* folded_const_233 = (float*)&__uninitialized_data[122880UL];
  int8_t* folded_const_287 = (int8_t*)&__uninitialized_data[2554880UL];
  float* folded_const_197 = (float*)&__uninitialized_data[46080UL];
  float* folded_const_253 = (float*)&__uninitialized_data[166912UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[4127744UL];
  float* folded_const_198 = (float*)&__uninitialized_data[50176UL];
  float* folded_const_234 = (float*)&__uninitialized_data[123904UL];
  int8_t* folded_const_301 = (int8_t*)&__uninitialized_data[8059904UL];
  float* folded_const_199 = (float*)&__uninitialized_data[51200UL];
  float* folded_const_235 = (float*)&__uninitialized_data[124928UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[2817024UL];
  float* folded_const_200 = (float*)&__uninitialized_data[52224UL];
  float* folded_const_254 = (float*)&__uninitialized_data[171008UL];
  int8_t* folded_const_294 = (int8_t*)&__uninitialized_data[4389888UL];
  float* folded_const_201 = (float*)&__uninitialized_data[56320UL];
  float* folded_const_236 = (float*)&__uninitialized_data[125952UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[8649728UL];
  float* folded_const_202 = (float*)&__uninitialized_data[57344UL];
  float* folded_const_237 = (float*)&__uninitialized_data[126976UL];
  int8_t* folded_const_289 = (int8_t*)&__uninitialized_data[3079168UL];
  float* folded_const_203 = (float*)&__uninitialized_data[58368UL];
  float* folded_const_255 = (float*)&__uninitialized_data[175104UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[14482432UL];
  float* folded_const_204 = (float*)&__uninitialized_data[62464UL];
  float* folded_const_256 = (float*)&__uninitialized_data[179200UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[5176320UL];
  float* folded_const_205 = (float*)&__uninitialized_data[70656UL];
  float* folded_const_243 = (float*)&__uninitialized_data[138240UL];
  int8_t* folded_const_309 = (int8_t*)&__uninitialized_data[16579584UL];
  float* folded_const_206 = (float*)&__uninitialized_data[72704UL];
  float* folded_const_244 = (float*)&__uninitialized_data[140288UL];
  int8_t* folded_const_303 = (int8_t*)&__uninitialized_data[9239552UL];
  float* folded_const_207 = (float*)&__uninitialized_data[74752UL];
  float* folded_const_257 = (float*)&__uninitialized_data[187392UL];
  int8_t* folded_const_306 = (int8_t*)&__uninitialized_data[12385280UL];
  float* folded_const_208 = (float*)&__uninitialized_data[82944UL];
  float* folded_const_245 = (float*)&__uninitialized_data[142336UL];
  int8_t* folded_const_310 = (int8_t*)&__uninitialized_data[18938880UL];
  float* folded_const_209 = (float*)&__uninitialized_data[84992UL];
  float* folded_const_246 = (float*)&__uninitialized_data[144384UL];
  int8_t* folded_const_304 = (int8_t*)&__uninitialized_data[10288128UL];
  float* folded_const_210 = (float*)&__uninitialized_data[87040UL];
  float* folded_const_258 = (float*)&__uninitialized_data[195584UL];
  int8_t* folded_const_307 = (int8_t*)&__uninitialized_data[13433856UL];
  float* folded_const_211 = (float*)&__uninitialized_data[95232UL];
  float* folded_const_247 = (float*)&__uninitialized_data[146432UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[21298176UL];
  float* folded_const_212 = (float*)&__uninitialized_data[97280UL];
  float* folded_const_248 = (float*)&__uninitialized_data[148480UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[11336704UL];
  float* folded_const_213 = (float*)&__uninitialized_data[99328UL];
  float* folded_const_259 = (float*)&__uninitialized_data[203776UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 154140672UL);
  if (!is_init) {
    __init_const_globals(backbone_output, backbone_input, res2a_weight_b, res2a_bias_b, res2a_weight_0, res2a_bias_0, res2a_weight_1, res2a_bias_1, res2a_weight_2, res2a_bias_2, res2b_weight_0, res2b_bias_0, res2b_weight_1, res2b_bias_1, res2b_weight_2, res2b_bias_2, res2c_weight_0, res2c_bias_0, res2c_weight_1, res2c_bias_1, res2c_weight_2, res2c_bias_2, res3a_weight_b, res3a_bias_b, res3a_weight_0, res3a_bias_0, res3a_weight_1, res3a_bias_1, res3a_weight_2, res3a_bias_2, res3b_weight_0, res3b_bias_0, res3b_weight_1, res3b_bias_1, res3b_weight_2, res3b_bias_2, res3c_weight_0, res3c_bias_0, res3c_weight_1, res3c_bias_1, res3c_weight_2, res3c_bias_2, res3d_weight_0, res3d_bias_0, res3d_weight_1, res3d_bias_1, res3d_weight_2, res3d_bias_2, res4a_weight_b, res4a_bias_b, res4a_weight_0, res4a_bias_0, res4a_weight_1, res4a_bias_1, res4a_weight_2, res4a_bias_2, res4b_weight_0, res4b_bias_0, res4b_weight_1, res4b_bias_1, res4b_weight_2, res4b_bias_2, res4c_weight_0, res4c_bias_0, res4c_weight_1, res4c_bias_1, res4c_weight_2, res4c_bias_2, res4d_weight_0, res4d_bias_0, res4d_weight_1, res4d_bias_1, res4d_weight_2, res4d_bias_2, res4e_weight_0, res4e_bias_0, res4e_weight_1, res4e_bias_1, res4e_weight_2, res4e_bias_2, res4f_weight_0, res4f_bias_0, res4f_weight_1, res4f_bias_1, res4f_weight_2, res4f_bias_2, res5a_weight_b, res5a_bias_b, res5a_weight_0, res5a_bias_0, res5a_weight_1, res5a_bias_1, res5a_weight_2, res5a_bias_2, res5b_weight_0, res5b_bias_0, res5b_weight_1, res5b_bias_1, res5b_weight_2, res5b_bias_2, res5c_weight_0, res5c_bias_0, res5c_weight_1, res5c_bias_1, res5c_weight_2, res5c_bias_2);
  }
  // [s8 [256, 1, 8, 28, 28, 64] @ A1aBCD64b]
  int8_t* buffer_611 = (int8_t*)&__rescheduled_0[0UL];
  batchwise_256_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(buffer_611, &backbone_input[0UL], folded_const_261, folded_const_162, folded_const_214, folded_const_260, folded_const_163, folded_const_156, folded_const_268, folded_const_164, folded_const_157, folded_const_262, folded_const_165, folded_const_215, folded_const_265, folded_const_166, folded_const_158, folded_const_269, folded_const_167, folded_const_159, folded_const_263, folded_const_168, folded_const_216, folded_const_266, folded_const_169, folded_const_160, folded_const_270, folded_const_170, folded_const_161, folded_const_264, folded_const_171, folded_const_217, folded_const_278, folded_const_172, folded_const_238, folded_const_267, folded_const_173, folded_const_218, folded_const_280, folded_const_174, folded_const_219, folded_const_271, folded_const_175, folded_const_239, folded_const_275, folded_const_176, folded_const_220, folded_const_281, folded_const_177, folded_const_221, folded_const_272, folded_const_178, folded_const_240, folded_const_276, folded_const_179, folded_const_222, folded_const_282, folded_const_180, folded_const_223, folded_const_273, folded_const_181, folded_const_241, folded_const_277, folded_const_182, folded_const_224, folded_const_283, folded_const_183, folded_const_225, folded_const_274, folded_const_184, folded_const_242);
  // [s8 [128, 2, 16, 14, 14, 64] @ A2aBCD64b]
  int8_t* buffer_612 = (int8_t*)&__rescheduled_0[102760448UL];
  batchwise_128_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(buffer_612, &buffer_611[0UL], folded_const_295, folded_const_185, folded_const_249, folded_const_279, folded_const_186, folded_const_226, folded_const_297, folded_const_187, folded_const_227, folded_const_284, folded_const_188, folded_const_250, folded_const_290, folded_const_189, folded_const_228, folded_const_298, folded_const_190, folded_const_229, folded_const_285, folded_const_191, folded_const_251, folded_const_291, folded_const_192, folded_const_230, folded_const_299, folded_const_193, folded_const_231, folded_const_286, folded_const_194, folded_const_252, folded_const_292, folded_const_195, folded_const_232, folded_const_300, folded_const_196, folded_const_233, folded_const_287, folded_const_197, folded_const_253, folded_const_293, folded_const_198, folded_const_234, folded_const_301, folded_const_199, folded_const_235, folded_const_288, folded_const_200, folded_const_254, folded_const_294, folded_const_201, folded_const_236, folded_const_302, folded_const_202, folded_const_237, folded_const_289, folded_const_203, folded_const_255);
  // [s8 [256, 1, 4, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_613 = (int8_t*)&__rescheduled_0[0UL];
  res5a_conv_b_cast_mul_add_cast__683(buffer_613, &buffer_612[0UL], folded_const_308, folded_const_204, folded_const_256);
  // [s8 [256, 1, 8, 16, 16, 64] @ A1aBCD64b]
  int8_t* buffer_614 = (int8_t*)&__rescheduled_0[42467328UL];
  res5a_conv_0_cast_mul_add_cast_relu_reorder__682(buffer_614, &buffer_612[0UL], folded_const_296, folded_const_205, folded_const_243);
  // [s8 [256, 1, 1, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_615 = (int8_t*)&__rescheduled_0[76021760UL];
  res5a_conv_1_cast_mul_add_cast_relu_reorder__681(buffer_615, buffer_614, folded_const_309, folded_const_206, folded_const_244);
  res5a_conv_2_cast_mul_add_cast_add_relu__680(buffer_613, buffer_615, folded_const_303, folded_const_207, folded_const_257, buffer_613);
  res5b_conv_0_cast_mul_add_cast_relu__679(buffer_615, buffer_613, folded_const_306, folded_const_208, folded_const_245);
  res5b_conv_1_cast_mul_add_cast_relu_reorder__678(buffer_614, buffer_615, folded_const_310, folded_const_209, folded_const_246);
  res5b_conv_2_cast_mul_add_cast_add_relu__677(buffer_613, buffer_614, folded_const_304, folded_const_210, folded_const_258, buffer_613);
  res5c_conv_0_cast_mul_add_cast_relu_reorder__676(buffer_615, buffer_613, folded_const_307, folded_const_211, folded_const_247);
  res5c_conv_1_cast_mul_add_cast_relu__675(buffer_614, buffer_615, folded_const_311, folded_const_212, folded_const_248);
  res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(backbone_output, buffer_614, folded_const_305, folded_const_213, folded_const_259, buffer_613);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_5 = 0UL; _fuseiter_5 < 64UL; _fuseiter_5 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_5]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_5]);
  }
  return true;
}

static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_12 = 0UL; _fuseiter_12 < 64UL; _fuseiter_12 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_12]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_12]);
  }
  return true;
}

static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_19 = 0UL; _fuseiter_19 < 64UL; _fuseiter_19 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_19]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_19]);
  }
  return true;
}

static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_26 = 0UL; _fuseiter_26 < 64UL; _fuseiter_26 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_26]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_26]);
  }
  return true;
}

static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_33 = 0UL; _fuseiter_33 < 64UL; _fuseiter_33 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_33]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_33]);
  }
  return true;
}

static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_40 = 0UL; _fuseiter_40 < 64UL; _fuseiter_40 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_40]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_40]);
  }
  return true;
}

static bool reorder__419(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 += 1UL) {
    for (uint64_t _fuseiter_47 = 0UL; _fuseiter_47 < 64UL; _fuseiter_47 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 / 4UL) * 256UL) + (_fuseiter_47 + ((fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_42___fuseiter_43_24___fuseiter_44_25___fuseiter_45_26___fuseiter_46_27 % 4UL) * 64UL) + _fuseiter_47))] = __cached_1;
    }
  }
  return true;
}

static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 += 1UL) {
    for (uint64_t _fuseiter_53 = 0UL; _fuseiter_53 < 64UL; _fuseiter_53 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 % 4UL) * 64UL)) + _fuseiter_53)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_28____itr_2_29____itr_3_30____itr_4_31 % 4UL) * 64UL)) + _fuseiter_53)]);
    }
  }
  return true;
}

static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_60 = 0UL; _fuseiter_60 < 64UL; _fuseiter_60 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_60]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_60]);
  }
  return true;
}

static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_67 = 0UL; _fuseiter_67 < 64UL; _fuseiter_67 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_67]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_67]);
  }
  return true;
}

static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 += 1UL) {
    for (uint64_t _fuseiter_74 = 0UL; _fuseiter_74 < 64UL; _fuseiter_74 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 / 4UL) * 256UL) + (_fuseiter_74 + ((fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_69___fuseiter_70_40___fuseiter_71_41___fuseiter_72_42___fuseiter_73_43 % 4UL) * 64UL) + _fuseiter_74))] = __cached_1;
    }
  }
  return true;
}

static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 += 1UL) {
    for (uint64_t _fuseiter_80 = 0UL; _fuseiter_80 < 64UL; _fuseiter_80 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 % 4UL) * 64UL)) + _fuseiter_80)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_44____itr_2_45____itr_3_46____itr_4_47 % 4UL) * 64UL)) + _fuseiter_80)]);
    }
  }
  return true;
}

static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_87 = 0UL; _fuseiter_87 < 64UL; _fuseiter_87 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_87]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_87]);
  }
  return true;
}

static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_94 = 0UL; _fuseiter_94 < 64UL; _fuseiter_94 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_94]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_94]);
  }
  return true;
}

static bool reorder__429(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 += 1UL) {
    for (uint64_t _fuseiter_101 = 0UL; _fuseiter_101 < 64UL; _fuseiter_101 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 / 4UL) * 256UL) + (_fuseiter_101 + ((fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_96___fuseiter_97_56___fuseiter_98_57___fuseiter_99_58___fuseiter_100_59 % 4UL) * 64UL) + _fuseiter_101))] = __cached_1;
    }
  }
  return true;
}

static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 += 1UL) {
    for (uint64_t _fuseiter_107 = 0UL; _fuseiter_107 < 64UL; _fuseiter_107 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 % 4UL) * 64UL)) + _fuseiter_107)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_60____itr_2_61____itr_3_62____itr_4_63 % 4UL) * 64UL)) + _fuseiter_107)]);
    }
  }
  return true;
}

static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_114 = 0UL; _fuseiter_114 < 64UL; _fuseiter_114 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_114]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_114]);
  }
  return true;
}

static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_121 = 0UL; _fuseiter_121 < 64UL; _fuseiter_121 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_121]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_121]);
  }
  return true;
}

static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 += 1UL) {
    for (uint64_t _fuseiter_128 = 0UL; _fuseiter_128 < 64UL; _fuseiter_128 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 / 4UL) * 256UL) + (_fuseiter_128 + ((fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_123___fuseiter_124_72___fuseiter_125_73___fuseiter_126_74___fuseiter_127_75 % 4UL) * 64UL) + _fuseiter_128))] = __cached_1;
    }
  }
  return true;
}

static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 += 1UL) {
    for (uint64_t _fuseiter_134 = 0UL; _fuseiter_134 < 64UL; _fuseiter_134 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 % 4UL) * 64UL)) + _fuseiter_134)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_76____itr_2_77____itr_3_78____itr_4_79 % 4UL) * 64UL)) + _fuseiter_134)]);
    }
  }
  return true;
}

static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 += 1UL) {
    for (uint64_t _fuseiter_141 = 0UL; _fuseiter_141 < 64UL; _fuseiter_141 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 / 8UL) * 512UL) + (_fuseiter_141 + ((fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_136___fuseiter_137_80___fuseiter_138_81___fuseiter_139_82___fuseiter_140_83 % 8UL) * 64UL) + _fuseiter_141))] = __cached_1;
    }
  }
  return true;
}

static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 += 1UL) {
    for (uint64_t _fuseiter_147 = 0UL; _fuseiter_147 < 64UL; _fuseiter_147 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 % 8UL) * 64UL)) + _fuseiter_147)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_84____itr_2_85____itr_3_86____itr_4_87 % 8UL) * 64UL)) + _fuseiter_147)]);
    }
  }
  return true;
}

static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 += 1UL) {
    for (uint64_t _fuseiter_154 = 0UL; _fuseiter_154 < 64UL; _fuseiter_154 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 / 2UL) * 128UL) + (_fuseiter_154 + ((fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_149___fuseiter_150_88___fuseiter_151_89___fuseiter_152_90___fuseiter_153_91 % 2UL) * 64UL) + _fuseiter_154))] = __cached_1;
    }
  }
  return true;
}

static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 += 1UL) {
    for (uint64_t _fuseiter_160 = 0UL; _fuseiter_160 < 64UL; _fuseiter_160 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 % 2UL) * 64UL)) + _fuseiter_160)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_92____itr_2_93____itr_3_94____itr_4_95 % 2UL) * 64UL)) + _fuseiter_160)]);
    }
  }
  return true;
}

static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 += 1UL) {
    for (uint64_t _fuseiter_167 = 0UL; _fuseiter_167 < 64UL; _fuseiter_167 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 / 2UL) * 128UL) + (_fuseiter_167 + ((fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_162___fuseiter_163_96___fuseiter_164_97___fuseiter_165_98___fuseiter_166_99 % 2UL) * 64UL) + _fuseiter_167))] = __cached_1;
    }
  }
  return true;
}

static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 += 1UL) {
    for (uint64_t _fuseiter_173 = 0UL; _fuseiter_173 < 64UL; _fuseiter_173 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 % 2UL) * 64UL)) + _fuseiter_173)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_100____itr_2_101____itr_3_102____itr_4_103 % 2UL) * 64UL)) + _fuseiter_173)]);
    }
  }
  return true;
}

static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 += 1UL) {
    for (uint64_t _fuseiter_180 = 0UL; _fuseiter_180 < 64UL; _fuseiter_180 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 / 8UL) * 512UL) + (_fuseiter_180 + ((fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_175___fuseiter_176_104___fuseiter_177_105___fuseiter_178_106___fuseiter_179_107 % 8UL) * 64UL) + _fuseiter_180))] = __cached_1;
    }
  }
  return true;
}

static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 += 1UL) {
    for (uint64_t _fuseiter_186 = 0UL; _fuseiter_186 < 64UL; _fuseiter_186 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 % 8UL) * 64UL)) + _fuseiter_186)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_108____itr_2_109____itr_3_110____itr_4_111 % 8UL) * 64UL)) + _fuseiter_186)]);
    }
  }
  return true;
}

static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 += 1UL) {
    for (uint64_t _fuseiter_193 = 0UL; _fuseiter_193 < 64UL; _fuseiter_193 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 / 2UL) * 128UL) + (_fuseiter_193 + ((fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_188___fuseiter_189_112___fuseiter_190_113___fuseiter_191_114___fuseiter_192_115 % 2UL) * 64UL) + _fuseiter_193))] = __cached_1;
    }
  }
  return true;
}

static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 += 1UL) {
    for (uint64_t _fuseiter_199 = 0UL; _fuseiter_199 < 64UL; _fuseiter_199 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 % 2UL) * 64UL)) + _fuseiter_199)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_116____itr_2_117____itr_3_118____itr_4_119 % 2UL) * 64UL)) + _fuseiter_199)]);
    }
  }
  return true;
}

static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 += 1UL) {
    for (uint64_t _fuseiter_206 = 0UL; _fuseiter_206 < 64UL; _fuseiter_206 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 / 2UL) * 128UL) + (_fuseiter_206 + ((fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_201___fuseiter_202_120___fuseiter_203_121___fuseiter_204_122___fuseiter_205_123 % 2UL) * 64UL) + _fuseiter_206))] = __cached_1;
    }
  }
  return true;
}

static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 += 1UL) {
    for (uint64_t _fuseiter_212 = 0UL; _fuseiter_212 < 64UL; _fuseiter_212 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 % 2UL) * 64UL)) + _fuseiter_212)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_124____itr_2_125____itr_3_126____itr_4_127 % 2UL) * 64UL)) + _fuseiter_212)]);
    }
  }
  return true;
}

static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 += 1UL) {
    for (uint64_t _fuseiter_219 = 0UL; _fuseiter_219 < 64UL; _fuseiter_219 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 / 8UL) * 512UL) + (_fuseiter_219 + ((fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_214___fuseiter_215_128___fuseiter_216_129___fuseiter_217_130___fuseiter_218_131 % 8UL) * 64UL) + _fuseiter_219))] = __cached_1;
    }
  }
  return true;
}

static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 += 1UL) {
    for (uint64_t _fuseiter_225 = 0UL; _fuseiter_225 < 64UL; _fuseiter_225 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 % 8UL) * 64UL)) + _fuseiter_225)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_132____itr_2_133____itr_3_134____itr_4_135 % 8UL) * 64UL)) + _fuseiter_225)]);
    }
  }
  return true;
}

static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 += 1UL) {
    for (uint64_t _fuseiter_232 = 0UL; _fuseiter_232 < 64UL; _fuseiter_232 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 / 2UL) * 128UL) + (_fuseiter_232 + ((fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_227___fuseiter_228_136___fuseiter_229_137___fuseiter_230_138___fuseiter_231_139 % 2UL) * 64UL) + _fuseiter_232))] = __cached_1;
    }
  }
  return true;
}

static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 += 1UL) {
    for (uint64_t _fuseiter_238 = 0UL; _fuseiter_238 < 64UL; _fuseiter_238 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 % 2UL) * 64UL)) + _fuseiter_238)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_140____itr_2_141____itr_3_142____itr_4_143 % 2UL) * 64UL)) + _fuseiter_238)]);
    }
  }
  return true;
}

static bool reorder__461(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 += 1UL) {
    for (uint64_t _fuseiter_245 = 0UL; _fuseiter_245 < 64UL; _fuseiter_245 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 / 2UL) * 128UL) + (_fuseiter_245 + ((fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_240___fuseiter_241_144___fuseiter_242_145___fuseiter_243_146___fuseiter_244_147 % 2UL) * 64UL) + _fuseiter_245))] = __cached_1;
    }
  }
  return true;
}

static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 += 1UL) {
    for (uint64_t _fuseiter_251 = 0UL; _fuseiter_251 < 64UL; _fuseiter_251 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 % 2UL) * 64UL)) + _fuseiter_251)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_148____itr_2_149____itr_3_150____itr_4_151 % 2UL) * 64UL)) + _fuseiter_251)]);
    }
  }
  return true;
}

static bool reorder__464(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 += 1UL) {
    for (uint64_t _fuseiter_258 = 0UL; _fuseiter_258 < 64UL; _fuseiter_258 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 / 8UL) * 512UL) + (_fuseiter_258 + ((fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_253___fuseiter_254_152___fuseiter_255_153___fuseiter_256_154___fuseiter_257_155 % 8UL) * 64UL) + _fuseiter_258))] = __cached_1;
    }
  }
  return true;
}

static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 += 1UL) {
    for (uint64_t _fuseiter_264 = 0UL; _fuseiter_264 < 64UL; _fuseiter_264 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 % 8UL) * 64UL)) + _fuseiter_264)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_156____itr_2_157____itr_3_158____itr_4_159 % 8UL) * 64UL)) + _fuseiter_264)]);
    }
  }
  return true;
}

static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 += 1UL) {
    for (uint64_t _fuseiter_271 = 0UL; _fuseiter_271 < 64UL; _fuseiter_271 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 / 2UL) * 128UL) + (_fuseiter_271 + ((fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_266___fuseiter_267_160___fuseiter_268_161___fuseiter_269_162___fuseiter_270_163 % 2UL) * 64UL) + _fuseiter_271))] = __cached_1;
    }
  }
  return true;
}

static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 += 1UL) {
    for (uint64_t _fuseiter_277 = 0UL; _fuseiter_277 < 64UL; _fuseiter_277 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 % 2UL) * 64UL)) + _fuseiter_277)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_164____itr_2_165____itr_3_166____itr_4_167 % 2UL) * 64UL)) + _fuseiter_277)]);
    }
  }
  return true;
}

static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 += 1UL) {
    for (uint64_t _fuseiter_284 = 0UL; _fuseiter_284 < 64UL; _fuseiter_284 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 / 2UL) * 128UL) + (_fuseiter_284 + ((fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_279___fuseiter_280_168___fuseiter_281_169___fuseiter_282_170___fuseiter_283_171 % 2UL) * 64UL) + _fuseiter_284))] = __cached_1;
    }
  }
  return true;
}

static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 += 1UL) {
    for (uint64_t _fuseiter_290 = 0UL; _fuseiter_290 < 64UL; _fuseiter_290 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 % 2UL) * 64UL)) + _fuseiter_290)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_172____itr_2_173____itr_3_174____itr_4_175 % 2UL) * 64UL)) + _fuseiter_290)]);
    }
  }
  return true;
}

static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 += 1UL) {
    for (uint64_t _fuseiter_297 = 0UL; _fuseiter_297 < 64UL; _fuseiter_297 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 / 8UL) * 512UL) + (_fuseiter_297 + ((fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_292___fuseiter_293_176___fuseiter_294_177___fuseiter_295_178___fuseiter_296_179 % 8UL) * 64UL) + _fuseiter_297))] = __cached_1;
    }
  }
  return true;
}

static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 += 1UL) {
    for (uint64_t _fuseiter_303 = 0UL; _fuseiter_303 < 64UL; _fuseiter_303 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 % 8UL) * 64UL)) + _fuseiter_303)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_180____itr_2_181____itr_3_182____itr_4_183 % 8UL) * 64UL)) + _fuseiter_303)]);
    }
  }
  return true;
}

static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 = 0UL; fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 < 16UL; fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 += 1UL) {
    for (uint64_t _fuseiter_310 = 0UL; _fuseiter_310 < 64UL; _fuseiter_310 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 / 16UL) * 1024UL) + (_fuseiter_310 + ((fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_305___fuseiter_306_184___fuseiter_307_185 % 16UL) * 64UL) + _fuseiter_310))] = __cached_1;
    }
  }
  return true;
}

static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 += 1UL) {
    for (uint64_t _fuseiter_316 = 0UL; _fuseiter_316 < 64UL; _fuseiter_316 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 % 16UL) * 64UL)) + _fuseiter_316)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_186____itr_2_187____itr_3_188____itr_4_189 % 16UL) * 64UL)) + _fuseiter_316)]);
    }
  }
  return true;
}

static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 += 1UL) {
    for (uint64_t _fuseiter_323 = 0UL; _fuseiter_323 < 64UL; _fuseiter_323 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 / 4UL) * 256UL) + (_fuseiter_323 + ((fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_318___fuseiter_319_190___fuseiter_320_191___fuseiter_321_192___fuseiter_322_193 % 4UL) * 64UL) + _fuseiter_323))] = __cached_1;
    }
  }
  return true;
}

static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 += 1UL) {
    for (uint64_t _fuseiter_329 = 0UL; _fuseiter_329 < 64UL; _fuseiter_329 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 % 4UL) * 64UL)) + _fuseiter_329)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_194____itr_2_195____itr_3_196____itr_4_197 % 4UL) * 64UL)) + _fuseiter_329)]);
    }
  }
  return true;
}

static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 += 1UL) {
    for (uint64_t _fuseiter_336 = 0UL; _fuseiter_336 < 64UL; _fuseiter_336 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 / 4UL) * 256UL) + (_fuseiter_336 + ((fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_331___fuseiter_332_198___fuseiter_333_199___fuseiter_334_200___fuseiter_335_201 % 4UL) * 64UL) + _fuseiter_336))] = __cached_1;
    }
  }
  return true;
}

static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 += 1UL) {
    for (uint64_t _fuseiter_342 = 0UL; _fuseiter_342 < 64UL; _fuseiter_342 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 % 4UL) * 64UL)) + _fuseiter_342)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_202____itr_2_203____itr_3_204____itr_4_205 % 4UL) * 64UL)) + _fuseiter_342)]);
    }
  }
  return true;
}

static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 = 0UL; fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 < 16UL; fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 += 1UL) {
    for (uint64_t _fuseiter_349 = 0UL; _fuseiter_349 < 64UL; _fuseiter_349 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 / 16UL) * 1024UL) + (_fuseiter_349 + ((fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_344___fuseiter_345_206___fuseiter_346_207 % 16UL) * 64UL) + _fuseiter_349))] = __cached_1;
    }
  }
  return true;
}

static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 += 1UL) {
    for (uint64_t _fuseiter_355 = 0UL; _fuseiter_355 < 64UL; _fuseiter_355 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 % 16UL) * 64UL)) + _fuseiter_355)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_208____itr_2_209____itr_3_210____itr_4_211 % 16UL) * 64UL)) + _fuseiter_355)]);
    }
  }
  return true;
}

static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 += 1UL) {
    for (uint64_t _fuseiter_362 = 0UL; _fuseiter_362 < 64UL; _fuseiter_362 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 / 4UL) * 256UL) + (_fuseiter_362 + ((fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_357___fuseiter_358_212___fuseiter_359_213___fuseiter_360_214___fuseiter_361_215 % 4UL) * 64UL) + _fuseiter_362))] = __cached_1;
    }
  }
  return true;
}

static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 += 1UL) {
    for (uint64_t _fuseiter_368 = 0UL; _fuseiter_368 < 64UL; _fuseiter_368 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 % 4UL) * 64UL)) + _fuseiter_368)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_216____itr_2_217____itr_3_218____itr_4_219 % 4UL) * 64UL)) + _fuseiter_368)]);
    }
  }
  return true;
}

static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 += 1UL) {
    for (uint64_t _fuseiter_375 = 0UL; _fuseiter_375 < 64UL; _fuseiter_375 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 / 4UL) * 256UL) + (_fuseiter_375 + ((fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_370___fuseiter_371_220___fuseiter_372_221___fuseiter_373_222___fuseiter_374_223 % 4UL) * 64UL) + _fuseiter_375))] = __cached_1;
    }
  }
  return true;
}

static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 += 1UL) {
    for (uint64_t _fuseiter_381 = 0UL; _fuseiter_381 < 64UL; _fuseiter_381 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 % 4UL) * 64UL)) + _fuseiter_381)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_224____itr_2_225____itr_3_226____itr_4_227 % 4UL) * 64UL)) + _fuseiter_381)]);
    }
  }
  return true;
}

static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 = 0UL; fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 < 16UL; fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 += 1UL) {
    for (uint64_t _fuseiter_388 = 0UL; _fuseiter_388 < 64UL; _fuseiter_388 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 / 16UL) * 1024UL) + (_fuseiter_388 + ((fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_383___fuseiter_384_228___fuseiter_385_229 % 16UL) * 64UL) + _fuseiter_388))] = __cached_1;
    }
  }
  return true;
}

static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 += 1UL) {
    for (uint64_t _fuseiter_394 = 0UL; _fuseiter_394 < 64UL; _fuseiter_394 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 % 16UL) * 64UL)) + _fuseiter_394)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_230____itr_2_231____itr_3_232____itr_4_233 % 16UL) * 64UL)) + _fuseiter_394)]);
    }
  }
  return true;
}

static bool reorder__497(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 += 1UL) {
    for (uint64_t _fuseiter_401 = 0UL; _fuseiter_401 < 64UL; _fuseiter_401 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 / 4UL) * 256UL) + (_fuseiter_401 + ((fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_396___fuseiter_397_234___fuseiter_398_235___fuseiter_399_236___fuseiter_400_237 % 4UL) * 64UL) + _fuseiter_401))] = __cached_1;
    }
  }
  return true;
}

static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 += 1UL) {
    for (uint64_t _fuseiter_407 = 0UL; _fuseiter_407 < 64UL; _fuseiter_407 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 % 4UL) * 64UL)) + _fuseiter_407)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_238____itr_2_239____itr_3_240____itr_4_241 % 4UL) * 64UL)) + _fuseiter_407)]);
    }
  }
  return true;
}

static bool reorder__500(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 += 1UL) {
    for (uint64_t _fuseiter_414 = 0UL; _fuseiter_414 < 64UL; _fuseiter_414 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 / 4UL) * 256UL) + (_fuseiter_414 + ((fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_409___fuseiter_410_242___fuseiter_411_243___fuseiter_412_244___fuseiter_413_245 % 4UL) * 64UL) + _fuseiter_414))] = __cached_1;
    }
  }
  return true;
}

static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 += 1UL) {
    for (uint64_t _fuseiter_420 = 0UL; _fuseiter_420 < 64UL; _fuseiter_420 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 % 4UL) * 64UL)) + _fuseiter_420)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_246____itr_2_247____itr_3_248____itr_4_249 % 4UL) * 64UL)) + _fuseiter_420)]);
    }
  }
  return true;
}

static bool reorder__503(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 = 0UL; fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 < 16UL; fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 += 1UL) {
    for (uint64_t _fuseiter_427 = 0UL; _fuseiter_427 < 64UL; _fuseiter_427 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 / 16UL) * 1024UL) + (_fuseiter_427 + ((fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_422___fuseiter_423_250___fuseiter_424_251 % 16UL) * 64UL) + _fuseiter_427))] = __cached_1;
    }
  }
  return true;
}

static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 += 1UL) {
    for (uint64_t _fuseiter_433 = 0UL; _fuseiter_433 < 64UL; _fuseiter_433 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 % 16UL) * 64UL)) + _fuseiter_433)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_252____itr_2_253____itr_3_254____itr_4_255 % 16UL) * 64UL)) + _fuseiter_433)]);
    }
  }
  return true;
}

static bool reorder__506(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 += 1UL) {
    for (uint64_t _fuseiter_440 = 0UL; _fuseiter_440 < 64UL; _fuseiter_440 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 / 4UL) * 256UL) + (_fuseiter_440 + ((fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_435___fuseiter_436_256___fuseiter_437_257___fuseiter_438_258___fuseiter_439_259 % 4UL) * 64UL) + _fuseiter_440))] = __cached_1;
    }
  }
  return true;
}

static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 += 1UL) {
    for (uint64_t _fuseiter_446 = 0UL; _fuseiter_446 < 64UL; _fuseiter_446 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 % 4UL) * 64UL)) + _fuseiter_446)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_260____itr_2_261____itr_3_262____itr_4_263 % 4UL) * 64UL)) + _fuseiter_446)]);
    }
  }
  return true;
}

static bool reorder__509(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 += 1UL) {
    for (uint64_t _fuseiter_453 = 0UL; _fuseiter_453 < 64UL; _fuseiter_453 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 / 4UL) * 256UL) + (_fuseiter_453 + ((fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_448___fuseiter_449_264___fuseiter_450_265___fuseiter_451_266___fuseiter_452_267 % 4UL) * 64UL) + _fuseiter_453))] = __cached_1;
    }
  }
  return true;
}

static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 += 1UL) {
    for (uint64_t _fuseiter_459 = 0UL; _fuseiter_459 < 64UL; _fuseiter_459 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 % 4UL) * 64UL)) + _fuseiter_459)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_268____itr_2_269____itr_3_270____itr_4_271 % 4UL) * 64UL)) + _fuseiter_459)]);
    }
  }
  return true;
}

static bool reorder__512(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 = 0UL; fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 < 16UL; fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 += 1UL) {
    for (uint64_t _fuseiter_466 = 0UL; _fuseiter_466 < 64UL; _fuseiter_466 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 / 16UL) * 1024UL) + (_fuseiter_466 + ((fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_461___fuseiter_462_272___fuseiter_463_273 % 16UL) * 64UL) + _fuseiter_466))] = __cached_1;
    }
  }
  return true;
}

static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 += 1UL) {
    for (uint64_t _fuseiter_472 = 0UL; _fuseiter_472 < 64UL; _fuseiter_472 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 % 16UL) * 64UL)) + _fuseiter_472)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_274____itr_2_275____itr_3_276____itr_4_277 % 16UL) * 64UL)) + _fuseiter_472)]);
    }
  }
  return true;
}

static bool reorder__515(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 += 1UL) {
    for (uint64_t _fuseiter_479 = 0UL; _fuseiter_479 < 64UL; _fuseiter_479 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 / 4UL) * 256UL) + (_fuseiter_479 + ((fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_474___fuseiter_475_278___fuseiter_476_279___fuseiter_477_280___fuseiter_478_281 % 4UL) * 64UL) + _fuseiter_479))] = __cached_1;
    }
  }
  return true;
}

static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 += 1UL) {
    for (uint64_t _fuseiter_485 = 0UL; _fuseiter_485 < 64UL; _fuseiter_485 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 % 4UL) * 64UL)) + _fuseiter_485)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_282____itr_2_283____itr_3_284____itr_4_285 % 4UL) * 64UL)) + _fuseiter_485)]);
    }
  }
  return true;
}

static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 += 1UL) {
    for (uint64_t _fuseiter_492 = 0UL; _fuseiter_492 < 64UL; _fuseiter_492 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 / 4UL) * 256UL) + (_fuseiter_492 + ((fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_487___fuseiter_488_286___fuseiter_489_287___fuseiter_490_288___fuseiter_491_289 % 4UL) * 64UL) + _fuseiter_492))] = __cached_1;
    }
  }
  return true;
}

static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 += 1UL) {
    for (uint64_t _fuseiter_498 = 0UL; _fuseiter_498 < 64UL; _fuseiter_498 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 % 4UL) * 64UL)) + _fuseiter_498)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_290____itr_2_291____itr_3_292____itr_4_293 % 4UL) * 64UL)) + _fuseiter_498)]);
    }
  }
  return true;
}

static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 = 0UL; fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 < 16UL; fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 += 1UL) {
    for (uint64_t _fuseiter_505 = 0UL; _fuseiter_505 < 64UL; _fuseiter_505 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 / 16UL) * 1024UL) + (_fuseiter_505 + ((fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_500___fuseiter_501_294___fuseiter_502_295 % 16UL) * 64UL) + _fuseiter_505))] = __cached_1;
    }
  }
  return true;
}

static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 += 1UL) {
    for (uint64_t _fuseiter_511 = 0UL; _fuseiter_511 < 64UL; _fuseiter_511 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 % 16UL) * 64UL)) + _fuseiter_511)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_296____itr_2_297____itr_3_298____itr_4_299 % 16UL) * 64UL)) + _fuseiter_511)]);
    }
  }
  return true;
}

static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 += 1UL) {
    for (uint64_t _fuseiter_518 = 0UL; _fuseiter_518 < 64UL; _fuseiter_518 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 / 4UL) * 256UL) + (_fuseiter_518 + ((fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_513___fuseiter_514_300___fuseiter_515_301___fuseiter_516_302___fuseiter_517_303 % 4UL) * 64UL) + _fuseiter_518))] = __cached_1;
    }
  }
  return true;
}

static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 += 1UL) {
    for (uint64_t _fuseiter_524 = 0UL; _fuseiter_524 < 64UL; _fuseiter_524 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 % 4UL) * 64UL)) + _fuseiter_524)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_304____itr_2_305____itr_3_306____itr_4_307 % 4UL) * 64UL)) + _fuseiter_524)]);
    }
  }
  return true;
}

static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 += 1UL) {
    for (uint64_t _fuseiter_531 = 0UL; _fuseiter_531 < 64UL; _fuseiter_531 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 / 4UL) * 256UL) + (_fuseiter_531 + ((fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_526___fuseiter_527_308___fuseiter_528_309___fuseiter_529_310___fuseiter_530_311 % 4UL) * 64UL) + _fuseiter_531))] = __cached_1;
    }
  }
  return true;
}

static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 += 1UL) {
    for (uint64_t _fuseiter_537 = 0UL; _fuseiter_537 < 64UL; _fuseiter_537 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 % 4UL) * 64UL)) + _fuseiter_537)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_312____itr_2_313____itr_3_314____itr_4_315 % 4UL) * 64UL)) + _fuseiter_537)]);
    }
  }
  return true;
}

static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 = 0UL; fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 < 16UL; fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 += 1UL) {
    for (uint64_t _fuseiter_544 = 0UL; _fuseiter_544 < 64UL; _fuseiter_544 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 / 16UL) * 1024UL) + (_fuseiter_544 + ((fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_539___fuseiter_540_316___fuseiter_541_317 % 16UL) * 64UL) + _fuseiter_544))] = __cached_1;
    }
  }
  return true;
}

static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 += 1UL) {
    for (uint64_t _fuseiter_550 = 0UL; _fuseiter_550 < 64UL; _fuseiter_550 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 % 16UL) * 64UL)) + _fuseiter_550)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_318____itr_2_319____itr_3_320____itr_4_321 % 16UL) * 64UL)) + _fuseiter_550)]);
    }
  }
  return true;
}

static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 += 1UL) {
    for (uint64_t _fuseiter_557 = 0UL; _fuseiter_557 < 512UL; _fuseiter_557 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 / 4UL) * 2048UL) + (_fuseiter_557 + ((fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_552___fuseiter_553_322___fuseiter_554_323___fuseiter_555_324___fuseiter_556_325 % 4UL) * 512UL) + _fuseiter_557))] = __cached_1;
    }
  }
  return true;
}

static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 += 1UL) {
    for (uint64_t _fuseiter_563 = 0UL; _fuseiter_563 < 512UL; _fuseiter_563 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 % 4UL) * 512UL)) + _fuseiter_563)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_326____itr_2_327____itr_3_328____itr_4_329 % 4UL) * 512UL)) + _fuseiter_563)]);
    }
  }
  return true;
}

static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_570 = 0UL; _fuseiter_570 < 512UL; _fuseiter_570 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_570]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_570]);
  }
  return true;
}

static bool reorder__537(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 += 1UL) {
    for (uint64_t _fuseiter_577 = 0UL; _fuseiter_577 < 256UL; _fuseiter_577 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 / 2UL) * 512UL) + (_fuseiter_577 + ((fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 % 2UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_572___fuseiter_573_334___fuseiter_574_335___fuseiter_575_336___fuseiter_576_337 % 2UL) * 256UL) + _fuseiter_577))] = __cached_1;
    }
  }
  return true;
}

static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 += 1UL) {
    for (uint64_t _fuseiter_583 = 0UL; _fuseiter_583 < 256UL; _fuseiter_583 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 % 2UL) * 256UL)) + _fuseiter_583)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_338____itr_2_339____itr_3_340____itr_4_341 % 2UL) * 256UL)) + _fuseiter_583)]);
    }
  }
  return true;
}

static bool reorder__540(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 += 1UL) {
    for (uint64_t _fuseiter_590 = 0UL; _fuseiter_590 < 512UL; _fuseiter_590 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 / 4UL) * 2048UL) + (_fuseiter_590 + ((fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_585___fuseiter_586_342___fuseiter_587_343___fuseiter_588_344___fuseiter_589_345 % 4UL) * 512UL) + _fuseiter_590))] = __cached_1;
    }
  }
  return true;
}

static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 += 1UL) {
    for (uint64_t _fuseiter_596 = 0UL; _fuseiter_596 < 512UL; _fuseiter_596 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 % 4UL) * 512UL)) + _fuseiter_596)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_346____itr_2_347____itr_3_348____itr_4_349 % 4UL) * 512UL)) + _fuseiter_596)]);
    }
  }
  return true;
}

static bool reorder__543(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 += 1UL) {
    for (uint64_t _fuseiter_603 = 0UL; _fuseiter_603 < 64UL; _fuseiter_603 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 / 8UL) * 512UL) + (_fuseiter_603 + ((fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_598___fuseiter_599_350___fuseiter_600_351___fuseiter_601_352___fuseiter_602_353 % 8UL) * 64UL) + _fuseiter_603))] = __cached_1;
    }
  }
  return true;
}

static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 += 1UL) {
    for (uint64_t _fuseiter_609 = 0UL; _fuseiter_609 < 64UL; _fuseiter_609 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 % 8UL) * 64UL)) + _fuseiter_609)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_354____itr_2_355____itr_3_356____itr_4_357 % 8UL) * 64UL)) + _fuseiter_609)]);
    }
  }
  return true;
}

static bool reorder__546(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 += 1UL) {
    for (uint64_t _fuseiter_616 = 0UL; _fuseiter_616 < 128UL; _fuseiter_616 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 / 4UL) * 512UL) + (_fuseiter_616 + ((fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 / 4UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_611___fuseiter_612_358___fuseiter_613_359___fuseiter_614_360___fuseiter_615_361 % 4UL) * 128UL) + _fuseiter_616))] = __cached_1;
    }
  }
  return true;
}

static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 += 1UL) {
    for (uint64_t _fuseiter_622 = 0UL; _fuseiter_622 < 128UL; _fuseiter_622 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 % 4UL) * 128UL)) + _fuseiter_622)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_362____itr_2_363____itr_3_364____itr_4_365 % 4UL) * 128UL)) + _fuseiter_622)]);
    }
  }
  return true;
}

static bool reorder__549(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 += 1UL) {
    for (uint64_t _fuseiter_629 = 0UL; _fuseiter_629 < 512UL; _fuseiter_629 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 / 4UL) * 2048UL) + (_fuseiter_629 + ((fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_624___fuseiter_625_366___fuseiter_626_367___fuseiter_627_368___fuseiter_628_369 % 4UL) * 512UL) + _fuseiter_629))] = __cached_1;
    }
  }
  return true;
}

static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 += 1UL) {
    for (uint64_t _fuseiter_635 = 0UL; _fuseiter_635 < 512UL; _fuseiter_635 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 % 4UL) * 512UL)) + _fuseiter_635)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_370____itr_2_371____itr_3_372____itr_4_373 % 4UL) * 512UL)) + _fuseiter_635)]);
    }
  }
  return true;
}

static bool reorder__552(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 += 1UL) {
    for (uint64_t _fuseiter_642 = 0UL; _fuseiter_642 < 256UL; _fuseiter_642 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 / 2UL) * 512UL) + (_fuseiter_642 + ((fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 % 2UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_637___fuseiter_638_374___fuseiter_639_375___fuseiter_640_376___fuseiter_641_377 % 2UL) * 256UL) + _fuseiter_642))] = __cached_1;
    }
  }
  return true;
}

static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 += 1UL) {
    for (uint64_t _fuseiter_648 = 0UL; _fuseiter_648 < 256UL; _fuseiter_648 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 % 2UL) * 256UL)) + _fuseiter_648)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_378____itr_2_379____itr_3_380____itr_4_381 % 2UL) * 256UL)) + _fuseiter_648)]);
    }
  }
  return true;
}

static bool reorder__555(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 += 1UL) {
    for (uint64_t _fuseiter_655 = 0UL; _fuseiter_655 < 64UL; _fuseiter_655 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 / 8UL) * 512UL) + (_fuseiter_655 + ((fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_650___fuseiter_651_382___fuseiter_652_383___fuseiter_653_384___fuseiter_654_385 % 8UL) * 64UL) + _fuseiter_655))] = __cached_1;
    }
  }
  return true;
}

static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 += 1UL) {
    for (uint64_t _fuseiter_661 = 0UL; _fuseiter_661 < 64UL; _fuseiter_661 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 % 8UL) * 64UL)) + _fuseiter_661)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_386____itr_2_387____itr_3_388____itr_4_389 % 8UL) * 64UL)) + _fuseiter_661)]);
    }
  }
  return true;
}

static bool reorder__558(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 += 1UL) {
    for (uint64_t _fuseiter_668 = 0UL; _fuseiter_668 < 512UL; _fuseiter_668 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 / 4UL) * 2048UL) + (_fuseiter_668 + ((fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_663___fuseiter_664_390___fuseiter_665_391___fuseiter_666_392___fuseiter_667_393 % 4UL) * 512UL) + _fuseiter_668))] = __cached_1;
    }
  }
  return true;
}

static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 += 1UL) {
    for (uint64_t _fuseiter_674 = 0UL; _fuseiter_674 < 512UL; _fuseiter_674 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 % 4UL) * 512UL)) + _fuseiter_674)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_394____itr_2_395____itr_3_396____itr_4_397 % 4UL) * 512UL)) + _fuseiter_674)]);
    }
  }
  return true;
}

static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 += 1UL) {
    for (uint64_t _fuseiter_681 = 0UL; _fuseiter_681 < 64UL; _fuseiter_681 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 / 4UL) * 256UL) + (_fuseiter_681 + ((fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_676___fuseiter_677_398___fuseiter_678_399___fuseiter_679_400___fuseiter_680_401 % 4UL) * 64UL) + _fuseiter_681))]);
    }
  }
  return true;
}

static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 += 1UL) {
    for (uint64_t _fuseiter_687 = 0UL; _fuseiter_687 < 64UL; _fuseiter_687 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 % 4UL) * 64UL)) + _fuseiter_687)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_402____itr_2_403____itr_3_404____itr_4_405 % 4UL) * 64UL)) + _fuseiter_687)]);
    }
  }
  return true;
}

static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 += 1UL) {
    for (uint64_t _fuseiter_694 = 0UL; _fuseiter_694 < 64UL; _fuseiter_694 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 / 4UL) * 256UL) + (_fuseiter_694 + ((fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_689___fuseiter_690_406___fuseiter_691_407___fuseiter_692_408___fuseiter_693_409 % 4UL) * 64UL) + _fuseiter_694))]);
    }
  }
  return true;
}

static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 += 1UL) {
    for (uint64_t _fuseiter_700 = 0UL; _fuseiter_700 < 64UL; _fuseiter_700 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 % 4UL) * 64UL)) + _fuseiter_700)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_410____itr_2_411____itr_3_412____itr_4_413 % 4UL) * 64UL)) + _fuseiter_700)]);
    }
  }
  return true;
}

static bool reorder__430(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 += 1UL) {
    for (uint64_t _fuseiter_707 = 0UL; _fuseiter_707 < 64UL; _fuseiter_707 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 / 4UL) * 256UL) + (_fuseiter_707 + ((fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_702___fuseiter_703_414___fuseiter_704_415___fuseiter_705_416___fuseiter_706_417 % 4UL) * 64UL) + _fuseiter_707))]);
    }
  }
  return true;
}

static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 += 1UL) {
    for (uint64_t _fuseiter_713 = 0UL; _fuseiter_713 < 64UL; _fuseiter_713 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 % 4UL) * 64UL)) + _fuseiter_713)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_418____itr_2_419____itr_3_420____itr_4_421 % 4UL) * 64UL)) + _fuseiter_713)]);
    }
  }
  return true;
}

static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 += 1UL) {
    for (uint64_t _fuseiter_720 = 0UL; _fuseiter_720 < 64UL; _fuseiter_720 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 / 4UL) * 256UL) + (_fuseiter_720 + ((fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_715___fuseiter_716_422___fuseiter_717_423___fuseiter_718_424___fuseiter_719_425 % 4UL) * 64UL) + _fuseiter_720))]);
    }
  }
  return true;
}

static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 += 1UL) {
    for (uint64_t _fuseiter_726 = 0UL; _fuseiter_726 < 64UL; _fuseiter_726 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 % 4UL) * 64UL)) + _fuseiter_726)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_426____itr_2_427____itr_3_428____itr_4_429 % 4UL) * 64UL)) + _fuseiter_726)]);
    }
  }
  return true;
}

static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 += 1UL) {
    for (uint64_t _fuseiter_733 = 0UL; _fuseiter_733 < 64UL; _fuseiter_733 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 / 2UL) * 128UL) + (_fuseiter_733 + ((fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_728___fuseiter_729_430___fuseiter_730_431___fuseiter_731_432___fuseiter_732_433 % 2UL) * 64UL) + _fuseiter_733))]);
    }
  }
  return true;
}

static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 += 1UL) {
    for (uint64_t _fuseiter_739 = 0UL; _fuseiter_739 < 64UL; _fuseiter_739 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 % 2UL) * 64UL)) + _fuseiter_739)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436____itr_4_437 % 2UL) * 64UL)) + _fuseiter_739)]);
    }
  }
  return true;
}

static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 += 1UL) {
    for (uint64_t _fuseiter_746 = 0UL; _fuseiter_746 < 64UL; _fuseiter_746 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 / 2UL) * 128UL) + (_fuseiter_746 + ((fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_741___fuseiter_742_438___fuseiter_743_439___fuseiter_744_440___fuseiter_745_441 % 2UL) * 64UL) + _fuseiter_746))]);
    }
  }
  return true;
}

static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 += 1UL) {
    for (uint64_t _fuseiter_752 = 0UL; _fuseiter_752 < 64UL; _fuseiter_752 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 % 2UL) * 64UL)) + _fuseiter_752)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_442____itr_2_443____itr_3_444____itr_4_445 % 2UL) * 64UL)) + _fuseiter_752)]);
    }
  }
  return true;
}

static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 += 1UL) {
    for (uint64_t _fuseiter_759 = 0UL; _fuseiter_759 < 64UL; _fuseiter_759 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 / 2UL) * 128UL) + (_fuseiter_759 + ((fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_754___fuseiter_755_446___fuseiter_756_447___fuseiter_757_448___fuseiter_758_449 % 2UL) * 64UL) + _fuseiter_759))]);
    }
  }
  return true;
}

static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 += 1UL) {
    for (uint64_t _fuseiter_765 = 0UL; _fuseiter_765 < 64UL; _fuseiter_765 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 % 2UL) * 64UL)) + _fuseiter_765)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_450____itr_2_451____itr_3_452____itr_4_453 % 2UL) * 64UL)) + _fuseiter_765)]);
    }
  }
  return true;
}

static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 += 1UL) {
    for (uint64_t _fuseiter_772 = 0UL; _fuseiter_772 < 64UL; _fuseiter_772 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 / 2UL) * 128UL) + (_fuseiter_772 + ((fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_767___fuseiter_768_454___fuseiter_769_455___fuseiter_770_456___fuseiter_771_457 % 2UL) * 64UL) + _fuseiter_772))]);
    }
  }
  return true;
}

static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 += 1UL) {
    for (uint64_t _fuseiter_778 = 0UL; _fuseiter_778 < 64UL; _fuseiter_778 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 % 2UL) * 64UL)) + _fuseiter_778)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460____itr_4_461 % 2UL) * 64UL)) + _fuseiter_778)]);
    }
  }
  return true;
}

static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 += 1UL) {
    for (uint64_t _fuseiter_785 = 0UL; _fuseiter_785 < 64UL; _fuseiter_785 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 / 2UL) * 128UL) + (_fuseiter_785 + ((fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_780___fuseiter_781_462___fuseiter_782_463___fuseiter_783_464___fuseiter_784_465 % 2UL) * 64UL) + _fuseiter_785))]);
    }
  }
  return true;
}

static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 += 1UL) {
    for (uint64_t _fuseiter_791 = 0UL; _fuseiter_791 < 64UL; _fuseiter_791 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 % 2UL) * 64UL)) + _fuseiter_791)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_466____itr_2_467____itr_3_468____itr_4_469 % 2UL) * 64UL)) + _fuseiter_791)]);
    }
  }
  return true;
}

static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 += 1UL) {
    for (uint64_t _fuseiter_798 = 0UL; _fuseiter_798 < 64UL; _fuseiter_798 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 / 2UL) * 128UL) + (_fuseiter_798 + ((fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_793___fuseiter_794_470___fuseiter_795_471___fuseiter_796_472___fuseiter_797_473 % 2UL) * 64UL) + _fuseiter_798))]);
    }
  }
  return true;
}

static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 += 1UL) {
    for (uint64_t _fuseiter_804 = 0UL; _fuseiter_804 < 64UL; _fuseiter_804 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 % 2UL) * 64UL)) + _fuseiter_804)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_474____itr_2_475____itr_3_476____itr_4_477 % 2UL) * 64UL)) + _fuseiter_804)]);
    }
  }
  return true;
}

static bool reorder__468(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 += 1UL) {
    for (uint64_t _fuseiter_811 = 0UL; _fuseiter_811 < 64UL; _fuseiter_811 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 / 2UL) * 128UL) + (_fuseiter_811 + ((fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_806___fuseiter_807_478___fuseiter_808_479___fuseiter_809_480___fuseiter_810_481 % 2UL) * 64UL) + _fuseiter_811))]);
    }
  }
  return true;
}

static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 += 1UL) {
    for (uint64_t _fuseiter_817 = 0UL; _fuseiter_817 < 64UL; _fuseiter_817 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 % 2UL) * 64UL)) + _fuseiter_817)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484____itr_4_485 % 2UL) * 64UL)) + _fuseiter_817)]);
    }
  }
  return true;
}

static bool reorder__471(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 += 1UL) {
    for (uint64_t _fuseiter_824 = 0UL; _fuseiter_824 < 64UL; _fuseiter_824 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 / 2UL) * 128UL) + (_fuseiter_824 + ((fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_819___fuseiter_820_486___fuseiter_821_487___fuseiter_822_488___fuseiter_823_489 % 2UL) * 64UL) + _fuseiter_824))]);
    }
  }
  return true;
}

static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 += 1UL) {
    for (uint64_t _fuseiter_830 = 0UL; _fuseiter_830 < 64UL; _fuseiter_830 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 % 2UL) * 64UL)) + _fuseiter_830)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_490____itr_2_491____itr_3_492____itr_4_493 % 2UL) * 64UL)) + _fuseiter_830)]);
    }
  }
  return true;
}

static bool reorder__480(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 += 1UL) {
    for (uint64_t _fuseiter_837 = 0UL; _fuseiter_837 < 64UL; _fuseiter_837 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 / 4UL) * 256UL) + (_fuseiter_837 + ((fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_832___fuseiter_833_494___fuseiter_834_495___fuseiter_835_496___fuseiter_836_497 % 4UL) * 64UL) + _fuseiter_837))]);
    }
  }
  return true;
}

static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 += 1UL) {
    for (uint64_t _fuseiter_843 = 0UL; _fuseiter_843 < 64UL; _fuseiter_843 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 % 4UL) * 64UL)) + _fuseiter_843)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_498____itr_2_499____itr_3_500____itr_4_501 % 4UL) * 64UL)) + _fuseiter_843)]);
    }
  }
  return true;
}

static bool reorder__483(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 += 1UL) {
    for (uint64_t _fuseiter_850 = 0UL; _fuseiter_850 < 64UL; _fuseiter_850 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 / 4UL) * 256UL) + (_fuseiter_850 + ((fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_845___fuseiter_846_502___fuseiter_847_503___fuseiter_848_504___fuseiter_849_505 % 4UL) * 64UL) + _fuseiter_850))]);
    }
  }
  return true;
}

static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 += 1UL) {
    for (uint64_t _fuseiter_856 = 0UL; _fuseiter_856 < 64UL; _fuseiter_856 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 % 4UL) * 64UL)) + _fuseiter_856)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508____itr_4_509 % 4UL) * 64UL)) + _fuseiter_856)]);
    }
  }
  return true;
}

static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 += 1UL) {
    for (uint64_t _fuseiter_863 = 0UL; _fuseiter_863 < 64UL; _fuseiter_863 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 / 4UL) * 256UL) + (_fuseiter_863 + ((fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_858___fuseiter_859_510___fuseiter_860_511___fuseiter_861_512___fuseiter_862_513 % 4UL) * 64UL) + _fuseiter_863))]);
    }
  }
  return true;
}

static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 += 1UL) {
    for (uint64_t _fuseiter_869 = 0UL; _fuseiter_869 < 64UL; _fuseiter_869 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 % 4UL) * 64UL)) + _fuseiter_869)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_514____itr_2_515____itr_3_516____itr_4_517 % 4UL) * 64UL)) + _fuseiter_869)]);
    }
  }
  return true;
}

static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 += 1UL) {
    for (uint64_t _fuseiter_876 = 0UL; _fuseiter_876 < 64UL; _fuseiter_876 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 / 4UL) * 256UL) + (_fuseiter_876 + ((fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_871___fuseiter_872_518___fuseiter_873_519___fuseiter_874_520___fuseiter_875_521 % 4UL) * 64UL) + _fuseiter_876))]);
    }
  }
  return true;
}

static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 += 1UL) {
    for (uint64_t _fuseiter_882 = 0UL; _fuseiter_882 < 64UL; _fuseiter_882 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 % 4UL) * 64UL)) + _fuseiter_882)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_522____itr_2_523____itr_3_524____itr_4_525 % 4UL) * 64UL)) + _fuseiter_882)]);
    }
  }
  return true;
}

static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 += 1UL) {
    for (uint64_t _fuseiter_889 = 0UL; _fuseiter_889 < 64UL; _fuseiter_889 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 / 4UL) * 256UL) + (_fuseiter_889 + ((fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_884___fuseiter_885_526___fuseiter_886_527___fuseiter_887_528___fuseiter_888_529 % 4UL) * 64UL) + _fuseiter_889))]);
    }
  }
  return true;
}

static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 += 1UL) {
    for (uint64_t _fuseiter_895 = 0UL; _fuseiter_895 < 64UL; _fuseiter_895 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 % 4UL) * 64UL)) + _fuseiter_895)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532____itr_4_533 % 4UL) * 64UL)) + _fuseiter_895)]);
    }
  }
  return true;
}

static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 += 1UL) {
    for (uint64_t _fuseiter_902 = 0UL; _fuseiter_902 < 64UL; _fuseiter_902 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 / 4UL) * 256UL) + (_fuseiter_902 + ((fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_897___fuseiter_898_534___fuseiter_899_535___fuseiter_900_536___fuseiter_901_537 % 4UL) * 64UL) + _fuseiter_902))]);
    }
  }
  return true;
}

static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 += 1UL) {
    for (uint64_t _fuseiter_908 = 0UL; _fuseiter_908 < 64UL; _fuseiter_908 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 % 4UL) * 64UL)) + _fuseiter_908)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_538____itr_2_539____itr_3_540____itr_4_541 % 4UL) * 64UL)) + _fuseiter_908)]);
    }
  }
  return true;
}

static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 += 1UL) {
    for (uint64_t _fuseiter_915 = 0UL; _fuseiter_915 < 64UL; _fuseiter_915 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 / 4UL) * 256UL) + (_fuseiter_915 + ((fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_910___fuseiter_911_542___fuseiter_912_543___fuseiter_913_544___fuseiter_914_545 % 4UL) * 64UL) + _fuseiter_915))]);
    }
  }
  return true;
}

static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 += 1UL) {
    for (uint64_t _fuseiter_921 = 0UL; _fuseiter_921 < 64UL; _fuseiter_921 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 % 4UL) * 64UL)) + _fuseiter_921)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_546____itr_2_547____itr_3_548____itr_4_549 % 4UL) * 64UL)) + _fuseiter_921)]);
    }
  }
  return true;
}

static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 += 1UL) {
    for (uint64_t _fuseiter_928 = 0UL; _fuseiter_928 < 64UL; _fuseiter_928 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 / 4UL) * 256UL) + (_fuseiter_928 + ((fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_923___fuseiter_924_550___fuseiter_925_551___fuseiter_926_552___fuseiter_927_553 % 4UL) * 64UL) + _fuseiter_928))]);
    }
  }
  return true;
}

static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 += 1UL) {
    for (uint64_t _fuseiter_934 = 0UL; _fuseiter_934 < 64UL; _fuseiter_934 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 % 4UL) * 64UL)) + _fuseiter_934)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556____itr_4_557 % 4UL) * 64UL)) + _fuseiter_934)]);
    }
  }
  return true;
}

static bool reorder__516(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 += 1UL) {
    for (uint64_t _fuseiter_941 = 0UL; _fuseiter_941 < 64UL; _fuseiter_941 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 / 4UL) * 256UL) + (_fuseiter_941 + ((fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_936___fuseiter_937_558___fuseiter_938_559___fuseiter_939_560___fuseiter_940_561 % 4UL) * 64UL) + _fuseiter_941))]);
    }
  }
  return true;
}

static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 += 1UL) {
    for (uint64_t _fuseiter_947 = 0UL; _fuseiter_947 < 64UL; _fuseiter_947 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 % 4UL) * 64UL)) + _fuseiter_947)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_562____itr_2_563____itr_3_564____itr_4_565 % 4UL) * 64UL)) + _fuseiter_947)]);
    }
  }
  return true;
}

static bool reorder__519(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 += 1UL) {
    for (uint64_t _fuseiter_954 = 0UL; _fuseiter_954 < 64UL; _fuseiter_954 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 / 4UL) * 256UL) + (_fuseiter_954 + ((fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_949___fuseiter_950_566___fuseiter_951_567___fuseiter_952_568___fuseiter_953_569 % 4UL) * 64UL) + _fuseiter_954))]);
    }
  }
  return true;
}

static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 += 1UL) {
    for (uint64_t _fuseiter_960 = 0UL; _fuseiter_960 < 64UL; _fuseiter_960 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 % 4UL) * 64UL)) + _fuseiter_960)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_570____itr_2_571____itr_3_572____itr_4_573 % 4UL) * 64UL)) + _fuseiter_960)]);
    }
  }
  return true;
}

static bool reorder__525(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 += 1UL) {
    for (uint64_t _fuseiter_967 = 0UL; _fuseiter_967 < 64UL; _fuseiter_967 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 / 4UL) * 256UL) + (_fuseiter_967 + ((fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_962___fuseiter_963_574___fuseiter_964_575___fuseiter_965_576___fuseiter_966_577 % 4UL) * 64UL) + _fuseiter_967))]);
    }
  }
  return true;
}

static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 += 1UL) {
    for (uint64_t _fuseiter_973 = 0UL; _fuseiter_973 < 64UL; _fuseiter_973 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 % 4UL) * 64UL)) + _fuseiter_973)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580____itr_4_581 % 4UL) * 64UL)) + _fuseiter_973)]);
    }
  }
  return true;
}

static bool reorder__528(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 += 1UL) {
    for (uint64_t _fuseiter_980 = 0UL; _fuseiter_980 < 64UL; _fuseiter_980 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 / 4UL) * 256UL) + (_fuseiter_980 + ((fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_975___fuseiter_976_582___fuseiter_977_583___fuseiter_978_584___fuseiter_979_585 % 4UL) * 64UL) + _fuseiter_980))]);
    }
  }
  return true;
}

static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 += 1UL) {
    for (uint64_t _fuseiter_986 = 0UL; _fuseiter_986 < 64UL; _fuseiter_986 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 % 4UL) * 64UL)) + _fuseiter_986)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_586____itr_2_587____itr_3_588____itr_4_589 % 4UL) * 64UL)) + _fuseiter_986)]);
    }
  }
  return true;
}

static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 += 1UL) {
    for (uint64_t _fuseiter_993 = 0UL; _fuseiter_993 < 64UL; _fuseiter_993 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 / 8UL) * 512UL) + (_fuseiter_993 + ((fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_988___fuseiter_989_590___fuseiter_990_591___fuseiter_991_592___fuseiter_992_593 % 8UL) * 64UL) + _fuseiter_993))]);
    }
  }
  return true;
}

static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 += 1UL) {
    for (uint64_t _fuseiter_999 = 0UL; _fuseiter_999 < 64UL; _fuseiter_999 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 % 8UL) * 64UL)) + _fuseiter_999)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_594____itr_2_595____itr_3_596____itr_4_597 % 8UL) * 64UL)) + _fuseiter_999)]);
    }
  }
  return true;
}

static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 += 1UL) {
    for (uint64_t _fuseiter_1006 = 0UL; _fuseiter_1006 < 64UL; _fuseiter_1006 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 / 8UL) * 512UL) + (_fuseiter_1006 + ((fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1001___fuseiter_1002_598___fuseiter_1003_599___fuseiter_1004_600___fuseiter_1005_601 % 8UL) * 64UL) + _fuseiter_1006))]);
    }
  }
  return true;
}

static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 += 1UL) {
    for (uint64_t _fuseiter_1012 = 0UL; _fuseiter_1012 < 64UL; _fuseiter_1012 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 % 8UL) * 64UL)) + _fuseiter_1012)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_602____itr_2_603____itr_3_604____itr_4_605 % 8UL) * 64UL)) + _fuseiter_1012)]);
    }
  }
  return true;
}

static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 += 1UL) {
    for (uint64_t _fuseiter_1019 = 0UL; _fuseiter_1019 < 64UL; _fuseiter_1019 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 / 8UL) * 512UL) + (_fuseiter_1019 + ((fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1014___fuseiter_1015_606___fuseiter_1016_607___fuseiter_1017_608___fuseiter_1018_609 % 8UL) * 64UL) + _fuseiter_1019))]);
    }
  }
  return true;
}

static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 += 1UL) {
    for (uint64_t _fuseiter_1025 = 0UL; _fuseiter_1025 < 64UL; _fuseiter_1025 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 % 8UL) * 64UL)) + _fuseiter_1025)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_610____itr_2_611____itr_3_612____itr_4_613 % 8UL) * 64UL)) + _fuseiter_1025)]);
    }
  }
  return true;
}

static bool reorder__465(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 += 1UL) {
    for (uint64_t _fuseiter_1032 = 0UL; _fuseiter_1032 < 64UL; _fuseiter_1032 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 / 8UL) * 512UL) + (_fuseiter_1032 + ((fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1027___fuseiter_1028_614___fuseiter_1029_615___fuseiter_1030_616___fuseiter_1031_617 % 8UL) * 64UL) + _fuseiter_1032))]);
    }
  }
  return true;
}

static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 += 1UL) {
    for (uint64_t _fuseiter_1038 = 0UL; _fuseiter_1038 < 64UL; _fuseiter_1038 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 % 8UL) * 64UL)) + _fuseiter_1038)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_618____itr_2_619____itr_3_620____itr_4_621 % 8UL) * 64UL)) + _fuseiter_1038)]);
    }
  }
  return true;
}

static bool reorder__474(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 += 1UL) {
    for (uint64_t _fuseiter_1045 = 0UL; _fuseiter_1045 < 64UL; _fuseiter_1045 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 / 8UL) * 512UL) + (_fuseiter_1045 + ((fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1040___fuseiter_1041_622___fuseiter_1042_623___fuseiter_1043_624___fuseiter_1044_625 % 8UL) * 64UL) + _fuseiter_1045))]);
    }
  }
  return true;
}

static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 += 1UL) {
    for (uint64_t _fuseiter_1051 = 0UL; _fuseiter_1051 < 64UL; _fuseiter_1051 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 % 8UL) * 64UL)) + _fuseiter_1051)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628____itr_4_629 % 8UL) * 64UL)) + _fuseiter_1051)]);
    }
  }
  return true;
}

static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1058 = 0UL; _fuseiter_1058 < 512UL; _fuseiter_1058 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1058]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1058]);
  }
  return true;
}

static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 += 1UL) {
    for (uint64_t _fuseiter_1065 = 0UL; _fuseiter_1065 < 256UL; _fuseiter_1065 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 / 2UL) * 512UL) + (_fuseiter_1065 + ((fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 % 2UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1060___fuseiter_1061_634___fuseiter_1062_635___fuseiter_1063_636___fuseiter_1064_637 % 2UL) * 256UL) + _fuseiter_1065))]);
    }
  }
  return true;
}

static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 += 1UL) {
    for (uint64_t _fuseiter_1071 = 0UL; _fuseiter_1071 < 256UL; _fuseiter_1071 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 % 2UL) * 256UL)) + _fuseiter_1071)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640____itr_4_641 % 2UL) * 256UL)) + _fuseiter_1071)]);
    }
  }
  return true;
}

static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 += 1UL) {
    for (uint64_t _fuseiter_1078 = 0UL; _fuseiter_1078 < 64UL; _fuseiter_1078 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 / 8UL) * 512UL) + (_fuseiter_1078 + ((fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1073___fuseiter_1074_642___fuseiter_1075_643___fuseiter_1076_644___fuseiter_1077_645 % 8UL) * 64UL) + _fuseiter_1078))]);
    }
  }
  return true;
}

static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 += 1UL) {
    for (uint64_t _fuseiter_1084 = 0UL; _fuseiter_1084 < 64UL; _fuseiter_1084 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 % 8UL) * 64UL)) + _fuseiter_1084)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_646____itr_2_647____itr_3_648____itr_4_649 % 8UL) * 64UL)) + _fuseiter_1084)]);
    }
  }
  return true;
}

static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 += 1UL) {
    for (uint64_t _fuseiter_1091 = 0UL; _fuseiter_1091 < 128UL; _fuseiter_1091 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 / 4UL) * 512UL) + (_fuseiter_1091 + ((fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 / 4UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1086___fuseiter_1087_650___fuseiter_1088_651___fuseiter_1089_652___fuseiter_1090_653 % 4UL) * 128UL) + _fuseiter_1091))]);
    }
  }
  return true;
}

static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 += 1UL) {
    for (uint64_t _fuseiter_1097 = 0UL; _fuseiter_1097 < 128UL; _fuseiter_1097 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 % 4UL) * 128UL)) + _fuseiter_1097)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_654____itr_2_655____itr_3_656____itr_4_657 % 4UL) * 128UL)) + _fuseiter_1097)]);
    }
  }
  return true;
}

static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 += 1UL) {
    for (uint64_t _fuseiter_1104 = 0UL; _fuseiter_1104 < 256UL; _fuseiter_1104 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 / 2UL) * 512UL) + (_fuseiter_1104 + ((fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 % 2UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1099___fuseiter_1100_658___fuseiter_1101_659___fuseiter_1102_660___fuseiter_1103_661 % 2UL) * 256UL) + _fuseiter_1104))]);
    }
  }
  return true;
}

static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 += 1UL) {
    for (uint64_t _fuseiter_1110 = 0UL; _fuseiter_1110 < 256UL; _fuseiter_1110 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 % 2UL) * 256UL)) + _fuseiter_1110)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664____itr_4_665 % 2UL) * 256UL)) + _fuseiter_1110)]);
    }
  }
  return true;
}

static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 += 1UL) {
    for (uint64_t _fuseiter_1117 = 0UL; _fuseiter_1117 < 64UL; _fuseiter_1117 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 / 8UL) * 512UL) + (_fuseiter_1117 + ((fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1112___fuseiter_1113_666___fuseiter_1114_667___fuseiter_1115_668___fuseiter_1116_669 % 8UL) * 64UL) + _fuseiter_1117))]);
    }
  }
  return true;
}

static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 += 1UL) {
    for (uint64_t _fuseiter_1123 = 0UL; _fuseiter_1123 < 64UL; _fuseiter_1123 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 % 8UL) * 64UL)) + _fuseiter_1123)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_670____itr_2_671____itr_3_672____itr_4_673 % 8UL) * 64UL)) + _fuseiter_1123)]);
    }
  }
  return true;
}

static bool reorder__477(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 = 0UL; fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 < 16UL; fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 += 1UL) {
    for (uint64_t _fuseiter_1130 = 0UL; _fuseiter_1130 < 64UL; _fuseiter_1130 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 / 16UL) * 1024UL) + (_fuseiter_1130 + ((fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1125___fuseiter_1126_674___fuseiter_1127_675 % 16UL) * 64UL) + _fuseiter_1130))]);
    }
  }
  return true;
}

static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 += 1UL) {
    for (uint64_t _fuseiter_1136 = 0UL; _fuseiter_1136 < 64UL; _fuseiter_1136 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 % 16UL) * 64UL)) + _fuseiter_1136)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_676____itr_2_677____itr_3_678____itr_4_679 % 16UL) * 64UL)) + _fuseiter_1136)]);
    }
  }
  return true;
}

static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 = 0UL; fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 < 16UL; fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 += 1UL) {
    for (uint64_t _fuseiter_1143 = 0UL; _fuseiter_1143 < 64UL; _fuseiter_1143 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 / 16UL) * 1024UL) + (_fuseiter_1143 + ((fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1138___fuseiter_1139_680___fuseiter_1140_681 % 16UL) * 64UL) + _fuseiter_1143))]);
    }
  }
  return true;
}

static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 += 1UL) {
    for (uint64_t _fuseiter_1149 = 0UL; _fuseiter_1149 < 64UL; _fuseiter_1149 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 % 16UL) * 64UL)) + _fuseiter_1149)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_682____itr_2_683____itr_3_684____itr_4_685 % 16UL) * 64UL)) + _fuseiter_1149)]);
    }
  }
  return true;
}

static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 = 0UL; fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 < 16UL; fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 += 1UL) {
    for (uint64_t _fuseiter_1156 = 0UL; _fuseiter_1156 < 64UL; _fuseiter_1156 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 / 16UL) * 1024UL) + (_fuseiter_1156 + ((fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1151___fuseiter_1152_686___fuseiter_1153_687 % 16UL) * 64UL) + _fuseiter_1156))]);
    }
  }
  return true;
}

static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 += 1UL) {
    for (uint64_t _fuseiter_1162 = 0UL; _fuseiter_1162 < 64UL; _fuseiter_1162 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 % 16UL) * 64UL)) + _fuseiter_1162)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_688____itr_2_689____itr_3_690____itr_4_691 % 16UL) * 64UL)) + _fuseiter_1162)]);
    }
  }
  return true;
}

static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 = 0UL; fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 < 16UL; fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 += 1UL) {
    for (uint64_t _fuseiter_1169 = 0UL; _fuseiter_1169 < 64UL; _fuseiter_1169 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 / 16UL) * 1024UL) + (_fuseiter_1169 + ((fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1164___fuseiter_1165_692___fuseiter_1166_693 % 16UL) * 64UL) + _fuseiter_1169))]);
    }
  }
  return true;
}

static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 += 1UL) {
    for (uint64_t _fuseiter_1175 = 0UL; _fuseiter_1175 < 64UL; _fuseiter_1175 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 % 16UL) * 64UL)) + _fuseiter_1175)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_694____itr_2_695____itr_3_696____itr_4_697 % 16UL) * 64UL)) + _fuseiter_1175)]);
    }
  }
  return true;
}

static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 = 0UL; fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 < 16UL; fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 += 1UL) {
    for (uint64_t _fuseiter_1182 = 0UL; _fuseiter_1182 < 64UL; _fuseiter_1182 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 / 16UL) * 1024UL) + (_fuseiter_1182 + ((fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1177___fuseiter_1178_698___fuseiter_1179_699 % 16UL) * 64UL) + _fuseiter_1182))]);
    }
  }
  return true;
}

static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 += 1UL) {
    for (uint64_t _fuseiter_1188 = 0UL; _fuseiter_1188 < 64UL; _fuseiter_1188 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 % 16UL) * 64UL)) + _fuseiter_1188)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_700____itr_2_701____itr_3_702____itr_4_703 % 16UL) * 64UL)) + _fuseiter_1188)]);
    }
  }
  return true;
}

static bool reorder__522(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 = 0UL; fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 < 16UL; fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 += 1UL) {
    for (uint64_t _fuseiter_1195 = 0UL; _fuseiter_1195 < 64UL; _fuseiter_1195 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 / 16UL) * 1024UL) + (_fuseiter_1195 + ((fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1190___fuseiter_1191_704___fuseiter_1192_705 % 16UL) * 64UL) + _fuseiter_1195))]);
    }
  }
  return true;
}

static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 += 1UL) {
    for (uint64_t _fuseiter_1201 = 0UL; _fuseiter_1201 < 64UL; _fuseiter_1201 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 % 16UL) * 64UL)) + _fuseiter_1201)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_706____itr_2_707____itr_3_708____itr_4_709 % 16UL) * 64UL)) + _fuseiter_1201)]);
    }
  }
  return true;
}

static bool reorder__531(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 = 0UL; fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 < 16UL; fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 += 1UL) {
    for (uint64_t _fuseiter_1208 = 0UL; _fuseiter_1208 < 64UL; _fuseiter_1208 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 / 16UL) * 1024UL) + (_fuseiter_1208 + ((fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_1203___fuseiter_1204_710___fuseiter_1205_711 % 16UL) * 64UL) + _fuseiter_1208))]);
    }
  }
  return true;
}

static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 += 1UL) {
    for (uint64_t _fuseiter_1214 = 0UL; _fuseiter_1214 < 64UL; _fuseiter_1214 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 % 16UL) * 64UL)) + _fuseiter_1214)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_712____itr_2_713____itr_3_714____itr_4_715 % 16UL) * 64UL)) + _fuseiter_1214)]);
    }
  }
  return true;
}

static bool reorder__534(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 += 1UL) {
    for (uint64_t _fuseiter_1221 = 0UL; _fuseiter_1221 < 512UL; _fuseiter_1221 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 / 4UL) * 2048UL) + (_fuseiter_1221 + ((fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1216___fuseiter_1217_716___fuseiter_1218_717___fuseiter_1219_718___fuseiter_1220_719 % 4UL) * 512UL) + _fuseiter_1221))]);
    }
  }
  return true;
}

static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 += 1UL) {
    for (uint64_t _fuseiter_1227 = 0UL; _fuseiter_1227 < 512UL; _fuseiter_1227 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 % 4UL) * 512UL)) + _fuseiter_1227)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_720____itr_2_721____itr_3_722____itr_4_723 % 4UL) * 512UL)) + _fuseiter_1227)]);
    }
  }
  return true;
}

static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 += 1UL) {
    for (uint64_t _fuseiter_1234 = 0UL; _fuseiter_1234 < 512UL; _fuseiter_1234 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 / 4UL) * 2048UL) + (_fuseiter_1234 + ((fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1229___fuseiter_1230_724___fuseiter_1231_725___fuseiter_1232_726___fuseiter_1233_727 % 4UL) * 512UL) + _fuseiter_1234))]);
    }
  }
  return true;
}

static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 += 1UL) {
    for (uint64_t _fuseiter_1240 = 0UL; _fuseiter_1240 < 512UL; _fuseiter_1240 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 % 4UL) * 512UL)) + _fuseiter_1240)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_728____itr_2_729____itr_3_730____itr_4_731 % 4UL) * 512UL)) + _fuseiter_1240)]);
    }
  }
  return true;
}

static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 += 1UL) {
    for (uint64_t _fuseiter_1247 = 0UL; _fuseiter_1247 < 512UL; _fuseiter_1247 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 / 4UL) * 2048UL) + (_fuseiter_1247 + ((fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1242___fuseiter_1243_732___fuseiter_1244_733___fuseiter_1245_734___fuseiter_1246_735 % 4UL) * 512UL) + _fuseiter_1247))]);
    }
  }
  return true;
}

static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 += 1UL) {
    for (uint64_t _fuseiter_1253 = 0UL; _fuseiter_1253 < 512UL; _fuseiter_1253 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 % 4UL) * 512UL)) + _fuseiter_1253)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_736____itr_2_737____itr_3_738____itr_4_739 % 4UL) * 512UL)) + _fuseiter_1253)]);
    }
  }
  return true;
}

static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 += 1UL) {
    for (uint64_t _fuseiter_1260 = 0UL; _fuseiter_1260 < 512UL; _fuseiter_1260 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 / 4UL) * 2048UL) + (_fuseiter_1260 + ((fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1255___fuseiter_1256_740___fuseiter_1257_741___fuseiter_1258_742___fuseiter_1259_743 % 4UL) * 512UL) + _fuseiter_1260))]);
    }
  }
  return true;
}

static bool mul__673(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 += 1UL) {
    for (uint64_t _fuseiter_1266 = 0UL; _fuseiter_1266 < 512UL; _fuseiter_1266 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 % 4UL) * 512UL)) + _fuseiter_1266)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_744____itr_2_745____itr_3_746____itr_4_747 % 4UL) * 512UL)) + _fuseiter_1266)]);
    }
  }
  return true;
}

static bool mul__110(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_748____itr_2_749 = 0UL; fused_0fused_0__itr_0____itr_1_748____itr_2_749 < 4096UL; fused_0fused_0__itr_0____itr_1_748____itr_2_749 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_748____itr_2_749 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_748____itr_2_749 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_748____itr_2_749 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_748____itr_2_749 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_748____itr_2_749 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__111(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_750____itr_2_751 = 0UL; fused_0fused_0__itr_0____itr_1_750____itr_2_751 < 4096UL; fused_0fused_0__itr_0____itr_1_750____itr_2_751 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_750____itr_2_751 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_750____itr_2_751 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_750____itr_2_751 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_750____itr_2_751 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__421(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 < 16UL; fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 += 1UL) {
    for (uint64_t _fuseiter_1283 = 0UL; _fuseiter_1283 < 64UL; _fuseiter_1283 += 1UL) {
      for (uint64_t _fuseiter_1284 = 0UL; _fuseiter_1284 < 4UL; _fuseiter_1284 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1283 + ((fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1284 + ((fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1278___fuseiter_1279_752___fuseiter_1280_753___fuseiter_1281_754___fuseiter_1282_755 % 16UL) * 256UL) + ((_fuseiter_1283 * 4UL) + _fuseiter_1284)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__107(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_756____itr_2_757 = 0UL; fused_0fused_0__itr_0____itr_1_756____itr_2_757 < 16384UL; fused_0fused_0__itr_0____itr_1_756____itr_2_757 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_756____itr_2_757 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_756____itr_2_757 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_756____itr_2_757 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_756____itr_2_757 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_756____itr_2_757 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__108(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_758____itr_2_759 = 0UL; fused_0fused_0__itr_0____itr_1_758____itr_2_759 < 16384UL; fused_0fused_0__itr_0____itr_1_758____itr_2_759 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_758____itr_2_759 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_758____itr_2_759 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_758____itr_2_759 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_758____itr_2_759 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__418(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 += 1UL) {
    for (uint64_t _fuseiter_1300 = 0UL; _fuseiter_1300 < 64UL; _fuseiter_1300 += 1UL) {
      for (uint64_t _fuseiter_1301 = 0UL; _fuseiter_1301 < 4UL; _fuseiter_1301 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1300 + ((fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1301 + ((fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1295___fuseiter_1296_760___fuseiter_1297_761___fuseiter_1298_762___fuseiter_1299_763 % 16UL) * 256UL) + ((_fuseiter_1300 * 4UL) + _fuseiter_1301)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__116(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_764____itr_2_765 = 0UL; fused_0fused_0__itr_0____itr_1_764____itr_2_765 < 16384UL; fused_0fused_0__itr_0____itr_1_764____itr_2_765 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_764____itr_2_765 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_764____itr_2_765 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_764____itr_2_765 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_764____itr_2_765 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_764____itr_2_765 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__117(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_766____itr_2_767 = 0UL; fused_0fused_0__itr_0____itr_1_766____itr_2_767 < 16384UL; fused_0fused_0__itr_0____itr_1_766____itr_2_767 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_766____itr_2_767 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_766____itr_2_767 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_766____itr_2_767 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_766____itr_2_767 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 += 1UL) {
    for (uint64_t _fuseiter_1317 = 0UL; _fuseiter_1317 < 64UL; _fuseiter_1317 += 1UL) {
      for (uint64_t _fuseiter_1318 = 0UL; _fuseiter_1318 < 4UL; _fuseiter_1318 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1317 + ((fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1318 + ((fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1312___fuseiter_1313_768___fuseiter_1314_769___fuseiter_1315_770___fuseiter_1316_771 % 16UL) * 256UL) + ((_fuseiter_1317 * 4UL) + _fuseiter_1318)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__125(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_772____itr_2_773 = 0UL; fused_0fused_0__itr_0____itr_1_772____itr_2_773 < 16384UL; fused_0fused_0__itr_0____itr_1_772____itr_2_773 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_772____itr_2_773 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_772____itr_2_773 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_772____itr_2_773 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_772____itr_2_773 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_772____itr_2_773 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__126(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_774____itr_2_775 = 0UL; fused_0fused_0__itr_0____itr_1_774____itr_2_775 < 16384UL; fused_0fused_0__itr_0____itr_1_774____itr_2_775 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_774____itr_2_775 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_774____itr_2_775 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_774____itr_2_775 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_774____itr_2_775 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__428(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 += 1UL) {
    for (uint64_t _fuseiter_1334 = 0UL; _fuseiter_1334 < 64UL; _fuseiter_1334 += 1UL) {
      for (uint64_t _fuseiter_1335 = 0UL; _fuseiter_1335 < 4UL; _fuseiter_1335 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1334 + ((fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1335 + ((fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1329___fuseiter_1330_776___fuseiter_1331_777___fuseiter_1332_778___fuseiter_1333_779 % 16UL) * 256UL) + ((_fuseiter_1334 * 4UL) + _fuseiter_1335)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__134(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_780____itr_2_781 = 0UL; fused_0fused_0__itr_0____itr_1_780____itr_2_781 < 16384UL; fused_0fused_0__itr_0____itr_1_780____itr_2_781 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_780____itr_2_781 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_780____itr_2_781 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_780____itr_2_781 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_780____itr_2_781 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_780____itr_2_781 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__135(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_782____itr_2_783 = 0UL; fused_0fused_0__itr_0____itr_1_782____itr_2_783 < 16384UL; fused_0fused_0__itr_0____itr_1_782____itr_2_783 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_782____itr_2_783 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_782____itr_2_783 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_782____itr_2_783 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_782____itr_2_783 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 += 1UL) {
    for (uint64_t _fuseiter_1351 = 0UL; _fuseiter_1351 < 64UL; _fuseiter_1351 += 1UL) {
      for (uint64_t _fuseiter_1352 = 0UL; _fuseiter_1352 < 4UL; _fuseiter_1352 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1351 + ((fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1352 + ((fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1346___fuseiter_1347_784___fuseiter_1348_785___fuseiter_1349_786___fuseiter_1350_787 % 16UL) * 256UL) + ((_fuseiter_1351 * 4UL) + _fuseiter_1352)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__119(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_788____itr_2_789 = 0UL; fused_0fused_0__itr_0____itr_1_788____itr_2_789 < 16384UL; fused_0fused_0__itr_0____itr_1_788____itr_2_789 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_788____itr_2_789 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_788____itr_2_789 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_788____itr_2_789 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_788____itr_2_789 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_788____itr_2_789 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__120(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_790____itr_2_791 = 0UL; fused_0fused_0__itr_0____itr_1_790____itr_2_791 < 16384UL; fused_0fused_0__itr_0____itr_1_790____itr_2_791 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_790____itr_2_791 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_790____itr_2_791 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_790____itr_2_791 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_790____itr_2_791 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 += 1UL) {
    for (uint64_t _fuseiter_1368 = 0UL; _fuseiter_1368 < 64UL; _fuseiter_1368 += 1UL) {
      for (uint64_t _fuseiter_1369 = 0UL; _fuseiter_1369 < 4UL; _fuseiter_1369 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1368 + ((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_1369 + ((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1363___fuseiter_1364_792___fuseiter_1365_793___fuseiter_1366_794___fuseiter_1367_795 % 16UL) * 256UL) + ((_fuseiter_1368 * 4UL) + _fuseiter_1369))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__128(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_796____itr_2_797 = 0UL; fused_0fused_0__itr_0____itr_1_796____itr_2_797 < 16384UL; fused_0fused_0__itr_0____itr_1_796____itr_2_797 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_796____itr_2_797 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_796____itr_2_797 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_796____itr_2_797 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_796____itr_2_797 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_796____itr_2_797 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__129(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_798____itr_2_799 = 0UL; fused_0fused_0__itr_0____itr_1_798____itr_2_799 < 16384UL; fused_0fused_0__itr_0____itr_1_798____itr_2_799 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_798____itr_2_799 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_798____itr_2_799 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_798____itr_2_799 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_798____itr_2_799 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__431(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 += 1UL) {
    for (uint64_t _fuseiter_1385 = 0UL; _fuseiter_1385 < 64UL; _fuseiter_1385 += 1UL) {
      for (uint64_t _fuseiter_1386 = 0UL; _fuseiter_1386 < 4UL; _fuseiter_1386 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1385 + ((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_1386 + ((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1380___fuseiter_1381_800___fuseiter_1382_801___fuseiter_1383_802___fuseiter_1384_803 % 16UL) * 256UL) + ((_fuseiter_1385 * 4UL) + _fuseiter_1386))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__140(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_804____itr_2_805 = 0UL; fused_0fused_0__itr_0____itr_1_804____itr_2_805 < 32768UL; fused_0fused_0__itr_0____itr_1_804____itr_2_805 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_804____itr_2_805 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_804____itr_2_805 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_804____itr_2_805 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_804____itr_2_805 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_804____itr_2_805 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__141(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_806____itr_2_807 = 0UL; fused_0fused_0__itr_0____itr_1_806____itr_2_807 < 32768UL; fused_0fused_0__itr_0____itr_1_806____itr_2_807 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_806____itr_2_807 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_806____itr_2_807 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_806____itr_2_807 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_806____itr_2_807 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 += 1UL) {
    for (uint64_t _fuseiter_1402 = 0UL; _fuseiter_1402 < 64UL; _fuseiter_1402 += 1UL) {
      for (uint64_t _fuseiter_1403 = 0UL; _fuseiter_1403 < 4UL; _fuseiter_1403 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1402 + ((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_1403 + ((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1397___fuseiter_1398_808___fuseiter_1399_809___fuseiter_1400_810___fuseiter_1401_811 % 16UL) * 256UL) + ((_fuseiter_1402 * 4UL) + _fuseiter_1403))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__113(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_812____itr_2_813 = 0UL; fused_0fused_0__itr_0____itr_1_812____itr_2_813 < 12288UL; fused_0fused_0__itr_0____itr_1_812____itr_2_813 += 1UL) {
    for (uint64_t _fuseiter_1408 = 0UL; _fuseiter_1408 < 3UL; _fuseiter_1408 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_812____itr_2_813 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_812____itr_2_813 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_812____itr_2_813 % 3UL) * 3UL))) + _fuseiter_1408)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_812____itr_2_813 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_812____itr_2_813 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_812____itr_2_813 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_812____itr_2_813 % 3UL) * 3UL))) + _fuseiter_1408)] = __cached_2;
    }
  }
  return true;
}

static bool cast__114(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_814____itr_2_815 = 0UL; fused_0fused_0__itr_0____itr_1_814____itr_2_815 < 12288UL; fused_0fused_0__itr_0____itr_1_814____itr_2_815 += 1UL) {
    for (uint64_t _fuseiter1413 = 0UL; _fuseiter1413 < 3UL; _fuseiter1413 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_814____itr_2_815 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_814____itr_2_815 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_814____itr_2_815 % 3UL) * 3UL))) + _fuseiter1413)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_814____itr_2_815 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_814____itr_2_815 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_814____itr_2_815 % 3UL) * 3UL))) + _fuseiter1413)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 += 1UL) {
    for (uint64_t _fuseiter_1419 = 0UL; _fuseiter_1419 < 64UL; _fuseiter_1419 += 1UL) {
      for (uint64_t _fuseiter_1420 = 0UL; _fuseiter_1420 < 4UL; _fuseiter_1420 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1419 + ((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_1420 + ((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1414___fuseiter_1415_816___fuseiter_1416_817___fuseiter_1417_818___fuseiter_1418_819 % 16UL) * 256UL) + ((_fuseiter_1419 * 4UL) + _fuseiter_1420)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__122(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_820____itr_2_821 = 0UL; fused_0fused_0__itr_0____itr_1_820____itr_2_821 < 12288UL; fused_0fused_0__itr_0____itr_1_820____itr_2_821 += 1UL) {
    for (uint64_t _fuseiter_1425 = 0UL; _fuseiter_1425 < 3UL; _fuseiter_1425 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_820____itr_2_821 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_820____itr_2_821 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_820____itr_2_821 % 3UL) * 3UL))) + _fuseiter_1425)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_820____itr_2_821 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_820____itr_2_821 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_820____itr_2_821 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_820____itr_2_821 % 3UL) * 3UL))) + _fuseiter_1425)] = __cached_2;
    }
  }
  return true;
}

static bool cast__123(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_822____itr_2_823 = 0UL; fused_0fused_0__itr_0____itr_1_822____itr_2_823 < 12288UL; fused_0fused_0__itr_0____itr_1_822____itr_2_823 += 1UL) {
    for (uint64_t _fuseiter1430 = 0UL; _fuseiter1430 < 3UL; _fuseiter1430 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_822____itr_2_823 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_822____itr_2_823 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_822____itr_2_823 % 3UL) * 3UL))) + _fuseiter1430)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_822____itr_2_823 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_822____itr_2_823 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_822____itr_2_823 % 3UL) * 3UL))) + _fuseiter1430)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__427(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 += 1UL) {
    for (uint64_t _fuseiter_1436 = 0UL; _fuseiter_1436 < 64UL; _fuseiter_1436 += 1UL) {
      for (uint64_t _fuseiter_1437 = 0UL; _fuseiter_1437 < 4UL; _fuseiter_1437 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1436 + ((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_1437 + ((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1431___fuseiter_1432_824___fuseiter_1433_825___fuseiter_1434_826___fuseiter_1435_827 % 16UL) * 256UL) + ((_fuseiter_1436 * 4UL) + _fuseiter_1437)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__131(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_828____itr_2_829 = 0UL; fused_0fused_0__itr_0____itr_1_828____itr_2_829 < 12288UL; fused_0fused_0__itr_0____itr_1_828____itr_2_829 += 1UL) {
    for (uint64_t _fuseiter_1442 = 0UL; _fuseiter_1442 < 3UL; _fuseiter_1442 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_828____itr_2_829 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_828____itr_2_829 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_828____itr_2_829 % 3UL) * 3UL))) + _fuseiter_1442)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_828____itr_2_829 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_828____itr_2_829 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_828____itr_2_829 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_828____itr_2_829 % 3UL) * 3UL))) + _fuseiter_1442)] = __cached_2;
    }
  }
  return true;
}

static bool cast__132(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_830____itr_2_831 = 0UL; fused_0fused_0__itr_0____itr_1_830____itr_2_831 < 12288UL; fused_0fused_0__itr_0____itr_1_830____itr_2_831 += 1UL) {
    for (uint64_t _fuseiter1447 = 0UL; _fuseiter1447 < 3UL; _fuseiter1447 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_830____itr_2_831 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_830____itr_2_831 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_830____itr_2_831 % 3UL) * 3UL))) + _fuseiter1447)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_830____itr_2_831 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_830____itr_2_831 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_830____itr_2_831 % 3UL) * 3UL))) + _fuseiter1447)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__432(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 += 1UL) {
    for (uint64_t _fuseiter_1453 = 0UL; _fuseiter_1453 < 64UL; _fuseiter_1453 += 1UL) {
      for (uint64_t _fuseiter_1454 = 0UL; _fuseiter_1454 < 4UL; _fuseiter_1454 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1453 + ((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_1454 + ((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1448___fuseiter_1449_832___fuseiter_1450_833___fuseiter_1451_834___fuseiter_1452_835 % 16UL) * 256UL) + ((_fuseiter_1453 * 4UL) + _fuseiter_1454)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__146(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_836____itr_2_837 = 0UL; fused_0fused_0__itr_0____itr_1_836____itr_2_837 < 65536UL; fused_0fused_0__itr_0____itr_1_836____itr_2_837 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_836____itr_2_837 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_836____itr_2_837 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_836____itr_2_837 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_836____itr_2_837 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_836____itr_2_837 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__147(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_838____itr_2_839 = 0UL; fused_0fused_0__itr_0____itr_1_838____itr_2_839 < 65536UL; fused_0fused_0__itr_0____itr_1_838____itr_2_839 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_838____itr_2_839 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_838____itr_2_839 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_838____itr_2_839 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_838____itr_2_839 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1465___fuseiter_1466_840 = 0UL; fused_0_fuseiter_1465___fuseiter_1466_840 < 16UL; fused_0_fuseiter_1465___fuseiter_1466_840 += 1UL) {
    for (uint64_t _fuseiter_1469 = 0UL; _fuseiter_1469 < 16UL; _fuseiter_1469 += 1UL) {
      for (uint64_t _fuseiter_1470 = 0UL; _fuseiter_1470 < 64UL; _fuseiter_1470 += 1UL) {
        for (uint64_t _fuseiter_1471 = 0UL; _fuseiter_1471 < 4UL; _fuseiter_1471 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1470 + ((fused_0_fuseiter_1465___fuseiter_1466_840 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_1471 + (_fuseiter_1469 * 4UL)) + ((fused_0_fuseiter_1465___fuseiter_1466_840 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1465___fuseiter_1466_840 / 2UL) * 8192UL) + (((fused_0_fuseiter_1465___fuseiter_1466_840 % 2UL) * 4096UL) + ((_fuseiter_1469 * 256UL) + ((_fuseiter_1470 * 4UL) + _fuseiter_1471))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__155(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_841____itr_2_842 = 0UL; fused_0fused_0__itr_0____itr_1_841____itr_2_842 < 65536UL; fused_0fused_0__itr_0____itr_1_841____itr_2_842 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_841____itr_2_842 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_841____itr_2_842 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_841____itr_2_842 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_841____itr_2_842 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_841____itr_2_842 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__156(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_843____itr_2_844 = 0UL; fused_0fused_0__itr_0____itr_1_843____itr_2_844 < 65536UL; fused_0fused_0__itr_0____itr_1_843____itr_2_844 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_843____itr_2_844 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_843____itr_2_844 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_843____itr_2_844 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_843____itr_2_844 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1482___fuseiter_1483_845 = 0UL; fused_0_fuseiter_1482___fuseiter_1483_845 < 16UL; fused_0_fuseiter_1482___fuseiter_1483_845 += 1UL) {
    for (uint64_t _fuseiter_1486 = 0UL; _fuseiter_1486 < 16UL; _fuseiter_1486 += 1UL) {
      for (uint64_t _fuseiter_1487 = 0UL; _fuseiter_1487 < 64UL; _fuseiter_1487 += 1UL) {
        for (uint64_t _fuseiter_1488 = 0UL; _fuseiter_1488 < 4UL; _fuseiter_1488 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1487 + ((fused_0_fuseiter_1482___fuseiter_1483_845 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_1488 + (_fuseiter_1486 * 4UL)) + ((fused_0_fuseiter_1482___fuseiter_1483_845 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1482___fuseiter_1483_845 / 2UL) * 8192UL) + (((fused_0_fuseiter_1482___fuseiter_1483_845 % 2UL) * 4096UL) + ((_fuseiter_1486 * 256UL) + ((_fuseiter_1487 * 4UL) + _fuseiter_1488))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__164(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_846____itr_2_847 = 0UL; fused_0fused_0__itr_0____itr_1_846____itr_2_847 < 65536UL; fused_0fused_0__itr_0____itr_1_846____itr_2_847 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_846____itr_2_847 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_846____itr_2_847 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_846____itr_2_847 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_846____itr_2_847 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_846____itr_2_847 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__165(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_848____itr_2_849 = 0UL; fused_0fused_0__itr_0____itr_1_848____itr_2_849 < 65536UL; fused_0fused_0__itr_0____itr_1_848____itr_2_849 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_848____itr_2_849 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_848____itr_2_849 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_848____itr_2_849 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_848____itr_2_849 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__463(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1499___fuseiter_1500_850 = 0UL; fused_0_fuseiter_1499___fuseiter_1500_850 < 16UL; fused_0_fuseiter_1499___fuseiter_1500_850 += 1UL) {
    for (uint64_t _fuseiter_1503 = 0UL; _fuseiter_1503 < 16UL; _fuseiter_1503 += 1UL) {
      for (uint64_t _fuseiter_1504 = 0UL; _fuseiter_1504 < 64UL; _fuseiter_1504 += 1UL) {
        for (uint64_t _fuseiter_1505 = 0UL; _fuseiter_1505 < 4UL; _fuseiter_1505 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1504 + ((fused_0_fuseiter_1499___fuseiter_1500_850 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_1505 + (_fuseiter_1503 * 4UL)) + ((fused_0_fuseiter_1499___fuseiter_1500_850 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1499___fuseiter_1500_850 / 2UL) * 8192UL) + (((fused_0_fuseiter_1499___fuseiter_1500_850 % 2UL) * 4096UL) + ((_fuseiter_1503 * 256UL) + ((_fuseiter_1504 * 4UL) + _fuseiter_1505))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__173(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_851____itr_2_852 = 0UL; fused_0fused_0__itr_0____itr_1_851____itr_2_852 < 65536UL; fused_0fused_0__itr_0____itr_1_851____itr_2_852 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_851____itr_2_852 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_851____itr_2_852 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_851____itr_2_852 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_851____itr_2_852 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_851____itr_2_852 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__174(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_853____itr_2_854 = 0UL; fused_0fused_0__itr_0____itr_1_853____itr_2_854 < 65536UL; fused_0fused_0__itr_0____itr_1_853____itr_2_854 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_853____itr_2_854 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_853____itr_2_854 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_853____itr_2_854 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_853____itr_2_854 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__472(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1516___fuseiter_1517_855 = 0UL; fused_0_fuseiter_1516___fuseiter_1517_855 < 16UL; fused_0_fuseiter_1516___fuseiter_1517_855 += 1UL) {
    for (uint64_t _fuseiter_1520 = 0UL; _fuseiter_1520 < 16UL; _fuseiter_1520 += 1UL) {
      for (uint64_t _fuseiter_1521 = 0UL; _fuseiter_1521 < 64UL; _fuseiter_1521 += 1UL) {
        for (uint64_t _fuseiter_1522 = 0UL; _fuseiter_1522 < 4UL; _fuseiter_1522 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1521 + ((fused_0_fuseiter_1516___fuseiter_1517_855 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_1522 + (_fuseiter_1520 * 4UL)) + ((fused_0_fuseiter_1516___fuseiter_1517_855 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1516___fuseiter_1517_855 / 2UL) * 8192UL) + (((fused_0_fuseiter_1516___fuseiter_1517_855 % 2UL) * 4096UL) + ((_fuseiter_1520 * 256UL) + ((_fuseiter_1521 * 4UL) + _fuseiter_1522))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__149(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_856____itr_2_857 = 0UL; fused_0fused_0__itr_0____itr_1_856____itr_2_857 < 65536UL; fused_0fused_0__itr_0____itr_1_856____itr_2_857 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_856____itr_2_857 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_856____itr_2_857 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_856____itr_2_857 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_856____itr_2_857 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_856____itr_2_857 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__150(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_858____itr_2_859 = 0UL; fused_0fused_0__itr_0____itr_1_858____itr_2_859 < 65536UL; fused_0fused_0__itr_0____itr_1_858____itr_2_859 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_858____itr_2_859 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_858____itr_2_859 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_858____itr_2_859 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_858____itr_2_859 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1533___fuseiter_1534_860 = 0UL; fused_0_fuseiter_1533___fuseiter_1534_860 < 16UL; fused_0_fuseiter_1533___fuseiter_1534_860 += 1UL) {
    for (uint64_t _fuseiter_1537 = 0UL; _fuseiter_1537 < 16UL; _fuseiter_1537 += 1UL) {
      for (uint64_t _fuseiter_1538 = 0UL; _fuseiter_1538 < 64UL; _fuseiter_1538 += 1UL) {
        for (uint64_t _fuseiter_1539 = 0UL; _fuseiter_1539 < 4UL; _fuseiter_1539 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1538 + ((fused_0_fuseiter_1533___fuseiter_1534_860 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_1539 + (_fuseiter_1537 * 4UL)) + ((fused_0_fuseiter_1533___fuseiter_1534_860 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1533___fuseiter_1534_860 / 8UL) * 32768UL) + (((fused_0_fuseiter_1533___fuseiter_1534_860 % 8UL) * 4096UL) + ((_fuseiter_1537 * 256UL) + ((_fuseiter_1538 * 4UL) + _fuseiter_1539))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__158(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_861____itr_2_862 = 0UL; fused_0fused_0__itr_0____itr_1_861____itr_2_862 < 65536UL; fused_0fused_0__itr_0____itr_1_861____itr_2_862 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_861____itr_2_862 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_861____itr_2_862 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_861____itr_2_862 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_861____itr_2_862 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_861____itr_2_862 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__159(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_863____itr_2_864 = 0UL; fused_0fused_0__itr_0____itr_1_863____itr_2_864 < 65536UL; fused_0fused_0__itr_0____itr_1_863____itr_2_864 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_863____itr_2_864 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_863____itr_2_864 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_863____itr_2_864 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_863____itr_2_864 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1550___fuseiter_1551_865 = 0UL; fused_0_fuseiter_1550___fuseiter_1551_865 < 16UL; fused_0_fuseiter_1550___fuseiter_1551_865 += 1UL) {
    for (uint64_t _fuseiter_1554 = 0UL; _fuseiter_1554 < 16UL; _fuseiter_1554 += 1UL) {
      for (uint64_t _fuseiter_1555 = 0UL; _fuseiter_1555 < 64UL; _fuseiter_1555 += 1UL) {
        for (uint64_t _fuseiter_1556 = 0UL; _fuseiter_1556 < 4UL; _fuseiter_1556 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1555 + ((fused_0_fuseiter_1550___fuseiter_1551_865 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_1556 + (_fuseiter_1554 * 4UL)) + ((fused_0_fuseiter_1550___fuseiter_1551_865 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1550___fuseiter_1551_865 / 8UL) * 32768UL) + (((fused_0_fuseiter_1550___fuseiter_1551_865 % 8UL) * 4096UL) + ((_fuseiter_1554 * 256UL) + ((_fuseiter_1555 * 4UL) + _fuseiter_1556))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__167(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_866____itr_2_867 = 0UL; fused_0fused_0__itr_0____itr_1_866____itr_2_867 < 65536UL; fused_0fused_0__itr_0____itr_1_866____itr_2_867 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_866____itr_2_867 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_866____itr_2_867 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_866____itr_2_867 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_866____itr_2_867 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_866____itr_2_867 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__168(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_868____itr_2_869 = 0UL; fused_0fused_0__itr_0____itr_1_868____itr_2_869 < 65536UL; fused_0fused_0__itr_0____itr_1_868____itr_2_869 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_868____itr_2_869 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_868____itr_2_869 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_868____itr_2_869 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_868____itr_2_869 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__466(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1567___fuseiter_1568_870 = 0UL; fused_0_fuseiter_1567___fuseiter_1568_870 < 16UL; fused_0_fuseiter_1567___fuseiter_1568_870 += 1UL) {
    for (uint64_t _fuseiter_1571 = 0UL; _fuseiter_1571 < 16UL; _fuseiter_1571 += 1UL) {
      for (uint64_t _fuseiter_1572 = 0UL; _fuseiter_1572 < 64UL; _fuseiter_1572 += 1UL) {
        for (uint64_t _fuseiter_1573 = 0UL; _fuseiter_1573 < 4UL; _fuseiter_1573 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1572 + ((fused_0_fuseiter_1567___fuseiter_1568_870 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_1573 + (_fuseiter_1571 * 4UL)) + ((fused_0_fuseiter_1567___fuseiter_1568_870 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1567___fuseiter_1568_870 / 8UL) * 32768UL) + (((fused_0_fuseiter_1567___fuseiter_1568_870 % 8UL) * 4096UL) + ((_fuseiter_1571 * 256UL) + ((_fuseiter_1572 * 4UL) + _fuseiter_1573))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__137(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_871____itr_2_872 = 0UL; fused_0fused_0__itr_0____itr_1_871____itr_2_872 < 131072UL; fused_0fused_0__itr_0____itr_1_871____itr_2_872 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_871____itr_2_872 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_871____itr_2_872 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_871____itr_2_872 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_871____itr_2_872 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_871____itr_2_872 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__138(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_873____itr_2_874 = 0UL; fused_0fused_0__itr_0____itr_1_873____itr_2_874 < 131072UL; fused_0fused_0__itr_0____itr_1_873____itr_2_874 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_873____itr_2_874 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_873____itr_2_874 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_873____itr_2_874 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_873____itr_2_874 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1584___fuseiter_1585_875 = 0UL; fused_0_fuseiter_1584___fuseiter_1585_875 < 32UL; fused_0_fuseiter_1584___fuseiter_1585_875 += 1UL) {
    for (uint64_t _fuseiter_1588 = 0UL; _fuseiter_1588 < 16UL; _fuseiter_1588 += 1UL) {
      for (uint64_t _fuseiter_1589 = 0UL; _fuseiter_1589 < 64UL; _fuseiter_1589 += 1UL) {
        for (uint64_t _fuseiter_1590 = 0UL; _fuseiter_1590 < 4UL; _fuseiter_1590 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1589 + ((fused_0_fuseiter_1584___fuseiter_1585_875 / 4UL) * 64UL)) * 256UL) + ((_fuseiter_1590 + (_fuseiter_1588 * 4UL)) + ((fused_0_fuseiter_1584___fuseiter_1585_875 % 4UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1584___fuseiter_1585_875 / 4UL) * 16384UL) + (((fused_0_fuseiter_1584___fuseiter_1585_875 % 4UL) * 4096UL) + ((_fuseiter_1588 * 256UL) + ((_fuseiter_1589 * 4UL) + _fuseiter_1590))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__179(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_876____itr_2_877 = 0UL; fused_0fused_0__itr_0____itr_1_876____itr_2_877 < 131072UL; fused_0fused_0__itr_0____itr_1_876____itr_2_877 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_876____itr_2_877 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_876____itr_2_877 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_876____itr_2_877 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_876____itr_2_877 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_876____itr_2_877 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__180(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_878____itr_2_879 = 0UL; fused_0fused_0__itr_0____itr_1_878____itr_2_879 < 131072UL; fused_0fused_0__itr_0____itr_1_878____itr_2_879 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_878____itr_2_879 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_878____itr_2_879 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_878____itr_2_879 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_878____itr_2_879 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__478(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1601___fuseiter_1602_880 = 0UL; fused_0_fuseiter_1601___fuseiter_1602_880 < 32UL; fused_0_fuseiter_1601___fuseiter_1602_880 += 1UL) {
    for (uint64_t _fuseiter_1605 = 0UL; _fuseiter_1605 < 16UL; _fuseiter_1605 += 1UL) {
      for (uint64_t _fuseiter_1606 = 0UL; _fuseiter_1606 < 64UL; _fuseiter_1606 += 1UL) {
        for (uint64_t _fuseiter_1607 = 0UL; _fuseiter_1607 < 4UL; _fuseiter_1607 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1606 + ((fused_0_fuseiter_1601___fuseiter_1602_880 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_1607 + (_fuseiter_1605 * 4UL)) + ((fused_0_fuseiter_1601___fuseiter_1602_880 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1601___fuseiter_1602_880 / 8UL) * 32768UL) + (((fused_0_fuseiter_1601___fuseiter_1602_880 % 8UL) * 4096UL) + ((_fuseiter_1605 * 256UL) + ((_fuseiter_1606 * 4UL) + _fuseiter_1607))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__143(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_881____itr_2_882 = 0UL; fused_0fused_0__itr_0____itr_1_881____itr_2_882 < 49152UL; fused_0fused_0__itr_0____itr_1_881____itr_2_882 += 1UL) {
    for (uint64_t _fuseiter_1612 = 0UL; _fuseiter_1612 < 3UL; _fuseiter_1612 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_881____itr_2_882 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_881____itr_2_882 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_881____itr_2_882 % 3UL) * 3UL))) + _fuseiter_1612)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_881____itr_2_882 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_881____itr_2_882 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_881____itr_2_882 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_881____itr_2_882 % 3UL) * 3UL))) + _fuseiter_1612)] = __cached_2;
    }
  }
  return true;
}

static bool cast__144(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_883____itr_2_884 = 0UL; fused_0fused_0__itr_0____itr_1_883____itr_2_884 < 49152UL; fused_0fused_0__itr_0____itr_1_883____itr_2_884 += 1UL) {
    for (uint64_t _fuseiter1617 = 0UL; _fuseiter1617 < 3UL; _fuseiter1617 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_883____itr_2_884 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_883____itr_2_884 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_883____itr_2_884 % 3UL) * 3UL))) + _fuseiter1617)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_883____itr_2_884 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_883____itr_2_884 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_883____itr_2_884 % 3UL) * 3UL))) + _fuseiter1617)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 = 0UL; fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 < 12UL; fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 += 1UL) {
    for (uint64_t _fuseiter_1621 = 0UL; _fuseiter_1621 < 3UL; _fuseiter_1621 += 1UL) {
      for (uint64_t _fuseiter_1622 = 0UL; _fuseiter_1622 < 16UL; _fuseiter_1622 += 1UL) {
        for (uint64_t _fuseiter_1623 = 0UL; _fuseiter_1623 < 64UL; _fuseiter_1623 += 1UL) {
          for (uint64_t _fuseiter_1624 = 0UL; _fuseiter_1624 < 4UL; _fuseiter_1624 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1623 + ((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_1624 + (_fuseiter_1622 * 4UL)) + (((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 % 3UL) * 3UL) + _fuseiter_1621)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_1618___fuseiter_1619_885___fuseiter_1620_886 % 3UL) * 12288UL) + ((_fuseiter_1621 * 4096UL) + ((_fuseiter_1622 * 256UL) + ((_fuseiter_1623 * 4UL) + _fuseiter_1624))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__152(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_887____itr_2_888 = 0UL; fused_0fused_0__itr_0____itr_1_887____itr_2_888 < 49152UL; fused_0fused_0__itr_0____itr_1_887____itr_2_888 += 1UL) {
    for (uint64_t _fuseiter_1629 = 0UL; _fuseiter_1629 < 3UL; _fuseiter_1629 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_887____itr_2_888 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_887____itr_2_888 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_887____itr_2_888 % 3UL) * 3UL))) + _fuseiter_1629)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_887____itr_2_888 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_887____itr_2_888 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_887____itr_2_888 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_887____itr_2_888 % 3UL) * 3UL))) + _fuseiter_1629)] = __cached_2;
    }
  }
  return true;
}

static bool cast__153(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_889____itr_2_890 = 0UL; fused_0fused_0__itr_0____itr_1_889____itr_2_890 < 49152UL; fused_0fused_0__itr_0____itr_1_889____itr_2_890 += 1UL) {
    for (uint64_t _fuseiter1634 = 0UL; _fuseiter1634 < 3UL; _fuseiter1634 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_889____itr_2_890 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_889____itr_2_890 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_889____itr_2_890 % 3UL) * 3UL))) + _fuseiter1634)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_889____itr_2_890 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_889____itr_2_890 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_889____itr_2_890 % 3UL) * 3UL))) + _fuseiter1634)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 = 0UL; fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 < 12UL; fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 += 1UL) {
    for (uint64_t _fuseiter_1638 = 0UL; _fuseiter_1638 < 3UL; _fuseiter_1638 += 1UL) {
      for (uint64_t _fuseiter_1639 = 0UL; _fuseiter_1639 < 16UL; _fuseiter_1639 += 1UL) {
        for (uint64_t _fuseiter_1640 = 0UL; _fuseiter_1640 < 64UL; _fuseiter_1640 += 1UL) {
          for (uint64_t _fuseiter_1641 = 0UL; _fuseiter_1641 < 4UL; _fuseiter_1641 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1640 + ((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_1641 + (_fuseiter_1639 * 4UL)) + (((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 % 3UL) * 3UL) + _fuseiter_1638)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_1635___fuseiter_1636_891___fuseiter_1637_892 % 3UL) * 12288UL) + ((_fuseiter_1638 * 4096UL) + ((_fuseiter_1639 * 256UL) + ((_fuseiter_1640 * 4UL) + _fuseiter_1641))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__161(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_893____itr_2_894 = 0UL; fused_0fused_0__itr_0____itr_1_893____itr_2_894 < 49152UL; fused_0fused_0__itr_0____itr_1_893____itr_2_894 += 1UL) {
    for (uint64_t _fuseiter_1646 = 0UL; _fuseiter_1646 < 3UL; _fuseiter_1646 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_893____itr_2_894 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_893____itr_2_894 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_893____itr_2_894 % 3UL) * 3UL))) + _fuseiter_1646)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_893____itr_2_894 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_893____itr_2_894 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_893____itr_2_894 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_893____itr_2_894 % 3UL) * 3UL))) + _fuseiter_1646)] = __cached_2;
    }
  }
  return true;
}

static bool cast__162(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_895____itr_2_896 = 0UL; fused_0fused_0__itr_0____itr_1_895____itr_2_896 < 49152UL; fused_0fused_0__itr_0____itr_1_895____itr_2_896 += 1UL) {
    for (uint64_t _fuseiter1651 = 0UL; _fuseiter1651 < 3UL; _fuseiter1651 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_895____itr_2_896 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_895____itr_2_896 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_895____itr_2_896 % 3UL) * 3UL))) + _fuseiter1651)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_895____itr_2_896 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_895____itr_2_896 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_895____itr_2_896 % 3UL) * 3UL))) + _fuseiter1651)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 = 0UL; fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 < 12UL; fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 += 1UL) {
    for (uint64_t _fuseiter_1655 = 0UL; _fuseiter_1655 < 3UL; _fuseiter_1655 += 1UL) {
      for (uint64_t _fuseiter_1656 = 0UL; _fuseiter_1656 < 16UL; _fuseiter_1656 += 1UL) {
        for (uint64_t _fuseiter_1657 = 0UL; _fuseiter_1657 < 64UL; _fuseiter_1657 += 1UL) {
          for (uint64_t _fuseiter_1658 = 0UL; _fuseiter_1658 < 4UL; _fuseiter_1658 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1657 + ((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_1658 + (_fuseiter_1656 * 4UL)) + (((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 % 3UL) * 3UL) + _fuseiter_1655)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_1652___fuseiter_1653_897___fuseiter_1654_898 % 3UL) * 12288UL) + ((_fuseiter_1655 * 4096UL) + ((_fuseiter_1656 * 256UL) + ((_fuseiter_1657 * 4UL) + _fuseiter_1658))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__170(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_899____itr_2_900 = 0UL; fused_0fused_0__itr_0____itr_1_899____itr_2_900 < 49152UL; fused_0fused_0__itr_0____itr_1_899____itr_2_900 += 1UL) {
    for (uint64_t _fuseiter_1663 = 0UL; _fuseiter_1663 < 3UL; _fuseiter_1663 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_899____itr_2_900 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_899____itr_2_900 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_899____itr_2_900 % 3UL) * 3UL))) + _fuseiter_1663)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_899____itr_2_900 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_899____itr_2_900 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_899____itr_2_900 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_899____itr_2_900 % 3UL) * 3UL))) + _fuseiter_1663)] = __cached_2;
    }
  }
  return true;
}

static bool cast__171(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_901____itr_2_902 = 0UL; fused_0fused_0__itr_0____itr_1_901____itr_2_902 < 49152UL; fused_0fused_0__itr_0____itr_1_901____itr_2_902 += 1UL) {
    for (uint64_t _fuseiter1668 = 0UL; _fuseiter1668 < 3UL; _fuseiter1668 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_901____itr_2_902 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_901____itr_2_902 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_901____itr_2_902 % 3UL) * 3UL))) + _fuseiter1668)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_901____itr_2_902 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_901____itr_2_902 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_901____itr_2_902 % 3UL) * 3UL))) + _fuseiter1668)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__469(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 = 0UL; fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 < 12UL; fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 += 1UL) {
    for (uint64_t _fuseiter_1672 = 0UL; _fuseiter_1672 < 3UL; _fuseiter_1672 += 1UL) {
      for (uint64_t _fuseiter_1673 = 0UL; _fuseiter_1673 < 16UL; _fuseiter_1673 += 1UL) {
        for (uint64_t _fuseiter_1674 = 0UL; _fuseiter_1674 < 64UL; _fuseiter_1674 += 1UL) {
          for (uint64_t _fuseiter_1675 = 0UL; _fuseiter_1675 < 4UL; _fuseiter_1675 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1674 + ((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_1675 + (_fuseiter_1673 * 4UL)) + (((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 % 3UL) * 3UL) + _fuseiter_1672)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_1669___fuseiter_1670_903___fuseiter_1671_904 % 3UL) * 12288UL) + ((_fuseiter_1672 * 4096UL) + ((_fuseiter_1673 * 256UL) + ((_fuseiter_1674 * 4UL) + _fuseiter_1675))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__185(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_905____itr_2_906 = 0UL; fused_0fused_0__itr_0____itr_1_905____itr_2_906 < 262144UL; fused_0fused_0__itr_0____itr_1_905____itr_2_906 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_905____itr_2_906 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_905____itr_2_906 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__186(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_907____itr_2_908 = 0UL; fused_0fused_0__itr_0____itr_1_907____itr_2_908 < 262144UL; fused_0fused_0__itr_0____itr_1_907____itr_2_908 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_907____itr_2_908 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_907____itr_2_908 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1686 = 0UL; _fuseiter_1686 < 16UL; _fuseiter_1686 += 1UL) {
    for (uint64_t _fuseiter_1687 = 0UL; _fuseiter_1687 < 4UL; _fuseiter_1687 += 1UL) {
      for (uint64_t _fuseiter_1690 = 0UL; _fuseiter_1690 < 16UL; _fuseiter_1690 += 1UL) {
        for (uint64_t _fuseiter_1691 = 0UL; _fuseiter_1691 < 64UL; _fuseiter_1691 += 1UL) {
          for (uint64_t _fuseiter_1692 = 0UL; _fuseiter_1692 < 4UL; _fuseiter_1692 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1691 + (_fuseiter_1686 * 64UL)) * 256UL) + ((_fuseiter_1692 + (_fuseiter_1690 * 4UL)) + (_fuseiter_1687 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1686 * 16384UL) + ((_fuseiter_1687 * 4096UL) + ((_fuseiter_1690 * 256UL) + ((_fuseiter_1691 * 4UL) + _fuseiter_1692))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__194(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_909____itr_2_910 = 0UL; fused_0fused_0__itr_0____itr_1_909____itr_2_910 < 262144UL; fused_0fused_0__itr_0____itr_1_909____itr_2_910 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_909____itr_2_910 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_909____itr_2_910 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__195(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_911____itr_2_912 = 0UL; fused_0fused_0__itr_0____itr_1_911____itr_2_912 < 262144UL; fused_0fused_0__itr_0____itr_1_911____itr_2_912 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_911____itr_2_912 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_911____itr_2_912 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1703 = 0UL; _fuseiter_1703 < 16UL; _fuseiter_1703 += 1UL) {
    for (uint64_t _fuseiter_1704 = 0UL; _fuseiter_1704 < 4UL; _fuseiter_1704 += 1UL) {
      for (uint64_t _fuseiter_1707 = 0UL; _fuseiter_1707 < 16UL; _fuseiter_1707 += 1UL) {
        for (uint64_t _fuseiter_1708 = 0UL; _fuseiter_1708 < 64UL; _fuseiter_1708 += 1UL) {
          for (uint64_t _fuseiter_1709 = 0UL; _fuseiter_1709 < 4UL; _fuseiter_1709 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1708 + (_fuseiter_1703 * 64UL)) * 256UL) + ((_fuseiter_1709 + (_fuseiter_1707 * 4UL)) + (_fuseiter_1704 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1703 * 16384UL) + ((_fuseiter_1704 * 4096UL) + ((_fuseiter_1707 * 256UL) + ((_fuseiter_1708 * 4UL) + _fuseiter_1709))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__203(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_913____itr_2_914 = 0UL; fused_0fused_0__itr_0____itr_1_913____itr_2_914 < 262144UL; fused_0fused_0__itr_0____itr_1_913____itr_2_914 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_913____itr_2_914 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_913____itr_2_914 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_913____itr_2_914 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_913____itr_2_914 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_913____itr_2_914 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__204(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_915____itr_2_916 = 0UL; fused_0fused_0__itr_0____itr_1_915____itr_2_916 < 262144UL; fused_0fused_0__itr_0____itr_1_915____itr_2_916 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_915____itr_2_916 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_915____itr_2_916 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_915____itr_2_916 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_915____itr_2_916 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__502(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1720 = 0UL; _fuseiter_1720 < 16UL; _fuseiter_1720 += 1UL) {
    for (uint64_t _fuseiter_1721 = 0UL; _fuseiter_1721 < 4UL; _fuseiter_1721 += 1UL) {
      for (uint64_t _fuseiter_1724 = 0UL; _fuseiter_1724 < 16UL; _fuseiter_1724 += 1UL) {
        for (uint64_t _fuseiter_1725 = 0UL; _fuseiter_1725 < 64UL; _fuseiter_1725 += 1UL) {
          for (uint64_t _fuseiter_1726 = 0UL; _fuseiter_1726 < 4UL; _fuseiter_1726 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1725 + (_fuseiter_1720 * 64UL)) * 256UL) + ((_fuseiter_1726 + (_fuseiter_1724 * 4UL)) + (_fuseiter_1721 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1720 * 16384UL) + ((_fuseiter_1721 * 4096UL) + ((_fuseiter_1724 * 256UL) + ((_fuseiter_1725 * 4UL) + _fuseiter_1726))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__212(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_917____itr_2_918 = 0UL; fused_0fused_0__itr_0____itr_1_917____itr_2_918 < 262144UL; fused_0fused_0__itr_0____itr_1_917____itr_2_918 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_917____itr_2_918 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_917____itr_2_918 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_917____itr_2_918 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_917____itr_2_918 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_917____itr_2_918 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__213(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_919____itr_2_920 = 0UL; fused_0fused_0__itr_0____itr_1_919____itr_2_920 < 262144UL; fused_0fused_0__itr_0____itr_1_919____itr_2_920 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_919____itr_2_920 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_919____itr_2_920 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_919____itr_2_920 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_919____itr_2_920 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__511(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1737 = 0UL; _fuseiter_1737 < 16UL; _fuseiter_1737 += 1UL) {
    for (uint64_t _fuseiter_1738 = 0UL; _fuseiter_1738 < 4UL; _fuseiter_1738 += 1UL) {
      for (uint64_t _fuseiter_1741 = 0UL; _fuseiter_1741 < 16UL; _fuseiter_1741 += 1UL) {
        for (uint64_t _fuseiter_1742 = 0UL; _fuseiter_1742 < 64UL; _fuseiter_1742 += 1UL) {
          for (uint64_t _fuseiter_1743 = 0UL; _fuseiter_1743 < 4UL; _fuseiter_1743 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1742 + (_fuseiter_1737 * 64UL)) * 256UL) + ((_fuseiter_1743 + (_fuseiter_1741 * 4UL)) + (_fuseiter_1738 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1737 * 16384UL) + ((_fuseiter_1738 * 4096UL) + ((_fuseiter_1741 * 256UL) + ((_fuseiter_1742 * 4UL) + _fuseiter_1743))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__221(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_921____itr_2_922 = 0UL; fused_0fused_0__itr_0____itr_1_921____itr_2_922 < 262144UL; fused_0fused_0__itr_0____itr_1_921____itr_2_922 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_921____itr_2_922 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_921____itr_2_922 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_921____itr_2_922 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_921____itr_2_922 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_921____itr_2_922 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__222(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_923____itr_2_924 = 0UL; fused_0fused_0__itr_0____itr_1_923____itr_2_924 < 262144UL; fused_0fused_0__itr_0____itr_1_923____itr_2_924 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_923____itr_2_924 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_923____itr_2_924 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_923____itr_2_924 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_923____itr_2_924 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__520(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1754 = 0UL; _fuseiter_1754 < 16UL; _fuseiter_1754 += 1UL) {
    for (uint64_t _fuseiter_1755 = 0UL; _fuseiter_1755 < 4UL; _fuseiter_1755 += 1UL) {
      for (uint64_t _fuseiter_1758 = 0UL; _fuseiter_1758 < 16UL; _fuseiter_1758 += 1UL) {
        for (uint64_t _fuseiter_1759 = 0UL; _fuseiter_1759 < 64UL; _fuseiter_1759 += 1UL) {
          for (uint64_t _fuseiter_1760 = 0UL; _fuseiter_1760 < 4UL; _fuseiter_1760 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1759 + (_fuseiter_1754 * 64UL)) * 256UL) + ((_fuseiter_1760 + (_fuseiter_1758 * 4UL)) + (_fuseiter_1755 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1754 * 16384UL) + ((_fuseiter_1755 * 4096UL) + ((_fuseiter_1758 * 256UL) + ((_fuseiter_1759 * 4UL) + _fuseiter_1760))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__230(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_925____itr_2_926 = 0UL; fused_0fused_0__itr_0____itr_1_925____itr_2_926 < 262144UL; fused_0fused_0__itr_0____itr_1_925____itr_2_926 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_925____itr_2_926 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_925____itr_2_926 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_925____itr_2_926 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_925____itr_2_926 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_925____itr_2_926 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__231(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_927____itr_2_928 = 0UL; fused_0fused_0__itr_0____itr_1_927____itr_2_928 < 262144UL; fused_0fused_0__itr_0____itr_1_927____itr_2_928 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_927____itr_2_928 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_927____itr_2_928 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_927____itr_2_928 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_927____itr_2_928 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__529(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1771 = 0UL; _fuseiter_1771 < 16UL; _fuseiter_1771 += 1UL) {
    for (uint64_t _fuseiter_1772 = 0UL; _fuseiter_1772 < 4UL; _fuseiter_1772 += 1UL) {
      for (uint64_t _fuseiter_1775 = 0UL; _fuseiter_1775 < 16UL; _fuseiter_1775 += 1UL) {
        for (uint64_t _fuseiter_1776 = 0UL; _fuseiter_1776 < 64UL; _fuseiter_1776 += 1UL) {
          for (uint64_t _fuseiter_1777 = 0UL; _fuseiter_1777 < 4UL; _fuseiter_1777 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1776 + (_fuseiter_1771 * 64UL)) * 256UL) + ((_fuseiter_1777 + (_fuseiter_1775 * 4UL)) + (_fuseiter_1772 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1771 * 16384UL) + ((_fuseiter_1772 * 4096UL) + ((_fuseiter_1775 * 256UL) + ((_fuseiter_1776 * 4UL) + _fuseiter_1777))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__188(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_929____itr_2_930 = 0UL; fused_0fused_0__itr_0____itr_1_929____itr_2_930 < 262144UL; fused_0fused_0__itr_0____itr_1_929____itr_2_930 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_929____itr_2_930 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_929____itr_2_930 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_929____itr_2_930 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_929____itr_2_930 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_929____itr_2_930 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__189(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_931____itr_2_932 = 0UL; fused_0fused_0__itr_0____itr_1_931____itr_2_932 < 262144UL; fused_0fused_0__itr_0____itr_1_931____itr_2_932 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_931____itr_2_932 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_931____itr_2_932 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_931____itr_2_932 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_931____itr_2_932 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1788___fuseiter_1789_933 = 0UL; fused_0_fuseiter_1788___fuseiter_1789_933 < 64UL; fused_0_fuseiter_1788___fuseiter_1789_933 += 1UL) {
    for (uint64_t _fuseiter_1792 = 0UL; _fuseiter_1792 < 16UL; _fuseiter_1792 += 1UL) {
      for (uint64_t _fuseiter_1793 = 0UL; _fuseiter_1793 < 64UL; _fuseiter_1793 += 1UL) {
        for (uint64_t _fuseiter_1794 = 0UL; _fuseiter_1794 < 4UL; _fuseiter_1794 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1793 + ((fused_0_fuseiter_1788___fuseiter_1789_933 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_1794 + (_fuseiter_1792 * 4UL)) + ((fused_0_fuseiter_1788___fuseiter_1789_933 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1788___fuseiter_1789_933 / 16UL) * 65536UL) + (((fused_0_fuseiter_1788___fuseiter_1789_933 % 16UL) * 4096UL) + ((_fuseiter_1792 * 256UL) + ((_fuseiter_1793 * 4UL) + _fuseiter_1794))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__197(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_934____itr_2_935 = 0UL; fused_0fused_0__itr_0____itr_1_934____itr_2_935 < 262144UL; fused_0fused_0__itr_0____itr_1_934____itr_2_935 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_934____itr_2_935 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_934____itr_2_935 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_934____itr_2_935 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_934____itr_2_935 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_934____itr_2_935 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__198(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_936____itr_2_937 = 0UL; fused_0fused_0__itr_0____itr_1_936____itr_2_937 < 262144UL; fused_0fused_0__itr_0____itr_1_936____itr_2_937 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_936____itr_2_937 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_936____itr_2_937 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_936____itr_2_937 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_936____itr_2_937 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1805___fuseiter_1806_938 = 0UL; fused_0_fuseiter_1805___fuseiter_1806_938 < 64UL; fused_0_fuseiter_1805___fuseiter_1806_938 += 1UL) {
    for (uint64_t _fuseiter_1809 = 0UL; _fuseiter_1809 < 16UL; _fuseiter_1809 += 1UL) {
      for (uint64_t _fuseiter_1810 = 0UL; _fuseiter_1810 < 64UL; _fuseiter_1810 += 1UL) {
        for (uint64_t _fuseiter_1811 = 0UL; _fuseiter_1811 < 4UL; _fuseiter_1811 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1810 + ((fused_0_fuseiter_1805___fuseiter_1806_938 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_1811 + (_fuseiter_1809 * 4UL)) + ((fused_0_fuseiter_1805___fuseiter_1806_938 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1805___fuseiter_1806_938 / 16UL) * 65536UL) + (((fused_0_fuseiter_1805___fuseiter_1806_938 % 16UL) * 4096UL) + ((_fuseiter_1809 * 256UL) + ((_fuseiter_1810 * 4UL) + _fuseiter_1811))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__206(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_939____itr_2_940 = 0UL; fused_0fused_0__itr_0____itr_1_939____itr_2_940 < 262144UL; fused_0fused_0__itr_0____itr_1_939____itr_2_940 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_939____itr_2_940 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_939____itr_2_940 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_939____itr_2_940 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_939____itr_2_940 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_939____itr_2_940 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__207(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_941____itr_2_942 = 0UL; fused_0fused_0__itr_0____itr_1_941____itr_2_942 < 262144UL; fused_0fused_0__itr_0____itr_1_941____itr_2_942 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_941____itr_2_942 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_941____itr_2_942 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_941____itr_2_942 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_941____itr_2_942 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__505(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1822___fuseiter_1823_943 = 0UL; fused_0_fuseiter_1822___fuseiter_1823_943 < 64UL; fused_0_fuseiter_1822___fuseiter_1823_943 += 1UL) {
    for (uint64_t _fuseiter_1826 = 0UL; _fuseiter_1826 < 16UL; _fuseiter_1826 += 1UL) {
      for (uint64_t _fuseiter_1827 = 0UL; _fuseiter_1827 < 64UL; _fuseiter_1827 += 1UL) {
        for (uint64_t _fuseiter_1828 = 0UL; _fuseiter_1828 < 4UL; _fuseiter_1828 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1827 + ((fused_0_fuseiter_1822___fuseiter_1823_943 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_1828 + (_fuseiter_1826 * 4UL)) + ((fused_0_fuseiter_1822___fuseiter_1823_943 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1822___fuseiter_1823_943 / 16UL) * 65536UL) + (((fused_0_fuseiter_1822___fuseiter_1823_943 % 16UL) * 4096UL) + ((_fuseiter_1826 * 256UL) + ((_fuseiter_1827 * 4UL) + _fuseiter_1828))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__215(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_944____itr_2_945 = 0UL; fused_0fused_0__itr_0____itr_1_944____itr_2_945 < 262144UL; fused_0fused_0__itr_0____itr_1_944____itr_2_945 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_944____itr_2_945 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_944____itr_2_945 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_944____itr_2_945 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_944____itr_2_945 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_944____itr_2_945 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__216(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_946____itr_2_947 = 0UL; fused_0fused_0__itr_0____itr_1_946____itr_2_947 < 262144UL; fused_0fused_0__itr_0____itr_1_946____itr_2_947 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_946____itr_2_947 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_946____itr_2_947 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_946____itr_2_947 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_946____itr_2_947 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__514(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1839___fuseiter_1840_948 = 0UL; fused_0_fuseiter_1839___fuseiter_1840_948 < 64UL; fused_0_fuseiter_1839___fuseiter_1840_948 += 1UL) {
    for (uint64_t _fuseiter_1843 = 0UL; _fuseiter_1843 < 16UL; _fuseiter_1843 += 1UL) {
      for (uint64_t _fuseiter_1844 = 0UL; _fuseiter_1844 < 64UL; _fuseiter_1844 += 1UL) {
        for (uint64_t _fuseiter_1845 = 0UL; _fuseiter_1845 < 4UL; _fuseiter_1845 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1844 + ((fused_0_fuseiter_1839___fuseiter_1840_948 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_1845 + (_fuseiter_1843 * 4UL)) + ((fused_0_fuseiter_1839___fuseiter_1840_948 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1839___fuseiter_1840_948 / 16UL) * 65536UL) + (((fused_0_fuseiter_1839___fuseiter_1840_948 % 16UL) * 4096UL) + ((_fuseiter_1843 * 256UL) + ((_fuseiter_1844 * 4UL) + _fuseiter_1845))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__224(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_949____itr_2_950 = 0UL; fused_0fused_0__itr_0____itr_1_949____itr_2_950 < 262144UL; fused_0fused_0__itr_0____itr_1_949____itr_2_950 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_949____itr_2_950 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_949____itr_2_950 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_949____itr_2_950 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_949____itr_2_950 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_949____itr_2_950 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__225(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_951____itr_2_952 = 0UL; fused_0fused_0__itr_0____itr_1_951____itr_2_952 < 262144UL; fused_0fused_0__itr_0____itr_1_951____itr_2_952 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_951____itr_2_952 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_951____itr_2_952 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_951____itr_2_952 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_951____itr_2_952 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__523(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1856___fuseiter_1857_953 = 0UL; fused_0_fuseiter_1856___fuseiter_1857_953 < 64UL; fused_0_fuseiter_1856___fuseiter_1857_953 += 1UL) {
    for (uint64_t _fuseiter_1860 = 0UL; _fuseiter_1860 < 16UL; _fuseiter_1860 += 1UL) {
      for (uint64_t _fuseiter_1861 = 0UL; _fuseiter_1861 < 64UL; _fuseiter_1861 += 1UL) {
        for (uint64_t _fuseiter_1862 = 0UL; _fuseiter_1862 < 4UL; _fuseiter_1862 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1861 + ((fused_0_fuseiter_1856___fuseiter_1857_953 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_1862 + (_fuseiter_1860 * 4UL)) + ((fused_0_fuseiter_1856___fuseiter_1857_953 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1856___fuseiter_1857_953 / 16UL) * 65536UL) + (((fused_0_fuseiter_1856___fuseiter_1857_953 % 16UL) * 4096UL) + ((_fuseiter_1860 * 256UL) + ((_fuseiter_1861 * 4UL) + _fuseiter_1862))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__176(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_954____itr_2_955 = 0UL; fused_0fused_0__itr_0____itr_1_954____itr_2_955 < 524288UL; fused_0fused_0__itr_0____itr_1_954____itr_2_955 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_954____itr_2_955 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_954____itr_2_955 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_954____itr_2_955 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_954____itr_2_955 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_954____itr_2_955 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__177(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_956____itr_2_957 = 0UL; fused_0fused_0__itr_0____itr_1_956____itr_2_957 < 524288UL; fused_0fused_0__itr_0____itr_1_956____itr_2_957 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_956____itr_2_957 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_956____itr_2_957 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_956____itr_2_957 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_956____itr_2_957 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__475(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1873 = 0UL; _fuseiter_1873 < 16UL; _fuseiter_1873 += 1UL) {
    for (uint64_t _fuseiter_1874 = 0UL; _fuseiter_1874 < 8UL; _fuseiter_1874 += 1UL) {
      for (uint64_t _fuseiter_1877 = 0UL; _fuseiter_1877 < 16UL; _fuseiter_1877 += 1UL) {
        for (uint64_t _fuseiter_1878 = 0UL; _fuseiter_1878 < 64UL; _fuseiter_1878 += 1UL) {
          for (uint64_t _fuseiter_1879 = 0UL; _fuseiter_1879 < 4UL; _fuseiter_1879 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1878 + (_fuseiter_1873 * 64UL)) * 512UL) + ((_fuseiter_1879 + (_fuseiter_1877 * 4UL)) + (_fuseiter_1874 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1873 * 32768UL) + ((_fuseiter_1874 * 4096UL) + ((_fuseiter_1877 * 256UL) + ((_fuseiter_1878 * 4UL) + _fuseiter_1879))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__236(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_958____itr_2_959 = 0UL; fused_0fused_0__itr_0____itr_1_958____itr_2_959 < 524288UL; fused_0fused_0__itr_0____itr_1_958____itr_2_959 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_958____itr_2_959 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_958____itr_2_959 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_958____itr_2_959 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_958____itr_2_959 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_958____itr_2_959 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__237(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_960____itr_2_961 = 0UL; fused_0fused_0__itr_0____itr_1_960____itr_2_961 < 524288UL; fused_0fused_0__itr_0____itr_1_960____itr_2_961 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_960____itr_2_961 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_960____itr_2_961 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_960____itr_2_961 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_960____itr_2_961 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__535(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1890___fuseiter_1891_962 = 0UL; fused_0_fuseiter_1890___fuseiter_1891_962 < 16UL; fused_0_fuseiter_1890___fuseiter_1891_962 += 1UL) {
    for (uint64_t _fuseiter_1894 = 0UL; _fuseiter_1894 < 16UL; _fuseiter_1894 += 1UL) {
      for (uint64_t _fuseiter_1895 = 0UL; _fuseiter_1895 < 512UL; _fuseiter_1895 += 1UL) {
        for (uint64_t _fuseiter_1896 = 0UL; _fuseiter_1896 < 4UL; _fuseiter_1896 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1895 + ((fused_0_fuseiter_1890___fuseiter_1891_962 / 16UL) * 512UL)) * 1024UL) + ((_fuseiter_1896 + (_fuseiter_1894 * 4UL)) + ((fused_0_fuseiter_1890___fuseiter_1891_962 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1890___fuseiter_1891_962 / 16UL) * 524288UL) + (((fused_0_fuseiter_1890___fuseiter_1891_962 % 16UL) * 32768UL) + ((_fuseiter_1894 * 2048UL) + ((_fuseiter_1895 * 4UL) + _fuseiter_1896))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__182(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_963____itr_2_964 = 0UL; fused_0fused_0__itr_0____itr_1_963____itr_2_964 < 196608UL; fused_0fused_0__itr_0____itr_1_963____itr_2_964 += 1UL) {
    for (uint64_t _fuseiter_1901 = 0UL; _fuseiter_1901 < 3UL; _fuseiter_1901 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_963____itr_2_964 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_963____itr_2_964 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_963____itr_2_964 % 3UL) * 3UL))) + _fuseiter_1901)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_963____itr_2_964 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_963____itr_2_964 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_963____itr_2_964 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_963____itr_2_964 % 3UL) * 3UL))) + _fuseiter_1901)] = __cached_2;
    }
  }
  return true;
}

static bool cast__183(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_965____itr_2_966 = 0UL; fused_0fused_0__itr_0____itr_1_965____itr_2_966 < 196608UL; fused_0fused_0__itr_0____itr_1_965____itr_2_966 += 1UL) {
    for (uint64_t _fuseiter1906 = 0UL; _fuseiter1906 < 3UL; _fuseiter1906 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_965____itr_2_966 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_965____itr_2_966 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_965____itr_2_966 % 3UL) * 3UL))) + _fuseiter1906)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_965____itr_2_966 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_965____itr_2_966 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_965____itr_2_966 % 3UL) * 3UL))) + _fuseiter1906)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__481(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1907___fuseiter_1908_967 = 0UL; fused_0_fuseiter_1907___fuseiter_1908_967 < 16UL; fused_0_fuseiter_1907___fuseiter_1908_967 += 1UL) {
    for (uint64_t _fuseiter_1909 = 0UL; _fuseiter_1909 < 3UL; _fuseiter_1909 += 1UL) {
      for (uint64_t _fuseiter_1910 = 0UL; _fuseiter_1910 < 3UL; _fuseiter_1910 += 1UL) {
        for (uint64_t _fuseiter_1911 = 0UL; _fuseiter_1911 < 16UL; _fuseiter_1911 += 1UL) {
          for (uint64_t _fuseiter_1912 = 0UL; _fuseiter_1912 < 64UL; _fuseiter_1912 += 1UL) {
            for (uint64_t _fuseiter_1913 = 0UL; _fuseiter_1913 < 4UL; _fuseiter_1913 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1912 + ((fused_0_fuseiter_1907___fuseiter_1908_967 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1913 + (_fuseiter_1911 * 4UL)) + ((fused_0_fuseiter_1907___fuseiter_1908_967 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1909 * 3UL) + _fuseiter_1910)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1907___fuseiter_1908_967 / 4UL) * 147456UL) + (((fused_0_fuseiter_1907___fuseiter_1908_967 % 4UL) * 36864UL) + ((_fuseiter_1909 * 12288UL) + ((_fuseiter_1910 * 4096UL) + ((_fuseiter_1911 * 256UL) + ((_fuseiter_1912 * 4UL) + _fuseiter_1913))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__191(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_968____itr_2_969 = 0UL; fused_0fused_0__itr_0____itr_1_968____itr_2_969 < 196608UL; fused_0fused_0__itr_0____itr_1_968____itr_2_969 += 1UL) {
    for (uint64_t _fuseiter_1918 = 0UL; _fuseiter_1918 < 3UL; _fuseiter_1918 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_968____itr_2_969 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_968____itr_2_969 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_968____itr_2_969 % 3UL) * 3UL))) + _fuseiter_1918)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_968____itr_2_969 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_968____itr_2_969 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_968____itr_2_969 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_968____itr_2_969 % 3UL) * 3UL))) + _fuseiter_1918)] = __cached_2;
    }
  }
  return true;
}

static bool cast__192(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_970____itr_2_971 = 0UL; fused_0fused_0__itr_0____itr_1_970____itr_2_971 < 196608UL; fused_0fused_0__itr_0____itr_1_970____itr_2_971 += 1UL) {
    for (uint64_t _fuseiter1923 = 0UL; _fuseiter1923 < 3UL; _fuseiter1923 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_970____itr_2_971 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_970____itr_2_971 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_970____itr_2_971 % 3UL) * 3UL))) + _fuseiter1923)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_970____itr_2_971 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_970____itr_2_971 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_970____itr_2_971 % 3UL) * 3UL))) + _fuseiter1923)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1924___fuseiter_1925_972 = 0UL; fused_0_fuseiter_1924___fuseiter_1925_972 < 16UL; fused_0_fuseiter_1924___fuseiter_1925_972 += 1UL) {
    for (uint64_t _fuseiter_1926 = 0UL; _fuseiter_1926 < 3UL; _fuseiter_1926 += 1UL) {
      for (uint64_t _fuseiter_1927 = 0UL; _fuseiter_1927 < 3UL; _fuseiter_1927 += 1UL) {
        for (uint64_t _fuseiter_1928 = 0UL; _fuseiter_1928 < 16UL; _fuseiter_1928 += 1UL) {
          for (uint64_t _fuseiter_1929 = 0UL; _fuseiter_1929 < 64UL; _fuseiter_1929 += 1UL) {
            for (uint64_t _fuseiter_1930 = 0UL; _fuseiter_1930 < 4UL; _fuseiter_1930 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1929 + ((fused_0_fuseiter_1924___fuseiter_1925_972 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1930 + (_fuseiter_1928 * 4UL)) + ((fused_0_fuseiter_1924___fuseiter_1925_972 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1926 * 3UL) + _fuseiter_1927)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1924___fuseiter_1925_972 / 4UL) * 147456UL) + (((fused_0_fuseiter_1924___fuseiter_1925_972 % 4UL) * 36864UL) + ((_fuseiter_1926 * 12288UL) + ((_fuseiter_1927 * 4096UL) + ((_fuseiter_1928 * 256UL) + ((_fuseiter_1929 * 4UL) + _fuseiter_1930))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__200(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_973____itr_2_974 = 0UL; fused_0fused_0__itr_0____itr_1_973____itr_2_974 < 196608UL; fused_0fused_0__itr_0____itr_1_973____itr_2_974 += 1UL) {
    for (uint64_t _fuseiter_1935 = 0UL; _fuseiter_1935 < 3UL; _fuseiter_1935 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_973____itr_2_974 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_973____itr_2_974 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_973____itr_2_974 % 3UL) * 3UL))) + _fuseiter_1935)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_973____itr_2_974 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_973____itr_2_974 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_973____itr_2_974 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_973____itr_2_974 % 3UL) * 3UL))) + _fuseiter_1935)] = __cached_2;
    }
  }
  return true;
}

static bool cast__201(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_975____itr_2_976 = 0UL; fused_0fused_0__itr_0____itr_1_975____itr_2_976 < 196608UL; fused_0fused_0__itr_0____itr_1_975____itr_2_976 += 1UL) {
    for (uint64_t _fuseiter1940 = 0UL; _fuseiter1940 < 3UL; _fuseiter1940 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_975____itr_2_976 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_975____itr_2_976 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_975____itr_2_976 % 3UL) * 3UL))) + _fuseiter1940)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_975____itr_2_976 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_975____itr_2_976 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_975____itr_2_976 % 3UL) * 3UL))) + _fuseiter1940)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__499(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1941___fuseiter_1942_977 = 0UL; fused_0_fuseiter_1941___fuseiter_1942_977 < 16UL; fused_0_fuseiter_1941___fuseiter_1942_977 += 1UL) {
    for (uint64_t _fuseiter_1943 = 0UL; _fuseiter_1943 < 3UL; _fuseiter_1943 += 1UL) {
      for (uint64_t _fuseiter_1944 = 0UL; _fuseiter_1944 < 3UL; _fuseiter_1944 += 1UL) {
        for (uint64_t _fuseiter_1945 = 0UL; _fuseiter_1945 < 16UL; _fuseiter_1945 += 1UL) {
          for (uint64_t _fuseiter_1946 = 0UL; _fuseiter_1946 < 64UL; _fuseiter_1946 += 1UL) {
            for (uint64_t _fuseiter_1947 = 0UL; _fuseiter_1947 < 4UL; _fuseiter_1947 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1946 + ((fused_0_fuseiter_1941___fuseiter_1942_977 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1947 + (_fuseiter_1945 * 4UL)) + ((fused_0_fuseiter_1941___fuseiter_1942_977 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1943 * 3UL) + _fuseiter_1944)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1941___fuseiter_1942_977 / 4UL) * 147456UL) + (((fused_0_fuseiter_1941___fuseiter_1942_977 % 4UL) * 36864UL) + ((_fuseiter_1943 * 12288UL) + ((_fuseiter_1944 * 4096UL) + ((_fuseiter_1945 * 256UL) + ((_fuseiter_1946 * 4UL) + _fuseiter_1947))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__209(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_978____itr_2_979 = 0UL; fused_0fused_0__itr_0____itr_1_978____itr_2_979 < 196608UL; fused_0fused_0__itr_0____itr_1_978____itr_2_979 += 1UL) {
    for (uint64_t _fuseiter_1952 = 0UL; _fuseiter_1952 < 3UL; _fuseiter_1952 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_978____itr_2_979 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_978____itr_2_979 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_978____itr_2_979 % 3UL) * 3UL))) + _fuseiter_1952)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_978____itr_2_979 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_978____itr_2_979 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_978____itr_2_979 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_978____itr_2_979 % 3UL) * 3UL))) + _fuseiter_1952)] = __cached_2;
    }
  }
  return true;
}

static bool cast__210(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_980____itr_2_981 = 0UL; fused_0fused_0__itr_0____itr_1_980____itr_2_981 < 196608UL; fused_0fused_0__itr_0____itr_1_980____itr_2_981 += 1UL) {
    for (uint64_t _fuseiter1957 = 0UL; _fuseiter1957 < 3UL; _fuseiter1957 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_980____itr_2_981 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_980____itr_2_981 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_980____itr_2_981 % 3UL) * 3UL))) + _fuseiter1957)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_980____itr_2_981 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_980____itr_2_981 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_980____itr_2_981 % 3UL) * 3UL))) + _fuseiter1957)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__508(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1958___fuseiter_1959_982 = 0UL; fused_0_fuseiter_1958___fuseiter_1959_982 < 16UL; fused_0_fuseiter_1958___fuseiter_1959_982 += 1UL) {
    for (uint64_t _fuseiter_1960 = 0UL; _fuseiter_1960 < 3UL; _fuseiter_1960 += 1UL) {
      for (uint64_t _fuseiter_1961 = 0UL; _fuseiter_1961 < 3UL; _fuseiter_1961 += 1UL) {
        for (uint64_t _fuseiter_1962 = 0UL; _fuseiter_1962 < 16UL; _fuseiter_1962 += 1UL) {
          for (uint64_t _fuseiter_1963 = 0UL; _fuseiter_1963 < 64UL; _fuseiter_1963 += 1UL) {
            for (uint64_t _fuseiter_1964 = 0UL; _fuseiter_1964 < 4UL; _fuseiter_1964 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1963 + ((fused_0_fuseiter_1958___fuseiter_1959_982 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1964 + (_fuseiter_1962 * 4UL)) + ((fused_0_fuseiter_1958___fuseiter_1959_982 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1960 * 3UL) + _fuseiter_1961)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1958___fuseiter_1959_982 / 4UL) * 147456UL) + (((fused_0_fuseiter_1958___fuseiter_1959_982 % 4UL) * 36864UL) + ((_fuseiter_1960 * 12288UL) + ((_fuseiter_1961 * 4096UL) + ((_fuseiter_1962 * 256UL) + ((_fuseiter_1963 * 4UL) + _fuseiter_1964))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__218(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_983____itr_2_984 = 0UL; fused_0fused_0__itr_0____itr_1_983____itr_2_984 < 196608UL; fused_0fused_0__itr_0____itr_1_983____itr_2_984 += 1UL) {
    for (uint64_t _fuseiter_1969 = 0UL; _fuseiter_1969 < 3UL; _fuseiter_1969 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_983____itr_2_984 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_983____itr_2_984 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_983____itr_2_984 % 3UL) * 3UL))) + _fuseiter_1969)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_983____itr_2_984 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_983____itr_2_984 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_983____itr_2_984 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_983____itr_2_984 % 3UL) * 3UL))) + _fuseiter_1969)] = __cached_2;
    }
  }
  return true;
}

static bool cast__219(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_985____itr_2_986 = 0UL; fused_0fused_0__itr_0____itr_1_985____itr_2_986 < 196608UL; fused_0fused_0__itr_0____itr_1_985____itr_2_986 += 1UL) {
    for (uint64_t _fuseiter1974 = 0UL; _fuseiter1974 < 3UL; _fuseiter1974 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_985____itr_2_986 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_985____itr_2_986 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_985____itr_2_986 % 3UL) * 3UL))) + _fuseiter1974)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_985____itr_2_986 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_985____itr_2_986 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_985____itr_2_986 % 3UL) * 3UL))) + _fuseiter1974)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__517(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1975___fuseiter_1976_987 = 0UL; fused_0_fuseiter_1975___fuseiter_1976_987 < 16UL; fused_0_fuseiter_1975___fuseiter_1976_987 += 1UL) {
    for (uint64_t _fuseiter_1977 = 0UL; _fuseiter_1977 < 3UL; _fuseiter_1977 += 1UL) {
      for (uint64_t _fuseiter_1978 = 0UL; _fuseiter_1978 < 3UL; _fuseiter_1978 += 1UL) {
        for (uint64_t _fuseiter_1979 = 0UL; _fuseiter_1979 < 16UL; _fuseiter_1979 += 1UL) {
          for (uint64_t _fuseiter_1980 = 0UL; _fuseiter_1980 < 64UL; _fuseiter_1980 += 1UL) {
            for (uint64_t _fuseiter_1981 = 0UL; _fuseiter_1981 < 4UL; _fuseiter_1981 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1980 + ((fused_0_fuseiter_1975___fuseiter_1976_987 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1981 + (_fuseiter_1979 * 4UL)) + ((fused_0_fuseiter_1975___fuseiter_1976_987 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1977 * 3UL) + _fuseiter_1978)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1975___fuseiter_1976_987 / 4UL) * 147456UL) + (((fused_0_fuseiter_1975___fuseiter_1976_987 % 4UL) * 36864UL) + ((_fuseiter_1977 * 12288UL) + ((_fuseiter_1978 * 4096UL) + ((_fuseiter_1979 * 256UL) + ((_fuseiter_1980 * 4UL) + _fuseiter_1981))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__227(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_988____itr_2_989 = 0UL; fused_0fused_0__itr_0____itr_1_988____itr_2_989 < 196608UL; fused_0fused_0__itr_0____itr_1_988____itr_2_989 += 1UL) {
    for (uint64_t _fuseiter_1986 = 0UL; _fuseiter_1986 < 3UL; _fuseiter_1986 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_988____itr_2_989 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_988____itr_2_989 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_988____itr_2_989 % 3UL) * 3UL))) + _fuseiter_1986)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_988____itr_2_989 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_988____itr_2_989 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_988____itr_2_989 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_988____itr_2_989 % 3UL) * 3UL))) + _fuseiter_1986)] = __cached_2;
    }
  }
  return true;
}

static bool cast__228(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_990____itr_2_991 = 0UL; fused_0fused_0__itr_0____itr_1_990____itr_2_991 < 196608UL; fused_0fused_0__itr_0____itr_1_990____itr_2_991 += 1UL) {
    for (uint64_t _fuseiter1991 = 0UL; _fuseiter1991 < 3UL; _fuseiter1991 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_990____itr_2_991 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_990____itr_2_991 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_990____itr_2_991 % 3UL) * 3UL))) + _fuseiter1991)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_990____itr_2_991 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_990____itr_2_991 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_990____itr_2_991 % 3UL) * 3UL))) + _fuseiter1991)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__526(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1992___fuseiter_1993_992 = 0UL; fused_0_fuseiter_1992___fuseiter_1993_992 < 16UL; fused_0_fuseiter_1992___fuseiter_1993_992 += 1UL) {
    for (uint64_t _fuseiter_1994 = 0UL; _fuseiter_1994 < 3UL; _fuseiter_1994 += 1UL) {
      for (uint64_t _fuseiter_1995 = 0UL; _fuseiter_1995 < 3UL; _fuseiter_1995 += 1UL) {
        for (uint64_t _fuseiter_1996 = 0UL; _fuseiter_1996 < 16UL; _fuseiter_1996 += 1UL) {
          for (uint64_t _fuseiter_1997 = 0UL; _fuseiter_1997 < 64UL; _fuseiter_1997 += 1UL) {
            for (uint64_t _fuseiter_1998 = 0UL; _fuseiter_1998 < 4UL; _fuseiter_1998 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1997 + ((fused_0_fuseiter_1992___fuseiter_1993_992 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1998 + (_fuseiter_1996 * 4UL)) + ((fused_0_fuseiter_1992___fuseiter_1993_992 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1994 * 3UL) + _fuseiter_1995)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1992___fuseiter_1993_992 / 4UL) * 147456UL) + (((fused_0_fuseiter_1992___fuseiter_1993_992 % 4UL) * 36864UL) + ((_fuseiter_1994 * 12288UL) + ((_fuseiter_1995 * 4096UL) + ((_fuseiter_1996 * 256UL) + ((_fuseiter_1997 * 4UL) + _fuseiter_1998))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__242(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_993____itr_2_994 = 0UL; fused_0fused_0__itr_0____itr_1_993____itr_2_994 < 1048576UL; fused_0fused_0__itr_0____itr_1_993____itr_2_994 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_993____itr_2_994 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_993____itr_2_994 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_993____itr_2_994 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_993____itr_2_994 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_993____itr_2_994 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__243(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_995____itr_2_996 = 0UL; fused_0fused_0__itr_0____itr_1_995____itr_2_996 < 1048576UL; fused_0fused_0__itr_0____itr_1_995____itr_2_996 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_995____itr_2_996 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_995____itr_2_996 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_995____itr_2_996 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_995____itr_2_996 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__539(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 += 1UL) {
    for (uint64_t _fuseiter_2014 = 0UL; _fuseiter_2014 < 512UL; _fuseiter_2014 += 1UL) {
      for (uint64_t _fuseiter_2015 = 0UL; _fuseiter_2015 < 4UL; _fuseiter_2015 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_2014 + ((fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 / 128UL) * 512UL)) * 512UL) + (_fuseiter_2015 + ((fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_2009___fuseiter_2010_997___fuseiter_2011_998___fuseiter_2012_999___fuseiter_2013_1000 % 128UL) * 2048UL) + ((_fuseiter_2014 * 4UL) + _fuseiter_2015)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__251(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 = 0UL; fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 < 1048576UL; fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1001____itr_2_1002 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__252(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 = 0UL; fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 < 1048576UL; fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1003____itr_2_1004 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__548(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 += 1UL) {
    for (uint64_t _fuseiter_2031 = 0UL; _fuseiter_2031 < 512UL; _fuseiter_2031 += 1UL) {
      for (uint64_t _fuseiter_2032 = 0UL; _fuseiter_2032 < 4UL; _fuseiter_2032 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_2031 + ((fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 / 128UL) * 512UL)) * 512UL) + (_fuseiter_2032 + ((fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_2026___fuseiter_2027_1005___fuseiter_2028_1006___fuseiter_2029_1007___fuseiter_2030_1008 % 128UL) * 2048UL) + ((_fuseiter_2031 * 4UL) + _fuseiter_2032)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__260(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 = 0UL; fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 < 1048576UL; fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1009____itr_2_1010 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__261(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 = 0UL; fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 < 1048576UL; fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_1011____itr_2_1012 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__557(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_2043___fuseiter_2044_1013 = 0UL; fused_0_fuseiter_2043___fuseiter_2044_1013 < 32UL; fused_0_fuseiter_2043___fuseiter_2044_1013 += 1UL) {
    for (uint64_t _fuseiter_2047 = 0UL; _fuseiter_2047 < 16UL; _fuseiter_2047 += 1UL) {
      for (uint64_t _fuseiter_2048 = 0UL; _fuseiter_2048 < 512UL; _fuseiter_2048 += 1UL) {
        for (uint64_t _fuseiter_2049 = 0UL; _fuseiter_2049 < 4UL; _fuseiter_2049 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_2048 + ((fused_0_fuseiter_2043___fuseiter_2044_1013 / 8UL) * 512UL)) * 512UL) + ((_fuseiter_2049 + (_fuseiter_2047 * 4UL)) + ((fused_0_fuseiter_2043___fuseiter_2044_1013 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_2043___fuseiter_2044_1013 / 8UL) * 262144UL) + (((fused_0_fuseiter_2043___fuseiter_2044_1013 % 8UL) * 32768UL) + ((_fuseiter_2047 * 2048UL) + ((_fuseiter_2048 * 4UL) + _fuseiter_2049))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__245(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 = 0UL; fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 < 1048576UL; fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1014____itr_2_1015 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__246(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 = 0UL; fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 < 1048576UL; fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1016____itr_2_1017 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool reorder__542(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_2060___fuseiter_2061_1018 = 0UL; fused_0_fuseiter_2060___fuseiter_2061_1018 < 32UL; fused_0_fuseiter_2060___fuseiter_2061_1018 += 1UL) {
    for (uint64_t _fuseiter_2064 = 0UL; _fuseiter_2064 < 128UL; _fuseiter_2064 += 1UL) {
      for (uint64_t _fuseiter_2065 = 0UL; _fuseiter_2065 < 64UL; _fuseiter_2065 += 1UL) {
        for (uint64_t _fuseiter_2066 = 0UL; _fuseiter_2066 < 4UL; _fuseiter_2066 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_2065 + ((fused_0_fuseiter_2060___fuseiter_2061_1018 / 4UL) * 64UL)) * 2048UL) + ((_fuseiter_2066 + (_fuseiter_2064 * 4UL)) + ((fused_0_fuseiter_2060___fuseiter_2061_1018 % 4UL) * 512UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_2060___fuseiter_2061_1018 / 4UL) * 131072UL) + (((fused_0_fuseiter_2060___fuseiter_2061_1018 % 4UL) * 32768UL) + ((_fuseiter_2064 * 256UL) + ((_fuseiter_2065 * 4UL) + _fuseiter_2066))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__254(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 = 0UL; fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 < 1048576UL; fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1019____itr_2_1020 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__255(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 = 0UL; fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 < 1048576UL; fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_1021____itr_2_1022 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool reorder__551(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 < 1024UL; fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 += 1UL) {
    for (uint64_t _fuseiter_2082 = 0UL; _fuseiter_2082 < 256UL; _fuseiter_2082 += 1UL) {
      for (uint64_t _fuseiter_2083 = 0UL; _fuseiter_2083 < 4UL; _fuseiter_2083 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_2082 + ((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 / 512UL) * 256UL)) * 2048UL) + ((_fuseiter_2083 + ((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 % 128UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 / 128UL) % 4UL) * 512UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 / 512UL) * 524288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 / 128UL) % 4UL) * 131072UL) + (((fused_0fused_0fused_0fused_0_fuseiter_2077___fuseiter_2078_1023___fuseiter_2079_1024___fuseiter_2080_1025___fuseiter_2081_1026 % 128UL) * 1024UL) + ((_fuseiter_2082 * 4UL) + _fuseiter_2083))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__233(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 = 0UL; fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 < 2097152UL; fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_1027____itr_2_1028 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__234(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 = 0UL; fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 < 2097152UL; fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_1029____itr_2_1030 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__532(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_2094___fuseiter_2095_1031 = 0UL; fused_0_fuseiter_2094___fuseiter_2095_1031 < 64UL; fused_0_fuseiter_2094___fuseiter_2095_1031 += 1UL) {
    for (uint64_t _fuseiter_2098 = 0UL; _fuseiter_2098 < 16UL; _fuseiter_2098 += 1UL) {
      for (uint64_t _fuseiter_2099 = 0UL; _fuseiter_2099 < 512UL; _fuseiter_2099 += 1UL) {
        for (uint64_t _fuseiter_2100 = 0UL; _fuseiter_2100 < 4UL; _fuseiter_2100 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_2099 + ((fused_0_fuseiter_2094___fuseiter_2095_1031 / 16UL) * 512UL)) * 1024UL) + ((_fuseiter_2100 + (_fuseiter_2098 * 4UL)) + ((fused_0_fuseiter_2094___fuseiter_2095_1031 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_2094___fuseiter_2095_1031 / 16UL) * 524288UL) + (((fused_0_fuseiter_2094___fuseiter_2095_1031 % 16UL) * 32768UL) + ((_fuseiter_2098 * 2048UL) + ((_fuseiter_2099 * 4UL) + _fuseiter_2100))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__239(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 = 0UL; fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 < 786432UL; fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 += 1UL) {
    for (uint64_t _fuseiter_2105 = 0UL; _fuseiter_2105 < 3UL; _fuseiter_2105 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 % 3UL) * 3UL))) + _fuseiter_2105)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1032____itr_2_1033 % 3UL) * 3UL))) + _fuseiter_2105)] = __cached_2;
    }
  }
  return true;
}

static bool cast__240(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 = 0UL; fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 < 786432UL; fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 += 1UL) {
    for (uint64_t _fuseiter2110 = 0UL; _fuseiter2110 < 3UL; _fuseiter2110 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 % 3UL) * 3UL))) + _fuseiter2110)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1034____itr_2_1035 % 3UL) * 3UL))) + _fuseiter2110)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__536(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_2111___fuseiter_2112_1036 = 0UL; fused_0_fuseiter_2111___fuseiter_2112_1036 < 16UL; fused_0_fuseiter_2111___fuseiter_2112_1036 += 1UL) {
    for (uint64_t _fuseiter_2113 = 0UL; _fuseiter_2113 < 3UL; _fuseiter_2113 += 1UL) {
      for (uint64_t _fuseiter_2114 = 0UL; _fuseiter_2114 < 3UL; _fuseiter_2114 += 1UL) {
        for (uint64_t _fuseiter_2115 = 0UL; _fuseiter_2115 < 16UL; _fuseiter_2115 += 1UL) {
          for (uint64_t _fuseiter_2116 = 0UL; _fuseiter_2116 < 256UL; _fuseiter_2116 += 1UL) {
            for (uint64_t _fuseiter_2117 = 0UL; _fuseiter_2117 < 4UL; _fuseiter_2117 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_2116 + ((fused_0_fuseiter_2111___fuseiter_2112_1036 / 8UL) * 256UL)) * 4608UL) + ((((_fuseiter_2117 + (_fuseiter_2115 * 4UL)) + ((fused_0_fuseiter_2111___fuseiter_2112_1036 % 8UL) * 64UL)) * 9UL) + ((_fuseiter_2113 * 3UL) + _fuseiter_2114)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_2111___fuseiter_2112_1036 / 8UL) * 1179648UL) + (((fused_0_fuseiter_2111___fuseiter_2112_1036 % 8UL) * 147456UL) + ((_fuseiter_2113 * 49152UL) + ((_fuseiter_2114 * 16384UL) + ((_fuseiter_2115 * 1024UL) + ((_fuseiter_2116 * 4UL) + _fuseiter_2117))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__248(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 = 0UL; fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 < 786432UL; fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 += 1UL) {
    for (uint64_t _fuseiter_2122 = 0UL; _fuseiter_2122 < 3UL; _fuseiter_2122 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 % 3UL) * 3UL))) + _fuseiter_2122)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1037____itr_2_1038 % 3UL) * 3UL))) + _fuseiter_2122)] = __cached_2;
    }
  }
  return true;
}

static bool cast__249(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 = 0UL; fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 < 786432UL; fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 += 1UL) {
    for (uint64_t _fuseiter2127 = 0UL; _fuseiter2127 < 3UL; _fuseiter2127 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 % 3UL) * 3UL))) + _fuseiter2127)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1039____itr_2_1040 % 3UL) * 3UL))) + _fuseiter2127)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__545(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_2128___fuseiter_2129_1041 = 0UL; fused_0_fuseiter_2128___fuseiter_2129_1041 < 32UL; fused_0_fuseiter_2128___fuseiter_2129_1041 += 1UL) {
    for (uint64_t _fuseiter_2130 = 0UL; _fuseiter_2130 < 3UL; _fuseiter_2130 += 1UL) {
      for (uint64_t _fuseiter_2131 = 0UL; _fuseiter_2131 < 3UL; _fuseiter_2131 += 1UL) {
        for (uint64_t _fuseiter_2132 = 0UL; _fuseiter_2132 < 16UL; _fuseiter_2132 += 1UL) {
          for (uint64_t _fuseiter_2133 = 0UL; _fuseiter_2133 < 128UL; _fuseiter_2133 += 1UL) {
            for (uint64_t _fuseiter_2134 = 0UL; _fuseiter_2134 < 4UL; _fuseiter_2134 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_2133 + ((fused_0_fuseiter_2128___fuseiter_2129_1041 / 8UL) * 128UL)) * 4608UL) + ((((_fuseiter_2134 + (_fuseiter_2132 * 4UL)) + ((fused_0_fuseiter_2128___fuseiter_2129_1041 % 8UL) * 64UL)) * 9UL) + ((_fuseiter_2130 * 3UL) + _fuseiter_2131)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_2128___fuseiter_2129_1041 / 8UL) * 589824UL) + (((fused_0_fuseiter_2128___fuseiter_2129_1041 % 8UL) * 73728UL) + ((_fuseiter_2130 * 24576UL) + ((_fuseiter_2131 * 8192UL) + ((_fuseiter_2132 * 512UL) + ((_fuseiter_2133 * 4UL) + _fuseiter_2134))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__257(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 = 0UL; fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 < 786432UL; fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 += 1UL) {
    for (uint64_t _fuseiter_2139 = 0UL; _fuseiter_2139 < 3UL; _fuseiter_2139 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 % 3UL) * 3UL))) + _fuseiter_2139)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1042____itr_2_1043 % 3UL) * 3UL))) + _fuseiter_2139)] = __cached_2;
    }
  }
  return true;
}

static bool cast__258(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 = 0UL; fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 < 786432UL; fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 += 1UL) {
    for (uint64_t _fuseiter2144 = 0UL; _fuseiter2144 < 3UL; _fuseiter2144 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 % 3UL) * 3UL))) + _fuseiter2144)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1044____itr_2_1045 % 3UL) * 3UL))) + _fuseiter2144)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__554(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 = 0UL; fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 < 24UL; fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 += 1UL) {
    for (uint64_t _fuseiter_2148 = 0UL; _fuseiter_2148 < 3UL; _fuseiter_2148 += 1UL) {
      for (uint64_t _fuseiter_2149 = 0UL; _fuseiter_2149 < 128UL; _fuseiter_2149 += 1UL) {
        for (uint64_t _fuseiter_2150 = 0UL; _fuseiter_2150 < 64UL; _fuseiter_2150 += 1UL) {
          for (uint64_t _fuseiter_2151 = 0UL; _fuseiter_2151 < 4UL; _fuseiter_2151 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_2150 + ((fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 / 3UL) * 64UL)) * 4608UL) + (((_fuseiter_2151 + (_fuseiter_2149 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 % 3UL) * 3UL) + _fuseiter_2148)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 / 3UL) * 294912UL) + (((fused_0fused_0_fuseiter_2145___fuseiter_2146_1046___fuseiter_2147_1047 % 3UL) * 98304UL) + ((_fuseiter_2148 * 32768UL) + ((_fuseiter_2149 * 256UL) + ((_fuseiter_2150 * 4UL) + _fuseiter_2151)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool batchwise_256_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 256UL; __batchwise_iter_0 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 1634816UL);
    // [s8 [1, 1, 4, 56, 56, 64] @ A1aBCD64b]
    int8_t* buffer_70 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_b_cast_mul_add_cast__4(buffer_70, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_1[0UL], &__ins_2[0UL], &__ins_3[0UL]);
    // [s8 [1, 1, 1, 58, 58, 64] @ A1aBCD64b]
    int8_t* buffer_71 = (int8_t*)&__rescheduled_1[802816UL];
    res2a_conv_0_cast_mul_add_cast_relu__8(buffer_71, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_4[0UL], &__ins_5[0UL], &__ins_6[0UL]);
    // [s8 [1, 1, 1, 56, 56, 64] @ A1aBCD64b]
    int8_t* buffer_72 = (int8_t*)&__rescheduled_1[1233408UL];
    res2a_conv_1_cast_mul_add_cast_relu__12(buffer_72, buffer_71, &__ins_7[0UL], &__ins_8[0UL], &__ins_9[0UL]);
    res2a_conv_2_cast_mul_add_cast_add_relu__16(buffer_70, buffer_72, &__ins_10[0UL], &__ins_11[0UL], &__ins_12[0UL], buffer_70);
    res2b_conv_0_cast_mul_add_cast_relu__20(buffer_72, buffer_70, &__ins_13[0UL], &__ins_14[0UL], &__ins_15[0UL]);
    res2b_conv_1_cast_mul_add_cast_relu__24(buffer_71, buffer_72, &__ins_16[0UL], &__ins_17[0UL], &__ins_18[0UL]);
    res2b_conv_2_cast_mul_add_cast_add_relu__28(buffer_70, buffer_71, &__ins_19[0UL], &__ins_20[0UL], &__ins_21[0UL], buffer_70);
    res2c_conv_0_cast_mul_add_cast_relu__32(buffer_72, buffer_70, &__ins_22[0UL], &__ins_23[0UL], &__ins_24[0UL]);
    res2c_conv_1_cast_mul_add_cast_relu__36(buffer_71, buffer_72, &__ins_25[0UL], &__ins_26[0UL], &__ins_27[0UL]);
    res2c_conv_2_cast_mul_add_cast_add_relu__40(buffer_70, buffer_71, &__ins_28[0UL], &__ins_29[0UL], &__ins_30[0UL], buffer_70);
    res3a_conv_b_cast_mul_add_cast__44(buffer_72, buffer_70, &__ins_31[0UL], &__ins_32[0UL], &__ins_33[0UL]);
    res3a_conv_0_cast_mul_add_cast_relu__48(buffer_71, buffer_70, &__ins_34[0UL], &__ins_35[0UL], &__ins_36[0UL]);
    res3a_conv_1_cast_mul_add_cast_relu__52(buffer_70, buffer_71, &__ins_37[0UL], &__ins_38[0UL], &__ins_39[0UL]);
    res3a_conv_2_cast_mul_add_cast_add_relu__56(buffer_72, buffer_70, &__ins_40[0UL], &__ins_41[0UL], &__ins_42[0UL], buffer_72);
    res3b_conv_0_cast_mul_add_cast_relu__60(buffer_70, buffer_72, &__ins_43[0UL], &__ins_44[0UL], &__ins_45[0UL]);
    res3b_conv_1_cast_mul_add_cast_relu__64(buffer_71, buffer_70, &__ins_46[0UL], &__ins_47[0UL], &__ins_48[0UL]);
    res3b_conv_2_cast_mul_add_cast_add_relu__68(buffer_72, buffer_71, &__ins_49[0UL], &__ins_50[0UL], &__ins_51[0UL], buffer_72);
    res3c_conv_0_cast_mul_add_cast_relu__72(buffer_70, buffer_72, &__ins_52[0UL], &__ins_53[0UL], &__ins_54[0UL]);
    res3c_conv_1_cast_mul_add_cast_relu__76(buffer_71, buffer_70, &__ins_55[0UL], &__ins_56[0UL], &__ins_57[0UL]);
    res3c_conv_2_cast_mul_add_cast_add_relu__80(buffer_72, buffer_71, &__ins_58[0UL], &__ins_59[0UL], &__ins_60[0UL], buffer_72);
    res3d_conv_0_cast_mul_add_cast_relu__84(buffer_70, buffer_72, &__ins_61[0UL], &__ins_62[0UL], &__ins_63[0UL]);
    res3d_conv_1_cast_mul_add_cast_relu__88(buffer_71, buffer_70, &__ins_64[0UL], &__ins_65[0UL], &__ins_66[0UL]);
    res3d_conv_2_cast_mul_add_cast_add_relu__93(&__outs_0[(__batchwise_iter_0 * 401408UL)], buffer_71, &__ins_67[0UL], &__ins_68[0UL], &__ins_69[0UL], buffer_72);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}


static bool res2a_conv_b_cast_mul_add_cast__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1048__k_1049 = 0UL; fused_0fused_0n__n_i_1048__k_1049 < 4UL; fused_0fused_0n__n_i_1048__k_1049 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_520_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1048__k_1049 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0fused_0n__n_i_1048__k_1049 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_0 = &__origouts_520_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_520_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2156 = 0UL; _fuseiter2156 < 56UL; _fuseiter2156 += 1UL) {
        for (uint64_t _fuseiter2157 = 0UL; _fuseiter2157 < 64UL; _fuseiter2157 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_520_shr[((_fuseiter2156 * 64UL) + _fuseiter2157)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1048__k_1049 / 4UL) * 256UL) + ((fused_0fused_0n__n_i_1048__k_1049 % 4UL) * 64UL)) + _fuseiter2157)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1048__k_1049 % 4UL) * 64UL) + _fuseiter2157)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[((((fused_0fused_0n__n_i_1048__k_1049 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_1048__k_1049 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter2156 * 64UL) + _fuseiter2157))]);
        }
      }
      sc_aligned_free(__stream, __origouts_520_shr);
    }
  }
  return true;
}

static bool res2a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_3 = *(void**)(__module_data + 16);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
    int32_t* __origouts_530_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[(p_o * 7168UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[0UL];
    B_list[0UL] = __cached_1;
    void* _arg_cache_1 = &__origouts_530_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_3, A_list, B_list, &__origouts_530_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter2183 = 0UL; _fuseiter2183 < 2UL; _fuseiter2183 += 1UL) {
      for (uint64_t _fuseiter2184 = 0UL; _fuseiter2184 < 56UL; _fuseiter2184 += 1UL) {
        for (uint64_t _fuseiter2185 = 0UL; _fuseiter2185 < 64UL; _fuseiter2185 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_530_shr[((_fuseiter2183 * 3584UL) + ((_fuseiter2184 * 64UL) + _fuseiter2185))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2185]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2185]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((p_o * 2UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter2183 * 3712UL) + ((_fuseiter2184 * 64UL) + _fuseiter2185)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_530_shr);
  }
  return true;
}

static bool res2a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_540_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((p_o + r) * 3712UL) + (s * 64UL))];
        A_list[((r * 3UL) + s)] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((r * 12288UL) + (s * 4096UL))];
        B_list[((r * 3UL) + s)] = __cached_1;
      }
    }
    void* _arg_cache_2 = &__origouts_540_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_5, A_list, B_list, &__origouts_540_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter2219 = 0UL; _fuseiter2219 < 56UL; _fuseiter2219 += 1UL) {
      for (uint64_t _fuseiter2220 = 0UL; _fuseiter2220 < 64UL; _fuseiter2220 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_540_shr[((_fuseiter2219 * 64UL) + _fuseiter2220)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2220]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2220]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter2219 * 64UL) + _fuseiter2220))]);
      }
    }
    sc_aligned_free(__stream, __origouts_540_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_1054__n_i_1055 = 0UL; fused_0fused_0k__n_1054__n_i_1055 < 4UL; fused_0fused_0k__n_1054__n_i_1055 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_550_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0fused_0k__n_1054__n_i_1055 * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_3 = &__origouts_550_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_550_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2254 = 0UL; _fuseiter2254 < 56UL; _fuseiter2254 += 1UL) {
        for (uint64_t _fuseiter2255 = 0UL; _fuseiter2255 < 64UL; _fuseiter2255 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_550_shr[((_fuseiter2254 * 64UL) + _fuseiter2255)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_1054__n_i_1055 * 64UL) + _fuseiter2255)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_1054__n_i_1055 * 64UL) + _fuseiter2255)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_1054__n_i_1055 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter2254 * 64UL) + _fuseiter2255))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_1054__n_i_1055 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter2254 * 64UL) + _fuseiter2255))]);
        }
      }
      sc_aligned_free(__stream, __origouts_550_shr);
    }
  }
  return true;
}

static bool res2b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    int32_t* __origouts_560_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 200704UL) + (p_o * 3584UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 4096UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_4 = &__origouts_560_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_7, A_list, B_list, &__origouts_560_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter2296 = 0UL; _fuseiter2296 < 56UL; _fuseiter2296 += 1UL) {
      for (uint64_t _fuseiter2297 = 0UL; _fuseiter2297 < 64UL; _fuseiter2297 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_560_shr[((_fuseiter2296 * 64UL) + _fuseiter2297)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2297]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2297]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((((p_o + 1UL) * 3712UL) + 64UL) + ((_fuseiter2296 * 64UL) + _fuseiter2297))]);
      }
    }
    sc_aligned_free(__stream, __origouts_560_shr);
  }
  return true;
}

static bool res2b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_570_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((p_o + r) * 3712UL) + (s * 64UL))];
        A_list[((r * 3UL) + s)] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((r * 12288UL) + (s * 4096UL))];
        B_list[((r * 3UL) + s)] = __cached_1;
      }
    }
    void* _arg_cache_5 = &__origouts_570_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_5, A_list, B_list, &__origouts_570_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter2331 = 0UL; _fuseiter2331 < 56UL; _fuseiter2331 += 1UL) {
      for (uint64_t _fuseiter2332 = 0UL; _fuseiter2332 < 64UL; _fuseiter2332 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_570_shr[((_fuseiter2331 * 64UL) + _fuseiter2332)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2332]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2332]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter2331 * 64UL) + _fuseiter2332))]);
      }
    }
    sc_aligned_free(__stream, __origouts_570_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_1060__n_i_1061 = 0UL; fused_0fused_0k__n_1060__n_i_1061 < 4UL; fused_0fused_0k__n_1060__n_i_1061 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_580_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0fused_0k__n_1060__n_i_1061 * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_6 = &__origouts_580_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_580_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2366 = 0UL; _fuseiter2366 < 56UL; _fuseiter2366 += 1UL) {
        for (uint64_t _fuseiter2367 = 0UL; _fuseiter2367 < 64UL; _fuseiter2367 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_580_shr[((_fuseiter2366 * 64UL) + _fuseiter2367)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_1060__n_i_1061 * 64UL) + _fuseiter2367)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_1060__n_i_1061 * 64UL) + _fuseiter2367)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_1060__n_i_1061 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter2366 * 64UL) + _fuseiter2367))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_1060__n_i_1061 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter2366 * 64UL) + _fuseiter2367))]);
        }
      }
      sc_aligned_free(__stream, __origouts_580_shr);
    }
  }
  return true;
}

static bool res2c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    int32_t* __origouts_590_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 200704UL) + (p_o * 3584UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 4096UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_7 = &__origouts_590_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_7, A_list, B_list, &__origouts_590_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter2408 = 0UL; _fuseiter2408 < 56UL; _fuseiter2408 += 1UL) {
      for (uint64_t _fuseiter2409 = 0UL; _fuseiter2409 < 64UL; _fuseiter2409 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_590_shr[((_fuseiter2408 * 64UL) + _fuseiter2409)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2409]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2409]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((((p_o + 1UL) * 3712UL) + 64UL) + ((_fuseiter2408 * 64UL) + _fuseiter2409))]);
      }
    }
    sc_aligned_free(__stream, __origouts_590_shr);
  }
  return true;
}

static bool res2c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_600_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((p_o + r) * 3712UL) + (s * 64UL))];
        A_list[((r * 3UL) + s)] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((r * 12288UL) + (s * 4096UL))];
        B_list[((r * 3UL) + s)] = __cached_1;
      }
    }
    void* _arg_cache_8 = &__origouts_600_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_5, A_list, B_list, &__origouts_600_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter2443 = 0UL; _fuseiter2443 < 56UL; _fuseiter2443 += 1UL) {
      for (uint64_t _fuseiter2444 = 0UL; _fuseiter2444 < 64UL; _fuseiter2444 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_600_shr[((_fuseiter2443 * 64UL) + _fuseiter2444)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2444]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2444]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter2443 * 64UL) + _fuseiter2444))]);
      }
    }
    sc_aligned_free(__stream, __origouts_600_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1066__k_1067 = 0UL; fused_0fused_0n__n_i_1066__k_1067 < 4UL; fused_0fused_0n__n_i_1066__k_1067 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_610_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1066__k_1067 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0fused_0n__n_i_1066__k_1067 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_9 = &__origouts_610_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_610_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2478 = 0UL; _fuseiter2478 < 56UL; _fuseiter2478 += 1UL) {
        for (uint64_t _fuseiter2479 = 0UL; _fuseiter2479 < 64UL; _fuseiter2479 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_610_shr[((_fuseiter2478 * 64UL) + _fuseiter2479)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1066__k_1067 / 4UL) * 256UL) + ((fused_0fused_0n__n_i_1066__k_1067 % 4UL) * 64UL)) + _fuseiter2479)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1066__k_1067 % 4UL) * 64UL) + _fuseiter2479)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1066__k_1067 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_1066__k_1067 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter2478 * 64UL) + _fuseiter2479))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1066__k_1067 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_1066__k_1067 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter2478 * 64UL) + _fuseiter2479))]);
        }
      }
      sc_aligned_free(__stream, __origouts_610_shr);
    }
  }
  return true;
}

static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  int8_t* input_tmp = (int8_t*)sc_aligned_malloc(__stream, 200704UL);
  for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
    for (uint64_t p = 0UL; p < 28UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 28UL; q += 1UL) {
        vec_s8x64 __cached_0;
        __cached_0 = vec_s8x64::load(&__ins_0[((c_o * 200704UL) + ((p * 7168UL) + (q * 128UL)))]);
        vec_s8x64 __cached_1;
        __cached_1 = __cached_0;
        vec_s8x64::store(__cached_1, &input_tmp[((c_o * 50176UL) + ((p * 1792UL) + (q * 64UL)))]);
      }
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1068__k_1069 = 0UL; fused_0fused_0n__n_i_1068__k_1069 < 8UL; fused_0fused_0n__n_i_1068__k_1069 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_620_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[(((fused_0fused_0n__n_i_1068__k_1069 / 8UL) * 200704UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[(((fused_0fused_0n__n_i_1068__k_1069 % 8UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_10 = &__origouts_620_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_7, A_list, B_list, &__origouts_620_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter2519 = 0UL; _fuseiter2519 < 2UL; _fuseiter2519 += 1UL) {
        for (uint64_t _fuseiter2520 = 0UL; _fuseiter2520 < 28UL; _fuseiter2520 += 1UL) {
          for (uint64_t _fuseiter2521 = 0UL; _fuseiter2521 < 64UL; _fuseiter2521 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_620_shr[((_fuseiter2519 * 1792UL) + ((_fuseiter2520 * 64UL) + _fuseiter2521))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1068__k_1069 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_1068__k_1069 % 8UL) * 64UL)) + _fuseiter2521)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1068__k_1069 % 8UL) * 64UL) + _fuseiter2521)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1068__k_1069 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1068__k_1069 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter2519 * 1792UL) + ((_fuseiter2520 * 64UL) + _fuseiter2521)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_620_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res3a_conv_0_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 215296UL)], 0, 3712UL);
    for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
      memset(&__outs_0[((k * 215296UL) + ((p1 + 1UL) * 3712UL))], 0, 64UL);
      memset(&__outs_0[(((k * 215296UL) + ((p1 + 1UL) * 3712UL)) + 3648UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 215296UL) + 211584UL)], 0, 3712UL);
  }
  for (uint64_t fused_0fused_0n__n_i_1070__k_1071 = 0UL; fused_0fused_0n__n_i_1070__k_1071 < 2UL; fused_0fused_0n__n_i_1070__k_1071 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      for (uint64_t q_i = 0UL; q_i < 2UL; q_i += 1UL) {
        int32_t* __origouts_630_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
        void** A_list = (void**)&__rescheduled_0[0UL];
        void** B_list = (void**)&__rescheduled_0[64UL];
        for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1070__k_1071 / 2UL) * 802816UL) + ((c * 200704UL) + ((p_o * 3584UL) + (q_i * 1792UL))))];
          A_list[c] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1070__k_1071 % 2UL) * 16384UL) + (c * 4096UL))];
          B_list[c] = __cached_1;
        }
        void* _arg_cache_11 = &__origouts_630_shr[0UL];
        dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_630_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
        for (uint64_t _fuseiter2548 = 0UL; _fuseiter2548 < 28UL; _fuseiter2548 += 1UL) {
          for (uint64_t _fuseiter2549 = 0UL; _fuseiter2549 < 64UL; _fuseiter2549 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_630_shr[((_fuseiter2548 * 64UL) + _fuseiter2549)]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1070__k_1071 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_1070__k_1071 % 2UL) * 64UL)) + _fuseiter2549)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1070__k_1071 % 2UL) * 64UL) + _fuseiter2549)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[((((fused_0fused_0n__n_i_1070__k_1071 / 2UL) * 430592UL) + (((fused_0fused_0n__n_i_1070__k_1071 % 2UL) * 215296UL) + (((p_o + 1UL) * 3712UL) + (((q_i * 28UL) + 1UL) * 64UL)))) + ((_fuseiter2548 * 64UL) + _fuseiter2549))]);
          }
        }
        sc_aligned_free(__stream, __origouts_630_shr);
      }
    }
  }
  return true;
}

static bool res3a_conv_1_cast_mul_add_cast_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_11 = *(void**)(__module_data + 48);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  for (uint64_t fused_0fused_0k_o__n_1072__n_i_1073 = 0UL; fused_0fused_0k_o__n_1072__n_i_1073 < 2UL; fused_0fused_0k_o__n_1072__n_i_1073 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[192UL];
      int32_t* __origouts_640_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[((c_o * 215296UL) + ((((p_o * 2UL) + r) * 3712UL) + (s * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[((fused_0fused_0k_o__n_1072__n_i_1073 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_12 = &__origouts_640_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_11, A_list, B_list, &__origouts_640_shr[0UL], 1, 64, 4096, 18, 7, 7, __stream);
      for (uint64_t _fuseiter2583 = 0UL; _fuseiter2583 < 28UL; _fuseiter2583 += 1UL) {
        for (uint64_t _fuseiter2584 = 0UL; _fuseiter2584 < 64UL; _fuseiter2584 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_640_shr[((_fuseiter2583 * 64UL) + _fuseiter2584)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_1072__n_i_1073 * 64UL) + _fuseiter2584)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_1072__n_i_1073 * 64UL) + _fuseiter2584)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0k_o__n_1072__n_i_1073 * 50176UL) + (p_o * 1792UL)) + ((_fuseiter2583 * 64UL) + _fuseiter2584))]);
        }
      }
      sc_aligned_free(__stream, __origouts_640_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3a_conv_2_cast_mul_add_cast_add_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_13 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_1074__n_i_1075 = 0UL; fused_0fused_0k__n_1074__n_i_1075 < 8UL; fused_0fused_0k__n_1074__n_i_1075 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_650_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 50176UL) + (p_o * 3584UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0fused_0k__n_1074__n_i_1075 * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_13 = &__origouts_650_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_13, A_list, B_list, &__origouts_650_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter2617 = 0UL; _fuseiter2617 < 2UL; _fuseiter2617 += 1UL) {
        for (uint64_t _fuseiter2618 = 0UL; _fuseiter2618 < 28UL; _fuseiter2618 += 1UL) {
          for (uint64_t _fuseiter2619 = 0UL; _fuseiter2619 < 64UL; _fuseiter2619 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_650_shr[((_fuseiter2617 * 1792UL) + ((_fuseiter2618 * 64UL) + _fuseiter2619))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_1074__n_i_1075 * 64UL) + _fuseiter2619)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_1074__n_i_1075 * 64UL) + _fuseiter2619)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_1074__n_i_1075 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter2617 * 1792UL) + ((_fuseiter2618 * 64UL) + _fuseiter2619)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_1074__n_i_1075 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter2617 * 1792UL) + ((_fuseiter2618 * 64UL) + _fuseiter2619)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_650_shr);
    }
  }
  return true;
}

static bool res3b_conv_0_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_15 = *(void**)(__module_data + 64);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_1076__k_1077 = 0UL; fused_0fused_0n__n_i_1076__k_1077 < 2UL; fused_0fused_0n__n_i_1076__k_1077 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_660_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1076__k_1077 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1076__k_1077 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_14 = &__origouts_660_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_15, A_list, B_list, &__origouts_660_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter2660 = 0UL; _fuseiter2660 < 28UL; _fuseiter2660 += 1UL) {
        for (uint64_t _fuseiter2661 = 0UL; _fuseiter2661 < 64UL; _fuseiter2661 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_660_shr[((_fuseiter2660 * 64UL) + _fuseiter2661)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1076__k_1077 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_1076__k_1077 % 2UL) * 64UL)) + _fuseiter2661)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1076__k_1077 % 2UL) * 64UL) + _fuseiter2661)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1076__k_1077 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_1076__k_1077 % 2UL) * 57600UL) + ((p_o + 1UL) * 1920UL))) + 64UL) + ((_fuseiter2660 * 64UL) + _fuseiter2661))]);
        }
      }
      sc_aligned_free(__stream, __origouts_660_shr);
    }
  }
  return true;
}

static bool res3b_conv_1_cast_mul_add_cast_relu__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 448UL);
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t __cached_0;
  __cached_0 = 0;
  conv_os_acc_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 392;
  conv_os_acc_size[1] = __cached_1;
  for (uint64_t fused_0fused_0k_o__n_1078__n_i_1079 = 0UL; fused_0fused_0k_o__n_1078__n_i_1079 < 2UL; fused_0fused_0k_o__n_1078__n_i_1079 += 1UL) {
    int32_t* __origouts_670_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    for (uint64_t o_o = 0UL; o_o < 2UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[64UL];
      void** B_list = (void**)&__rescheduled_0[256UL];
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_2;
            __cached_2 = &__ins_0[((c_o * 57600UL) + (((((o_o * 419UL) / 30UL) + r) * 1920UL) + ((((o_o * 419UL) % 30UL) + s) * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
            void* __cached_3;
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_1078__n_i_1079 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_15 = &__origouts_670_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_670_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2694 = 0UL; _fuseiter2694 < 28UL; _fuseiter2694 += 1UL) {
      for (uint64_t _fuseiter2695 = 0UL; _fuseiter2695 < 28UL; _fuseiter2695 += 1UL) {
        for (uint64_t _fuseiter2696 = 0UL; _fuseiter2696 < 64UL; _fuseiter2696 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_670_shr[((_fuseiter2694 * 1792UL) + ((_fuseiter2695 * 64UL) + _fuseiter2696))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_1078__n_i_1079 * 64UL) + _fuseiter2696)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_1078__n_i_1079 * 64UL) + _fuseiter2696)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_1078__n_i_1079 * 50176UL) + ((_fuseiter2694 * 1792UL) + ((_fuseiter2695 * 64UL) + _fuseiter2696)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_670_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3b_conv_2_cast_mul_add_cast_add_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_18 = *(void**)(__module_data + 72);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1080__k_1081 = 0UL; fused_0fused_0n__n_i_1080__k_1081 < 8UL; fused_0fused_0n__n_i_1080__k_1081 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_680_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1080__k_1081 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1080__k_1081 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_16 = &__origouts_680_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_18, A_list, B_list, &__origouts_680_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter2730 = 0UL; _fuseiter2730 < 28UL; _fuseiter2730 += 1UL) {
        for (uint64_t _fuseiter2731 = 0UL; _fuseiter2731 < 64UL; _fuseiter2731 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_680_shr[((_fuseiter2730 * 64UL) + _fuseiter2731)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1080__k_1081 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_1080__k_1081 % 8UL) * 64UL)) + _fuseiter2731)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1080__k_1081 % 8UL) * 64UL) + _fuseiter2731)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1080__k_1081 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1080__k_1081 % 8UL) * 50176UL) + (p_o * 1792UL))) + ((_fuseiter2730 * 64UL) + _fuseiter2731))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1080__k_1081 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1080__k_1081 % 8UL) * 50176UL) + (p_o * 1792UL))) + ((_fuseiter2730 * 64UL) + _fuseiter2731))]);
        }
      }
      sc_aligned_free(__stream, __origouts_680_shr);
    }
  }
  return true;
}

static bool res3c_conv_0_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_20 = *(void**)(__module_data + 80);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_1082__k_1083 = 0UL; fused_0fused_0n__n_i_1082__k_1083 < 2UL; fused_0fused_0n__n_i_1082__k_1083 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_690_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1082__k_1083 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 25088UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1082__k_1083 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_17 = &__origouts_690_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_20, A_list, B_list, &__origouts_690_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter2771 = 0UL; _fuseiter2771 < 14UL; _fuseiter2771 += 1UL) {
        for (uint64_t _fuseiter2772 = 0UL; _fuseiter2772 < 28UL; _fuseiter2772 += 1UL) {
          for (uint64_t _fuseiter2773 = 0UL; _fuseiter2773 < 64UL; _fuseiter2773 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_690_shr[((_fuseiter2771 * 1792UL) + ((_fuseiter2772 * 64UL) + _fuseiter2773))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1082__k_1083 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_1082__k_1083 % 2UL) * 64UL)) + _fuseiter2773)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1082__k_1083 % 2UL) * 64UL) + _fuseiter2773)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1082__k_1083 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_1082__k_1083 % 2UL) * 57600UL) + (((p_o * 14UL) + 1UL) * 1920UL))) + 64UL) + ((_fuseiter2771 * 1920UL) + ((_fuseiter2772 * 64UL) + _fuseiter2773)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_690_shr);
    }
  }
  return true;
}

static bool res3c_conv_1_cast_mul_add_cast_relu__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 448UL);
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t __cached_0;
  __cached_0 = 0;
  conv_os_acc_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 392;
  conv_os_acc_size[1] = __cached_1;
  for (uint64_t fused_0fused_0k_o__n_1084__n_i_1085 = 0UL; fused_0fused_0k_o__n_1084__n_i_1085 < 2UL; fused_0fused_0k_o__n_1084__n_i_1085 += 1UL) {
    int32_t* __origouts_700_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    for (uint64_t o_o = 0UL; o_o < 2UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[64UL];
      void** B_list = (void**)&__rescheduled_0[256UL];
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_2;
            __cached_2 = &__ins_0[((c_o * 57600UL) + (((((o_o * 419UL) / 30UL) + r) * 1920UL) + ((((o_o * 419UL) % 30UL) + s) * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
            void* __cached_3;
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_1084__n_i_1085 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_18 = &__origouts_700_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_700_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2806 = 0UL; _fuseiter2806 < 28UL; _fuseiter2806 += 1UL) {
      for (uint64_t _fuseiter2807 = 0UL; _fuseiter2807 < 28UL; _fuseiter2807 += 1UL) {
        for (uint64_t _fuseiter2808 = 0UL; _fuseiter2808 < 64UL; _fuseiter2808 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_700_shr[((_fuseiter2806 * 1792UL) + ((_fuseiter2807 * 64UL) + _fuseiter2808))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_1084__n_i_1085 * 64UL) + _fuseiter2808)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_1084__n_i_1085 * 64UL) + _fuseiter2808)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_1084__n_i_1085 * 50176UL) + ((_fuseiter2806 * 1792UL) + ((_fuseiter2807 * 64UL) + _fuseiter2808)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_700_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3c_conv_2_cast_mul_add_cast_add_relu__80(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_22 = *(void**)(__module_data + 88);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1086__k_1087 = 0UL; fused_0fused_0n__n_i_1086__k_1087 < 8UL; fused_0fused_0n__n_i_1086__k_1087 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_710_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1086__k_1087 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 7168UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1086__k_1087 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_19 = &__origouts_710_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_22, A_list, B_list, &__origouts_710_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter2841 = 0UL; _fuseiter2841 < 4UL; _fuseiter2841 += 1UL) {
        for (uint64_t _fuseiter2842 = 0UL; _fuseiter2842 < 28UL; _fuseiter2842 += 1UL) {
          for (uint64_t _fuseiter2843 = 0UL; _fuseiter2843 < 64UL; _fuseiter2843 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_710_shr[((_fuseiter2841 * 1792UL) + ((_fuseiter2842 * 64UL) + _fuseiter2843))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1086__k_1087 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_1086__k_1087 % 8UL) * 64UL)) + _fuseiter2843)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1086__k_1087 % 8UL) * 64UL) + _fuseiter2843)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1086__k_1087 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1086__k_1087 % 8UL) * 50176UL) + (p_o * 7168UL))) + ((_fuseiter2841 * 1792UL) + ((_fuseiter2842 * 64UL) + _fuseiter2843)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1086__k_1087 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1086__k_1087 % 8UL) * 50176UL) + (p_o * 7168UL))) + ((_fuseiter2841 * 1792UL) + ((_fuseiter2842 * 64UL) + _fuseiter2843)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_710_shr);
    }
  }
  return true;
}

static bool res3d_conv_0_cast_mul_add_cast_relu__84(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_24 = *(void**)(__module_data + 96);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_1088__k_1089 = 0UL; fused_0fused_0n__n_i_1088__k_1089 < 2UL; fused_0fused_0n__n_i_1088__k_1089 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_720_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1088__k_1089 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1088__k_1089 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_20 = &__origouts_720_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_24, A_list, B_list, &__origouts_720_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter2883 = 0UL; _fuseiter2883 < 2UL; _fuseiter2883 += 1UL) {
        for (uint64_t _fuseiter2884 = 0UL; _fuseiter2884 < 28UL; _fuseiter2884 += 1UL) {
          for (uint64_t _fuseiter2885 = 0UL; _fuseiter2885 < 64UL; _fuseiter2885 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_720_shr[((_fuseiter2883 * 1792UL) + ((_fuseiter2884 * 64UL) + _fuseiter2885))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1088__k_1089 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_1088__k_1089 % 2UL) * 64UL)) + _fuseiter2885)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1088__k_1089 % 2UL) * 64UL) + _fuseiter2885)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1088__k_1089 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_1088__k_1089 % 2UL) * 57600UL) + (((p_o * 2UL) + 1UL) * 1920UL))) + 64UL) + ((_fuseiter2883 * 1920UL) + ((_fuseiter2884 * 64UL) + _fuseiter2885)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_720_shr);
    }
  }
  return true;
}

static bool res3d_conv_1_cast_mul_add_cast_relu__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 448UL);
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t __cached_0;
  __cached_0 = 0;
  conv_os_acc_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 392;
  conv_os_acc_size[1] = __cached_1;
  for (uint64_t fused_0fused_0k_o__n_1090__n_i_1091 = 0UL; fused_0fused_0k_o__n_1090__n_i_1091 < 2UL; fused_0fused_0k_o__n_1090__n_i_1091 += 1UL) {
    int32_t* __origouts_730_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    for (uint64_t o_o = 0UL; o_o < 2UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[64UL];
      void** B_list = (void**)&__rescheduled_0[256UL];
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_2;
            __cached_2 = &__ins_0[((c_o * 57600UL) + (((((o_o * 419UL) / 30UL) + r) * 1920UL) + ((((o_o * 419UL) % 30UL) + s) * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
            void* __cached_3;
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_1090__n_i_1091 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_21 = &__origouts_730_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_730_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2918 = 0UL; _fuseiter2918 < 28UL; _fuseiter2918 += 1UL) {
      for (uint64_t _fuseiter2919 = 0UL; _fuseiter2919 < 28UL; _fuseiter2919 += 1UL) {
        for (uint64_t _fuseiter2920 = 0UL; _fuseiter2920 < 64UL; _fuseiter2920 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_730_shr[((_fuseiter2918 * 1792UL) + ((_fuseiter2919 * 64UL) + _fuseiter2920))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_1090__n_i_1091 * 64UL) + _fuseiter2920)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_1090__n_i_1091 * 64UL) + _fuseiter2920)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_1090__n_i_1091 * 50176UL) + ((_fuseiter2918 * 1792UL) + ((_fuseiter2919 * 64UL) + _fuseiter2920)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_730_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3d_conv_2_cast_mul_add_cast_add_relu__93(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_13 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1092__k_1093 = 0UL; fused_0fused_0n__n_i_1092__k_1093 < 8UL; fused_0fused_0n__n_i_1092__k_1093 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_740_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1092__k_1093 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1092__k_1093 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_22 = &__origouts_740_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_13, A_list, B_list, &__origouts_740_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter2953 = 0UL; _fuseiter2953 < 2UL; _fuseiter2953 += 1UL) {
        for (uint64_t _fuseiter2954 = 0UL; _fuseiter2954 < 28UL; _fuseiter2954 += 1UL) {
          for (uint64_t _fuseiter2955 = 0UL; _fuseiter2955 < 64UL; _fuseiter2955 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_740_shr[((_fuseiter2953 * 1792UL) + ((_fuseiter2954 * 64UL) + _fuseiter2955))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_1092__k_1093 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_1092__k_1093 % 8UL) * 64UL)) + _fuseiter2955)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1092__k_1093 % 8UL) * 64UL) + _fuseiter2955)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1092__k_1093 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1092__k_1093 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter2953 * 1792UL) + ((_fuseiter2954 * 64UL) + _fuseiter2955)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1092__k_1093 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_1092__k_1093 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter2953 * 1792UL) + ((_fuseiter2954 * 64UL) + _fuseiter2955)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_740_shr);
    }
  }
  return true;
}

static bool batchwise_128_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 128UL; __batchwise_iter_0 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 993280UL);
    // [s8 [1, 2, 16, 14, 14, 64] @ A2aBCD64b]
    int8_t* buffer_58 = (int8_t*)&__rescheduled_1[0UL];
    res4a_conv_b_cast_mul_add_cast__4(buffer_58, &__ins_0[(__batchwise_iter_0 * 802816UL)], &__ins_1[0UL], &__ins_2[0UL], &__ins_3[0UL]);
    // [s8 [1, 2, 4, 30, 30, 64] @ A2aBCD64b]
    int8_t* buffer_59 = (int8_t*)&__rescheduled_1[401408UL];
    res4a_conv_0_cast_mul_add_cast_relu__8(buffer_59, &__ins_0[(__batchwise_iter_0 * 802816UL)], &__ins_4[0UL], &__ins_5[0UL], &__ins_6[0UL]);
    // [s8 [1, 2, 4, 14, 14, 64] @ A2aBCD64b]
    int8_t* buffer_60 = (int8_t*)&__rescheduled_1[862208UL];
    res4a_conv_1_cast_mul_add_cast_relu__12(buffer_60, buffer_59, &__ins_7[0UL], &__ins_8[0UL], &__ins_9[0UL]);
    res4a_conv_2_cast_mul_add_cast_add_relu__16(buffer_58, buffer_60, &__ins_10[0UL], &__ins_11[0UL], &__ins_12[0UL], buffer_58);
    res4b_conv_0_cast_mul_add_cast_relu__20(buffer_60, buffer_58, &__ins_13[0UL], &__ins_14[0UL], &__ins_15[0UL]);
    res4b_conv_1_cast_mul_add_cast_relu__24(buffer_59, buffer_60, &__ins_16[0UL], &__ins_17[0UL], &__ins_18[0UL]);
    res4b_conv_2_cast_mul_add_cast_add_relu__28(buffer_58, buffer_59, &__ins_19[0UL], &__ins_20[0UL], &__ins_21[0UL], buffer_58);
    res4c_conv_0_cast_mul_add_cast_relu__32(buffer_60, buffer_58, &__ins_22[0UL], &__ins_23[0UL], &__ins_24[0UL]);
    res4c_conv_1_cast_mul_add_cast_relu__36(buffer_59, buffer_60, &__ins_25[0UL], &__ins_26[0UL], &__ins_27[0UL]);
    res4c_conv_2_cast_mul_add_cast_add_relu__40(buffer_58, buffer_59, &__ins_28[0UL], &__ins_29[0UL], &__ins_30[0UL], buffer_58);
    res4d_conv_0_cast_mul_add_cast_relu__44(buffer_60, buffer_58, &__ins_31[0UL], &__ins_32[0UL], &__ins_33[0UL]);
    res4d_conv_1_cast_mul_add_cast_relu__48(buffer_59, buffer_60, &__ins_34[0UL], &__ins_35[0UL], &__ins_36[0UL]);
    res4d_conv_2_cast_mul_add_cast_add_relu__52(buffer_58, buffer_59, &__ins_37[0UL], &__ins_38[0UL], &__ins_39[0UL], buffer_58);
    res4e_conv_0_cast_mul_add_cast_relu__56(buffer_60, buffer_58, &__ins_40[0UL], &__ins_41[0UL], &__ins_42[0UL]);
    res4e_conv_1_cast_mul_add_cast_relu__60(buffer_59, buffer_60, &__ins_43[0UL], &__ins_44[0UL], &__ins_45[0UL]);
    res4e_conv_2_cast_mul_add_cast_add_relu__64(buffer_58, buffer_59, &__ins_46[0UL], &__ins_47[0UL], &__ins_48[0UL], buffer_58);
    res4f_conv_0_cast_mul_add_cast_relu__68(buffer_60, buffer_58, &__ins_49[0UL], &__ins_50[0UL], &__ins_51[0UL]);
    res4f_conv_1_cast_mul_add_cast_relu__72(buffer_59, buffer_60, &__ins_52[0UL], &__ins_53[0UL], &__ins_54[0UL]);
    res4f_conv_2_cast_mul_add_cast_add_relu__77(&__outs_0[(__batchwise_iter_0 * 401408UL)], buffer_59, &__ins_55[0UL], &__ins_56[0UL], &__ins_57[0UL], buffer_58);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

extern "C" void main_entry_1(int8_t* __restrict__ buffer_61, int8_t* __restrict__ buffer_57, int8_t* __restrict__ buffer_56, float* __restrict__ buffer_55, float* __restrict__ buffer_54, int8_t* __restrict__ buffer_53, float* __restrict__ buffer_52, float* __restrict__ buffer_51, int8_t* __restrict__ buffer_50, float* __restrict__ buffer_49, float* __restrict__ buffer_48, int8_t* __restrict__ buffer_47, float* __restrict__ buffer_46, float* __restrict__ buffer_45, int8_t* __restrict__ buffer_44, float* __restrict__ buffer_43, float* __restrict__ buffer_42, int8_t* __restrict__ buffer_41, float* __restrict__ buffer_40, float* __restrict__ buffer_39, int8_t* __restrict__ buffer_38, float* __restrict__ buffer_37, float* __restrict__ buffer_36, int8_t* __restrict__ buffer_35, float* __restrict__ buffer_34, float* __restrict__ buffer_33, int8_t* __restrict__ buffer_32, float* __restrict__ buffer_31, float* __restrict__ buffer_30, int8_t* __restrict__ buffer_29, float* __restrict__ buffer_28, float* __restrict__ buffer_27, int8_t* __restrict__ buffer_26, float* __restrict__ buffer_25, float* __restrict__ buffer_24, int8_t* __restrict__ buffer_23, float* __restrict__ buffer_22, float* __restrict__ buffer_21, int8_t* __restrict__ buffer_20, float* __restrict__ buffer_19, float* __restrict__ buffer_18, int8_t* __restrict__ buffer_17, float* __restrict__ buffer_16, float* __restrict__ buffer_15, int8_t* __restrict__ buffer_14, float* __restrict__ buffer_13, float* __restrict__ buffer_12, int8_t* __restrict__ buffer_11, float* __restrict__ buffer_10, float* __restrict__ buffer_9, int8_t* __restrict__ buffer_8, float* __restrict__ buffer_7, float* __restrict__ buffer_6, int8_t* __restrict__ buffer_5, float* __restrict__ buffer_4, float* __restrict__ buffer_3, int8_t* __restrict__ buffer_2, float* __restrict__ buffer_1, float* __restrict__ buffer_0) noexcept{
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 993280UL);
  // [s8 [1, 2, 16, 14, 14, 64] @ A2aBCD64b]
  int8_t* buffer_58 = (int8_t*)&__rescheduled_0[0UL];
  res4a_conv_b_cast_mul_add_cast__4(buffer_58, buffer_57, buffer_56, buffer_55, buffer_54);
  // [s8 [1, 2, 4, 30, 30, 64] @ A2aBCD64b]
  int8_t* buffer_59 = (int8_t*)&__rescheduled_0[401408UL];
  res4a_conv_0_cast_mul_add_cast_relu__8(buffer_59, buffer_57, buffer_53, buffer_52, buffer_51);
  // [s8 [1, 2, 4, 14, 14, 64] @ A2aBCD64b]
  int8_t* buffer_60 = (int8_t*)&__rescheduled_0[862208UL];
  res4a_conv_1_cast_mul_add_cast_relu__12(buffer_60, buffer_59, buffer_50, buffer_49, buffer_48);
  res4a_conv_2_cast_mul_add_cast_add_relu__16(buffer_58, buffer_60, buffer_47, buffer_46, buffer_45, buffer_58);
  res4b_conv_0_cast_mul_add_cast_relu__20(buffer_60, buffer_58, buffer_44, buffer_43, buffer_42);
  res4b_conv_1_cast_mul_add_cast_relu__24(buffer_59, buffer_60, buffer_41, buffer_40, buffer_39);
  res4b_conv_2_cast_mul_add_cast_add_relu__28(buffer_58, buffer_59, buffer_38, buffer_37, buffer_36, buffer_58);
  res4c_conv_0_cast_mul_add_cast_relu__32(buffer_60, buffer_58, buffer_35, buffer_34, buffer_33);
  res4c_conv_1_cast_mul_add_cast_relu__36(buffer_59, buffer_60, buffer_32, buffer_31, buffer_30);
  res4c_conv_2_cast_mul_add_cast_add_relu__40(buffer_58, buffer_59, buffer_29, buffer_28, buffer_27, buffer_58);
  res4d_conv_0_cast_mul_add_cast_relu__44(buffer_60, buffer_58, buffer_26, buffer_25, buffer_24);
  res4d_conv_1_cast_mul_add_cast_relu__48(buffer_59, buffer_60, buffer_23, buffer_22, buffer_21);
  res4d_conv_2_cast_mul_add_cast_add_relu__52(buffer_58, buffer_59, buffer_20, buffer_19, buffer_18, buffer_58);
  res4e_conv_0_cast_mul_add_cast_relu__56(buffer_60, buffer_58, buffer_17, buffer_16, buffer_15);
  res4e_conv_1_cast_mul_add_cast_relu__60(buffer_59, buffer_60, buffer_14, buffer_13, buffer_12);
  res4e_conv_2_cast_mul_add_cast_add_relu__64(buffer_58, buffer_59, buffer_11, buffer_10, buffer_9, buffer_58);
  res4f_conv_0_cast_mul_add_cast_relu__68(buffer_60, buffer_58, buffer_8, buffer_7, buffer_6);
  res4f_conv_1_cast_mul_add_cast_relu__72(buffer_59, buffer_60, buffer_5, buffer_4, buffer_3);
  res4f_conv_2_cast_mul_add_cast_add_relu__77(buffer_61, buffer_59, buffer_2, buffer_1, buffer_0, buffer_58);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool res4a_conv_b_cast_mul_add_cast__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_15 = *(void**)(__module_data + 64);
  alignas(64) int8_t __rescheduled_0[128UL];
  int8_t* input_tmp = (int8_t*)sc_aligned_malloc(__stream, 200704UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t c_o = 0UL; c_o < 8UL; c_o += 1UL) {
      for (uint64_t p = 0UL; p < 14UL; p += 1UL) {
        for (uint64_t q = 0UL; q < 14UL; q += 1UL) {
          vec_s8x64 __cached_0;
          __cached_0 = vec_s8x64::load(&__ins_0[((n_i * 401408UL) + ((c_o * 50176UL) + ((p * 3584UL) + (q * 128UL))))]);
          vec_s8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_s8x64::store(__cached_1, &input_tmp[((n_i * 100352UL) + ((c_o * 12544UL) + ((p * 896UL) + (q * 64UL))))]);
        }
      }
    }
  }
  for (uint64_t fused_0fused_0k__n_1094__n_i_1095 = 0UL; fused_0fused_0k__n_1094__n_i_1095 < 32UL; fused_0fused_0k__n_1094__n_i_1095 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_750_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[(((fused_0fused_0k__n_1094__n_i_1095 % 2UL) * 100352UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[(((fused_0fused_0k__n_1094__n_i_1095 / 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_23 = &__origouts_750_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_15, A_list, B_list, &__origouts_750_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter2995 = 0UL; _fuseiter2995 < 2UL; _fuseiter2995 += 1UL) {
        for (uint64_t _fuseiter2996 = 0UL; _fuseiter2996 < 14UL; _fuseiter2996 += 1UL) {
          for (uint64_t _fuseiter2997 = 0UL; _fuseiter2997 < 64UL; _fuseiter2997 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_750_shr[((_fuseiter2995 * 896UL) + ((_fuseiter2996 * 64UL) + _fuseiter2997))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_1094__n_i_1095 / 2UL) * 64UL) + _fuseiter2997)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_1094__n_i_1095 / 2UL) * 64UL) + _fuseiter2997)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_1094__n_i_1095 % 2UL) * 200704UL) + (((fused_0fused_0k__n_1094__n_i_1095 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter2995 * 896UL) + ((_fuseiter2996 * 64UL) + _fuseiter2997)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_750_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res4a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_24 = *(void**)(__module_data + 96);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 230400UL) + (k * 57600UL))], 0, 1920UL);
      for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 230400UL) + ((k * 57600UL) + ((p1 + 1UL) * 1920UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 230400UL) + ((k * 57600UL) + ((p1 + 1UL) * 1920UL))) + 1856UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 230400UL) + (k * 57600UL)) + 55680UL)], 0, 1920UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1096__k_1097 = 0UL; fused_0fused_0n__n_i_1096__k_1097 < 8UL; fused_0fused_0n__n_i_1096__k_1097 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_760_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1096__k_1097 / 8UL) * 802816UL) + ((((fused_0fused_0n__n_i_1096__k_1097 / 4UL) % 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 3584UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1096__k_1097 % 4UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_24 = &__origouts_760_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_24, A_list, B_list, &__origouts_760_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter3023 = 0UL; _fuseiter3023 < 2UL; _fuseiter3023 += 1UL) {
        for (uint64_t _fuseiter3024 = 0UL; _fuseiter3024 < 28UL; _fuseiter3024 += 1UL) {
          for (uint64_t _fuseiter3025 = 0UL; _fuseiter3025 < 64UL; _fuseiter3025 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_760_shr[((_fuseiter3023 * 1792UL) + ((_fuseiter3024 * 64UL) + _fuseiter3025))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1096__k_1097 % 4UL) * 64UL) + _fuseiter3025)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1096__k_1097 % 4UL) * 64UL) + _fuseiter3025)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1096__k_1097 / 8UL) * 460800UL) + ((((fused_0fused_0n__n_i_1096__k_1097 / 4UL) % 2UL) * 230400UL) + (((fused_0fused_0n__n_i_1096__k_1097 % 4UL) * 57600UL) + (((p_o * 2UL) + 1UL) * 1920UL)))) + 64UL) + ((_fuseiter3023 * 1920UL) + ((_fuseiter3024 * 64UL) + _fuseiter3025)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_760_shr);
    }
  }
  return true;
}

static bool res4a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_26 = *(void**)(__module_data + 104);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  for (uint64_t fused_0fused_0n__n_i_1098__k_o_1099 = 0UL; fused_0fused_0n__n_i_1098__k_o_1099 < 8UL; fused_0fused_0n__n_i_1098__k_o_1099 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[320UL];
      for (uint64_t p_i = 0UL; p_i < 2UL; p_i += 1UL) {
        int32_t* __origouts_770_shr = (int32_t*)sc_aligned_malloc(__stream, 3584UL);
        for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
          for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
            for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
              void* __cached_0;
              __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1098__k_o_1099 / 8UL) * 460800UL) + ((((fused_0fused_0n__n_i_1098__k_o_1099 / 4UL) % 2UL) * 230400UL) + ((c_o * 57600UL) + ((((((p_o * 2UL) + p_i) * 2UL) + r) * 1920UL) + (s * 64UL)))))];
              A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
              void* __cached_1;
              __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1098__k_o_1099 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
              B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
            }
          }
        }
        void* _arg_cache_25 = &__origouts_770_shr[0UL];
        dnnl_brgemm_list_call(__sc_kernel_cache_26, A_list, B_list, &__origouts_770_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
        for (uint64_t _fuseiter3059 = 0UL; _fuseiter3059 < 14UL; _fuseiter3059 += 1UL) {
          for (uint64_t _fuseiter3060 = 0UL; _fuseiter3060 < 64UL; _fuseiter3060 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_770_shr[((_fuseiter3059 * 64UL) + _fuseiter3060)]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1098__k_o_1099 % 4UL) * 64UL) + _fuseiter3060)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1098__k_o_1099 % 4UL) * 64UL) + _fuseiter3060)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[((((fused_0fused_0n__n_i_1098__k_o_1099 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_1098__k_o_1099 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_1098__k_o_1099 % 4UL) * 12544UL) + (((p_o * 2UL) + p_i) * 896UL)))) + ((_fuseiter3059 * 64UL) + _fuseiter3060))]);
          }
        }
        sc_aligned_free(__stream, __origouts_770_shr);
      }
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_1100__n_i_1101 = 0UL; fused_0fused_0k__n_1100__n_i_1101 < 32UL; fused_0fused_0k__n_1100__n_i_1101 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_780_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_1100__n_i_1101 % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_1100__n_i_1101 / 2UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_26 = &__origouts_780_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_780_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3093 = 0UL; _fuseiter3093 < 2UL; _fuseiter3093 += 1UL) {
        for (uint64_t _fuseiter3094 = 0UL; _fuseiter3094 < 14UL; _fuseiter3094 += 1UL) {
          for (uint64_t _fuseiter3095 = 0UL; _fuseiter3095 < 64UL; _fuseiter3095 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_780_shr[((_fuseiter3093 * 896UL) + ((_fuseiter3094 * 64UL) + _fuseiter3095))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_1100__n_i_1101 / 2UL) * 64UL) + _fuseiter3095)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_1100__n_i_1101 / 2UL) * 64UL) + _fuseiter3095)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_1100__n_i_1101 % 2UL) * 200704UL) + (((fused_0fused_0k__n_1100__n_i_1101 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter3093 * 896UL) + ((_fuseiter3094 * 64UL) + _fuseiter3095)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_1100__n_i_1101 % 2UL) * 200704UL) + (((fused_0fused_0k__n_1100__n_i_1101 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter3093 * 896UL) + ((_fuseiter3094 * 64UL) + _fuseiter3095)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_780_shr);
    }
  }
  return true;
}

static bool res4b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 65536UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 65536UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1102__k_1103 = 0UL; fused_0fused_0n__n_i_1102__k_1103 < 8UL; fused_0fused_0n__n_i_1102__k_1103 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_790_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1102__k_1103 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_1102__k_1103 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1102__k_1103 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_27 = &__origouts_790_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_790_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3135 = 0UL; _fuseiter3135 < 2UL; _fuseiter3135 += 1UL) {
        for (uint64_t _fuseiter3136 = 0UL; _fuseiter3136 < 14UL; _fuseiter3136 += 1UL) {
          for (uint64_t _fuseiter3137 = 0UL; _fuseiter3137 < 64UL; _fuseiter3137 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_790_shr[((_fuseiter3135 * 896UL) + ((_fuseiter3136 * 64UL) + _fuseiter3137))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1102__k_1103 % 4UL) * 64UL) + _fuseiter3137)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1102__k_1103 % 4UL) * 64UL) + _fuseiter3137)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1102__k_1103 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1102__k_1103 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_1102__k_1103 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter3135 * 1024UL) + ((_fuseiter3136 * 64UL) + _fuseiter3137)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_790_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_1104__k_o_1105 = 0UL; fused_0fused_0n__n_i_1104__k_o_1105 < 8UL; fused_0fused_0n__n_i_1104__k_o_1105 += 1UL) {
    int32_t* __origouts_800_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_1104__k_o_1105 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1104__k_o_1105 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_1104__k_o_1105 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_28 = &__origouts_800_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_32[0UL], A_list, B_list, &__origouts_800_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3170 = 0UL; _fuseiter3170 < 14UL; _fuseiter3170 += 1UL) {
      for (uint64_t _fuseiter3171 = 0UL; _fuseiter3171 < 14UL; _fuseiter3171 += 1UL) {
        for (uint64_t _fuseiter3172 = 0UL; _fuseiter3172 < 64UL; _fuseiter3172 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_800_shr[((_fuseiter3170 * 896UL) + ((_fuseiter3171 * 64UL) + _fuseiter3172))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1104__k_o_1105 % 4UL) * 64UL) + _fuseiter3172)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1104__k_o_1105 % 4UL) * 64UL) + _fuseiter3172)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_1104__k_o_1105 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_1104__k_o_1105 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_1104__k_o_1105 % 4UL) * 12544UL) + ((_fuseiter3170 * 896UL) + ((_fuseiter3171 * 64UL) + _fuseiter3172)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_800_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1106__k_1107 = 0UL; fused_0fused_0n__n_i_1106__k_1107 < 32UL; fused_0fused_0n__n_i_1106__k_1107 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_810_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1106__k_1107 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_1106__k_1107 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1106__k_1107 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_29 = &__origouts_810_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_810_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3205 = 0UL; _fuseiter3205 < 2UL; _fuseiter3205 += 1UL) {
        for (uint64_t _fuseiter3206 = 0UL; _fuseiter3206 < 14UL; _fuseiter3206 += 1UL) {
          for (uint64_t _fuseiter3207 = 0UL; _fuseiter3207 < 64UL; _fuseiter3207 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_810_shr[((_fuseiter3205 * 896UL) + ((_fuseiter3206 * 64UL) + _fuseiter3207))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1106__k_1107 % 16UL) * 64UL) + _fuseiter3207)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1106__k_1107 % 16UL) * 64UL) + _fuseiter3207)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1106__k_1107 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1106__k_1107 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1106__k_1107 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3205 * 896UL) + ((_fuseiter3206 * 64UL) + _fuseiter3207)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1106__k_1107 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1106__k_1107 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1106__k_1107 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3205 * 896UL) + ((_fuseiter3206 * 64UL) + _fuseiter3207)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_810_shr);
    }
  }
  return true;
}

static bool res4c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 65536UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 65536UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1108__k_1109 = 0UL; fused_0fused_0n__n_i_1108__k_1109 < 8UL; fused_0fused_0n__n_i_1108__k_1109 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_820_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1108__k_1109 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_1108__k_1109 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1108__k_1109 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_30 = &__origouts_820_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_820_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3247 = 0UL; _fuseiter3247 < 2UL; _fuseiter3247 += 1UL) {
        for (uint64_t _fuseiter3248 = 0UL; _fuseiter3248 < 14UL; _fuseiter3248 += 1UL) {
          for (uint64_t _fuseiter3249 = 0UL; _fuseiter3249 < 64UL; _fuseiter3249 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_820_shr[((_fuseiter3247 * 896UL) + ((_fuseiter3248 * 64UL) + _fuseiter3249))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1108__k_1109 % 4UL) * 64UL) + _fuseiter3249)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1108__k_1109 % 4UL) * 64UL) + _fuseiter3249)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1108__k_1109 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1108__k_1109 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_1108__k_1109 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter3247 * 1024UL) + ((_fuseiter3248 * 64UL) + _fuseiter3249)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_820_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_1110__k_o_1111 = 0UL; fused_0fused_0n__n_i_1110__k_o_1111 < 8UL; fused_0fused_0n__n_i_1110__k_o_1111 += 1UL) {
    int32_t* __origouts_830_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_1110__k_o_1111 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1110__k_o_1111 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_1110__k_o_1111 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_31 = &__origouts_830_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_32[0UL], A_list, B_list, &__origouts_830_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3282 = 0UL; _fuseiter3282 < 14UL; _fuseiter3282 += 1UL) {
      for (uint64_t _fuseiter3283 = 0UL; _fuseiter3283 < 14UL; _fuseiter3283 += 1UL) {
        for (uint64_t _fuseiter3284 = 0UL; _fuseiter3284 < 64UL; _fuseiter3284 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_830_shr[((_fuseiter3282 * 896UL) + ((_fuseiter3283 * 64UL) + _fuseiter3284))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1110__k_o_1111 % 4UL) * 64UL) + _fuseiter3284)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1110__k_o_1111 % 4UL) * 64UL) + _fuseiter3284)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_1110__k_o_1111 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_1110__k_o_1111 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_1110__k_o_1111 % 4UL) * 12544UL) + ((_fuseiter3282 * 896UL) + ((_fuseiter3283 * 64UL) + _fuseiter3284)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_830_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1112__k_1113 = 0UL; fused_0fused_0n__n_i_1112__k_1113 < 32UL; fused_0fused_0n__n_i_1112__k_1113 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_840_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1112__k_1113 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_1112__k_1113 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1112__k_1113 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_32 = &__origouts_840_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_840_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3317 = 0UL; _fuseiter3317 < 2UL; _fuseiter3317 += 1UL) {
        for (uint64_t _fuseiter3318 = 0UL; _fuseiter3318 < 14UL; _fuseiter3318 += 1UL) {
          for (uint64_t _fuseiter3319 = 0UL; _fuseiter3319 < 64UL; _fuseiter3319 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_840_shr[((_fuseiter3317 * 896UL) + ((_fuseiter3318 * 64UL) + _fuseiter3319))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1112__k_1113 % 16UL) * 64UL) + _fuseiter3319)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1112__k_1113 % 16UL) * 64UL) + _fuseiter3319)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1112__k_1113 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1112__k_1113 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1112__k_1113 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3317 * 896UL) + ((_fuseiter3318 * 64UL) + _fuseiter3319)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1112__k_1113 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1112__k_1113 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1112__k_1113 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3317 * 896UL) + ((_fuseiter3318 * 64UL) + _fuseiter3319)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_840_shr);
    }
  }
  return true;
}

static bool res4d_conv_0_cast_mul_add_cast_relu__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 65536UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 65536UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1114__k_1115 = 0UL; fused_0fused_0n__n_i_1114__k_1115 < 8UL; fused_0fused_0n__n_i_1114__k_1115 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_850_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1114__k_1115 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_1114__k_1115 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1114__k_1115 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_33 = &__origouts_850_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_850_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3359 = 0UL; _fuseiter3359 < 2UL; _fuseiter3359 += 1UL) {
        for (uint64_t _fuseiter3360 = 0UL; _fuseiter3360 < 14UL; _fuseiter3360 += 1UL) {
          for (uint64_t _fuseiter3361 = 0UL; _fuseiter3361 < 64UL; _fuseiter3361 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_850_shr[((_fuseiter3359 * 896UL) + ((_fuseiter3360 * 64UL) + _fuseiter3361))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1114__k_1115 % 4UL) * 64UL) + _fuseiter3361)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1114__k_1115 % 4UL) * 64UL) + _fuseiter3361)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1114__k_1115 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1114__k_1115 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_1114__k_1115 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter3359 * 1024UL) + ((_fuseiter3360 * 64UL) + _fuseiter3361)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_850_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_1_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_1116__k_o_1117 = 0UL; fused_0fused_0n__n_i_1116__k_o_1117 < 8UL; fused_0fused_0n__n_i_1116__k_o_1117 += 1UL) {
    int32_t* __origouts_860_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_1116__k_o_1117 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1116__k_o_1117 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_1116__k_o_1117 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_34 = &__origouts_860_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_32[0UL], A_list, B_list, &__origouts_860_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3394 = 0UL; _fuseiter3394 < 14UL; _fuseiter3394 += 1UL) {
      for (uint64_t _fuseiter3395 = 0UL; _fuseiter3395 < 14UL; _fuseiter3395 += 1UL) {
        for (uint64_t _fuseiter3396 = 0UL; _fuseiter3396 < 64UL; _fuseiter3396 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_860_shr[((_fuseiter3394 * 896UL) + ((_fuseiter3395 * 64UL) + _fuseiter3396))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1116__k_o_1117 % 4UL) * 64UL) + _fuseiter3396)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1116__k_o_1117 % 4UL) * 64UL) + _fuseiter3396)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_1116__k_o_1117 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_1116__k_o_1117 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_1116__k_o_1117 % 4UL) * 12544UL) + ((_fuseiter3394 * 896UL) + ((_fuseiter3395 * 64UL) + _fuseiter3396)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_860_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_2_cast_mul_add_cast_add_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1118__k_1119 = 0UL; fused_0fused_0n__n_i_1118__k_1119 < 32UL; fused_0fused_0n__n_i_1118__k_1119 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_870_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1118__k_1119 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_1118__k_1119 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1118__k_1119 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_35 = &__origouts_870_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_870_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3429 = 0UL; _fuseiter3429 < 2UL; _fuseiter3429 += 1UL) {
        for (uint64_t _fuseiter3430 = 0UL; _fuseiter3430 < 14UL; _fuseiter3430 += 1UL) {
          for (uint64_t _fuseiter3431 = 0UL; _fuseiter3431 < 64UL; _fuseiter3431 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_870_shr[((_fuseiter3429 * 896UL) + ((_fuseiter3430 * 64UL) + _fuseiter3431))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1118__k_1119 % 16UL) * 64UL) + _fuseiter3431)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1118__k_1119 % 16UL) * 64UL) + _fuseiter3431)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1118__k_1119 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1118__k_1119 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1118__k_1119 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3429 * 896UL) + ((_fuseiter3430 * 64UL) + _fuseiter3431)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1118__k_1119 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1118__k_1119 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1118__k_1119 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3429 * 896UL) + ((_fuseiter3430 * 64UL) + _fuseiter3431)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_870_shr);
    }
  }
  return true;
}

static bool res4e_conv_0_cast_mul_add_cast_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 65536UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 65536UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1120__k_1121 = 0UL; fused_0fused_0n__n_i_1120__k_1121 < 8UL; fused_0fused_0n__n_i_1120__k_1121 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_880_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1120__k_1121 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_1120__k_1121 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1120__k_1121 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_36 = &__origouts_880_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_880_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3471 = 0UL; _fuseiter3471 < 2UL; _fuseiter3471 += 1UL) {
        for (uint64_t _fuseiter3472 = 0UL; _fuseiter3472 < 14UL; _fuseiter3472 += 1UL) {
          for (uint64_t _fuseiter3473 = 0UL; _fuseiter3473 < 64UL; _fuseiter3473 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_880_shr[((_fuseiter3471 * 896UL) + ((_fuseiter3472 * 64UL) + _fuseiter3473))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1120__k_1121 % 4UL) * 64UL) + _fuseiter3473)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1120__k_1121 % 4UL) * 64UL) + _fuseiter3473)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1120__k_1121 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1120__k_1121 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_1120__k_1121 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter3471 * 1024UL) + ((_fuseiter3472 * 64UL) + _fuseiter3473)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_880_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_1_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_1122__k_o_1123 = 0UL; fused_0fused_0n__n_i_1122__k_o_1123 < 8UL; fused_0fused_0n__n_i_1122__k_o_1123 += 1UL) {
    int32_t* __origouts_890_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_1122__k_o_1123 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1122__k_o_1123 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_1122__k_o_1123 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_37 = &__origouts_890_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_32[0UL], A_list, B_list, &__origouts_890_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3506 = 0UL; _fuseiter3506 < 14UL; _fuseiter3506 += 1UL) {
      for (uint64_t _fuseiter3507 = 0UL; _fuseiter3507 < 14UL; _fuseiter3507 += 1UL) {
        for (uint64_t _fuseiter3508 = 0UL; _fuseiter3508 < 64UL; _fuseiter3508 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_890_shr[((_fuseiter3506 * 896UL) + ((_fuseiter3507 * 64UL) + _fuseiter3508))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1122__k_o_1123 % 4UL) * 64UL) + _fuseiter3508)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1122__k_o_1123 % 4UL) * 64UL) + _fuseiter3508)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_1122__k_o_1123 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_1122__k_o_1123 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_1122__k_o_1123 % 4UL) * 12544UL) + ((_fuseiter3506 * 896UL) + ((_fuseiter3507 * 64UL) + _fuseiter3508)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_890_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_2_cast_mul_add_cast_add_relu__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_1124__k_1125 = 0UL; fused_0fused_0n__n_i_1124__k_1125 < 32UL; fused_0fused_0n__n_i_1124__k_1125 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_900_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1124__k_1125 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_1124__k_1125 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1124__k_1125 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_38 = &__origouts_900_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_900_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3541 = 0UL; _fuseiter3541 < 2UL; _fuseiter3541 += 1UL) {
        for (uint64_t _fuseiter3542 = 0UL; _fuseiter3542 < 14UL; _fuseiter3542 += 1UL) {
          for (uint64_t _fuseiter3543 = 0UL; _fuseiter3543 < 64UL; _fuseiter3543 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_900_shr[((_fuseiter3541 * 896UL) + ((_fuseiter3542 * 64UL) + _fuseiter3543))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1124__k_1125 % 16UL) * 64UL) + _fuseiter3543)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1124__k_1125 % 16UL) * 64UL) + _fuseiter3543)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1124__k_1125 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1124__k_1125 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1124__k_1125 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3541 * 896UL) + ((_fuseiter3542 * 64UL) + _fuseiter3543)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1124__k_1125 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_1124__k_1125 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_1124__k_1125 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter3541 * 896UL) + ((_fuseiter3542 * 64UL) + _fuseiter3543)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_900_shr);
    }
  }
  return true;
}

static bool res4f_conv_0_cast_mul_add_cast_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t n_i = 0UL; n_i < 2UL; n_i += 1UL) {
    for (uint64_t k = 0UL; k < 4UL; k += 1UL) {
      memset(&__outs_0[((n_i * 65536UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n_i * 65536UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n_i * 65536UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1126__k_1127 = 0UL; fused_0fused_0n__n_i_1126__k_1127 < 8UL; fused_0fused_0n__n_i_1126__k_1127 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_910_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1126__k_1127 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_1126__k_1127 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1126__k_1127 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_39 = &__origouts_910_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_910_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3583 = 0UL; _fuseiter3583 < 2UL; _fuseiter3583 += 1UL) {
        for (uint64_t _fuseiter3584 = 0UL; _fuseiter3584 < 14UL; _fuseiter3584 += 1UL) {
          for (uint64_t _fuseiter3585 = 0UL; _fuseiter3585 < 64UL; _fuseiter3585 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_910_shr[((_fuseiter3583 * 896UL) + ((_fuseiter3584 * 64UL) + _fuseiter3585))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1126__k_1127 % 4UL) * 64UL) + _fuseiter3585)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1126__k_1127 % 4UL) * 64UL) + _fuseiter3585)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1126__k_1127 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_1126__k_1127 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_1126__k_1127 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter3583 * 1024UL) + ((_fuseiter3584 * 64UL) + _fuseiter3585)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_910_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_1_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0k_o__n_1128__n_i_1129 = 0UL; fused_0fused_0k_o__n_1128__n_i_1129 < 8UL; fused_0fused_0k_o__n_1128__n_i_1129 += 1UL) {
    int32_t* __origouts_920_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0k_o__n_1128__n_i_1129 % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0k_o__n_1128__n_i_1129 / 2UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_40 = &__origouts_920_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_32[0UL], A_list, B_list, &__origouts_920_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3618 = 0UL; _fuseiter3618 < 14UL; _fuseiter3618 += 1UL) {
      for (uint64_t _fuseiter3619 = 0UL; _fuseiter3619 < 14UL; _fuseiter3619 += 1UL) {
        for (uint64_t _fuseiter3620 = 0UL; _fuseiter3620 < 64UL; _fuseiter3620 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_920_shr[((_fuseiter3618 * 896UL) + ((_fuseiter3619 * 64UL) + _fuseiter3620))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k_o__n_1128__n_i_1129 / 2UL) * 64UL) + _fuseiter3620)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k_o__n_1128__n_i_1129 / 2UL) * 64UL) + _fuseiter3620)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k_o__n_1128__n_i_1129 % 2UL) * 50176UL) + (((fused_0fused_0k_o__n_1128__n_i_1129 / 2UL) * 12544UL) + ((_fuseiter3618 * 896UL) + ((_fuseiter3619 * 64UL) + _fuseiter3620))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_920_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_2_cast_mul_add_cast_add_relu__77(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_1130__n_i_1131 = 0UL; fused_0fused_0k__n_1130__n_i_1131 < 32UL; fused_0fused_0k__n_1130__n_i_1131 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_930_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_1130__n_i_1131 % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_1130__n_i_1131 / 2UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_41 = &__origouts_930_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_9, A_list, B_list, &__origouts_930_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter3653 = 0UL; _fuseiter3653 < 2UL; _fuseiter3653 += 1UL) {
        for (uint64_t _fuseiter3654 = 0UL; _fuseiter3654 < 14UL; _fuseiter3654 += 1UL) {
          for (uint64_t _fuseiter3655 = 0UL; _fuseiter3655 < 64UL; _fuseiter3655 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_930_shr[((_fuseiter3653 * 896UL) + ((_fuseiter3654 * 64UL) + _fuseiter3655))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_1130__n_i_1131 / 2UL) * 64UL) + _fuseiter3655)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_1130__n_i_1131 / 2UL) * 64UL) + _fuseiter3655)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_1130__n_i_1131 % 2UL) * 200704UL) + (((fused_0fused_0k__n_1130__n_i_1131 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter3653 * 896UL) + ((_fuseiter3654 * 64UL) + _fuseiter3655)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_1130__n_i_1131 % 2UL) * 200704UL) + (((fused_0fused_0k__n_1130__n_i_1131 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter3653 * 896UL) + ((_fuseiter3654 * 64UL) + _fuseiter3655)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_930_shr);
    }
  }
  return true;
}

static bool res5a_conv_b_cast_mul_add_cast__683(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_34 = *(void**)(__module_data + 120);
  int8_t* input_tmp = (int8_t*)sc_aligned_malloc(__stream, 12845056UL);
  for (uint64_t n = 0UL; n < 256UL; n += 1UL) {
    for (uint64_t c_o = 0UL; c_o < 16UL; c_o += 1UL) {
      for (uint64_t p = 0UL; p < 7UL; p += 1UL) {
        for (uint64_t q = 0UL; q < 7UL; q += 1UL) {
          vec_s8x64 __cached_0;
          __cached_0 = vec_s8x64::load(&__ins_0[((n * 200704UL) + ((c_o * 12544UL) + ((p * 1792UL) + (q * 128UL))))]);
          vec_s8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_s8x64::store(__cached_1, &input_tmp[((n * 50176UL) + ((c_o * 3136UL) + ((p * 448UL) + (q * 64UL))))]);
        }
      }
    }
  }
  for (uint64_t fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 = 0UL; fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 < 1024UL; fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_940_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[128UL];
    for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
      void* __cached_2;
      __cached_2 = &input_tmp[(((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 % 256UL) * 50176UL) + (c * 3136UL))];
      A_list[c] = __cached_2;
      void* __cached_3;
      __cached_3 = &__ins_1[(((((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 256UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 512UL) * 2UL)) * 524288UL) + (c * 32768UL))];
      B_list[c] = __cached_3;
    }
    void* _arg_cache_42 = &__origouts_940_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_34, A_list, B_list, &__origouts_940_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
    for (uint64_t _fuseiter3695 = 0UL; _fuseiter3695 < 7UL; _fuseiter3695 += 1UL) {
      for (uint64_t _fuseiter3696 = 0UL; _fuseiter3696 < 7UL; _fuseiter3696 += 1UL) {
        for (uint64_t _fuseiter3697 = 0UL; _fuseiter3697 < 512UL; _fuseiter3697 += 16UL) {
          vec_s32x16 __cached_4;
          __cached_4 = vec_s32x16::load(&__origouts_940_shr[((_fuseiter3695 * 3584UL) + ((_fuseiter3696 * 512UL) + _fuseiter3697))]);
          vec_f32x16 __cached_5;
          __cached_5 = (vec_f32x16)(__cached_4);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_2[(((((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 256UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 512UL) * 2UL)) * 512UL) + _fuseiter3697)]);
          __cached_5 = (__cached_5 * __cached_6);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_3[(((((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 256UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 512UL) * 2UL)) * 512UL) + _fuseiter3697)]);
          __cached_5 = (__cached_5 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 % 256UL) * 100352UL) + ((((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 256UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_1132__n_1133__n_i_1134 / 512UL) * 2UL)) * 25088UL)) + ((_fuseiter3695 * 3584UL) + ((_fuseiter3696 * 512UL) + _fuseiter3697)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_940_shr);
    sc_aligned_free(__stream, __rescheduled_2);
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res5a_conv_0_cast_mul_add_cast_relu_reorder__682(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_36 = *(void**)(__module_data + 128);
  for (uint64_t n = 0UL; n < 256UL; n += 1UL) {
    for (uint64_t k = 0UL; k < 8UL; k += 1UL) {
      memset(&__outs_0[((n * 131072UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n * 131072UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n * 131072UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n * 131072UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0k__n_1135__n_i_1136 = 0UL; fused_0fused_0k__n_1135__n_i_1136 < 256UL; fused_0fused_0k__n_1135__n_i_1136 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_950_shr = (int32_t*)sc_aligned_malloc(__stream, 57344UL);
      void** A_list = (void**)&__rescheduled_2[0UL];
      void** B_list = (void**)&__rescheduled_2[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_1135__n_i_1136 % 256UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_1135__n_i_1136 / 256UL) * 524288UL) + (c * 32768UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_43 = &__origouts_950_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_36, A_list, B_list, &__origouts_950_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter3723 = 0UL; _fuseiter3723 < 2UL; _fuseiter3723 += 1UL) {
        for (uint64_t _fuseiter3724 = 0UL; _fuseiter3724 < 14UL; _fuseiter3724 += 1UL) {
          for (uint64_t _fuseiter3725 = 0UL; _fuseiter3725 < 512UL; _fuseiter3725 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_950_shr[((_fuseiter3723 * 7168UL) + ((_fuseiter3724 * 512UL) + _fuseiter3725))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_1135__n_i_1136 / 256UL) * 512UL) + _fuseiter3725)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_1135__n_i_1136 / 256UL) * 512UL) + _fuseiter3725)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            __cached_6 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0k__n_1135__n_i_1136 % 256UL) * 131072UL) + ((((_fuseiter3725 + ((fused_0fused_0k__n_1135__n_i_1136 / 256UL) * 512UL)) / 64UL) * 16384UL) + ((((_fuseiter3723 + (p_o * 2UL)) + 1UL) * 1024UL) + (((_fuseiter3724 + 1UL) * 64UL) + ((_fuseiter3725 + ((fused_0fused_0k__n_1135__n_i_1136 / 256UL) * 512UL)) % 64UL)))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_950_shr);
    }
    sc_aligned_free(__stream, __rescheduled_2);
  }
  return true;
}

static bool res5a_conv_1_cast_mul_add_cast_relu_reorder__681(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_38 = *(void**)(__module_data + 136);
  for (uint64_t fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 = 0UL; fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 < 512UL; fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 1152UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[576UL];
    for (uint64_t p_i = 0UL; p_i < 7UL; p_i += 1UL) {
      int32_t* __origouts_960_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      for (uint64_t c_o = 0UL; c_o < 8UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[(((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 % 256UL) * 131072UL) + ((c_o * 16384UL) + ((((p_i * 2UL) + r) * 1024UL) + (s * 64UL))))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[(((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 / 256UL) * 1179648UL) + ((c_o * 147456UL) + ((r * 49152UL) + (s * 16384UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_44 = &__origouts_960_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_38, A_list, B_list, &__origouts_960_shr[0UL], 1, 64, 16384, 72, 7, 7, __stream);
      for (uint64_t _fuseiter3765 = 0UL; _fuseiter3765 < 7UL; _fuseiter3765 += 1UL) {
        for (uint64_t _fuseiter3766 = 0UL; _fuseiter3766 < 256UL; _fuseiter3766 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_960_shr[((_fuseiter3765 * 256UL) + _fuseiter3766)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 / 256UL) * 256UL) + _fuseiter3766)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 / 256UL) * 256UL) + _fuseiter3766)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 % 256UL) * 25088UL) + ((((_fuseiter3766 + ((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 / 256UL) * 256UL)) / 512UL) * 25088UL) + ((p_i * 3584UL) + ((_fuseiter3765 * 512UL) + ((_fuseiter3766 + ((fused_0fused_0fused_0oc_i__n_1137__n_i_1138__k_o_1139 / 256UL) * 256UL)) % 512UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_960_shr);
    }
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5a_conv_2_cast_mul_add_cast_add_relu__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_40 = *(void**)(__module_data + 144);
  for (uint64_t fused_0fused_0n__n_i_1140__k_1141 = 0UL; fused_0fused_0n__n_i_1140__k_1141 < 1024UL; fused_0fused_0n__n_i_1140__k_1141 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_970_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0fused_0n__n_i_1140__k_1141 / 4UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0fused_0n__n_i_1140__k_1141 % 4UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_45 = &__origouts_970_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_40, A_list, B_list, &__origouts_970_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter3805 = 0UL; _fuseiter3805 < 7UL; _fuseiter3805 += 1UL) {
      for (uint64_t _fuseiter3806 = 0UL; _fuseiter3806 < 7UL; _fuseiter3806 += 1UL) {
        for (uint64_t _fuseiter3807 = 0UL; _fuseiter3807 < 512UL; _fuseiter3807 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_970_shr[((_fuseiter3805 * 3584UL) + ((_fuseiter3806 * 512UL) + _fuseiter3807))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1140__k_1141 % 4UL) * 512UL) + _fuseiter3807)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1140__k_1141 % 4UL) * 512UL) + _fuseiter3807)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1140__k_1141 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_1140__k_1141 % 4UL) * 25088UL)) + ((_fuseiter3805 * 3584UL) + ((_fuseiter3806 * 512UL) + _fuseiter3807)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_1140__k_1141 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_1140__k_1141 % 4UL) * 25088UL)) + ((_fuseiter3805 * 3584UL) + ((_fuseiter3806 * 512UL) + _fuseiter3807)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_970_shr);
  }
  return true;
}

static bool res5b_conv_0_cast_mul_add_cast_relu__679(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_42 = *(void**)(__module_data + 152);
  for (uint64_t n = 0UL; n < 256UL; n += 1UL) {
    for (uint64_t k = 0UL; k < 8UL; k += 1UL) {
      memset(&__outs_0[((n * 41472UL) + (k * 5184UL))], 0, 576UL);
      for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
        memset(&__outs_0[((n * 41472UL) + ((k * 5184UL) + ((p1 + 1UL) * 576UL)))], 0, 64UL);
        memset(&__outs_0[(((n * 41472UL) + ((k * 5184UL) + ((p1 + 1UL) * 576UL))) + 512UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n * 41472UL) + (k * 5184UL)) + 4608UL)], 0, 576UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_1142__k_1143 = 0UL; fused_0fused_0n__n_i_1142__k_1143 < 2048UL; fused_0fused_0n__n_i_1142__k_1143 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_980_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1142__k_1143 / 8UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1142__k_1143 % 8UL) * 131072UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_46 = &__origouts_980_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_42, A_list, B_list, &__origouts_980_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter3847 = 0UL; _fuseiter3847 < 7UL; _fuseiter3847 += 1UL) {
      for (uint64_t _fuseiter3848 = 0UL; _fuseiter3848 < 7UL; _fuseiter3848 += 1UL) {
        for (uint64_t _fuseiter3849 = 0UL; _fuseiter3849 < 64UL; _fuseiter3849 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_980_shr[((_fuseiter3847 * 448UL) + ((_fuseiter3848 * 64UL) + _fuseiter3849))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1142__k_1143 % 8UL) * 64UL) + _fuseiter3849)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1142__k_1143 % 8UL) * 64UL) + _fuseiter3849)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_1142__k_1143 / 8UL) * 41472UL) + ((fused_0fused_0n__n_i_1142__k_1143 % 8UL) * 5184UL)) + 640UL) + ((_fuseiter3847 * 576UL) + ((_fuseiter3848 * 64UL) + _fuseiter3849)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_980_shr);
  }
  return true;
}

static bool res5b_conv_1_cast_mul_add_cast_relu_reorder__678(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_46 = (void**)&__uninitialized_data[23657528UL];
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 = 0UL; fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 < 1024UL; fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 1152UL);
    int32_t* __origouts_990_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[576UL];
    for (uint64_t c_o = 0UL; c_o < 8UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[((((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 2UL) % 256UL) * 41472UL) + ((c_o * 5184UL) + ((r * 576UL) + (s * 64UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[((((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 % 2UL) + ((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 512UL) * 2UL)) * 589824UL) + ((c_o * 73728UL) + ((r * 24576UL) + (s * 8192UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_47 = &__origouts_990_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_46[0UL], A_list, B_list, &__origouts_990_shr[0UL], 1, 64, 8192, 72, 7, 7, __stream);
    for (uint64_t _fuseiter3882 = 0UL; _fuseiter3882 < 7UL; _fuseiter3882 += 1UL) {
      for (uint64_t _fuseiter3883 = 0UL; _fuseiter3883 < 7UL; _fuseiter3883 += 1UL) {
        for (uint64_t _fuseiter3884 = 0UL; _fuseiter3884 < 128UL; _fuseiter3884 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_990_shr[((_fuseiter3882 * 896UL) + ((_fuseiter3883 * 128UL) + _fuseiter3884))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 % 2UL) + ((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 512UL) * 2UL)) * 128UL) + _fuseiter3884)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 % 2UL) + ((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 512UL) * 2UL)) * 128UL) + _fuseiter3884)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          __cached_7 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16 __cached_8;
          __cached_8 = __cached_7;
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 2UL) % 256UL) * 25088UL) + ((((_fuseiter3884 + (((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 % 2UL) + ((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 512UL) * 2UL)) * 128UL)) / 512UL) * 25088UL) + ((_fuseiter3882 * 3584UL) + ((_fuseiter3883 * 512UL) + ((_fuseiter3884 + (((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 % 2UL) + ((fused_0fused_0fused_0oc_i__n_1144__n_i_1145__k_o_1146 / 512UL) * 2UL)) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_990_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5b_conv_2_cast_mul_add_cast_add_relu__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_40 = *(void**)(__module_data + 144);
  for (uint64_t fused_0fused_0k__n_1147__n_i_1148 = 0UL; fused_0fused_0k__n_1147__n_i_1148 < 1024UL; fused_0fused_0k__n_1147__n_i_1148 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_1000_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0fused_0k__n_1147__n_i_1148 % 256UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0fused_0k__n_1147__n_i_1148 / 256UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_48 = &__origouts_1000_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_40, A_list, B_list, &__origouts_1000_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter3923 = 0UL; _fuseiter3923 < 7UL; _fuseiter3923 += 1UL) {
      for (uint64_t _fuseiter3924 = 0UL; _fuseiter3924 < 7UL; _fuseiter3924 += 1UL) {
        for (uint64_t _fuseiter3925 = 0UL; _fuseiter3925 < 512UL; _fuseiter3925 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1000_shr[((_fuseiter3923 * 3584UL) + ((_fuseiter3924 * 512UL) + _fuseiter3925))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_1147__n_i_1148 / 256UL) * 512UL) + _fuseiter3925)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_1147__n_i_1148 / 256UL) * 512UL) + _fuseiter3925)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_1147__n_i_1148 % 256UL) * 100352UL) + ((fused_0fused_0k__n_1147__n_i_1148 / 256UL) * 25088UL)) + ((_fuseiter3923 * 3584UL) + ((_fuseiter3924 * 512UL) + _fuseiter3925)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_1147__n_i_1148 % 256UL) * 100352UL) + ((fused_0fused_0k__n_1147__n_i_1148 / 256UL) * 25088UL)) + ((_fuseiter3923 * 3584UL) + ((_fuseiter3924 * 512UL) + _fuseiter3925)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1000_shr);
  }
  return true;
}

static bool res5c_conv_0_cast_mul_add_cast_relu_reorder__676(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_48 = *(void**)(__module_data + 160);
  for (uint64_t n = 0UL; n < 256UL; n += 1UL) {
    memset(&__outs_0[(n * 41472UL)], 0, 4608UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[((n * 41472UL) + ((p1 + 1UL) * 4608UL))], 0, 512UL);
      memset(&__outs_0[(((n * 41472UL) + ((p1 + 1UL) * 4608UL)) + 4096UL)], 0, 512UL);
    }
    memset(&__outs_0[((n * 41472UL) + 36864UL)], 0, 4608UL);
  }
  for (uint64_t fused_0fused_0n__n_i_1149__k_1150 = 0UL; fused_0fused_0n__n_i_1149__k_1150 < 512UL; fused_0fused_0n__n_i_1149__k_1150 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_1010_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1149__k_1150 / 2UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1149__k_1150 % 2UL) * 524288UL) + (c * 131072UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_49 = &__origouts_1010_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_48, A_list, B_list, &__origouts_1010_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter3965 = 0UL; _fuseiter3965 < 7UL; _fuseiter3965 += 1UL) {
      for (uint64_t _fuseiter3966 = 0UL; _fuseiter3966 < 7UL; _fuseiter3966 += 1UL) {
        for (uint64_t _fuseiter3967 = 0UL; _fuseiter3967 < 256UL; _fuseiter3967 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1010_shr[((_fuseiter3965 * 1792UL) + ((_fuseiter3966 * 256UL) + _fuseiter3967))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1149__k_1150 % 2UL) * 256UL) + _fuseiter3967)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1149__k_1150 % 2UL) * 256UL) + _fuseiter3967)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0n__n_i_1149__k_1150 / 2UL) * 41472UL) + ((((_fuseiter3967 + ((fused_0fused_0n__n_i_1149__k_1150 % 2UL) * 256UL)) / 512UL) * 41472UL) + (((_fuseiter3965 + 1UL) * 4608UL) + (((_fuseiter3966 + 1UL) * 512UL) + ((_fuseiter3967 + ((fused_0fused_0n__n_i_1149__k_1150 % 2UL) * 256UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1010_shr);
  }
  return true;
}

static bool res5c_conv_1_cast_mul_add_cast_relu__675(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_50 = (void**)&__uninitialized_data[23657536UL];
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 = 0UL; fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 < 2048UL; fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_1020_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[128UL];
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_1;
        __cached_1 = &__ins_0[((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 4UL) % 256UL) * 41472UL) + ((r * 4608UL) + (s * 512UL)))];
        A_list[((r * 3UL) + s)] = __cached_1;
        void* __cached_2;
        __cached_2 = &__ins_1[((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 % 4UL) + ((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 1024UL) * 4UL)) * 294912UL) + ((r * 98304UL) + (s * 32768UL)))];
        B_list[((r * 3UL) + s)] = __cached_2;
      }
    }
    void* _arg_cache_50 = &__origouts_1020_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_50[0UL], A_list, B_list, &__origouts_1020_shr[0UL], 1, 512, 32768, 9, 7, 7, __stream);
    for (uint64_t _fuseiter4006 = 0UL; _fuseiter4006 < 7UL; _fuseiter4006 += 1UL) {
      for (uint64_t _fuseiter4007 = 0UL; _fuseiter4007 < 7UL; _fuseiter4007 += 1UL) {
        for (uint64_t _fuseiter4008 = 0UL; _fuseiter4008 < 64UL; _fuseiter4008 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1020_shr[((_fuseiter4006 * 448UL) + ((_fuseiter4007 * 64UL) + _fuseiter4008))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 % 4UL) + ((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 1024UL) * 4UL)) * 64UL) + _fuseiter4008)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 % 4UL) + ((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 1024UL) * 4UL)) * 64UL) + _fuseiter4008)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 4UL) % 256UL) * 25088UL) + ((((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 % 4UL) + ((fused_0fused_0fused_0oc_i__n_1151__n_i_1152__k_o_1153 / 1024UL) * 4UL)) * 3136UL) + ((_fuseiter4006 * 448UL) + ((_fuseiter4007 * 64UL) + _fuseiter4008))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1020_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_52 = *(void**)(__module_data + 168);
  for (uint64_t fused_0fused_0n__n_i_1154__k_1155 = 0UL; fused_0fused_0n__n_i_1154__k_1155 < 1024UL; fused_0fused_0n__n_i_1154__k_1155 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_1030_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_1154__k_1155 / 4UL) * 25088UL) + (c * 3136UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_1154__k_1155 % 4UL) * 262144UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_51 = &__origouts_1030_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_52, A_list, B_list, &__origouts_1030_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
    for (uint64_t _fuseiter4041 = 0UL; _fuseiter4041 < 7UL; _fuseiter4041 += 1UL) {
      for (uint64_t _fuseiter4042 = 0UL; _fuseiter4042 < 7UL; _fuseiter4042 += 1UL) {
        for (uint64_t _fuseiter4043 = 0UL; _fuseiter4043 < 512UL; _fuseiter4043 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1030_shr[((_fuseiter4041 * 3584UL) + ((_fuseiter4042 * 512UL) + _fuseiter4043))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_1154__k_1155 % 4UL) * 512UL) + _fuseiter4043)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_1154__k_1155 % 4UL) * 512UL) + _fuseiter4043)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_1154__k_1155 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_1154__k_1155 % 4UL) * 25088UL)) + ((_fuseiter4041 * 3584UL) + ((_fuseiter4042 * 512UL) + _fuseiter4043)))]);
          __cached_6 = (__cached_6 + __cached_7);
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_8;
          __cached_8 = __cached_6;
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_1154__k_1155 / 4UL) * 100352UL) + ((_fuseiter4041 * 14336UL) + ((_fuseiter4042 * 2048UL) + (_fuseiter4043 + ((fused_0fused_0n__n_i_1154__k_1155 % 4UL) * 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1030_shr);
  }
  return true;
}

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  float* folded_const_156 = (float*)&__uninitialized_data[0UL];
  float* folded_const_81 = (float*)&__module_data[111296UL];
  float* folded_const_157 = (float*)&__uninitialized_data[256UL];
  float* folded_const_79 = (float*)&__module_data[110976UL];
  float* folded_const_158 = (float*)&__uninitialized_data[512UL];
  float* folded_const_76 = (float*)&__module_data[109632UL];
  float* folded_const_159 = (float*)&__uninitialized_data[768UL];
  float* folded_const_74 = (float*)&__module_data[109312UL];
  float* folded_const_160 = (float*)&__uninitialized_data[1024UL];
  float* folded_const_71 = (float*)&__module_data[107968UL];
  float* folded_const_161 = (float*)&__uninitialized_data[1280UL];
  float* folded_const_69 = (float*)&__module_data[107648UL];
  float* folded_const_103 = (float*)&__module_data[111680UL];
  float* folded_const_162 = (float*)&__uninitialized_data[1536UL];
  float* folded_const_101 = (float*)&__module_data[111376UL];
  float* folded_const_163 = (float*)&__uninitialized_data[2560UL];
  float* folded_const_102 = (float*)&__module_data[111424UL];
  float* folded_const_164 = (float*)&__uninitialized_data[2816UL];
  float* folded_const_80 = (float*)&__module_data[111040UL];
  float* folded_const_78 = (float*)&__module_data[109952UL];
  float* folded_const_165 = (float*)&__uninitialized_data[3072UL];
  float* folded_const_100 = (float*)&__module_data[111372UL];
  float* folded_const_166 = (float*)&__uninitialized_data[4096UL];
  float* folded_const_77 = (float*)&__module_data[109696UL];
  float* folded_const_167 = (float*)&__uninitialized_data[4352UL];
  float* folded_const_75 = (float*)&__module_data[109376UL];
  float* folded_const_73 = (float*)&__module_data[108288UL];
  float* folded_const_168 = (float*)&__uninitialized_data[4608UL];
  float* folded_const_99 = (float*)&__module_data[111368UL];
  float* folded_const_169 = (float*)&__uninitialized_data[5632UL];
  float* folded_const_72 = (float*)&__module_data[108032UL];
  float* folded_const_170 = (float*)&__uninitialized_data[5888UL];
  float* folded_const_70 = (float*)&__module_data[107712UL];
  float* folded_const_68 = (float*)&__module_data[106624UL];
  float* folded_const_171 = (float*)&__uninitialized_data[6144UL];
  float* folded_const_98 = (float*)&__module_data[111364UL];
  float* folded_const_67 = (float*)&__module_data[104576UL];
  float* folded_const_172 = (float*)&__uninitialized_data[7168UL];
  float* folded_const_97 = (float*)&__module_data[111360UL];
  float* folded_const_66 = (float*)&__module_data[104064UL];
  float* folded_const_173 = (float*)&__uninitialized_data[9216UL];
  float* folded_const_65 = (float*)&__module_data[104000UL];
  float* folded_const_64 = (float*)&__module_data[103488UL];
  float* folded_const_174 = (float*)&__uninitialized_data[9728UL];
  float* folded_const_63 = (float*)&__module_data[103424UL];
  float* folded_const_62 = (float*)&__module_data[101376UL];
  float* folded_const_175 = (float*)&__uninitialized_data[10240UL];
  float* folded_const_96 = (float*)&__module_data[111356UL];
  float* folded_const_61 = (float*)&__module_data[100864UL];
  float* folded_const_176 = (float*)&__uninitialized_data[12288UL];
  float* folded_const_60 = (float*)&__module_data[100800UL];
  float* folded_const_59 = (float*)&__module_data[100288UL];
  float* folded_const_177 = (float*)&__uninitialized_data[12800UL];
  float* folded_const_58 = (float*)&__module_data[100224UL];
  float* folded_const_57 = (float*)&__module_data[98176UL];
  float* folded_const_178 = (float*)&__uninitialized_data[13312UL];
  float* folded_const_95 = (float*)&__module_data[111352UL];
  float* folded_const_56 = (float*)&__module_data[97664UL];
  float* folded_const_179 = (float*)&__uninitialized_data[15360UL];
  float* folded_const_55 = (float*)&__module_data[97600UL];
  float* folded_const_54 = (float*)&__module_data[97088UL];
  float* folded_const_180 = (float*)&__uninitialized_data[15872UL];
  float* folded_const_53 = (float*)&__module_data[97024UL];
  float* folded_const_52 = (float*)&__module_data[94976UL];
  float* folded_const_181 = (float*)&__uninitialized_data[16384UL];
  float* folded_const_94 = (float*)&__module_data[111348UL];
  float* folded_const_51 = (float*)&__module_data[94464UL];
  float* folded_const_182 = (float*)&__uninitialized_data[18432UL];
  float* folded_const_50 = (float*)&__module_data[94400UL];
  float* folded_const_49 = (float*)&__module_data[93888UL];
  float* folded_const_183 = (float*)&__uninitialized_data[18944UL];
  float* folded_const_48 = (float*)&__module_data[93824UL];
  float* folded_const_47 = (float*)&__module_data[91776UL];
  float* folded_const_184 = (float*)&__uninitialized_data[19456UL];
  float* folded_const_93 = (float*)&__module_data[111344UL];
  float* folded_const_46 = (float*)&__module_data[87680UL];
  float* folded_const_185 = (float*)&__uninitialized_data[21504UL];
  float* folded_const_92 = (float*)&__module_data[111340UL];
  float* folded_const_45 = (float*)&__module_data[86656UL];
  float* folded_const_186 = (float*)&__uninitialized_data[25600UL];
  float* folded_const_44 = (float*)&__module_data[86592UL];
  float* folded_const_43 = (float*)&__module_data[85568UL];
  float* folded_const_187 = (float*)&__uninitialized_data[26624UL];
  float* folded_const_42 = (float*)&__module_data[85504UL];
  float* folded_const_41 = (float*)&__module_data[81408UL];
  float* folded_const_188 = (float*)&__uninitialized_data[27648UL];
  float* folded_const_91 = (float*)&__module_data[111336UL];
  float* folded_const_40 = (float*)&__module_data[80384UL];
  float* folded_const_189 = (float*)&__uninitialized_data[31744UL];
  float* folded_const_39 = (float*)&__module_data[80320UL];
  float* folded_const_38 = (float*)&__module_data[79296UL];
  float* folded_const_190 = (float*)&__uninitialized_data[32768UL];
  float* folded_const_37 = (float*)&__module_data[79232UL];
  float* folded_const_36 = (float*)&__module_data[75136UL];
  float* folded_const_191 = (float*)&__uninitialized_data[33792UL];
  float* folded_const_90 = (float*)&__module_data[111332UL];
  float* folded_const_35 = (float*)&__module_data[74112UL];
  float* folded_const_192 = (float*)&__uninitialized_data[37888UL];
  float* folded_const_34 = (float*)&__module_data[74048UL];
  float* folded_const_33 = (float*)&__module_data[73024UL];
  float* folded_const_193 = (float*)&__uninitialized_data[38912UL];
  float* folded_const_32 = (float*)&__module_data[72960UL];
  float* folded_const_31 = (float*)&__module_data[68864UL];
  float* folded_const_194 = (float*)&__uninitialized_data[39936UL];
  float* folded_const_89 = (float*)&__module_data[111328UL];
  float* folded_const_30 = (float*)&__module_data[67840UL];
  float* folded_const_195 = (float*)&__uninitialized_data[44032UL];
  float* folded_const_29 = (float*)&__module_data[67776UL];
  float* folded_const_28 = (float*)&__module_data[66752UL];
  float* folded_const_196 = (float*)&__uninitialized_data[45056UL];
  float* folded_const_27 = (float*)&__module_data[66688UL];
  float* folded_const_26 = (float*)&__module_data[62592UL];
  float* folded_const_197 = (float*)&__uninitialized_data[46080UL];
  float* folded_const_88 = (float*)&__module_data[111324UL];
  float* folded_const_25 = (float*)&__module_data[61568UL];
  float* folded_const_198 = (float*)&__uninitialized_data[50176UL];
  float* folded_const_24 = (float*)&__module_data[61504UL];
  float* folded_const_23 = (float*)&__module_data[60480UL];
  float* folded_const_199 = (float*)&__uninitialized_data[51200UL];
  float* folded_const_22 = (float*)&__module_data[60416UL];
  float* folded_const_21 = (float*)&__module_data[56320UL];
  float* folded_const_200 = (float*)&__uninitialized_data[52224UL];
  float* folded_const_87 = (float*)&__module_data[111320UL];
  float* folded_const_20 = (float*)&__module_data[55296UL];
  float* folded_const_201 = (float*)&__uninitialized_data[56320UL];
  float* folded_const_19 = (float*)&__module_data[55232UL];
  float* folded_const_18 = (float*)&__module_data[54208UL];
  float* folded_const_202 = (float*)&__uninitialized_data[57344UL];
  float* folded_const_17 = (float*)&__module_data[54144UL];
  float* folded_const_16 = (float*)&__module_data[50048UL];
  float* folded_const_203 = (float*)&__uninitialized_data[58368UL];
  float* folded_const_86 = (float*)&__module_data[111316UL];
  float* folded_const_15 = (float*)&__module_data[41856UL];
  float* folded_const_204 = (float*)&__uninitialized_data[62464UL];
  float* folded_const_85 = (float*)&__module_data[111312UL];
  float* folded_const_205 = (float*)&__uninitialized_data[70656UL];
  float* folded_const_14 = (float*)&__module_data[39808UL];
  float* folded_const_13 = (float*)&__module_data[39744UL];
  float* folded_const_12 = (float*)&__module_data[37696UL];
  float* folded_const_206 = (float*)&__uninitialized_data[72704UL];
  float* folded_const_11 = (float*)&__module_data[37632UL];
  float* folded_const_10 = (float*)&__module_data[29440UL];
  float* folded_const_207 = (float*)&__uninitialized_data[74752UL];
  float* folded_const_84 = (float*)&__module_data[111308UL];
  float* folded_const_9 = (float*)&__module_data[27392UL];
  float* folded_const_208 = (float*)&__uninitialized_data[82944UL];
  float* folded_const_8 = (float*)&__module_data[27328UL];
  float* folded_const_7 = (float*)&__module_data[25280UL];
  float* folded_const_209 = (float*)&__uninitialized_data[84992UL];
  float* folded_const_6 = (float*)&__module_data[25216UL];
  float* folded_const_5 = (float*)&__module_data[17024UL];
  float* folded_const_210 = (float*)&__uninitialized_data[87040UL];
  float* folded_const_83 = (float*)&__module_data[111304UL];
  float* folded_const_4 = (float*)&__module_data[14976UL];
  float* folded_const_211 = (float*)&__uninitialized_data[95232UL];
  float* folded_const_3 = (float*)&__module_data[14912UL];
  float* folded_const_2 = (float*)&__module_data[12864UL];
  float* folded_const_212 = (float*)&__uninitialized_data[97280UL];
  float* folded_const_1 = (float*)&__module_data[12800UL];
  float* folded_const_0 = (float*)&__module_data[4608UL];
  float* folded_const_213 = (float*)&__uninitialized_data[99328UL];
  float* folded_const_82 = (float*)&__module_data[111300UL];
  float* folded_const_214 = (float*)&__uninitialized_data[107520UL];
  float* folded_const_215 = (float*)&__uninitialized_data[108544UL];
  float* folded_const_216 = (float*)&__uninitialized_data[109568UL];
  float* folded_const_217 = (float*)&__uninitialized_data[110592UL];
  float* folded_const_218 = (float*)&__uninitialized_data[111616UL];
  float* folded_const_219 = (float*)&__uninitialized_data[112128UL];
  float* folded_const_220 = (float*)&__uninitialized_data[112640UL];
  float* folded_const_221 = (float*)&__uninitialized_data[113152UL];
  float* folded_const_222 = (float*)&__uninitialized_data[113664UL];
  float* folded_const_223 = (float*)&__uninitialized_data[114176UL];
  float* folded_const_224 = (float*)&__uninitialized_data[114688UL];
  float* folded_const_225 = (float*)&__uninitialized_data[115200UL];
  float* folded_const_226 = (float*)&__uninitialized_data[115712UL];
  float* folded_const_227 = (float*)&__uninitialized_data[116736UL];
  float* folded_const_228 = (float*)&__uninitialized_data[117760UL];
  float* folded_const_229 = (float*)&__uninitialized_data[118784UL];
  float* folded_const_230 = (float*)&__uninitialized_data[119808UL];
  float* folded_const_231 = (float*)&__uninitialized_data[120832UL];
  float* folded_const_232 = (float*)&__uninitialized_data[121856UL];
  float* folded_const_233 = (float*)&__uninitialized_data[122880UL];
  float* folded_const_234 = (float*)&__uninitialized_data[123904UL];
  float* folded_const_235 = (float*)&__uninitialized_data[124928UL];
  float* folded_const_236 = (float*)&__uninitialized_data[125952UL];
  float* folded_const_237 = (float*)&__uninitialized_data[126976UL];
  float* folded_const_238 = (float*)&__uninitialized_data[128000UL];
  float* folded_const_239 = (float*)&__uninitialized_data[130048UL];
  float* folded_const_240 = (float*)&__uninitialized_data[132096UL];
  float* folded_const_241 = (float*)&__uninitialized_data[134144UL];
  float* folded_const_242 = (float*)&__uninitialized_data[136192UL];
  float* folded_const_243 = (float*)&__uninitialized_data[138240UL];
  float* folded_const_244 = (float*)&__uninitialized_data[140288UL];
  float* folded_const_245 = (float*)&__uninitialized_data[142336UL];
  float* folded_const_246 = (float*)&__uninitialized_data[144384UL];
  float* folded_const_247 = (float*)&__uninitialized_data[146432UL];
  float* folded_const_248 = (float*)&__uninitialized_data[148480UL];
  float* folded_const_249 = (float*)&__uninitialized_data[150528UL];
  float* folded_const_250 = (float*)&__uninitialized_data[154624UL];
  float* folded_const_251 = (float*)&__uninitialized_data[158720UL];
  float* folded_const_252 = (float*)&__uninitialized_data[162816UL];
  float* folded_const_253 = (float*)&__uninitialized_data[166912UL];
  float* folded_const_254 = (float*)&__uninitialized_data[171008UL];
  float* folded_const_255 = (float*)&__uninitialized_data[175104UL];
  float* folded_const_256 = (float*)&__uninitialized_data[179200UL];
  float* folded_const_257 = (float*)&__uninitialized_data[187392UL];
  float* folded_const_258 = (float*)&__uninitialized_data[195584UL];
  float* folded_const_259 = (float*)&__uninitialized_data[203776UL];
  float* folded_const_154 = (float*)&__module_data[217408UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[211968UL];
  float* folded_const_155 = (float*)&__module_data[217664UL];
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[216064UL];
  float* folded_const_152 = (float*)&__module_data[216128UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[232448UL];
  float* folded_const_149 = (float*)&__module_data[214592UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[248832UL];
  float* folded_const_146 = (float*)&__module_data[213056UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[265216UL];
  float* folded_const_151 = (float*)&__module_data[215872UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[281600UL];
  float* folded_const_148 = (float*)&__module_data[214336UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[297984UL];
  float* folded_const_144 = (float*)&__module_data[210496UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[314368UL];
  float* folded_const_153 = (float*)&__module_data[217152UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[347136UL];
  float* folded_const_150 = (float*)&__module_data[215616UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[384000UL];
  float* folded_const_147 = (float*)&__module_data[214080UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[420864UL];
  float* folded_const_142 = (float*)&__module_data[207936UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[457728UL];
  float* folded_const_139 = (float*)&__module_data[204864UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[523264UL];
  float* folded_const_136 = (float*)&__module_data[201792UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[588800UL];
  float* folded_const_133 = (float*)&__module_data[198720UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[654336UL];
  float* folded_const_141 = (float*)&__module_data[207424UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[719872UL];
  float* folded_const_138 = (float*)&__module_data[204352UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[785408UL];
  float* folded_const_135 = (float*)&__module_data[201280UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[850944UL];
  float* folded_const_145 = (float*)&__module_data[211008UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[916480UL];
  float* folded_const_131 = (float*)&__module_data[193600UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[1047552UL];
  float* folded_const_143 = (float*)&__module_data[209984UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[1178624UL];
  float* folded_const_140 = (float*)&__module_data[206912UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[1326080UL];
  float* folded_const_137 = (float*)&__module_data[203840UL];
  int8_t* folded_const_282 = (int8_t*)&__uninitialized_data[1473536UL];
  float* folded_const_134 = (float*)&__module_data[200768UL];
  int8_t* folded_const_283 = (int8_t*)&__uninitialized_data[1620992UL];
  float* folded_const_129 = (float*)&__module_data[188480UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[1768448UL];
  float* folded_const_126 = (float*)&__module_data[182336UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[2030592UL];
  float* folded_const_123 = (float*)&__module_data[176192UL];
  int8_t* folded_const_286 = (int8_t*)&__uninitialized_data[2292736UL];
  float* folded_const_120 = (float*)&__module_data[170048UL];
  int8_t* folded_const_287 = (int8_t*)&__uninitialized_data[2554880UL];
  float* folded_const_117 = (float*)&__module_data[163904UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[2817024UL];
  float* folded_const_114 = (float*)&__module_data[157760UL];
  int8_t* folded_const_289 = (int8_t*)&__uninitialized_data[3079168UL];
  float* folded_const_128 = (float*)&__module_data[187456UL];
  int8_t* folded_const_290 = (int8_t*)&__uninitialized_data[3341312UL];
  float* folded_const_125 = (float*)&__module_data[181312UL];
  int8_t* folded_const_291 = (int8_t*)&__uninitialized_data[3603456UL];
  float* folded_const_122 = (float*)&__module_data[175168UL];
  int8_t* folded_const_292 = (int8_t*)&__uninitialized_data[3865600UL];
  float* folded_const_119 = (float*)&__module_data[169024UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[4127744UL];
  float* folded_const_116 = (float*)&__module_data[162880UL];
  int8_t* folded_const_294 = (int8_t*)&__uninitialized_data[4389888UL];
  float* folded_const_132 = (float*)&__module_data[194624UL];
  int8_t* folded_const_295 = (int8_t*)&__uninitialized_data[4652032UL];
  float* folded_const_112 = (float*)&__module_data[147520UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[5176320UL];
  float* folded_const_130 = (float*)&__module_data[192576UL];
  int8_t* folded_const_297 = (int8_t*)&__uninitialized_data[5700608UL];
  float* folded_const_127 = (float*)&__module_data[186432UL];
  int8_t* folded_const_298 = (int8_t*)&__uninitialized_data[6290432UL];
  float* folded_const_124 = (float*)&__module_data[180288UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[6880256UL];
  float* folded_const_121 = (float*)&__module_data[174144UL];
  int8_t* folded_const_300 = (int8_t*)&__uninitialized_data[7470080UL];
  float* folded_const_118 = (float*)&__module_data[168000UL];
  int8_t* folded_const_301 = (int8_t*)&__uninitialized_data[8059904UL];
  float* folded_const_115 = (float*)&__module_data[161856UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[8649728UL];
  float* folded_const_110 = (float*)&__module_data[137280UL];
  int8_t* folded_const_303 = (int8_t*)&__uninitialized_data[9239552UL];
  float* folded_const_107 = (float*)&__module_data[124992UL];
  int8_t* folded_const_304 = (int8_t*)&__uninitialized_data[10288128UL];
  float* folded_const_104 = (float*)&__module_data[112704UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[11336704UL];
  float* folded_const_109 = (float*)&__module_data[135232UL];
  int8_t* folded_const_306 = (int8_t*)&__uninitialized_data[12385280UL];
  float* folded_const_106 = (float*)&__module_data[122944UL];
  int8_t* folded_const_307 = (int8_t*)&__uninitialized_data[13433856UL];
  float* folded_const_113 = (float*)&__module_data[149568UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[14482432UL];
  float* folded_const_111 = (float*)&__module_data[145472UL];
  int8_t* folded_const_309 = (int8_t*)&__uninitialized_data[16579584UL];
  float* folded_const_108 = (float*)&__module_data[133184UL];
  int8_t* folded_const_310 = (int8_t*)&__uninitialized_data[18938880UL];
  float* folded_const_105 = (float*)&__module_data[120896UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[21298176UL];
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 11796480UL);
  mul__573(folded_const_156, &res2a_bias_0[0], folded_const_81);
  mul__575(folded_const_157, &res2a_bias_1[0], folded_const_79);
  mul__579(folded_const_158, &res2b_bias_0[0], folded_const_76);
  mul__581(folded_const_159, &res2b_bias_1[0], folded_const_74);
  mul__585(folded_const_160, &res2c_bias_0[0], folded_const_71);
  mul__587(folded_const_161, &res2c_bias_1[0], folded_const_69);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_267 = (float*)&__rescheduled_0[0UL];
  reorder__419(buffer_267, folded_const_103);
  mul__570(folded_const_162, buffer_267, folded_const_101);
  mul__572(folded_const_163, &folded_const_102[0UL], folded_const_81);
  mul__574(folded_const_164, &folded_const_80[0UL], folded_const_79);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_271 = (float*)&__rescheduled_0[0UL];
  reorder__424(buffer_271, folded_const_78);
  mul__576(folded_const_165, buffer_271, folded_const_100);
  mul__578(folded_const_166, &folded_const_77[0UL], folded_const_76);
  mul__580(folded_const_167, &folded_const_75[0UL], folded_const_74);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_275 = (float*)&__rescheduled_0[0UL];
  reorder__429(buffer_275, folded_const_73);
  mul__582(folded_const_168, buffer_275, folded_const_99);
  mul__584(folded_const_169, &folded_const_72[0UL], folded_const_71);
  mul__586(folded_const_170, &folded_const_70[0UL], folded_const_69);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_279 = (float*)&__rescheduled_0[0UL];
  reorder__434(buffer_279, folded_const_68);
  mul__588(folded_const_171, buffer_279, folded_const_98);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_281 = (float*)&__rescheduled_0[0UL];
  reorder__437(buffer_281, folded_const_67);
  mul__590(folded_const_172, buffer_281, folded_const_97);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_283 = (float*)&__rescheduled_0[0UL];
  reorder__440(buffer_283, folded_const_66);
  mul__592(folded_const_173, buffer_283, folded_const_65);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_285 = (float*)&__rescheduled_0[0UL];
  reorder__443(buffer_285, folded_const_64);
  mul__594(folded_const_174, buffer_285, folded_const_63);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_287 = (float*)&__rescheduled_0[0UL];
  reorder__446(buffer_287, folded_const_62);
  mul__596(folded_const_175, buffer_287, folded_const_96);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_289 = (float*)&__rescheduled_0[0UL];
  reorder__449(buffer_289, folded_const_61);
  mul__598(folded_const_176, buffer_289, folded_const_60);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_291 = (float*)&__rescheduled_0[0UL];
  reorder__452(buffer_291, folded_const_59);
  mul__600(folded_const_177, buffer_291, folded_const_58);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_293 = (float*)&__rescheduled_0[0UL];
  reorder__455(buffer_293, folded_const_57);
  mul__602(folded_const_178, buffer_293, folded_const_95);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_295 = (float*)&__rescheduled_0[0UL];
  reorder__458(buffer_295, folded_const_56);
  mul__604(folded_const_179, buffer_295, folded_const_55);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_297 = (float*)&__rescheduled_0[0UL];
  reorder__461(buffer_297, folded_const_54);
  mul__606(folded_const_180, buffer_297, folded_const_53);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_299 = (float*)&__rescheduled_0[0UL];
  reorder__464(buffer_299, folded_const_52);
  mul__608(folded_const_181, buffer_299, folded_const_94);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_301 = (float*)&__rescheduled_0[0UL];
  reorder__467(buffer_301, folded_const_51);
  mul__610(folded_const_182, buffer_301, folded_const_50);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_303 = (float*)&__rescheduled_0[0UL];
  reorder__470(buffer_303, folded_const_49);
  mul__612(folded_const_183, buffer_303, folded_const_48);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_305 = (float*)&__rescheduled_0[0UL];
  reorder__473(buffer_305, folded_const_47);
  mul__614(folded_const_184, buffer_305, folded_const_93);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_307 = (float*)&__rescheduled_0[0UL];
  reorder__476(buffer_307, folded_const_46);
  mul__616(folded_const_185, buffer_307, folded_const_92);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_309 = (float*)&__rescheduled_0[0UL];
  reorder__479(buffer_309, folded_const_45);
  mul__618(folded_const_186, buffer_309, folded_const_44);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_311 = (float*)&__rescheduled_0[0UL];
  reorder__482(buffer_311, folded_const_43);
  mul__620(folded_const_187, buffer_311, folded_const_42);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_313 = (float*)&__rescheduled_0[0UL];
  reorder__485(buffer_313, folded_const_41);
  mul__622(folded_const_188, buffer_313, folded_const_91);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_315 = (float*)&__rescheduled_0[0UL];
  reorder__488(buffer_315, folded_const_40);
  mul__624(folded_const_189, buffer_315, folded_const_39);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_317 = (float*)&__rescheduled_0[0UL];
  reorder__491(buffer_317, folded_const_38);
  mul__626(folded_const_190, buffer_317, folded_const_37);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_319 = (float*)&__rescheduled_0[0UL];
  reorder__494(buffer_319, folded_const_36);
  mul__628(folded_const_191, buffer_319, folded_const_90);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_321 = (float*)&__rescheduled_0[0UL];
  reorder__497(buffer_321, folded_const_35);
  mul__630(folded_const_192, buffer_321, folded_const_34);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_323 = (float*)&__rescheduled_0[0UL];
  reorder__500(buffer_323, folded_const_33);
  mul__632(folded_const_193, buffer_323, folded_const_32);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_325 = (float*)&__rescheduled_0[0UL];
  reorder__503(buffer_325, folded_const_31);
  mul__634(folded_const_194, buffer_325, folded_const_89);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_327 = (float*)&__rescheduled_0[0UL];
  reorder__506(buffer_327, folded_const_30);
  mul__636(folded_const_195, buffer_327, folded_const_29);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_329 = (float*)&__rescheduled_0[0UL];
  reorder__509(buffer_329, folded_const_28);
  mul__638(folded_const_196, buffer_329, folded_const_27);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_331 = (float*)&__rescheduled_0[0UL];
  reorder__512(buffer_331, folded_const_26);
  mul__640(folded_const_197, buffer_331, folded_const_88);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_333 = (float*)&__rescheduled_0[0UL];
  reorder__515(buffer_333, folded_const_25);
  mul__642(folded_const_198, buffer_333, folded_const_24);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_335 = (float*)&__rescheduled_0[0UL];
  reorder__518(buffer_335, folded_const_23);
  mul__644(folded_const_199, buffer_335, folded_const_22);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_337 = (float*)&__rescheduled_0[0UL];
  reorder__521(buffer_337, folded_const_21);
  mul__646(folded_const_200, buffer_337, folded_const_87);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_339 = (float*)&__rescheduled_0[0UL];
  reorder__524(buffer_339, folded_const_20);
  mul__648(folded_const_201, buffer_339, folded_const_19);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_341 = (float*)&__rescheduled_0[0UL];
  reorder__527(buffer_341, folded_const_18);
  mul__650(folded_const_202, buffer_341, folded_const_17);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_343 = (float*)&__rescheduled_0[0UL];
  reorder__530(buffer_343, folded_const_16);
  mul__652(folded_const_203, buffer_343, folded_const_86);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_345 = (float*)&__rescheduled_0[0UL];
  reorder__533(buffer_345, folded_const_15);
  mul__654(folded_const_204, buffer_345, folded_const_85);
  mul__656(folded_const_205, &folded_const_14[0UL], folded_const_13);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_348 = (float*)&__rescheduled_0[0UL];
  reorder__537(buffer_348, folded_const_12);
  mul__658(folded_const_206, buffer_348, folded_const_11);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_350 = (float*)&__rescheduled_0[0UL];
  reorder__540(buffer_350, folded_const_10);
  mul__660(folded_const_207, buffer_350, folded_const_84);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_352 = (float*)&__rescheduled_0[0UL];
  reorder__543(buffer_352, folded_const_9);
  mul__662(folded_const_208, buffer_352, folded_const_8);
  // [f32 [1, 1, 4, 1, 1, 128] @ A1aBCD128b]
  float* buffer_354 = (float*)&__rescheduled_0[0UL];
  reorder__546(buffer_354, folded_const_7);
  mul__664(folded_const_209, buffer_354, folded_const_6);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_356 = (float*)&__rescheduled_0[0UL];
  reorder__549(buffer_356, folded_const_5);
  mul__666(folded_const_210, buffer_356, folded_const_83);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_358 = (float*)&__rescheduled_0[0UL];
  reorder__552(buffer_358, folded_const_4);
  mul__668(folded_const_211, buffer_358, folded_const_3);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_360 = (float*)&__rescheduled_0[0UL];
  reorder__555(buffer_360, folded_const_2);
  mul__670(folded_const_212, buffer_360, folded_const_1);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_362 = (float*)&__rescheduled_0[0UL];
  reorder__558(buffer_362, folded_const_0);
  mul__672(folded_const_213, buffer_362, folded_const_82);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_364 = (float*)&__rescheduled_0[0UL];
  reorder__420(buffer_364, &res2a_bias_b[0]);
  mul__571(folded_const_214, buffer_364, folded_const_101);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_366 = (float*)&__rescheduled_0[0UL];
  reorder__425(buffer_366, &res2a_bias_2[0]);
  mul__577(folded_const_215, buffer_366, folded_const_100);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_368 = (float*)&__rescheduled_0[0UL];
  reorder__430(buffer_368, &res2b_bias_2[0]);
  mul__583(folded_const_216, buffer_368, folded_const_99);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_370 = (float*)&__rescheduled_0[0UL];
  reorder__435(buffer_370, &res2c_bias_2[0]);
  mul__589(folded_const_217, buffer_370, folded_const_98);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_372 = (float*)&__rescheduled_0[0UL];
  reorder__441(buffer_372, &res3a_bias_0[0]);
  mul__593(folded_const_218, buffer_372, folded_const_65);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_374 = (float*)&__rescheduled_0[0UL];
  reorder__444(buffer_374, &res3a_bias_1[0]);
  mul__595(folded_const_219, buffer_374, folded_const_63);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_376 = (float*)&__rescheduled_0[0UL];
  reorder__450(buffer_376, &res3b_bias_0[0]);
  mul__599(folded_const_220, buffer_376, folded_const_60);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_378 = (float*)&__rescheduled_0[0UL];
  reorder__453(buffer_378, &res3b_bias_1[0]);
  mul__601(folded_const_221, buffer_378, folded_const_58);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_380 = (float*)&__rescheduled_0[0UL];
  reorder__459(buffer_380, &res3c_bias_0[0]);
  mul__605(folded_const_222, buffer_380, folded_const_55);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_382 = (float*)&__rescheduled_0[0UL];
  reorder__462(buffer_382, &res3c_bias_1[0]);
  mul__607(folded_const_223, buffer_382, folded_const_53);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_384 = (float*)&__rescheduled_0[0UL];
  reorder__468(buffer_384, &res3d_bias_0[0]);
  mul__611(folded_const_224, buffer_384, folded_const_50);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_386 = (float*)&__rescheduled_0[0UL];
  reorder__471(buffer_386, &res3d_bias_1[0]);
  mul__613(folded_const_225, buffer_386, folded_const_48);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_388 = (float*)&__rescheduled_0[0UL];
  reorder__480(buffer_388, &res4a_bias_0[0]);
  mul__619(folded_const_226, buffer_388, folded_const_44);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_390 = (float*)&__rescheduled_0[0UL];
  reorder__483(buffer_390, &res4a_bias_1[0]);
  mul__621(folded_const_227, buffer_390, folded_const_42);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_392 = (float*)&__rescheduled_0[0UL];
  reorder__489(buffer_392, &res4b_bias_0[0]);
  mul__625(folded_const_228, buffer_392, folded_const_39);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_394 = (float*)&__rescheduled_0[0UL];
  reorder__492(buffer_394, &res4b_bias_1[0]);
  mul__627(folded_const_229, buffer_394, folded_const_37);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_396 = (float*)&__rescheduled_0[0UL];
  reorder__498(buffer_396, &res4c_bias_0[0]);
  mul__631(folded_const_230, buffer_396, folded_const_34);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_398 = (float*)&__rescheduled_0[0UL];
  reorder__501(buffer_398, &res4c_bias_1[0]);
  mul__633(folded_const_231, buffer_398, folded_const_32);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_400 = (float*)&__rescheduled_0[0UL];
  reorder__507(buffer_400, &res4d_bias_0[0]);
  mul__637(folded_const_232, buffer_400, folded_const_29);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_402 = (float*)&__rescheduled_0[0UL];
  reorder__510(buffer_402, &res4d_bias_1[0]);
  mul__639(folded_const_233, buffer_402, folded_const_27);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_404 = (float*)&__rescheduled_0[0UL];
  reorder__516(buffer_404, &res4e_bias_0[0]);
  mul__643(folded_const_234, buffer_404, folded_const_24);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_406 = (float*)&__rescheduled_0[0UL];
  reorder__519(buffer_406, &res4e_bias_1[0]);
  mul__645(folded_const_235, buffer_406, folded_const_22);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_408 = (float*)&__rescheduled_0[0UL];
  reorder__525(buffer_408, &res4f_bias_0[0]);
  mul__649(folded_const_236, buffer_408, folded_const_19);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_410 = (float*)&__rescheduled_0[0UL];
  reorder__528(buffer_410, &res4f_bias_1[0]);
  mul__651(folded_const_237, buffer_410, folded_const_17);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_412 = (float*)&__rescheduled_0[0UL];
  reorder__438(buffer_412, &res3a_bias_b[0]);
  mul__591(folded_const_238, buffer_412, folded_const_97);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_414 = (float*)&__rescheduled_0[0UL];
  reorder__447(buffer_414, &res3a_bias_2[0]);
  mul__597(folded_const_239, buffer_414, folded_const_96);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_416 = (float*)&__rescheduled_0[0UL];
  reorder__456(buffer_416, &res3b_bias_2[0]);
  mul__603(folded_const_240, buffer_416, folded_const_95);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_418 = (float*)&__rescheduled_0[0UL];
  reorder__465(buffer_418, &res3c_bias_2[0]);
  mul__609(folded_const_241, buffer_418, folded_const_94);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_420 = (float*)&__rescheduled_0[0UL];
  reorder__474(buffer_420, &res3d_bias_2[0]);
  mul__615(folded_const_242, buffer_420, folded_const_93);
  mul__657(folded_const_243, &res5a_bias_0[0], folded_const_13);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_423 = (float*)&__rescheduled_0[0UL];
  reorder__538(buffer_423, &res5a_bias_1[0]);
  mul__659(folded_const_244, buffer_423, folded_const_11);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_425 = (float*)&__rescheduled_0[0UL];
  reorder__544(buffer_425, &res5b_bias_0[0]);
  mul__663(folded_const_245, buffer_425, folded_const_8);
  // [f32 [1, 1, 4, 1, 1, 128] @ A1aBCD128b]
  float* buffer_427 = (float*)&__rescheduled_0[0UL];
  reorder__547(buffer_427, &res5b_bias_1[0]);
  mul__665(folded_const_246, buffer_427, folded_const_6);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_429 = (float*)&__rescheduled_0[0UL];
  reorder__553(buffer_429, &res5c_bias_0[0]);
  mul__669(folded_const_247, buffer_429, folded_const_3);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_431 = (float*)&__rescheduled_0[0UL];
  reorder__556(buffer_431, &res5c_bias_1[0]);
  mul__671(folded_const_248, buffer_431, folded_const_1);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_433 = (float*)&__rescheduled_0[0UL];
  reorder__477(buffer_433, &res4a_bias_b[0]);
  mul__617(folded_const_249, buffer_433, folded_const_92);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_435 = (float*)&__rescheduled_0[0UL];
  reorder__486(buffer_435, &res4a_bias_2[0]);
  mul__623(folded_const_250, buffer_435, folded_const_91);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_437 = (float*)&__rescheduled_0[0UL];
  reorder__495(buffer_437, &res4b_bias_2[0]);
  mul__629(folded_const_251, buffer_437, folded_const_90);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_439 = (float*)&__rescheduled_0[0UL];
  reorder__504(buffer_439, &res4c_bias_2[0]);
  mul__635(folded_const_252, buffer_439, folded_const_89);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_441 = (float*)&__rescheduled_0[0UL];
  reorder__513(buffer_441, &res4d_bias_2[0]);
  mul__641(folded_const_253, buffer_441, folded_const_88);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_443 = (float*)&__rescheduled_0[0UL];
  reorder__522(buffer_443, &res4e_bias_2[0]);
  mul__647(folded_const_254, buffer_443, folded_const_87);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_445 = (float*)&__rescheduled_0[0UL];
  reorder__531(buffer_445, &res4f_bias_2[0]);
  mul__653(folded_const_255, buffer_445, folded_const_86);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_447 = (float*)&__rescheduled_0[0UL];
  reorder__534(buffer_447, &res5a_bias_b[0]);
  mul__655(folded_const_256, buffer_447, folded_const_85);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_449 = (float*)&__rescheduled_0[0UL];
  reorder__541(buffer_449, &res5a_bias_2[0]);
  mul__661(folded_const_257, buffer_449, folded_const_84);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_451 = (float*)&__rescheduled_0[0UL];
  reorder__550(buffer_451, &res5b_bias_2[0]);
  mul__667(folded_const_258, buffer_451, folded_const_83);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_453 = (float*)&__rescheduled_0[0UL];
  reorder__559(buffer_453, &res5c_bias_2[0]);
  mul__673(folded_const_259, buffer_453, folded_const_82);
  // [f32 [64, 64, 1, 1] @ ABCD]
  float* buffer_455 = (float*)&__rescheduled_0[0UL];
  mul__110(buffer_455, res2a_weight_0, folded_const_154);
  // [s8 [64, 64, 1, 1] @ ABCD]
  int8_t* buffer_456 = (int8_t*)&__rescheduled_0[16384UL];
  cast__111(buffer_456, buffer_455);
  reorder__421(folded_const_260, buffer_456);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_458 = (float*)&__rescheduled_0[0UL];
  mul__107(buffer_458, res2a_weight_b, folded_const_155);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_459 = (int8_t*)&__rescheduled_0[65536UL];
  cast__108(buffer_459, buffer_458);
  reorder__418(folded_const_261, buffer_459);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_461 = (float*)&__rescheduled_0[0UL];
  mul__116(buffer_461, res2a_weight_2, folded_const_152);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_462 = (int8_t*)&__rescheduled_0[65536UL];
  cast__117(buffer_462, buffer_461);
  reorder__423(folded_const_262, buffer_462);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_464 = (float*)&__rescheduled_0[0UL];
  mul__125(buffer_464, res2b_weight_2, folded_const_149);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_465 = (int8_t*)&__rescheduled_0[65536UL];
  cast__126(buffer_465, buffer_464);
  reorder__428(folded_const_263, buffer_465);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_467 = (float*)&__rescheduled_0[0UL];
  mul__134(buffer_467, res2c_weight_2, folded_const_146);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_468 = (int8_t*)&__rescheduled_0[65536UL];
  cast__135(buffer_468, buffer_467);
  reorder__433(folded_const_264, buffer_468);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_470 = (float*)&__rescheduled_0[0UL];
  mul__119(buffer_470, res2b_weight_0, folded_const_151);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_471 = (int8_t*)&__rescheduled_0[65536UL];
  cast__120(buffer_471, buffer_470);
  reorder__426(folded_const_265, buffer_471);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_473 = (float*)&__rescheduled_0[0UL];
  mul__128(buffer_473, res2c_weight_0, folded_const_148);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_474 = (int8_t*)&__rescheduled_0[65536UL];
  cast__129(buffer_474, buffer_473);
  reorder__431(folded_const_266, buffer_474);
  // [f32 [128, 256, 1, 1] @ ABCD]
  float* buffer_476 = (float*)&__rescheduled_0[0UL];
  mul__140(buffer_476, res3a_weight_0, folded_const_144);
  // [s8 [128, 256, 1, 1] @ ABCD]
  int8_t* buffer_477 = (int8_t*)&__rescheduled_0[131072UL];
  cast__141(buffer_477, buffer_476);
  reorder__439(folded_const_267, buffer_477);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_479 = (float*)&__rescheduled_0[0UL];
  mul__113(buffer_479, res2a_weight_1, folded_const_153);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_480 = (int8_t*)&__rescheduled_0[147456UL];
  cast__114(buffer_480, buffer_479);
  reorder__422(folded_const_268, buffer_480);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_482 = (float*)&__rescheduled_0[0UL];
  mul__122(buffer_482, res2b_weight_1, folded_const_150);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_483 = (int8_t*)&__rescheduled_0[147456UL];
  cast__123(buffer_483, buffer_482);
  reorder__427(folded_const_269, buffer_483);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_485 = (float*)&__rescheduled_0[0UL];
  mul__131(buffer_485, res2c_weight_1, folded_const_147);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_486 = (int8_t*)&__rescheduled_0[147456UL];
  cast__132(buffer_486, buffer_485);
  reorder__432(folded_const_270, buffer_486);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_488 = (float*)&__rescheduled_0[0UL];
  mul__146(buffer_488, res3a_weight_2, folded_const_142);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_489 = (int8_t*)&__rescheduled_0[262144UL];
  cast__147(buffer_489, buffer_488);
  reorder__445(folded_const_271, buffer_489);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_491 = (float*)&__rescheduled_0[0UL];
  mul__155(buffer_491, res3b_weight_2, folded_const_139);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_492 = (int8_t*)&__rescheduled_0[262144UL];
  cast__156(buffer_492, buffer_491);
  reorder__454(folded_const_272, buffer_492);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_494 = (float*)&__rescheduled_0[0UL];
  mul__164(buffer_494, res3c_weight_2, folded_const_136);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_495 = (int8_t*)&__rescheduled_0[262144UL];
  cast__165(buffer_495, buffer_494);
  reorder__463(folded_const_273, buffer_495);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_497 = (float*)&__rescheduled_0[0UL];
  mul__173(buffer_497, res3d_weight_2, folded_const_133);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_498 = (int8_t*)&__rescheduled_0[262144UL];
  cast__174(buffer_498, buffer_497);
  reorder__472(folded_const_274, buffer_498);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_500 = (float*)&__rescheduled_0[0UL];
  mul__149(buffer_500, res3b_weight_0, folded_const_141);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_501 = (int8_t*)&__rescheduled_0[262144UL];
  cast__150(buffer_501, buffer_500);
  reorder__448(folded_const_275, buffer_501);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_503 = (float*)&__rescheduled_0[0UL];
  mul__158(buffer_503, res3c_weight_0, folded_const_138);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_504 = (int8_t*)&__rescheduled_0[262144UL];
  cast__159(buffer_504, buffer_503);
  reorder__457(folded_const_276, buffer_504);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_506 = (float*)&__rescheduled_0[0UL];
  mul__167(buffer_506, res3d_weight_0, folded_const_135);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_507 = (int8_t*)&__rescheduled_0[262144UL];
  cast__168(buffer_507, buffer_506);
  reorder__466(folded_const_277, buffer_507);
  // [f32 [512, 256, 1, 1] @ ABCD]
  float* buffer_509 = (float*)&__rescheduled_0[0UL];
  mul__137(buffer_509, res3a_weight_b, folded_const_145);
  // [s8 [512, 256, 1, 1] @ ABCD]
  int8_t* buffer_510 = (int8_t*)&__rescheduled_0[524288UL];
  cast__138(buffer_510, buffer_509);
  reorder__436(folded_const_278, buffer_510);
  // [f32 [256, 512, 1, 1] @ ABCD]
  float* buffer_512 = (float*)&__rescheduled_0[0UL];
  mul__179(buffer_512, res4a_weight_0, folded_const_131);
  // [s8 [256, 512, 1, 1] @ ABCD]
  int8_t* buffer_513 = (int8_t*)&__rescheduled_0[524288UL];
  cast__180(buffer_513, buffer_512);
  reorder__478(folded_const_279, buffer_513);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_515 = (float*)&__rescheduled_0[0UL];
  mul__143(buffer_515, res3a_weight_1, folded_const_143);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_516 = (int8_t*)&__rescheduled_0[589824UL];
  cast__144(buffer_516, buffer_515);
  reorder__442(folded_const_280, buffer_516);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_518 = (float*)&__rescheduled_0[0UL];
  mul__152(buffer_518, res3b_weight_1, folded_const_140);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_519 = (int8_t*)&__rescheduled_0[589824UL];
  cast__153(buffer_519, buffer_518);
  reorder__451(folded_const_281, buffer_519);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_521 = (float*)&__rescheduled_0[0UL];
  mul__161(buffer_521, res3c_weight_1, folded_const_137);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_522 = (int8_t*)&__rescheduled_0[589824UL];
  cast__162(buffer_522, buffer_521);
  reorder__460(folded_const_282, buffer_522);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_524 = (float*)&__rescheduled_0[0UL];
  mul__170(buffer_524, res3d_weight_1, folded_const_134);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_525 = (int8_t*)&__rescheduled_0[589824UL];
  cast__171(buffer_525, buffer_524);
  reorder__469(folded_const_283, buffer_525);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_527 = (float*)&__rescheduled_0[0UL];
  mul__185(buffer_527, res4a_weight_2, folded_const_129);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_528 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__186(buffer_528, buffer_527);
  reorder__484(folded_const_284, buffer_528);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_530 = (float*)&__rescheduled_0[0UL];
  mul__194(buffer_530, res4b_weight_2, folded_const_126);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_531 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__195(buffer_531, buffer_530);
  reorder__493(folded_const_285, buffer_531);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_533 = (float*)&__rescheduled_0[0UL];
  mul__203(buffer_533, res4c_weight_2, folded_const_123);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_534 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__204(buffer_534, buffer_533);
  reorder__502(folded_const_286, buffer_534);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_536 = (float*)&__rescheduled_0[0UL];
  mul__212(buffer_536, res4d_weight_2, folded_const_120);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_537 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__213(buffer_537, buffer_536);
  reorder__511(folded_const_287, buffer_537);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_539 = (float*)&__rescheduled_0[0UL];
  mul__221(buffer_539, res4e_weight_2, folded_const_117);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_540 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__222(buffer_540, buffer_539);
  reorder__520(folded_const_288, buffer_540);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_542 = (float*)&__rescheduled_0[0UL];
  mul__230(buffer_542, res4f_weight_2, folded_const_114);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_543 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__231(buffer_543, buffer_542);
  reorder__529(folded_const_289, buffer_543);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_545 = (float*)&__rescheduled_0[0UL];
  mul__188(buffer_545, res4b_weight_0, folded_const_128);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_546 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__189(buffer_546, buffer_545);
  reorder__487(folded_const_290, buffer_546);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_548 = (float*)&__rescheduled_0[0UL];
  mul__197(buffer_548, res4c_weight_0, folded_const_125);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_549 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__198(buffer_549, buffer_548);
  reorder__496(folded_const_291, buffer_549);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_551 = (float*)&__rescheduled_0[0UL];
  mul__206(buffer_551, res4d_weight_0, folded_const_122);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_552 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__207(buffer_552, buffer_551);
  reorder__505(folded_const_292, buffer_552);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_554 = (float*)&__rescheduled_0[0UL];
  mul__215(buffer_554, res4e_weight_0, folded_const_119);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_555 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__216(buffer_555, buffer_554);
  reorder__514(folded_const_293, buffer_555);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_557 = (float*)&__rescheduled_0[0UL];
  mul__224(buffer_557, res4f_weight_0, folded_const_116);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_558 = (int8_t*)&__rescheduled_0[1048576UL];
  cast__225(buffer_558, buffer_557);
  reorder__523(folded_const_294, buffer_558);
  // [f32 [1024, 512, 1, 1] @ ABCD]
  float* buffer_560 = (float*)&__rescheduled_0[0UL];
  mul__176(buffer_560, res4a_weight_b, folded_const_132);
  // [s8 [1024, 512, 1, 1] @ ABCD]
  int8_t* buffer_561 = (int8_t*)&__rescheduled_0[2097152UL];
  cast__177(buffer_561, buffer_560);
  reorder__475(folded_const_295, buffer_561);
  // [f32 [512, 1024, 1, 1] @ ABCD]
  float* buffer_563 = (float*)&__rescheduled_0[0UL];
  mul__236(buffer_563, res5a_weight_0, folded_const_112);
  // [s8 [512, 1024, 1, 1] @ ABCD]
  int8_t* buffer_564 = (int8_t*)&__rescheduled_0[2097152UL];
  cast__237(buffer_564, buffer_563);
  reorder__535(folded_const_296, buffer_564);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_566 = (float*)&__rescheduled_0[0UL];
  mul__182(buffer_566, res4a_weight_1, folded_const_130);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_567 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__183(buffer_567, buffer_566);
  reorder__481(folded_const_297, buffer_567);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_569 = (float*)&__rescheduled_0[0UL];
  mul__191(buffer_569, res4b_weight_1, folded_const_127);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_570 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__192(buffer_570, buffer_569);
  reorder__490(folded_const_298, buffer_570);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_572 = (float*)&__rescheduled_0[0UL];
  mul__200(buffer_572, res4c_weight_1, folded_const_124);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_573 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__201(buffer_573, buffer_572);
  reorder__499(folded_const_299, buffer_573);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_575 = (float*)&__rescheduled_0[0UL];
  mul__209(buffer_575, res4d_weight_1, folded_const_121);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_576 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__210(buffer_576, buffer_575);
  reorder__508(folded_const_300, buffer_576);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_578 = (float*)&__rescheduled_0[0UL];
  mul__218(buffer_578, res4e_weight_1, folded_const_118);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_579 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__219(buffer_579, buffer_578);
  reorder__517(folded_const_301, buffer_579);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_581 = (float*)&__rescheduled_0[0UL];
  mul__227(buffer_581, res4f_weight_1, folded_const_115);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_582 = (int8_t*)&__rescheduled_0[2359296UL];
  cast__228(buffer_582, buffer_581);
  reorder__526(folded_const_302, buffer_582);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_584 = (float*)&__rescheduled_0[0UL];
  mul__242(buffer_584, res5a_weight_2, folded_const_110);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_585 = (int8_t*)&__rescheduled_0[4194304UL];
  cast__243(buffer_585, buffer_584);
  reorder__539(folded_const_303, buffer_585);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_587 = (float*)&__rescheduled_0[0UL];
  mul__251(buffer_587, res5b_weight_2, folded_const_107);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_588 = (int8_t*)&__rescheduled_0[4194304UL];
  cast__252(buffer_588, buffer_587);
  reorder__548(folded_const_304, buffer_588);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_590 = (float*)&__rescheduled_0[0UL];
  mul__260(buffer_590, res5c_weight_2, folded_const_104);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_591 = (int8_t*)&__rescheduled_0[4194304UL];
  cast__261(buffer_591, buffer_590);
  reorder__557(folded_const_305, buffer_591);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_593 = (float*)&__rescheduled_0[0UL];
  mul__245(buffer_593, res5b_weight_0, folded_const_109);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_594 = (int8_t*)&__rescheduled_0[4194304UL];
  cast__246(buffer_594, buffer_593);
  reorder__542(folded_const_306, buffer_594);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_596 = (float*)&__rescheduled_0[0UL];
  mul__254(buffer_596, res5c_weight_0, folded_const_106);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_597 = (int8_t*)&__rescheduled_0[4194304UL];
  cast__255(buffer_597, buffer_596);
  reorder__551(folded_const_307, buffer_597);
  // [f32 [2048, 1024, 1, 1] @ ABCD]
  float* buffer_599 = (float*)&__rescheduled_0[0UL];
  mul__233(buffer_599, res5a_weight_b, folded_const_113);
  // [s8 [2048, 1024, 1, 1] @ ABCD]
  int8_t* buffer_600 = (int8_t*)&__rescheduled_0[8388608UL];
  cast__234(buffer_600, buffer_599);
  reorder__532(folded_const_308, buffer_600);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_602 = (float*)&__rescheduled_0[0UL];
  mul__239(buffer_602, res5a_weight_1, folded_const_111);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_603 = (int8_t*)&__rescheduled_0[9437184UL];
  cast__240(buffer_603, buffer_602);
  reorder__536(folded_const_309, buffer_603);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_605 = (float*)&__rescheduled_0[0UL];
  mul__248(buffer_605, res5b_weight_1, folded_const_108);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_606 = (int8_t*)&__rescheduled_0[9437184UL];
  cast__249(buffer_606, buffer_605);
  reorder__545(folded_const_310, buffer_606);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_608 = (float*)&__rescheduled_0[0UL];
  mul__257(buffer_608, res5c_weight_1, folded_const_105);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_609 = (int8_t*)&__rescheduled_0[9437184UL];
  cast__258(buffer_609, buffer_608);
  reorder__554(folded_const_311, buffer_609);
  is_init = true;
  sc_aligned_free(__stream, __rescheduled_0);
}

extern "C" void sc_init_rn50_backbone_bs256() {
  bool& is_init = *(bool*)(__module_data + 0);
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  uint8_t* __brgemm_attrs = (uint8_t*)&__module_data[192UL];
  void*& __sc_kernel_cache_3 = *(void**)(__module_data + 16);
  uint8_t* __brgemm_attrs_2 = (uint8_t*)&__module_data[320UL];
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  uint8_t* __brgemm_attrs_4 = (uint8_t*)&__module_data[448UL];
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 32);
  uint8_t* __brgemm_attrs_6 = (uint8_t*)&__module_data[576UL];
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 40);
  uint8_t* __brgemm_attrs_8 = (uint8_t*)&__module_data[704UL];
  void*& __sc_kernel_cache_11 = *(void**)(__module_data + 48);
  uint8_t* __brgemm_attrs_10 = (uint8_t*)&__module_data[832UL];
  void*& __sc_kernel_cache_13 = *(void**)(__module_data + 56);
  uint8_t* __brgemm_attrs_12 = (uint8_t*)&__module_data[960UL];
  void*& __sc_kernel_cache_15 = *(void**)(__module_data + 64);
  uint8_t* __brgemm_attrs_14 = (uint8_t*)&__module_data[1088UL];
  void*& __sc_kernel_cache_18 = *(void**)(__module_data + 72);
  uint8_t* __brgemm_attrs_17 = (uint8_t*)&__module_data[2240UL];
  void*& __sc_kernel_cache_20 = *(void**)(__module_data + 80);
  uint8_t* __brgemm_attrs_19 = (uint8_t*)&__module_data[2368UL];
  void*& __sc_kernel_cache_22 = *(void**)(__module_data + 88);
  uint8_t* __brgemm_attrs_21 = (uint8_t*)&__module_data[2496UL];
  void*& __sc_kernel_cache_24 = *(void**)(__module_data + 96);
  uint8_t* __brgemm_attrs_23 = (uint8_t*)&__module_data[2624UL];
  void*& __sc_kernel_cache_26 = *(void**)(__module_data + 104);
  uint8_t* __brgemm_attrs_25 = (uint8_t*)&__module_data[2752UL];
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 112);
  uint8_t* __brgemm_attrs_27 = (uint8_t*)&__module_data[2880UL];
  void*& __sc_kernel_cache_34 = *(void**)(__module_data + 120);
  uint8_t* __brgemm_attrs_33 = (uint8_t*)&__module_data[3392UL];
  void*& __sc_kernel_cache_36 = *(void**)(__module_data + 128);
  uint8_t* __brgemm_attrs_35 = (uint8_t*)&__module_data[3520UL];
  void*& __sc_kernel_cache_38 = *(void**)(__module_data + 136);
  uint8_t* __brgemm_attrs_37 = (uint8_t*)&__module_data[3648UL];
  void*& __sc_kernel_cache_40 = *(void**)(__module_data + 144);
  uint8_t* __brgemm_attrs_39 = (uint8_t*)&__module_data[3776UL];
  void*& __sc_kernel_cache_42 = *(void**)(__module_data + 152);
  uint8_t* __brgemm_attrs_41 = (uint8_t*)&__module_data[3904UL];
  void*& __sc_kernel_cache_48 = *(void**)(__module_data + 160);
  uint8_t* __brgemm_attrs_47 = (uint8_t*)&__module_data[4224UL];
  void*& __sc_kernel_cache_52 = *(void**)(__module_data + 168);
  uint8_t* __brgemm_attrs_51 = (uint8_t*)&__module_data[4480UL];
  void** __brgemm_bd_mask_arr = (void**)&__uninitialized_data[23657472UL];
  uint8_t* __brgemm_full_bd_mask = (uint8_t*)&__module_data[1344UL];
  void** __brgemm_bd_mask_arr_31 = (void**)&__uninitialized_data[23657504UL];
  uint8_t* __brgemm_full_bd_mask_30 = (uint8_t*)&__module_data[3136UL];
  void** __brgemm_bd_mask_arr_45 = (void**)&__uninitialized_data[23657520UL];
  uint8_t* __brgemm_full_bd_mask_44 = (uint8_t*)&__module_data[4152UL];
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  uint8_t* __brgemm_attrs_16 = (uint8_t*)&__module_data[1216UL];
  void** __sc_kernel_cache_arr_32 = (void**)&__uninitialized_data[23657512UL];
  uint8_t* __brgemm_attrs_29 = (uint8_t*)&__module_data[3008UL];
  void** __sc_kernel_cache_arr_46 = (void**)&__uninitialized_data[23657528UL];
  uint8_t* __brgemm_attrs_43 = (uint8_t*)&__module_data[4032UL];
  void** __sc_kernel_cache_arr_50 = (void**)&__uninitialized_data[23657536UL];
  uint8_t* __brgemm_attrs_49 = (uint8_t*)&__module_data[4352UL];
  is_init = false;
  __sc_kernel_cache = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs, ((void*)0), ((void*)0));
  __sc_kernel_cache_3 = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_2, ((void*)0), ((void*)0));
  __sc_kernel_cache_5 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_4, ((void*)0), ((void*)0));
  __sc_kernel_cache_7 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_6, ((void*)0), ((void*)0));
  __sc_kernel_cache_9 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_8, ((void*)0), ((void*)0));
  __sc_kernel_cache_11 = dnnl_brgemm_list_func(28, 64, 64, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_10, ((void*)0), ((void*)0));
  __sc_kernel_cache_13 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_12, ((void*)0), ((void*)0));
  __sc_kernel_cache_15 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_14, ((void*)0), ((void*)0));
  __sc_kernel_cache_18 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_17, ((void*)0), ((void*)0));
  __sc_kernel_cache_20 = dnnl_brgemm_list_func(392, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_19, ((void*)0), ((void*)0));
  __sc_kernel_cache_22 = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_21, ((void*)0), ((void*)0));
  __sc_kernel_cache_24 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_23, ((void*)0), ((void*)0));
  __sc_kernel_cache_26 = dnnl_brgemm_list_func(14, 64, 64, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_25, ((void*)0), ((void*)0));
  __sc_kernel_cache_28 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_27, ((void*)0), ((void*)0));
  __sc_kernel_cache_34 = dnnl_brgemm_list_func(49, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_33, ((void*)0), ((void*)0));
  __sc_kernel_cache_36 = dnnl_brgemm_list_func(28, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_35, ((void*)0), ((void*)0));
  __sc_kernel_cache_38 = dnnl_brgemm_list_func(7, 256, 64, 128, 256, 256, 0.f, 7, 7, __brgemm_attrs_37, ((void*)0), ((void*)0));
  __sc_kernel_cache_40 = dnnl_brgemm_list_func(49, 512, 512, 512, 512, 512, 0.f, 7, 7, __brgemm_attrs_39, ((void*)0), ((void*)0));
  __sc_kernel_cache_42 = dnnl_brgemm_list_func(49, 64, 512, 512, 64, 64, 0.f, 7, 7, __brgemm_attrs_41, ((void*)0), ((void*)0));
  __sc_kernel_cache_48 = dnnl_brgemm_list_func(49, 256, 512, 512, 256, 256, 0.f, 7, 7, __brgemm_attrs_47, ((void*)0), ((void*)0));
  __sc_kernel_cache_52 = dnnl_brgemm_list_func(49, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_51, ((void*)0), ((void*)0));
  __brgemm_bd_mask_arr[0] = &__brgemm_full_bd_mask[(0 * 419)];
  __brgemm_bd_mask_arr[1] = &__brgemm_full_bd_mask[(1 * 419)];
  __brgemm_bd_mask_arr_31[0] = &__brgemm_full_bd_mask_30[(0 * 222)];
  __brgemm_bd_mask_arr_45[0] = &__brgemm_full_bd_mask_44[(0 * 61)];
  __sc_kernel_cache_arr[0] = dnnl_brgemm_list_func(419, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_16, __brgemm_bd_mask_arr[0], ((void*)0));
  __sc_kernel_cache_arr[1] = dnnl_brgemm_list_func(419, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_16, __brgemm_bd_mask_arr[1], ((void*)0));
  __sc_kernel_cache_arr_32[0] = dnnl_brgemm_list_func(222, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_29, __brgemm_bd_mask_arr_31[0], ((void*)0));
  __sc_kernel_cache_arr_46[0] = dnnl_brgemm_list_func(61, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_43, __brgemm_bd_mask_arr_45[0], ((void*)0));
  __sc_kernel_cache_arr_50[0] = dnnl_brgemm_list_func(61, 64, 512, 512, 64, 64, 0.f, 7, 7, __brgemm_attrs_49, __brgemm_bd_mask_arr_45[0], ((void*)0));
}
