
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t rn50_backbone_bs8_data[218688];
static constexpr int8_t* __module_data = rn50_backbone_bs8_data;
alignas(64) static int8_t __uninitialized_data[23657544UL];

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106)));
static bool batchwise_8_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)));
static bool batchwise_4_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59)));
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
static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
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
static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__430(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
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


extern "C" void rn50_backbone_bs8(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[216064UL];
  float* folded_const_156 = (float*)&__uninitialized_data[0UL];
  float* folded_const_222 = (float*)&__uninitialized_data[111616UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[211968UL];
  float* folded_const_157 = (float*)&__uninitialized_data[1024UL];
  float* folded_const_208 = (float*)&__uninitialized_data[105984UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[347136UL];
  float* folded_const_158 = (float*)&__uninitialized_data[1280UL];
  float* folded_const_209 = (float*)&__uninitialized_data[106240UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[232448UL];
  float* folded_const_159 = (float*)&__uninitialized_data[1536UL];
  float* folded_const_223 = (float*)&__uninitialized_data[112640UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[281600UL];
  float* folded_const_160 = (float*)&__uninitialized_data[2560UL];
  float* folded_const_210 = (float*)&__uninitialized_data[106496UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[384000UL];
  float* folded_const_161 = (float*)&__uninitialized_data[2816UL];
  float* folded_const_211 = (float*)&__uninitialized_data[106752UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[248832UL];
  float* folded_const_162 = (float*)&__uninitialized_data[3072UL];
  float* folded_const_224 = (float*)&__uninitialized_data[113664UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[297984UL];
  float* folded_const_163 = (float*)&__uninitialized_data[4096UL];
  float* folded_const_212 = (float*)&__uninitialized_data[107008UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[420864UL];
  float* folded_const_164 = (float*)&__uninitialized_data[4352UL];
  float* folded_const_213 = (float*)&__uninitialized_data[107264UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[265216UL];
  float* folded_const_165 = (float*)&__uninitialized_data[4608UL];
  float* folded_const_225 = (float*)&__uninitialized_data[114688UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[916480UL];
  float* folded_const_166 = (float*)&__uninitialized_data[5632UL];
  float* folded_const_238 = (float*)&__uninitialized_data[128000UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[314368UL];
  float* folded_const_167 = (float*)&__uninitialized_data[7680UL];
  float* folded_const_214 = (float*)&__uninitialized_data[107520UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[1178624UL];
  float* folded_const_168 = (float*)&__uninitialized_data[8192UL];
  float* folded_const_215 = (float*)&__uninitialized_data[108032UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[457728UL];
  float* folded_const_169 = (float*)&__uninitialized_data[8704UL];
  float* folded_const_239 = (float*)&__uninitialized_data[130048UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[719872UL];
  float* folded_const_170 = (float*)&__uninitialized_data[10752UL];
  float* folded_const_216 = (float*)&__uninitialized_data[108544UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[1326080UL];
  float* folded_const_171 = (float*)&__uninitialized_data[11264UL];
  float* folded_const_217 = (float*)&__uninitialized_data[109056UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[523264UL];
  float* folded_const_172 = (float*)&__uninitialized_data[11776UL];
  float* folded_const_240 = (float*)&__uninitialized_data[132096UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[785408UL];
  float* folded_const_173 = (float*)&__uninitialized_data[13824UL];
  float* folded_const_218 = (float*)&__uninitialized_data[109568UL];
  int8_t* folded_const_282 = (int8_t*)&__uninitialized_data[1473536UL];
  float* folded_const_174 = (float*)&__uninitialized_data[14336UL];
  float* folded_const_219 = (float*)&__uninitialized_data[110080UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[588800UL];
  float* folded_const_175 = (float*)&__uninitialized_data[14848UL];
  float* folded_const_241 = (float*)&__uninitialized_data[134144UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[850944UL];
  float* folded_const_176 = (float*)&__uninitialized_data[16896UL];
  float* folded_const_220 = (float*)&__uninitialized_data[110592UL];
  int8_t* folded_const_283 = (int8_t*)&__uninitialized_data[1620992UL];
  float* folded_const_177 = (float*)&__uninitialized_data[17408UL];
  float* folded_const_221 = (float*)&__uninitialized_data[111104UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[654336UL];
  float* folded_const_178 = (float*)&__uninitialized_data[17920UL];
  float* folded_const_242 = (float*)&__uninitialized_data[136192UL];
  int8_t* folded_const_295 = (int8_t*)&__uninitialized_data[4652032UL];
  float* folded_const_179 = (float*)&__uninitialized_data[19968UL];
  float* folded_const_249 = (float*)&__uninitialized_data[150528UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[1047552UL];
  float* folded_const_180 = (float*)&__uninitialized_data[24064UL];
  float* folded_const_226 = (float*)&__uninitialized_data[115712UL];
  int8_t* folded_const_297 = (int8_t*)&__uninitialized_data[5700608UL];
  float* folded_const_181 = (float*)&__uninitialized_data[25088UL];
  float* folded_const_227 = (float*)&__uninitialized_data[116736UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[1768448UL];
  float* folded_const_182 = (float*)&__uninitialized_data[26112UL];
  float* folded_const_250 = (float*)&__uninitialized_data[154624UL];
  int8_t* folded_const_290 = (int8_t*)&__uninitialized_data[3341312UL];
  float* folded_const_183 = (float*)&__uninitialized_data[30208UL];
  float* folded_const_228 = (float*)&__uninitialized_data[117760UL];
  int8_t* folded_const_298 = (int8_t*)&__uninitialized_data[6290432UL];
  float* folded_const_184 = (float*)&__uninitialized_data[31232UL];
  float* folded_const_229 = (float*)&__uninitialized_data[118784UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[2030592UL];
  float* folded_const_185 = (float*)&__uninitialized_data[32256UL];
  float* folded_const_251 = (float*)&__uninitialized_data[158720UL];
  int8_t* folded_const_291 = (int8_t*)&__uninitialized_data[3603456UL];
  float* folded_const_186 = (float*)&__uninitialized_data[36352UL];
  float* folded_const_230 = (float*)&__uninitialized_data[119808UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[6880256UL];
  float* folded_const_187 = (float*)&__uninitialized_data[37376UL];
  float* folded_const_231 = (float*)&__uninitialized_data[120832UL];
  int8_t* folded_const_286 = (int8_t*)&__uninitialized_data[2292736UL];
  float* folded_const_188 = (float*)&__uninitialized_data[38400UL];
  float* folded_const_252 = (float*)&__uninitialized_data[162816UL];
  int8_t* folded_const_292 = (int8_t*)&__uninitialized_data[3865600UL];
  float* folded_const_189 = (float*)&__uninitialized_data[42496UL];
  float* folded_const_232 = (float*)&__uninitialized_data[121856UL];
  int8_t* folded_const_300 = (int8_t*)&__uninitialized_data[7470080UL];
  float* folded_const_190 = (float*)&__uninitialized_data[43520UL];
  float* folded_const_233 = (float*)&__uninitialized_data[122880UL];
  int8_t* folded_const_287 = (int8_t*)&__uninitialized_data[2554880UL];
  float* folded_const_191 = (float*)&__uninitialized_data[44544UL];
  float* folded_const_253 = (float*)&__uninitialized_data[166912UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[4127744UL];
  float* folded_const_192 = (float*)&__uninitialized_data[48640UL];
  float* folded_const_234 = (float*)&__uninitialized_data[123904UL];
  int8_t* folded_const_301 = (int8_t*)&__uninitialized_data[8059904UL];
  float* folded_const_193 = (float*)&__uninitialized_data[49664UL];
  float* folded_const_235 = (float*)&__uninitialized_data[124928UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[2817024UL];
  float* folded_const_194 = (float*)&__uninitialized_data[50688UL];
  float* folded_const_254 = (float*)&__uninitialized_data[171008UL];
  int8_t* folded_const_294 = (int8_t*)&__uninitialized_data[4389888UL];
  float* folded_const_195 = (float*)&__uninitialized_data[54784UL];
  float* folded_const_236 = (float*)&__uninitialized_data[125952UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[8649728UL];
  float* folded_const_196 = (float*)&__uninitialized_data[55808UL];
  float* folded_const_237 = (float*)&__uninitialized_data[126976UL];
  int8_t* folded_const_289 = (int8_t*)&__uninitialized_data[3079168UL];
  float* folded_const_197 = (float*)&__uninitialized_data[56832UL];
  float* folded_const_255 = (float*)&__uninitialized_data[175104UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[14482432UL];
  float* folded_const_198 = (float*)&__uninitialized_data[60928UL];
  float* folded_const_256 = (float*)&__uninitialized_data[179200UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[5176320UL];
  float* folded_const_199 = (float*)&__uninitialized_data[69120UL];
  float* folded_const_243 = (float*)&__uninitialized_data[138240UL];
  int8_t* folded_const_309 = (int8_t*)&__uninitialized_data[16579584UL];
  float* folded_const_200 = (float*)&__uninitialized_data[71168UL];
  float* folded_const_244 = (float*)&__uninitialized_data[140288UL];
  int8_t* folded_const_303 = (int8_t*)&__uninitialized_data[9239552UL];
  float* folded_const_201 = (float*)&__uninitialized_data[73216UL];
  float* folded_const_257 = (float*)&__uninitialized_data[187392UL];
  int8_t* folded_const_306 = (int8_t*)&__uninitialized_data[12385280UL];
  float* folded_const_202 = (float*)&__uninitialized_data[81408UL];
  float* folded_const_245 = (float*)&__uninitialized_data[142336UL];
  int8_t* folded_const_310 = (int8_t*)&__uninitialized_data[18938880UL];
  float* folded_const_203 = (float*)&__uninitialized_data[83456UL];
  float* folded_const_246 = (float*)&__uninitialized_data[144384UL];
  int8_t* folded_const_304 = (int8_t*)&__uninitialized_data[10288128UL];
  float* folded_const_204 = (float*)&__uninitialized_data[85504UL];
  float* folded_const_258 = (float*)&__uninitialized_data[195584UL];
  int8_t* folded_const_307 = (int8_t*)&__uninitialized_data[13433856UL];
  float* folded_const_205 = (float*)&__uninitialized_data[93696UL];
  float* folded_const_247 = (float*)&__uninitialized_data[146432UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[21298176UL];
  float* folded_const_206 = (float*)&__uninitialized_data[95744UL];
  float* folded_const_248 = (float*)&__uninitialized_data[148480UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[11336704UL];
  float* folded_const_207 = (float*)&__uninitialized_data[97792UL];
  float* folded_const_259 = (float*)&__uninitialized_data[203776UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 86638592UL);
  if (!is_init) {
    __init_const_globals(backbone_output, backbone_input, res2a_weight_b, res2a_bias_b, res2a_weight_0, res2a_bias_0, res2a_weight_1, res2a_bias_1, res2a_weight_2, res2a_bias_2, res2b_weight_0, res2b_bias_0, res2b_weight_1, res2b_bias_1, res2b_weight_2, res2b_bias_2, res2c_weight_0, res2c_bias_0, res2c_weight_1, res2c_bias_1, res2c_weight_2, res2c_bias_2, res3a_weight_b, res3a_bias_b, res3a_weight_0, res3a_bias_0, res3a_weight_1, res3a_bias_1, res3a_weight_2, res3a_bias_2, res3b_weight_0, res3b_bias_0, res3b_weight_1, res3b_bias_1, res3b_weight_2, res3b_bias_2, res3c_weight_0, res3c_bias_0, res3c_weight_1, res3c_bias_1, res3c_weight_2, res3c_bias_2, res3d_weight_0, res3d_bias_0, res3d_weight_1, res3d_bias_1, res3d_weight_2, res3d_bias_2, res4a_weight_b, res4a_bias_b, res4a_weight_0, res4a_bias_0, res4a_weight_1, res4a_bias_1, res4a_weight_2, res4a_bias_2, res4b_weight_0, res4b_bias_0, res4b_weight_1, res4b_bias_1, res4b_weight_2, res4b_bias_2, res4c_weight_0, res4c_bias_0, res4c_weight_1, res4c_bias_1, res4c_weight_2, res4c_bias_2, res4d_weight_0, res4d_bias_0, res4d_weight_1, res4d_bias_1, res4d_weight_2, res4d_bias_2, res4e_weight_0, res4e_bias_0, res4e_weight_1, res4e_bias_1, res4e_weight_2, res4e_bias_2, res4f_weight_0, res4f_bias_0, res4f_weight_1, res4f_bias_1, res4f_weight_2, res4f_bias_2, res5a_weight_b, res5a_bias_b, res5a_weight_0, res5a_bias_0, res5a_weight_1, res5a_bias_1, res5a_weight_2, res5a_bias_2, res5b_weight_0, res5b_bias_0, res5b_weight_1, res5b_bias_1, res5b_weight_2, res5b_bias_2, res5c_weight_0, res5c_bias_0, res5c_weight_1, res5c_bias_1, res5c_weight_2, res5c_bias_2);
  }
  // [s8 [8, 1, 8, 28, 28, 64] @ A1aBCD64b]
  int8_t* buffer_611 = (int8_t*)&__rescheduled_0[0UL];
  batchwise_8_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(buffer_611, &backbone_input[0UL], folded_const_261, folded_const_156, folded_const_222, folded_const_260, folded_const_157, folded_const_208, folded_const_268, folded_const_158, folded_const_209, folded_const_262, folded_const_159, folded_const_223, folded_const_265, folded_const_160, folded_const_210, folded_const_269, folded_const_161, folded_const_211, folded_const_263, folded_const_162, folded_const_224, folded_const_266, folded_const_163, folded_const_212, folded_const_270, folded_const_164, folded_const_213, folded_const_264, folded_const_165, folded_const_225, folded_const_278, folded_const_166, folded_const_238, folded_const_267, folded_const_167, folded_const_214, folded_const_280, folded_const_168, folded_const_215, folded_const_271, folded_const_169, folded_const_239, folded_const_275, folded_const_170, folded_const_216, folded_const_281, folded_const_171, folded_const_217, folded_const_272, folded_const_172, folded_const_240, folded_const_276, folded_const_173, folded_const_218, folded_const_282, folded_const_174, folded_const_219, folded_const_273, folded_const_175, folded_const_241, folded_const_277, folded_const_176, folded_const_220, folded_const_283, folded_const_177, folded_const_221, folded_const_274, folded_const_178, folded_const_242);
  // [s8 [4, 2, 16, 14, 14, 64] @ A2aBCD64b]
  int8_t* buffer_612 = (int8_t*)&__rescheduled_0[42467328UL];
  batchwise_4_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(buffer_612, &buffer_611[0UL], folded_const_295, folded_const_179, folded_const_249, folded_const_279, folded_const_180, folded_const_226, folded_const_297, folded_const_181, folded_const_227, folded_const_284, folded_const_182, folded_const_250, folded_const_290, folded_const_183, folded_const_228, folded_const_298, folded_const_184, folded_const_229, folded_const_285, folded_const_185, folded_const_251, folded_const_291, folded_const_186, folded_const_230, folded_const_299, folded_const_187, folded_const_231, folded_const_286, folded_const_188, folded_const_252, folded_const_292, folded_const_189, folded_const_232, folded_const_300, folded_const_190, folded_const_233, folded_const_287, folded_const_191, folded_const_253, folded_const_293, folded_const_192, folded_const_234, folded_const_301, folded_const_193, folded_const_235, folded_const_288, folded_const_194, folded_const_254, folded_const_294, folded_const_195, folded_const_236, folded_const_302, folded_const_196, folded_const_237, folded_const_289, folded_const_197, folded_const_255);
  // [s8 [8, 1, 4, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_613 = (int8_t*)&__rescheduled_0[0UL];
  res5a_conv_b_cast_mul_add_cast__683(buffer_613, &buffer_612[0UL], folded_const_308, folded_const_198, folded_const_256);
  // [s8 [8, 1, 8, 16, 16, 64] @ A1aBCD64b]
  int8_t* buffer_614 = (int8_t*)&__rescheduled_0[53084160UL];
  res5a_conv_0_cast_mul_add_cast_relu_reorder__682(buffer_614, &buffer_612[0UL], folded_const_296, folded_const_199, folded_const_243);
  // [s8 [8, 1, 1, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_615 = (int8_t*)&__rescheduled_0[42467328UL];
  res5a_conv_1_cast_mul_add_cast_relu_reorder__681(buffer_615, buffer_614, folded_const_309, folded_const_200, folded_const_244);
  res5a_conv_2_cast_mul_add_cast_add_relu__680(buffer_613, buffer_615, folded_const_303, folded_const_201, folded_const_257, buffer_613);
  res5b_conv_0_cast_mul_add_cast_relu__679(buffer_615, buffer_613, folded_const_306, folded_const_202, folded_const_245);
  res5b_conv_1_cast_mul_add_cast_relu_reorder__678(buffer_614, buffer_615, folded_const_310, folded_const_203, folded_const_246);
  res5b_conv_2_cast_mul_add_cast_add_relu__677(buffer_613, buffer_614, folded_const_304, folded_const_204, folded_const_258, buffer_613);
  res5c_conv_0_cast_mul_add_cast_relu_reorder__676(buffer_615, buffer_613, folded_const_307, folded_const_205, folded_const_247);
  res5c_conv_1_cast_mul_add_cast_relu__675(buffer_614, buffer_615, folded_const_311, folded_const_206, folded_const_248);
  res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(backbone_output, buffer_614, folded_const_305, folded_const_207, folded_const_259, buffer_613);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool reorder__419(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 += 1UL) {
    for (uint64_t _fuseiter_4091 = 0UL; _fuseiter_4091 < 64UL; _fuseiter_4091 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 / 4UL) * 256UL) + (_fuseiter_4091 + ((fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4086___fuseiter_4087_1156___fuseiter_4088_1157___fuseiter_4089_1158___fuseiter_4090_1159 % 4UL) * 64UL) + _fuseiter_4091))] = __cached_1;
    }
  }
  return true;
}

static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 += 1UL) {
    for (uint64_t _fuseiter_4097 = 0UL; _fuseiter_4097 < 64UL; _fuseiter_4097 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 % 4UL) * 64UL)) + _fuseiter_4097)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1160____itr_2_1161____itr_3_1162____itr_4_1163 % 4UL) * 64UL)) + _fuseiter_4097)]);
    }
  }
  return true;
}

static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4104 = 0UL; _fuseiter_4104 < 64UL; _fuseiter_4104 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4104]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4104]);
  }
  return true;
}

static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4111 = 0UL; _fuseiter_4111 < 64UL; _fuseiter_4111 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4111]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4111]);
  }
  return true;
}

static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 += 1UL) {
    for (uint64_t _fuseiter_4118 = 0UL; _fuseiter_4118 < 64UL; _fuseiter_4118 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 / 4UL) * 256UL) + (_fuseiter_4118 + ((fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4113___fuseiter_4114_1172___fuseiter_4115_1173___fuseiter_4116_1174___fuseiter_4117_1175 % 4UL) * 64UL) + _fuseiter_4118))] = __cached_1;
    }
  }
  return true;
}

static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 += 1UL) {
    for (uint64_t _fuseiter_4124 = 0UL; _fuseiter_4124 < 64UL; _fuseiter_4124 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 % 4UL) * 64UL)) + _fuseiter_4124)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1176____itr_2_1177____itr_3_1178____itr_4_1179 % 4UL) * 64UL)) + _fuseiter_4124)]);
    }
  }
  return true;
}

static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4131 = 0UL; _fuseiter_4131 < 64UL; _fuseiter_4131 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4131]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4131]);
  }
  return true;
}

static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4138 = 0UL; _fuseiter_4138 < 64UL; _fuseiter_4138 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4138]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4138]);
  }
  return true;
}

static bool reorder__429(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 += 1UL) {
    for (uint64_t _fuseiter_4145 = 0UL; _fuseiter_4145 < 64UL; _fuseiter_4145 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 / 4UL) * 256UL) + (_fuseiter_4145 + ((fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4140___fuseiter_4141_1188___fuseiter_4142_1189___fuseiter_4143_1190___fuseiter_4144_1191 % 4UL) * 64UL) + _fuseiter_4145))] = __cached_1;
    }
  }
  return true;
}

static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 += 1UL) {
    for (uint64_t _fuseiter_4151 = 0UL; _fuseiter_4151 < 64UL; _fuseiter_4151 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 % 4UL) * 64UL)) + _fuseiter_4151)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1192____itr_2_1193____itr_3_1194____itr_4_1195 % 4UL) * 64UL)) + _fuseiter_4151)]);
    }
  }
  return true;
}

static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4158 = 0UL; _fuseiter_4158 < 64UL; _fuseiter_4158 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4158]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4158]);
  }
  return true;
}

static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4165 = 0UL; _fuseiter_4165 < 64UL; _fuseiter_4165 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4165]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4165]);
  }
  return true;
}

static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 += 1UL) {
    for (uint64_t _fuseiter_4172 = 0UL; _fuseiter_4172 < 64UL; _fuseiter_4172 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 / 4UL) * 256UL) + (_fuseiter_4172 + ((fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4167___fuseiter_4168_1204___fuseiter_4169_1205___fuseiter_4170_1206___fuseiter_4171_1207 % 4UL) * 64UL) + _fuseiter_4172))] = __cached_1;
    }
  }
  return true;
}

static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 += 1UL) {
    for (uint64_t _fuseiter_4178 = 0UL; _fuseiter_4178 < 64UL; _fuseiter_4178 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 % 4UL) * 64UL)) + _fuseiter_4178)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1208____itr_2_1209____itr_3_1210____itr_4_1211 % 4UL) * 64UL)) + _fuseiter_4178)]);
    }
  }
  return true;
}

static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 += 1UL) {
    for (uint64_t _fuseiter_4185 = 0UL; _fuseiter_4185 < 64UL; _fuseiter_4185 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 / 8UL) * 512UL) + (_fuseiter_4185 + ((fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4180___fuseiter_4181_1212___fuseiter_4182_1213___fuseiter_4183_1214___fuseiter_4184_1215 % 8UL) * 64UL) + _fuseiter_4185))] = __cached_1;
    }
  }
  return true;
}

static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 += 1UL) {
    for (uint64_t _fuseiter_4191 = 0UL; _fuseiter_4191 < 64UL; _fuseiter_4191 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 % 8UL) * 64UL)) + _fuseiter_4191)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1216____itr_2_1217____itr_3_1218____itr_4_1219 % 8UL) * 64UL)) + _fuseiter_4191)]);
    }
  }
  return true;
}

static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 += 1UL) {
    for (uint64_t _fuseiter_4198 = 0UL; _fuseiter_4198 < 64UL; _fuseiter_4198 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 / 2UL) * 128UL) + (_fuseiter_4198 + ((fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4193___fuseiter_4194_1220___fuseiter_4195_1221___fuseiter_4196_1222___fuseiter_4197_1223 % 2UL) * 64UL) + _fuseiter_4198))] = __cached_1;
    }
  }
  return true;
}

static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 += 1UL) {
    for (uint64_t _fuseiter_4204 = 0UL; _fuseiter_4204 < 64UL; _fuseiter_4204 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 % 2UL) * 64UL)) + _fuseiter_4204)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1224____itr_2_1225____itr_3_1226____itr_4_1227 % 2UL) * 64UL)) + _fuseiter_4204)]);
    }
  }
  return true;
}

static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 += 1UL) {
    for (uint64_t _fuseiter_4211 = 0UL; _fuseiter_4211 < 64UL; _fuseiter_4211 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 / 2UL) * 128UL) + (_fuseiter_4211 + ((fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4206___fuseiter_4207_1228___fuseiter_4208_1229___fuseiter_4209_1230___fuseiter_4210_1231 % 2UL) * 64UL) + _fuseiter_4211))] = __cached_1;
    }
  }
  return true;
}

static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 += 1UL) {
    for (uint64_t _fuseiter_4217 = 0UL; _fuseiter_4217 < 64UL; _fuseiter_4217 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 % 2UL) * 64UL)) + _fuseiter_4217)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1232____itr_2_1233____itr_3_1234____itr_4_1235 % 2UL) * 64UL)) + _fuseiter_4217)]);
    }
  }
  return true;
}

static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 += 1UL) {
    for (uint64_t _fuseiter_4224 = 0UL; _fuseiter_4224 < 64UL; _fuseiter_4224 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 / 8UL) * 512UL) + (_fuseiter_4224 + ((fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4219___fuseiter_4220_1236___fuseiter_4221_1237___fuseiter_4222_1238___fuseiter_4223_1239 % 8UL) * 64UL) + _fuseiter_4224))] = __cached_1;
    }
  }
  return true;
}

static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 += 1UL) {
    for (uint64_t _fuseiter_4230 = 0UL; _fuseiter_4230 < 64UL; _fuseiter_4230 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 % 8UL) * 64UL)) + _fuseiter_4230)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1240____itr_2_1241____itr_3_1242____itr_4_1243 % 8UL) * 64UL)) + _fuseiter_4230)]);
    }
  }
  return true;
}

static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 += 1UL) {
    for (uint64_t _fuseiter_4237 = 0UL; _fuseiter_4237 < 64UL; _fuseiter_4237 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 / 2UL) * 128UL) + (_fuseiter_4237 + ((fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4232___fuseiter_4233_1244___fuseiter_4234_1245___fuseiter_4235_1246___fuseiter_4236_1247 % 2UL) * 64UL) + _fuseiter_4237))] = __cached_1;
    }
  }
  return true;
}

static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 += 1UL) {
    for (uint64_t _fuseiter_4243 = 0UL; _fuseiter_4243 < 64UL; _fuseiter_4243 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 % 2UL) * 64UL)) + _fuseiter_4243)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1248____itr_2_1249____itr_3_1250____itr_4_1251 % 2UL) * 64UL)) + _fuseiter_4243)]);
    }
  }
  return true;
}

static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 += 1UL) {
    for (uint64_t _fuseiter_4250 = 0UL; _fuseiter_4250 < 64UL; _fuseiter_4250 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 / 2UL) * 128UL) + (_fuseiter_4250 + ((fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4245___fuseiter_4246_1252___fuseiter_4247_1253___fuseiter_4248_1254___fuseiter_4249_1255 % 2UL) * 64UL) + _fuseiter_4250))] = __cached_1;
    }
  }
  return true;
}

static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 += 1UL) {
    for (uint64_t _fuseiter_4256 = 0UL; _fuseiter_4256 < 64UL; _fuseiter_4256 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 % 2UL) * 64UL)) + _fuseiter_4256)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1256____itr_2_1257____itr_3_1258____itr_4_1259 % 2UL) * 64UL)) + _fuseiter_4256)]);
    }
  }
  return true;
}

static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 += 1UL) {
    for (uint64_t _fuseiter_4263 = 0UL; _fuseiter_4263 < 64UL; _fuseiter_4263 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 / 8UL) * 512UL) + (_fuseiter_4263 + ((fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4258___fuseiter_4259_1260___fuseiter_4260_1261___fuseiter_4261_1262___fuseiter_4262_1263 % 8UL) * 64UL) + _fuseiter_4263))] = __cached_1;
    }
  }
  return true;
}

static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 += 1UL) {
    for (uint64_t _fuseiter_4269 = 0UL; _fuseiter_4269 < 64UL; _fuseiter_4269 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 % 8UL) * 64UL)) + _fuseiter_4269)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1264____itr_2_1265____itr_3_1266____itr_4_1267 % 8UL) * 64UL)) + _fuseiter_4269)]);
    }
  }
  return true;
}

static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 += 1UL) {
    for (uint64_t _fuseiter_4276 = 0UL; _fuseiter_4276 < 64UL; _fuseiter_4276 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 / 2UL) * 128UL) + (_fuseiter_4276 + ((fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4271___fuseiter_4272_1268___fuseiter_4273_1269___fuseiter_4274_1270___fuseiter_4275_1271 % 2UL) * 64UL) + _fuseiter_4276))] = __cached_1;
    }
  }
  return true;
}

static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 += 1UL) {
    for (uint64_t _fuseiter_4282 = 0UL; _fuseiter_4282 < 64UL; _fuseiter_4282 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 % 2UL) * 64UL)) + _fuseiter_4282)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1272____itr_2_1273____itr_3_1274____itr_4_1275 % 2UL) * 64UL)) + _fuseiter_4282)]);
    }
  }
  return true;
}

static bool reorder__461(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 += 1UL) {
    for (uint64_t _fuseiter_4289 = 0UL; _fuseiter_4289 < 64UL; _fuseiter_4289 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 / 2UL) * 128UL) + (_fuseiter_4289 + ((fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4284___fuseiter_4285_1276___fuseiter_4286_1277___fuseiter_4287_1278___fuseiter_4288_1279 % 2UL) * 64UL) + _fuseiter_4289))] = __cached_1;
    }
  }
  return true;
}

static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 += 1UL) {
    for (uint64_t _fuseiter_4295 = 0UL; _fuseiter_4295 < 64UL; _fuseiter_4295 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 % 2UL) * 64UL)) + _fuseiter_4295)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1280____itr_2_1281____itr_3_1282____itr_4_1283 % 2UL) * 64UL)) + _fuseiter_4295)]);
    }
  }
  return true;
}

static bool reorder__464(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 += 1UL) {
    for (uint64_t _fuseiter_4302 = 0UL; _fuseiter_4302 < 64UL; _fuseiter_4302 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 / 8UL) * 512UL) + (_fuseiter_4302 + ((fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4297___fuseiter_4298_1284___fuseiter_4299_1285___fuseiter_4300_1286___fuseiter_4301_1287 % 8UL) * 64UL) + _fuseiter_4302))] = __cached_1;
    }
  }
  return true;
}

static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 += 1UL) {
    for (uint64_t _fuseiter_4308 = 0UL; _fuseiter_4308 < 64UL; _fuseiter_4308 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 % 8UL) * 64UL)) + _fuseiter_4308)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1288____itr_2_1289____itr_3_1290____itr_4_1291 % 8UL) * 64UL)) + _fuseiter_4308)]);
    }
  }
  return true;
}

static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 += 1UL) {
    for (uint64_t _fuseiter_4315 = 0UL; _fuseiter_4315 < 64UL; _fuseiter_4315 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 / 2UL) * 128UL) + (_fuseiter_4315 + ((fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4310___fuseiter_4311_1292___fuseiter_4312_1293___fuseiter_4313_1294___fuseiter_4314_1295 % 2UL) * 64UL) + _fuseiter_4315))] = __cached_1;
    }
  }
  return true;
}

static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 += 1UL) {
    for (uint64_t _fuseiter_4321 = 0UL; _fuseiter_4321 < 64UL; _fuseiter_4321 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 % 2UL) * 64UL)) + _fuseiter_4321)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1296____itr_2_1297____itr_3_1298____itr_4_1299 % 2UL) * 64UL)) + _fuseiter_4321)]);
    }
  }
  return true;
}

static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 += 1UL) {
    for (uint64_t _fuseiter_4328 = 0UL; _fuseiter_4328 < 64UL; _fuseiter_4328 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 / 2UL) * 128UL) + (_fuseiter_4328 + ((fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4323___fuseiter_4324_1300___fuseiter_4325_1301___fuseiter_4326_1302___fuseiter_4327_1303 % 2UL) * 64UL) + _fuseiter_4328))] = __cached_1;
    }
  }
  return true;
}

static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 += 1UL) {
    for (uint64_t _fuseiter_4334 = 0UL; _fuseiter_4334 < 64UL; _fuseiter_4334 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 % 2UL) * 64UL)) + _fuseiter_4334)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1304____itr_2_1305____itr_3_1306____itr_4_1307 % 2UL) * 64UL)) + _fuseiter_4334)]);
    }
  }
  return true;
}

static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 += 1UL) {
    for (uint64_t _fuseiter_4341 = 0UL; _fuseiter_4341 < 64UL; _fuseiter_4341 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 / 8UL) * 512UL) + (_fuseiter_4341 + ((fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4336___fuseiter_4337_1308___fuseiter_4338_1309___fuseiter_4339_1310___fuseiter_4340_1311 % 8UL) * 64UL) + _fuseiter_4341))] = __cached_1;
    }
  }
  return true;
}

static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 += 1UL) {
    for (uint64_t _fuseiter_4347 = 0UL; _fuseiter_4347 < 64UL; _fuseiter_4347 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 % 8UL) * 64UL)) + _fuseiter_4347)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1312____itr_2_1313____itr_3_1314____itr_4_1315 % 8UL) * 64UL)) + _fuseiter_4347)]);
    }
  }
  return true;
}

static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 = 0UL; fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 < 16UL; fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 += 1UL) {
    for (uint64_t _fuseiter_4354 = 0UL; _fuseiter_4354 < 64UL; _fuseiter_4354 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 / 16UL) * 1024UL) + (_fuseiter_4354 + ((fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4349___fuseiter_4350_1316___fuseiter_4351_1317 % 16UL) * 64UL) + _fuseiter_4354))] = __cached_1;
    }
  }
  return true;
}

static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 += 1UL) {
    for (uint64_t _fuseiter_4360 = 0UL; _fuseiter_4360 < 64UL; _fuseiter_4360 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 % 16UL) * 64UL)) + _fuseiter_4360)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1318____itr_2_1319____itr_3_1320____itr_4_1321 % 16UL) * 64UL)) + _fuseiter_4360)]);
    }
  }
  return true;
}

static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 += 1UL) {
    for (uint64_t _fuseiter_4367 = 0UL; _fuseiter_4367 < 64UL; _fuseiter_4367 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 / 4UL) * 256UL) + (_fuseiter_4367 + ((fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4362___fuseiter_4363_1322___fuseiter_4364_1323___fuseiter_4365_1324___fuseiter_4366_1325 % 4UL) * 64UL) + _fuseiter_4367))] = __cached_1;
    }
  }
  return true;
}

static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 += 1UL) {
    for (uint64_t _fuseiter_4373 = 0UL; _fuseiter_4373 < 64UL; _fuseiter_4373 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 % 4UL) * 64UL)) + _fuseiter_4373)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1326____itr_2_1327____itr_3_1328____itr_4_1329 % 4UL) * 64UL)) + _fuseiter_4373)]);
    }
  }
  return true;
}

static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 += 1UL) {
    for (uint64_t _fuseiter_4380 = 0UL; _fuseiter_4380 < 64UL; _fuseiter_4380 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 / 4UL) * 256UL) + (_fuseiter_4380 + ((fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4375___fuseiter_4376_1330___fuseiter_4377_1331___fuseiter_4378_1332___fuseiter_4379_1333 % 4UL) * 64UL) + _fuseiter_4380))] = __cached_1;
    }
  }
  return true;
}

static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 += 1UL) {
    for (uint64_t _fuseiter_4386 = 0UL; _fuseiter_4386 < 64UL; _fuseiter_4386 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 % 4UL) * 64UL)) + _fuseiter_4386)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1334____itr_2_1335____itr_3_1336____itr_4_1337 % 4UL) * 64UL)) + _fuseiter_4386)]);
    }
  }
  return true;
}

static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 = 0UL; fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 < 16UL; fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 += 1UL) {
    for (uint64_t _fuseiter_4393 = 0UL; _fuseiter_4393 < 64UL; _fuseiter_4393 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 / 16UL) * 1024UL) + (_fuseiter_4393 + ((fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4388___fuseiter_4389_1338___fuseiter_4390_1339 % 16UL) * 64UL) + _fuseiter_4393))] = __cached_1;
    }
  }
  return true;
}

static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 += 1UL) {
    for (uint64_t _fuseiter_4399 = 0UL; _fuseiter_4399 < 64UL; _fuseiter_4399 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 % 16UL) * 64UL)) + _fuseiter_4399)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1340____itr_2_1341____itr_3_1342____itr_4_1343 % 16UL) * 64UL)) + _fuseiter_4399)]);
    }
  }
  return true;
}

static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 += 1UL) {
    for (uint64_t _fuseiter_4406 = 0UL; _fuseiter_4406 < 64UL; _fuseiter_4406 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 / 4UL) * 256UL) + (_fuseiter_4406 + ((fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4401___fuseiter_4402_1344___fuseiter_4403_1345___fuseiter_4404_1346___fuseiter_4405_1347 % 4UL) * 64UL) + _fuseiter_4406))] = __cached_1;
    }
  }
  return true;
}

static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 += 1UL) {
    for (uint64_t _fuseiter_4412 = 0UL; _fuseiter_4412 < 64UL; _fuseiter_4412 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 % 4UL) * 64UL)) + _fuseiter_4412)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1348____itr_2_1349____itr_3_1350____itr_4_1351 % 4UL) * 64UL)) + _fuseiter_4412)]);
    }
  }
  return true;
}

static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 += 1UL) {
    for (uint64_t _fuseiter_4419 = 0UL; _fuseiter_4419 < 64UL; _fuseiter_4419 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 / 4UL) * 256UL) + (_fuseiter_4419 + ((fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4414___fuseiter_4415_1352___fuseiter_4416_1353___fuseiter_4417_1354___fuseiter_4418_1355 % 4UL) * 64UL) + _fuseiter_4419))] = __cached_1;
    }
  }
  return true;
}

static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 += 1UL) {
    for (uint64_t _fuseiter_4425 = 0UL; _fuseiter_4425 < 64UL; _fuseiter_4425 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 % 4UL) * 64UL)) + _fuseiter_4425)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1356____itr_2_1357____itr_3_1358____itr_4_1359 % 4UL) * 64UL)) + _fuseiter_4425)]);
    }
  }
  return true;
}

static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 = 0UL; fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 < 16UL; fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 += 1UL) {
    for (uint64_t _fuseiter_4432 = 0UL; _fuseiter_4432 < 64UL; _fuseiter_4432 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 / 16UL) * 1024UL) + (_fuseiter_4432 + ((fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4427___fuseiter_4428_1360___fuseiter_4429_1361 % 16UL) * 64UL) + _fuseiter_4432))] = __cached_1;
    }
  }
  return true;
}

static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 += 1UL) {
    for (uint64_t _fuseiter_4438 = 0UL; _fuseiter_4438 < 64UL; _fuseiter_4438 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 % 16UL) * 64UL)) + _fuseiter_4438)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1362____itr_2_1363____itr_3_1364____itr_4_1365 % 16UL) * 64UL)) + _fuseiter_4438)]);
    }
  }
  return true;
}

static bool reorder__497(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 += 1UL) {
    for (uint64_t _fuseiter_4445 = 0UL; _fuseiter_4445 < 64UL; _fuseiter_4445 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 / 4UL) * 256UL) + (_fuseiter_4445 + ((fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4440___fuseiter_4441_1366___fuseiter_4442_1367___fuseiter_4443_1368___fuseiter_4444_1369 % 4UL) * 64UL) + _fuseiter_4445))] = __cached_1;
    }
  }
  return true;
}

static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 += 1UL) {
    for (uint64_t _fuseiter_4451 = 0UL; _fuseiter_4451 < 64UL; _fuseiter_4451 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 % 4UL) * 64UL)) + _fuseiter_4451)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1370____itr_2_1371____itr_3_1372____itr_4_1373 % 4UL) * 64UL)) + _fuseiter_4451)]);
    }
  }
  return true;
}

static bool reorder__500(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 += 1UL) {
    for (uint64_t _fuseiter_4458 = 0UL; _fuseiter_4458 < 64UL; _fuseiter_4458 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 / 4UL) * 256UL) + (_fuseiter_4458 + ((fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4453___fuseiter_4454_1374___fuseiter_4455_1375___fuseiter_4456_1376___fuseiter_4457_1377 % 4UL) * 64UL) + _fuseiter_4458))] = __cached_1;
    }
  }
  return true;
}

static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 += 1UL) {
    for (uint64_t _fuseiter_4464 = 0UL; _fuseiter_4464 < 64UL; _fuseiter_4464 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 % 4UL) * 64UL)) + _fuseiter_4464)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1378____itr_2_1379____itr_3_1380____itr_4_1381 % 4UL) * 64UL)) + _fuseiter_4464)]);
    }
  }
  return true;
}

static bool reorder__503(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 = 0UL; fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 < 16UL; fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 += 1UL) {
    for (uint64_t _fuseiter_4471 = 0UL; _fuseiter_4471 < 64UL; _fuseiter_4471 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 / 16UL) * 1024UL) + (_fuseiter_4471 + ((fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4466___fuseiter_4467_1382___fuseiter_4468_1383 % 16UL) * 64UL) + _fuseiter_4471))] = __cached_1;
    }
  }
  return true;
}

static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 += 1UL) {
    for (uint64_t _fuseiter_4477 = 0UL; _fuseiter_4477 < 64UL; _fuseiter_4477 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 % 16UL) * 64UL)) + _fuseiter_4477)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1384____itr_2_1385____itr_3_1386____itr_4_1387 % 16UL) * 64UL)) + _fuseiter_4477)]);
    }
  }
  return true;
}

static bool reorder__506(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 += 1UL) {
    for (uint64_t _fuseiter_4484 = 0UL; _fuseiter_4484 < 64UL; _fuseiter_4484 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 / 4UL) * 256UL) + (_fuseiter_4484 + ((fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4479___fuseiter_4480_1388___fuseiter_4481_1389___fuseiter_4482_1390___fuseiter_4483_1391 % 4UL) * 64UL) + _fuseiter_4484))] = __cached_1;
    }
  }
  return true;
}

static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 += 1UL) {
    for (uint64_t _fuseiter_4490 = 0UL; _fuseiter_4490 < 64UL; _fuseiter_4490 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 % 4UL) * 64UL)) + _fuseiter_4490)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1392____itr_2_1393____itr_3_1394____itr_4_1395 % 4UL) * 64UL)) + _fuseiter_4490)]);
    }
  }
  return true;
}

static bool reorder__509(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 += 1UL) {
    for (uint64_t _fuseiter_4497 = 0UL; _fuseiter_4497 < 64UL; _fuseiter_4497 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 / 4UL) * 256UL) + (_fuseiter_4497 + ((fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4492___fuseiter_4493_1396___fuseiter_4494_1397___fuseiter_4495_1398___fuseiter_4496_1399 % 4UL) * 64UL) + _fuseiter_4497))] = __cached_1;
    }
  }
  return true;
}

static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 += 1UL) {
    for (uint64_t _fuseiter_4503 = 0UL; _fuseiter_4503 < 64UL; _fuseiter_4503 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 % 4UL) * 64UL)) + _fuseiter_4503)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1400____itr_2_1401____itr_3_1402____itr_4_1403 % 4UL) * 64UL)) + _fuseiter_4503)]);
    }
  }
  return true;
}

static bool reorder__512(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 = 0UL; fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 < 16UL; fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 += 1UL) {
    for (uint64_t _fuseiter_4510 = 0UL; _fuseiter_4510 < 64UL; _fuseiter_4510 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 / 16UL) * 1024UL) + (_fuseiter_4510 + ((fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4505___fuseiter_4506_1404___fuseiter_4507_1405 % 16UL) * 64UL) + _fuseiter_4510))] = __cached_1;
    }
  }
  return true;
}

static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 += 1UL) {
    for (uint64_t _fuseiter_4516 = 0UL; _fuseiter_4516 < 64UL; _fuseiter_4516 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 % 16UL) * 64UL)) + _fuseiter_4516)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1406____itr_2_1407____itr_3_1408____itr_4_1409 % 16UL) * 64UL)) + _fuseiter_4516)]);
    }
  }
  return true;
}

static bool reorder__515(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 += 1UL) {
    for (uint64_t _fuseiter_4523 = 0UL; _fuseiter_4523 < 64UL; _fuseiter_4523 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 / 4UL) * 256UL) + (_fuseiter_4523 + ((fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4518___fuseiter_4519_1410___fuseiter_4520_1411___fuseiter_4521_1412___fuseiter_4522_1413 % 4UL) * 64UL) + _fuseiter_4523))] = __cached_1;
    }
  }
  return true;
}

static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 += 1UL) {
    for (uint64_t _fuseiter_4529 = 0UL; _fuseiter_4529 < 64UL; _fuseiter_4529 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 % 4UL) * 64UL)) + _fuseiter_4529)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1414____itr_2_1415____itr_3_1416____itr_4_1417 % 4UL) * 64UL)) + _fuseiter_4529)]);
    }
  }
  return true;
}

static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 += 1UL) {
    for (uint64_t _fuseiter_4536 = 0UL; _fuseiter_4536 < 64UL; _fuseiter_4536 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 / 4UL) * 256UL) + (_fuseiter_4536 + ((fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4531___fuseiter_4532_1418___fuseiter_4533_1419___fuseiter_4534_1420___fuseiter_4535_1421 % 4UL) * 64UL) + _fuseiter_4536))] = __cached_1;
    }
  }
  return true;
}

static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 += 1UL) {
    for (uint64_t _fuseiter_4542 = 0UL; _fuseiter_4542 < 64UL; _fuseiter_4542 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 % 4UL) * 64UL)) + _fuseiter_4542)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1422____itr_2_1423____itr_3_1424____itr_4_1425 % 4UL) * 64UL)) + _fuseiter_4542)]);
    }
  }
  return true;
}

static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 = 0UL; fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 < 16UL; fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 += 1UL) {
    for (uint64_t _fuseiter_4549 = 0UL; _fuseiter_4549 < 64UL; _fuseiter_4549 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 / 16UL) * 1024UL) + (_fuseiter_4549 + ((fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4544___fuseiter_4545_1426___fuseiter_4546_1427 % 16UL) * 64UL) + _fuseiter_4549))] = __cached_1;
    }
  }
  return true;
}

static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 += 1UL) {
    for (uint64_t _fuseiter_4555 = 0UL; _fuseiter_4555 < 64UL; _fuseiter_4555 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 % 16UL) * 64UL)) + _fuseiter_4555)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1428____itr_2_1429____itr_3_1430____itr_4_1431 % 16UL) * 64UL)) + _fuseiter_4555)]);
    }
  }
  return true;
}

static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 += 1UL) {
    for (uint64_t _fuseiter_4562 = 0UL; _fuseiter_4562 < 64UL; _fuseiter_4562 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 / 4UL) * 256UL) + (_fuseiter_4562 + ((fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4557___fuseiter_4558_1432___fuseiter_4559_1433___fuseiter_4560_1434___fuseiter_4561_1435 % 4UL) * 64UL) + _fuseiter_4562))] = __cached_1;
    }
  }
  return true;
}

static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 += 1UL) {
    for (uint64_t _fuseiter_4568 = 0UL; _fuseiter_4568 < 64UL; _fuseiter_4568 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 % 4UL) * 64UL)) + _fuseiter_4568)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1436____itr_2_1437____itr_3_1438____itr_4_1439 % 4UL) * 64UL)) + _fuseiter_4568)]);
    }
  }
  return true;
}

static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 += 1UL) {
    for (uint64_t _fuseiter_4575 = 0UL; _fuseiter_4575 < 64UL; _fuseiter_4575 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 / 4UL) * 256UL) + (_fuseiter_4575 + ((fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4570___fuseiter_4571_1440___fuseiter_4572_1441___fuseiter_4573_1442___fuseiter_4574_1443 % 4UL) * 64UL) + _fuseiter_4575))] = __cached_1;
    }
  }
  return true;
}

static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 += 1UL) {
    for (uint64_t _fuseiter_4581 = 0UL; _fuseiter_4581 < 64UL; _fuseiter_4581 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 % 4UL) * 64UL)) + _fuseiter_4581)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1444____itr_2_1445____itr_3_1446____itr_4_1447 % 4UL) * 64UL)) + _fuseiter_4581)]);
    }
  }
  return true;
}

static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 = 0UL; fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 < 16UL; fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 += 1UL) {
    for (uint64_t _fuseiter_4588 = 0UL; _fuseiter_4588 < 64UL; _fuseiter_4588 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 / 16UL) * 1024UL) + (_fuseiter_4588 + ((fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 % 16UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_4583___fuseiter_4584_1448___fuseiter_4585_1449 % 16UL) * 64UL) + _fuseiter_4588))] = __cached_1;
    }
  }
  return true;
}

static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 += 1UL) {
    for (uint64_t _fuseiter_4594 = 0UL; _fuseiter_4594 < 64UL; _fuseiter_4594 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 % 16UL) * 64UL)) + _fuseiter_4594)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1450____itr_2_1451____itr_3_1452____itr_4_1453 % 16UL) * 64UL)) + _fuseiter_4594)]);
    }
  }
  return true;
}

static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 += 1UL) {
    for (uint64_t _fuseiter_4601 = 0UL; _fuseiter_4601 < 512UL; _fuseiter_4601 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 / 4UL) * 2048UL) + (_fuseiter_4601 + ((fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4596___fuseiter_4597_1454___fuseiter_4598_1455___fuseiter_4599_1456___fuseiter_4600_1457 % 4UL) * 512UL) + _fuseiter_4601))] = __cached_1;
    }
  }
  return true;
}

static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 += 1UL) {
    for (uint64_t _fuseiter_4607 = 0UL; _fuseiter_4607 < 512UL; _fuseiter_4607 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 % 4UL) * 512UL)) + _fuseiter_4607)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1458____itr_2_1459____itr_3_1460____itr_4_1461 % 4UL) * 512UL)) + _fuseiter_4607)]);
    }
  }
  return true;
}

static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4614 = 0UL; _fuseiter_4614 < 512UL; _fuseiter_4614 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4614]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4614]);
  }
  return true;
}

static bool reorder__537(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 += 1UL) {
    for (uint64_t _fuseiter_4621 = 0UL; _fuseiter_4621 < 256UL; _fuseiter_4621 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 / 2UL) * 512UL) + (_fuseiter_4621 + ((fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 % 2UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4616___fuseiter_4617_1466___fuseiter_4618_1467___fuseiter_4619_1468___fuseiter_4620_1469 % 2UL) * 256UL) + _fuseiter_4621))] = __cached_1;
    }
  }
  return true;
}

static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 += 1UL) {
    for (uint64_t _fuseiter_4627 = 0UL; _fuseiter_4627 < 256UL; _fuseiter_4627 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 % 2UL) * 256UL)) + _fuseiter_4627)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1470____itr_2_1471____itr_3_1472____itr_4_1473 % 2UL) * 256UL)) + _fuseiter_4627)]);
    }
  }
  return true;
}

static bool reorder__540(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 += 1UL) {
    for (uint64_t _fuseiter_4634 = 0UL; _fuseiter_4634 < 512UL; _fuseiter_4634 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 / 4UL) * 2048UL) + (_fuseiter_4634 + ((fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4629___fuseiter_4630_1474___fuseiter_4631_1475___fuseiter_4632_1476___fuseiter_4633_1477 % 4UL) * 512UL) + _fuseiter_4634))] = __cached_1;
    }
  }
  return true;
}

static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 += 1UL) {
    for (uint64_t _fuseiter_4640 = 0UL; _fuseiter_4640 < 512UL; _fuseiter_4640 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 % 4UL) * 512UL)) + _fuseiter_4640)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1478____itr_2_1479____itr_3_1480____itr_4_1481 % 4UL) * 512UL)) + _fuseiter_4640)]);
    }
  }
  return true;
}

static bool reorder__543(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 += 1UL) {
    for (uint64_t _fuseiter_4647 = 0UL; _fuseiter_4647 < 64UL; _fuseiter_4647 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 / 8UL) * 512UL) + (_fuseiter_4647 + ((fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4642___fuseiter_4643_1482___fuseiter_4644_1483___fuseiter_4645_1484___fuseiter_4646_1485 % 8UL) * 64UL) + _fuseiter_4647))] = __cached_1;
    }
  }
  return true;
}

static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 += 1UL) {
    for (uint64_t _fuseiter_4653 = 0UL; _fuseiter_4653 < 64UL; _fuseiter_4653 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 % 8UL) * 64UL)) + _fuseiter_4653)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1486____itr_2_1487____itr_3_1488____itr_4_1489 % 8UL) * 64UL)) + _fuseiter_4653)]);
    }
  }
  return true;
}

static bool reorder__546(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 += 1UL) {
    for (uint64_t _fuseiter_4660 = 0UL; _fuseiter_4660 < 128UL; _fuseiter_4660 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 / 4UL) * 512UL) + (_fuseiter_4660 + ((fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 / 4UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4655___fuseiter_4656_1490___fuseiter_4657_1491___fuseiter_4658_1492___fuseiter_4659_1493 % 4UL) * 128UL) + _fuseiter_4660))] = __cached_1;
    }
  }
  return true;
}

static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 += 1UL) {
    for (uint64_t _fuseiter_4666 = 0UL; _fuseiter_4666 < 128UL; _fuseiter_4666 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 % 4UL) * 128UL)) + _fuseiter_4666)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1494____itr_2_1495____itr_3_1496____itr_4_1497 % 4UL) * 128UL)) + _fuseiter_4666)]);
    }
  }
  return true;
}

static bool reorder__549(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 += 1UL) {
    for (uint64_t _fuseiter_4673 = 0UL; _fuseiter_4673 < 512UL; _fuseiter_4673 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 / 4UL) * 2048UL) + (_fuseiter_4673 + ((fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4668___fuseiter_4669_1498___fuseiter_4670_1499___fuseiter_4671_1500___fuseiter_4672_1501 % 4UL) * 512UL) + _fuseiter_4673))] = __cached_1;
    }
  }
  return true;
}

static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 += 1UL) {
    for (uint64_t _fuseiter_4679 = 0UL; _fuseiter_4679 < 512UL; _fuseiter_4679 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 % 4UL) * 512UL)) + _fuseiter_4679)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1502____itr_2_1503____itr_3_1504____itr_4_1505 % 4UL) * 512UL)) + _fuseiter_4679)]);
    }
  }
  return true;
}

static bool reorder__552(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 += 1UL) {
    for (uint64_t _fuseiter_4686 = 0UL; _fuseiter_4686 < 256UL; _fuseiter_4686 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 / 2UL) * 512UL) + (_fuseiter_4686 + ((fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 % 2UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4681___fuseiter_4682_1506___fuseiter_4683_1507___fuseiter_4684_1508___fuseiter_4685_1509 % 2UL) * 256UL) + _fuseiter_4686))] = __cached_1;
    }
  }
  return true;
}

static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 += 1UL) {
    for (uint64_t _fuseiter_4692 = 0UL; _fuseiter_4692 < 256UL; _fuseiter_4692 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 % 2UL) * 256UL)) + _fuseiter_4692)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1510____itr_2_1511____itr_3_1512____itr_4_1513 % 2UL) * 256UL)) + _fuseiter_4692)]);
    }
  }
  return true;
}

static bool reorder__555(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 += 1UL) {
    for (uint64_t _fuseiter_4699 = 0UL; _fuseiter_4699 < 64UL; _fuseiter_4699 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 / 8UL) * 512UL) + (_fuseiter_4699 + ((fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4694___fuseiter_4695_1514___fuseiter_4696_1515___fuseiter_4697_1516___fuseiter_4698_1517 % 8UL) * 64UL) + _fuseiter_4699))] = __cached_1;
    }
  }
  return true;
}

static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 += 1UL) {
    for (uint64_t _fuseiter_4705 = 0UL; _fuseiter_4705 < 64UL; _fuseiter_4705 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 % 8UL) * 64UL)) + _fuseiter_4705)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1518____itr_2_1519____itr_3_1520____itr_4_1521 % 8UL) * 64UL)) + _fuseiter_4705)]);
    }
  }
  return true;
}

static bool reorder__558(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 += 1UL) {
    for (uint64_t _fuseiter_4712 = 0UL; _fuseiter_4712 < 512UL; _fuseiter_4712 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 / 4UL) * 2048UL) + (_fuseiter_4712 + ((fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4707___fuseiter_4708_1522___fuseiter_4709_1523___fuseiter_4710_1524___fuseiter_4711_1525 % 4UL) * 512UL) + _fuseiter_4712))] = __cached_1;
    }
  }
  return true;
}

static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 += 1UL) {
    for (uint64_t _fuseiter_4718 = 0UL; _fuseiter_4718 < 512UL; _fuseiter_4718 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 % 4UL) * 512UL)) + _fuseiter_4718)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1526____itr_2_1527____itr_3_1528____itr_4_1529 % 4UL) * 512UL)) + _fuseiter_4718)]);
    }
  }
  return true;
}

static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4725 = 0UL; _fuseiter_4725 < 64UL; _fuseiter_4725 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4725]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4725]);
  }
  return true;
}

static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4732 = 0UL; _fuseiter_4732 < 64UL; _fuseiter_4732 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4732]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4732]);
  }
  return true;
}

static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4739 = 0UL; _fuseiter_4739 < 64UL; _fuseiter_4739 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4739]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4739]);
  }
  return true;
}

static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4746 = 0UL; _fuseiter_4746 < 64UL; _fuseiter_4746 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4746]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4746]);
  }
  return true;
}

static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4753 = 0UL; _fuseiter_4753 < 64UL; _fuseiter_4753 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4753]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4753]);
  }
  return true;
}

static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_4760 = 0UL; _fuseiter_4760 < 64UL; _fuseiter_4760 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_4760]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_4760]);
  }
  return true;
}

static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 += 1UL) {
    for (uint64_t _fuseiter_4767 = 0UL; _fuseiter_4767 < 64UL; _fuseiter_4767 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 / 2UL) * 128UL) + (_fuseiter_4767 + ((fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4762___fuseiter_4763_1554___fuseiter_4764_1555___fuseiter_4765_1556___fuseiter_4766_1557 % 2UL) * 64UL) + _fuseiter_4767))]);
    }
  }
  return true;
}

static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 += 1UL) {
    for (uint64_t _fuseiter_4773 = 0UL; _fuseiter_4773 < 64UL; _fuseiter_4773 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 % 2UL) * 64UL)) + _fuseiter_4773)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1558____itr_2_1559____itr_3_1560____itr_4_1561 % 2UL) * 64UL)) + _fuseiter_4773)]);
    }
  }
  return true;
}

static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 += 1UL) {
    for (uint64_t _fuseiter_4780 = 0UL; _fuseiter_4780 < 64UL; _fuseiter_4780 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 / 2UL) * 128UL) + (_fuseiter_4780 + ((fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4775___fuseiter_4776_1562___fuseiter_4777_1563___fuseiter_4778_1564___fuseiter_4779_1565 % 2UL) * 64UL) + _fuseiter_4780))]);
    }
  }
  return true;
}

static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 += 1UL) {
    for (uint64_t _fuseiter_4786 = 0UL; _fuseiter_4786 < 64UL; _fuseiter_4786 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 % 2UL) * 64UL)) + _fuseiter_4786)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1566____itr_2_1567____itr_3_1568____itr_4_1569 % 2UL) * 64UL)) + _fuseiter_4786)]);
    }
  }
  return true;
}

static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 += 1UL) {
    for (uint64_t _fuseiter_4793 = 0UL; _fuseiter_4793 < 64UL; _fuseiter_4793 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 / 2UL) * 128UL) + (_fuseiter_4793 + ((fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4788___fuseiter_4789_1570___fuseiter_4790_1571___fuseiter_4791_1572___fuseiter_4792_1573 % 2UL) * 64UL) + _fuseiter_4793))]);
    }
  }
  return true;
}

static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 += 1UL) {
    for (uint64_t _fuseiter_4799 = 0UL; _fuseiter_4799 < 64UL; _fuseiter_4799 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 % 2UL) * 64UL)) + _fuseiter_4799)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1574____itr_2_1575____itr_3_1576____itr_4_1577 % 2UL) * 64UL)) + _fuseiter_4799)]);
    }
  }
  return true;
}

static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 += 1UL) {
    for (uint64_t _fuseiter_4806 = 0UL; _fuseiter_4806 < 64UL; _fuseiter_4806 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 / 2UL) * 128UL) + (_fuseiter_4806 + ((fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4801___fuseiter_4802_1578___fuseiter_4803_1579___fuseiter_4804_1580___fuseiter_4805_1581 % 2UL) * 64UL) + _fuseiter_4806))]);
    }
  }
  return true;
}

static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 += 1UL) {
    for (uint64_t _fuseiter_4812 = 0UL; _fuseiter_4812 < 64UL; _fuseiter_4812 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 % 2UL) * 64UL)) + _fuseiter_4812)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1582____itr_2_1583____itr_3_1584____itr_4_1585 % 2UL) * 64UL)) + _fuseiter_4812)]);
    }
  }
  return true;
}

static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 += 1UL) {
    for (uint64_t _fuseiter_4819 = 0UL; _fuseiter_4819 < 64UL; _fuseiter_4819 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 / 2UL) * 128UL) + (_fuseiter_4819 + ((fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4814___fuseiter_4815_1586___fuseiter_4816_1587___fuseiter_4817_1588___fuseiter_4818_1589 % 2UL) * 64UL) + _fuseiter_4819))]);
    }
  }
  return true;
}

static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 += 1UL) {
    for (uint64_t _fuseiter_4825 = 0UL; _fuseiter_4825 < 64UL; _fuseiter_4825 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 % 2UL) * 64UL)) + _fuseiter_4825)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1590____itr_2_1591____itr_3_1592____itr_4_1593 % 2UL) * 64UL)) + _fuseiter_4825)]);
    }
  }
  return true;
}

static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 += 1UL) {
    for (uint64_t _fuseiter_4832 = 0UL; _fuseiter_4832 < 64UL; _fuseiter_4832 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 / 2UL) * 128UL) + (_fuseiter_4832 + ((fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4827___fuseiter_4828_1594___fuseiter_4829_1595___fuseiter_4830_1596___fuseiter_4831_1597 % 2UL) * 64UL) + _fuseiter_4832))]);
    }
  }
  return true;
}

static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 += 1UL) {
    for (uint64_t _fuseiter_4838 = 0UL; _fuseiter_4838 < 64UL; _fuseiter_4838 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 % 2UL) * 64UL)) + _fuseiter_4838)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1598____itr_2_1599____itr_3_1600____itr_4_1601 % 2UL) * 64UL)) + _fuseiter_4838)]);
    }
  }
  return true;
}

static bool reorder__468(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 += 1UL) {
    for (uint64_t _fuseiter_4845 = 0UL; _fuseiter_4845 < 64UL; _fuseiter_4845 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 / 2UL) * 128UL) + (_fuseiter_4845 + ((fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4840___fuseiter_4841_1602___fuseiter_4842_1603___fuseiter_4843_1604___fuseiter_4844_1605 % 2UL) * 64UL) + _fuseiter_4845))]);
    }
  }
  return true;
}

static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 += 1UL) {
    for (uint64_t _fuseiter_4851 = 0UL; _fuseiter_4851 < 64UL; _fuseiter_4851 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 % 2UL) * 64UL)) + _fuseiter_4851)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1606____itr_2_1607____itr_3_1608____itr_4_1609 % 2UL) * 64UL)) + _fuseiter_4851)]);
    }
  }
  return true;
}

static bool reorder__471(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 += 1UL) {
    for (uint64_t _fuseiter_4858 = 0UL; _fuseiter_4858 < 64UL; _fuseiter_4858 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 / 2UL) * 128UL) + (_fuseiter_4858 + ((fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 / 2UL) * 128UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4853___fuseiter_4854_1610___fuseiter_4855_1611___fuseiter_4856_1612___fuseiter_4857_1613 % 2UL) * 64UL) + _fuseiter_4858))]);
    }
  }
  return true;
}

static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 += 1UL) {
    for (uint64_t _fuseiter_4864 = 0UL; _fuseiter_4864 < 64UL; _fuseiter_4864 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 % 2UL) * 64UL)) + _fuseiter_4864)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 / 2UL) * 128UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1614____itr_2_1615____itr_3_1616____itr_4_1617 % 2UL) * 64UL)) + _fuseiter_4864)]);
    }
  }
  return true;
}

static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 += 1UL) {
    for (uint64_t _fuseiter_4871 = 0UL; _fuseiter_4871 < 64UL; _fuseiter_4871 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 / 4UL) * 256UL) + (_fuseiter_4871 + ((fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4866___fuseiter_4867_1618___fuseiter_4868_1619___fuseiter_4869_1620___fuseiter_4870_1621 % 4UL) * 64UL) + _fuseiter_4871))]);
    }
  }
  return true;
}

static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 += 1UL) {
    for (uint64_t _fuseiter_4877 = 0UL; _fuseiter_4877 < 64UL; _fuseiter_4877 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 % 4UL) * 64UL)) + _fuseiter_4877)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1622____itr_2_1623____itr_3_1624____itr_4_1625 % 4UL) * 64UL)) + _fuseiter_4877)]);
    }
  }
  return true;
}

static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 += 1UL) {
    for (uint64_t _fuseiter_4884 = 0UL; _fuseiter_4884 < 64UL; _fuseiter_4884 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 / 4UL) * 256UL) + (_fuseiter_4884 + ((fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4879___fuseiter_4880_1626___fuseiter_4881_1627___fuseiter_4882_1628___fuseiter_4883_1629 % 4UL) * 64UL) + _fuseiter_4884))]);
    }
  }
  return true;
}

static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 += 1UL) {
    for (uint64_t _fuseiter_4890 = 0UL; _fuseiter_4890 < 64UL; _fuseiter_4890 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 % 4UL) * 64UL)) + _fuseiter_4890)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1630____itr_2_1631____itr_3_1632____itr_4_1633 % 4UL) * 64UL)) + _fuseiter_4890)]);
    }
  }
  return true;
}

static bool reorder__430(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 += 1UL) {
    for (uint64_t _fuseiter_4897 = 0UL; _fuseiter_4897 < 64UL; _fuseiter_4897 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 / 4UL) * 256UL) + (_fuseiter_4897 + ((fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4892___fuseiter_4893_1634___fuseiter_4894_1635___fuseiter_4895_1636___fuseiter_4896_1637 % 4UL) * 64UL) + _fuseiter_4897))]);
    }
  }
  return true;
}

static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 += 1UL) {
    for (uint64_t _fuseiter_4903 = 0UL; _fuseiter_4903 < 64UL; _fuseiter_4903 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 % 4UL) * 64UL)) + _fuseiter_4903)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1638____itr_2_1639____itr_3_1640____itr_4_1641 % 4UL) * 64UL)) + _fuseiter_4903)]);
    }
  }
  return true;
}

static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 += 1UL) {
    for (uint64_t _fuseiter_4910 = 0UL; _fuseiter_4910 < 64UL; _fuseiter_4910 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 / 4UL) * 256UL) + (_fuseiter_4910 + ((fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4905___fuseiter_4906_1642___fuseiter_4907_1643___fuseiter_4908_1644___fuseiter_4909_1645 % 4UL) * 64UL) + _fuseiter_4910))]);
    }
  }
  return true;
}

static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 += 1UL) {
    for (uint64_t _fuseiter_4916 = 0UL; _fuseiter_4916 < 64UL; _fuseiter_4916 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 % 4UL) * 64UL)) + _fuseiter_4916)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1646____itr_2_1647____itr_3_1648____itr_4_1649 % 4UL) * 64UL)) + _fuseiter_4916)]);
    }
  }
  return true;
}

static bool reorder__480(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 += 1UL) {
    for (uint64_t _fuseiter_4923 = 0UL; _fuseiter_4923 < 64UL; _fuseiter_4923 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 / 4UL) * 256UL) + (_fuseiter_4923 + ((fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4918___fuseiter_4919_1650___fuseiter_4920_1651___fuseiter_4921_1652___fuseiter_4922_1653 % 4UL) * 64UL) + _fuseiter_4923))]);
    }
  }
  return true;
}

static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 += 1UL) {
    for (uint64_t _fuseiter_4929 = 0UL; _fuseiter_4929 < 64UL; _fuseiter_4929 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 % 4UL) * 64UL)) + _fuseiter_4929)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1654____itr_2_1655____itr_3_1656____itr_4_1657 % 4UL) * 64UL)) + _fuseiter_4929)]);
    }
  }
  return true;
}

static bool reorder__483(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 += 1UL) {
    for (uint64_t _fuseiter_4936 = 0UL; _fuseiter_4936 < 64UL; _fuseiter_4936 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 / 4UL) * 256UL) + (_fuseiter_4936 + ((fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4931___fuseiter_4932_1658___fuseiter_4933_1659___fuseiter_4934_1660___fuseiter_4935_1661 % 4UL) * 64UL) + _fuseiter_4936))]);
    }
  }
  return true;
}

static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 += 1UL) {
    for (uint64_t _fuseiter_4942 = 0UL; _fuseiter_4942 < 64UL; _fuseiter_4942 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 % 4UL) * 64UL)) + _fuseiter_4942)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1662____itr_2_1663____itr_3_1664____itr_4_1665 % 4UL) * 64UL)) + _fuseiter_4942)]);
    }
  }
  return true;
}

static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 += 1UL) {
    for (uint64_t _fuseiter_4949 = 0UL; _fuseiter_4949 < 64UL; _fuseiter_4949 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 / 4UL) * 256UL) + (_fuseiter_4949 + ((fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4944___fuseiter_4945_1666___fuseiter_4946_1667___fuseiter_4947_1668___fuseiter_4948_1669 % 4UL) * 64UL) + _fuseiter_4949))]);
    }
  }
  return true;
}

static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 += 1UL) {
    for (uint64_t _fuseiter_4955 = 0UL; _fuseiter_4955 < 64UL; _fuseiter_4955 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 % 4UL) * 64UL)) + _fuseiter_4955)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1670____itr_2_1671____itr_3_1672____itr_4_1673 % 4UL) * 64UL)) + _fuseiter_4955)]);
    }
  }
  return true;
}

static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 += 1UL) {
    for (uint64_t _fuseiter_4962 = 0UL; _fuseiter_4962 < 64UL; _fuseiter_4962 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 / 4UL) * 256UL) + (_fuseiter_4962 + ((fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4957___fuseiter_4958_1674___fuseiter_4959_1675___fuseiter_4960_1676___fuseiter_4961_1677 % 4UL) * 64UL) + _fuseiter_4962))]);
    }
  }
  return true;
}

static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 += 1UL) {
    for (uint64_t _fuseiter_4968 = 0UL; _fuseiter_4968 < 64UL; _fuseiter_4968 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 % 4UL) * 64UL)) + _fuseiter_4968)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1678____itr_2_1679____itr_3_1680____itr_4_1681 % 4UL) * 64UL)) + _fuseiter_4968)]);
    }
  }
  return true;
}

static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 += 1UL) {
    for (uint64_t _fuseiter_4975 = 0UL; _fuseiter_4975 < 64UL; _fuseiter_4975 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 / 4UL) * 256UL) + (_fuseiter_4975 + ((fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4970___fuseiter_4971_1682___fuseiter_4972_1683___fuseiter_4973_1684___fuseiter_4974_1685 % 4UL) * 64UL) + _fuseiter_4975))]);
    }
  }
  return true;
}

static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 += 1UL) {
    for (uint64_t _fuseiter_4981 = 0UL; _fuseiter_4981 < 64UL; _fuseiter_4981 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 % 4UL) * 64UL)) + _fuseiter_4981)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1686____itr_2_1687____itr_3_1688____itr_4_1689 % 4UL) * 64UL)) + _fuseiter_4981)]);
    }
  }
  return true;
}

static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 += 1UL) {
    for (uint64_t _fuseiter_4988 = 0UL; _fuseiter_4988 < 64UL; _fuseiter_4988 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 / 4UL) * 256UL) + (_fuseiter_4988 + ((fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4983___fuseiter_4984_1690___fuseiter_4985_1691___fuseiter_4986_1692___fuseiter_4987_1693 % 4UL) * 64UL) + _fuseiter_4988))]);
    }
  }
  return true;
}

static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 += 1UL) {
    for (uint64_t _fuseiter_4994 = 0UL; _fuseiter_4994 < 64UL; _fuseiter_4994 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 % 4UL) * 64UL)) + _fuseiter_4994)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1694____itr_2_1695____itr_3_1696____itr_4_1697 % 4UL) * 64UL)) + _fuseiter_4994)]);
    }
  }
  return true;
}

static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 += 1UL) {
    for (uint64_t _fuseiter_5001 = 0UL; _fuseiter_5001 < 64UL; _fuseiter_5001 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 / 4UL) * 256UL) + (_fuseiter_5001 + ((fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_4996___fuseiter_4997_1698___fuseiter_4998_1699___fuseiter_4999_1700___fuseiter_5000_1701 % 4UL) * 64UL) + _fuseiter_5001))]);
    }
  }
  return true;
}

static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 += 1UL) {
    for (uint64_t _fuseiter_5007 = 0UL; _fuseiter_5007 < 64UL; _fuseiter_5007 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 % 4UL) * 64UL)) + _fuseiter_5007)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1702____itr_2_1703____itr_3_1704____itr_4_1705 % 4UL) * 64UL)) + _fuseiter_5007)]);
    }
  }
  return true;
}

static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 += 1UL) {
    for (uint64_t _fuseiter_5014 = 0UL; _fuseiter_5014 < 64UL; _fuseiter_5014 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 / 4UL) * 256UL) + (_fuseiter_5014 + ((fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5009___fuseiter_5010_1706___fuseiter_5011_1707___fuseiter_5012_1708___fuseiter_5013_1709 % 4UL) * 64UL) + _fuseiter_5014))]);
    }
  }
  return true;
}

static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 += 1UL) {
    for (uint64_t _fuseiter_5020 = 0UL; _fuseiter_5020 < 64UL; _fuseiter_5020 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 % 4UL) * 64UL)) + _fuseiter_5020)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1710____itr_2_1711____itr_3_1712____itr_4_1713 % 4UL) * 64UL)) + _fuseiter_5020)]);
    }
  }
  return true;
}

static bool reorder__516(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 += 1UL) {
    for (uint64_t _fuseiter_5027 = 0UL; _fuseiter_5027 < 64UL; _fuseiter_5027 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 / 4UL) * 256UL) + (_fuseiter_5027 + ((fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5022___fuseiter_5023_1714___fuseiter_5024_1715___fuseiter_5025_1716___fuseiter_5026_1717 % 4UL) * 64UL) + _fuseiter_5027))]);
    }
  }
  return true;
}

static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 += 1UL) {
    for (uint64_t _fuseiter_5033 = 0UL; _fuseiter_5033 < 64UL; _fuseiter_5033 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 % 4UL) * 64UL)) + _fuseiter_5033)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1718____itr_2_1719____itr_3_1720____itr_4_1721 % 4UL) * 64UL)) + _fuseiter_5033)]);
    }
  }
  return true;
}

static bool reorder__519(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 += 1UL) {
    for (uint64_t _fuseiter_5040 = 0UL; _fuseiter_5040 < 64UL; _fuseiter_5040 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 / 4UL) * 256UL) + (_fuseiter_5040 + ((fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5035___fuseiter_5036_1722___fuseiter_5037_1723___fuseiter_5038_1724___fuseiter_5039_1725 % 4UL) * 64UL) + _fuseiter_5040))]);
    }
  }
  return true;
}

static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 += 1UL) {
    for (uint64_t _fuseiter_5046 = 0UL; _fuseiter_5046 < 64UL; _fuseiter_5046 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 % 4UL) * 64UL)) + _fuseiter_5046)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1726____itr_2_1727____itr_3_1728____itr_4_1729 % 4UL) * 64UL)) + _fuseiter_5046)]);
    }
  }
  return true;
}

static bool reorder__525(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 += 1UL) {
    for (uint64_t _fuseiter_5053 = 0UL; _fuseiter_5053 < 64UL; _fuseiter_5053 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 / 4UL) * 256UL) + (_fuseiter_5053 + ((fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5048___fuseiter_5049_1730___fuseiter_5050_1731___fuseiter_5051_1732___fuseiter_5052_1733 % 4UL) * 64UL) + _fuseiter_5053))]);
    }
  }
  return true;
}

static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 += 1UL) {
    for (uint64_t _fuseiter_5059 = 0UL; _fuseiter_5059 < 64UL; _fuseiter_5059 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 % 4UL) * 64UL)) + _fuseiter_5059)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1734____itr_2_1735____itr_3_1736____itr_4_1737 % 4UL) * 64UL)) + _fuseiter_5059)]);
    }
  }
  return true;
}

static bool reorder__528(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 += 1UL) {
    for (uint64_t _fuseiter_5066 = 0UL; _fuseiter_5066 < 64UL; _fuseiter_5066 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 / 4UL) * 256UL) + (_fuseiter_5066 + ((fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 / 4UL) * 256UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5061___fuseiter_5062_1738___fuseiter_5063_1739___fuseiter_5064_1740___fuseiter_5065_1741 % 4UL) * 64UL) + _fuseiter_5066))]);
    }
  }
  return true;
}

static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 += 1UL) {
    for (uint64_t _fuseiter_5072 = 0UL; _fuseiter_5072 < 64UL; _fuseiter_5072 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 % 4UL) * 64UL)) + _fuseiter_5072)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 / 4UL) * 256UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1742____itr_2_1743____itr_3_1744____itr_4_1745 % 4UL) * 64UL)) + _fuseiter_5072)]);
    }
  }
  return true;
}

static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 += 1UL) {
    for (uint64_t _fuseiter_5079 = 0UL; _fuseiter_5079 < 64UL; _fuseiter_5079 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 / 8UL) * 512UL) + (_fuseiter_5079 + ((fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5074___fuseiter_5075_1746___fuseiter_5076_1747___fuseiter_5077_1748___fuseiter_5078_1749 % 8UL) * 64UL) + _fuseiter_5079))]);
    }
  }
  return true;
}

static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 += 1UL) {
    for (uint64_t _fuseiter_5085 = 0UL; _fuseiter_5085 < 64UL; _fuseiter_5085 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 % 8UL) * 64UL)) + _fuseiter_5085)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1750____itr_2_1751____itr_3_1752____itr_4_1753 % 8UL) * 64UL)) + _fuseiter_5085)]);
    }
  }
  return true;
}

static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 += 1UL) {
    for (uint64_t _fuseiter_5092 = 0UL; _fuseiter_5092 < 64UL; _fuseiter_5092 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 / 8UL) * 512UL) + (_fuseiter_5092 + ((fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5087___fuseiter_5088_1754___fuseiter_5089_1755___fuseiter_5090_1756___fuseiter_5091_1757 % 8UL) * 64UL) + _fuseiter_5092))]);
    }
  }
  return true;
}

static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 += 1UL) {
    for (uint64_t _fuseiter_5098 = 0UL; _fuseiter_5098 < 64UL; _fuseiter_5098 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 % 8UL) * 64UL)) + _fuseiter_5098)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1758____itr_2_1759____itr_3_1760____itr_4_1761 % 8UL) * 64UL)) + _fuseiter_5098)]);
    }
  }
  return true;
}

static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 += 1UL) {
    for (uint64_t _fuseiter_5105 = 0UL; _fuseiter_5105 < 64UL; _fuseiter_5105 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 / 8UL) * 512UL) + (_fuseiter_5105 + ((fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5100___fuseiter_5101_1762___fuseiter_5102_1763___fuseiter_5103_1764___fuseiter_5104_1765 % 8UL) * 64UL) + _fuseiter_5105))]);
    }
  }
  return true;
}

static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 += 1UL) {
    for (uint64_t _fuseiter_5111 = 0UL; _fuseiter_5111 < 64UL; _fuseiter_5111 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 % 8UL) * 64UL)) + _fuseiter_5111)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1766____itr_2_1767____itr_3_1768____itr_4_1769 % 8UL) * 64UL)) + _fuseiter_5111)]);
    }
  }
  return true;
}

static bool reorder__465(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 += 1UL) {
    for (uint64_t _fuseiter_5118 = 0UL; _fuseiter_5118 < 64UL; _fuseiter_5118 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 / 8UL) * 512UL) + (_fuseiter_5118 + ((fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5113___fuseiter_5114_1770___fuseiter_5115_1771___fuseiter_5116_1772___fuseiter_5117_1773 % 8UL) * 64UL) + _fuseiter_5118))]);
    }
  }
  return true;
}

static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 += 1UL) {
    for (uint64_t _fuseiter_5124 = 0UL; _fuseiter_5124 < 64UL; _fuseiter_5124 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 % 8UL) * 64UL)) + _fuseiter_5124)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1774____itr_2_1775____itr_3_1776____itr_4_1777 % 8UL) * 64UL)) + _fuseiter_5124)]);
    }
  }
  return true;
}

static bool reorder__474(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 += 1UL) {
    for (uint64_t _fuseiter_5131 = 0UL; _fuseiter_5131 < 64UL; _fuseiter_5131 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 / 8UL) * 512UL) + (_fuseiter_5131 + ((fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5126___fuseiter_5127_1778___fuseiter_5128_1779___fuseiter_5129_1780___fuseiter_5130_1781 % 8UL) * 64UL) + _fuseiter_5131))]);
    }
  }
  return true;
}

static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 += 1UL) {
    for (uint64_t _fuseiter_5137 = 0UL; _fuseiter_5137 < 64UL; _fuseiter_5137 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 % 8UL) * 64UL)) + _fuseiter_5137)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1782____itr_2_1783____itr_3_1784____itr_4_1785 % 8UL) * 64UL)) + _fuseiter_5137)]);
    }
  }
  return true;
}

static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_5144 = 0UL; _fuseiter_5144 < 512UL; _fuseiter_5144 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_5144]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_5144]);
  }
  return true;
}

static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 += 1UL) {
    for (uint64_t _fuseiter_5151 = 0UL; _fuseiter_5151 < 256UL; _fuseiter_5151 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 / 2UL) * 512UL) + (_fuseiter_5151 + ((fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 % 2UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5146___fuseiter_5147_1790___fuseiter_5148_1791___fuseiter_5149_1792___fuseiter_5150_1793 % 2UL) * 256UL) + _fuseiter_5151))]);
    }
  }
  return true;
}

static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 += 1UL) {
    for (uint64_t _fuseiter_5157 = 0UL; _fuseiter_5157 < 256UL; _fuseiter_5157 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 % 2UL) * 256UL)) + _fuseiter_5157)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1794____itr_2_1795____itr_3_1796____itr_4_1797 % 2UL) * 256UL)) + _fuseiter_5157)]);
    }
  }
  return true;
}

static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 += 1UL) {
    for (uint64_t _fuseiter_5164 = 0UL; _fuseiter_5164 < 64UL; _fuseiter_5164 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 / 8UL) * 512UL) + (_fuseiter_5164 + ((fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5159___fuseiter_5160_1798___fuseiter_5161_1799___fuseiter_5162_1800___fuseiter_5163_1801 % 8UL) * 64UL) + _fuseiter_5164))]);
    }
  }
  return true;
}

static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 += 1UL) {
    for (uint64_t _fuseiter_5170 = 0UL; _fuseiter_5170 < 64UL; _fuseiter_5170 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 % 8UL) * 64UL)) + _fuseiter_5170)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1802____itr_2_1803____itr_3_1804____itr_4_1805 % 8UL) * 64UL)) + _fuseiter_5170)]);
    }
  }
  return true;
}

static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 += 1UL) {
    for (uint64_t _fuseiter_5177 = 0UL; _fuseiter_5177 < 128UL; _fuseiter_5177 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 / 4UL) * 512UL) + (_fuseiter_5177 + ((fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 / 4UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5172___fuseiter_5173_1806___fuseiter_5174_1807___fuseiter_5175_1808___fuseiter_5176_1809 % 4UL) * 128UL) + _fuseiter_5177))]);
    }
  }
  return true;
}

static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 += 1UL) {
    for (uint64_t _fuseiter_5183 = 0UL; _fuseiter_5183 < 128UL; _fuseiter_5183 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 % 4UL) * 128UL)) + _fuseiter_5183)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 / 4UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1810____itr_2_1811____itr_3_1812____itr_4_1813 % 4UL) * 128UL)) + _fuseiter_5183)]);
    }
  }
  return true;
}

static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 < 2UL; fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 += 1UL) {
    for (uint64_t _fuseiter_5190 = 0UL; _fuseiter_5190 < 256UL; _fuseiter_5190 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 / 2UL) * 512UL) + (_fuseiter_5190 + ((fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 % 2UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 / 2UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5185___fuseiter_5186_1814___fuseiter_5187_1815___fuseiter_5188_1816___fuseiter_5189_1817 % 2UL) * 256UL) + _fuseiter_5190))]);
    }
  }
  return true;
}

static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 < 2UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 += 1UL) {
    for (uint64_t _fuseiter_5196 = 0UL; _fuseiter_5196 < 256UL; _fuseiter_5196 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 % 2UL) * 256UL)) + _fuseiter_5196)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 / 2UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1818____itr_2_1819____itr_3_1820____itr_4_1821 % 2UL) * 256UL)) + _fuseiter_5196)]);
    }
  }
  return true;
}

static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 < 8UL; fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 += 1UL) {
    for (uint64_t _fuseiter_5203 = 0UL; _fuseiter_5203 < 64UL; _fuseiter_5203 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 / 8UL) * 512UL) + (_fuseiter_5203 + ((fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 / 8UL) * 512UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5198___fuseiter_5199_1822___fuseiter_5200_1823___fuseiter_5201_1824___fuseiter_5202_1825 % 8UL) * 64UL) + _fuseiter_5203))]);
    }
  }
  return true;
}

static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 < 8UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 += 1UL) {
    for (uint64_t _fuseiter_5209 = 0UL; _fuseiter_5209 < 64UL; _fuseiter_5209 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 % 8UL) * 64UL)) + _fuseiter_5209)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 / 8UL) * 512UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1826____itr_2_1827____itr_3_1828____itr_4_1829 % 8UL) * 64UL)) + _fuseiter_5209)]);
    }
  }
  return true;
}

static bool reorder__477(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 = 0UL; fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 < 16UL; fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 += 1UL) {
    for (uint64_t _fuseiter_5216 = 0UL; _fuseiter_5216 < 64UL; _fuseiter_5216 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 / 16UL) * 1024UL) + (_fuseiter_5216 + ((fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5211___fuseiter_5212_1830___fuseiter_5213_1831 % 16UL) * 64UL) + _fuseiter_5216))]);
    }
  }
  return true;
}

static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 += 1UL) {
    for (uint64_t _fuseiter_5222 = 0UL; _fuseiter_5222 < 64UL; _fuseiter_5222 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 % 16UL) * 64UL)) + _fuseiter_5222)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1832____itr_2_1833____itr_3_1834____itr_4_1835 % 16UL) * 64UL)) + _fuseiter_5222)]);
    }
  }
  return true;
}

static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 = 0UL; fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 < 16UL; fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 += 1UL) {
    for (uint64_t _fuseiter_5229 = 0UL; _fuseiter_5229 < 64UL; _fuseiter_5229 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 / 16UL) * 1024UL) + (_fuseiter_5229 + ((fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5224___fuseiter_5225_1836___fuseiter_5226_1837 % 16UL) * 64UL) + _fuseiter_5229))]);
    }
  }
  return true;
}

static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 += 1UL) {
    for (uint64_t _fuseiter_5235 = 0UL; _fuseiter_5235 < 64UL; _fuseiter_5235 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 % 16UL) * 64UL)) + _fuseiter_5235)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1838____itr_2_1839____itr_3_1840____itr_4_1841 % 16UL) * 64UL)) + _fuseiter_5235)]);
    }
  }
  return true;
}

static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 = 0UL; fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 < 16UL; fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 += 1UL) {
    for (uint64_t _fuseiter_5242 = 0UL; _fuseiter_5242 < 64UL; _fuseiter_5242 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 / 16UL) * 1024UL) + (_fuseiter_5242 + ((fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5237___fuseiter_5238_1842___fuseiter_5239_1843 % 16UL) * 64UL) + _fuseiter_5242))]);
    }
  }
  return true;
}

static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 += 1UL) {
    for (uint64_t _fuseiter_5248 = 0UL; _fuseiter_5248 < 64UL; _fuseiter_5248 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 % 16UL) * 64UL)) + _fuseiter_5248)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1844____itr_2_1845____itr_3_1846____itr_4_1847 % 16UL) * 64UL)) + _fuseiter_5248)]);
    }
  }
  return true;
}

static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 = 0UL; fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 < 16UL; fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 += 1UL) {
    for (uint64_t _fuseiter_5255 = 0UL; _fuseiter_5255 < 64UL; _fuseiter_5255 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 / 16UL) * 1024UL) + (_fuseiter_5255 + ((fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5250___fuseiter_5251_1848___fuseiter_5252_1849 % 16UL) * 64UL) + _fuseiter_5255))]);
    }
  }
  return true;
}

static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 += 1UL) {
    for (uint64_t _fuseiter_5261 = 0UL; _fuseiter_5261 < 64UL; _fuseiter_5261 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 % 16UL) * 64UL)) + _fuseiter_5261)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1850____itr_2_1851____itr_3_1852____itr_4_1853 % 16UL) * 64UL)) + _fuseiter_5261)]);
    }
  }
  return true;
}

static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 = 0UL; fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 < 16UL; fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 += 1UL) {
    for (uint64_t _fuseiter_5268 = 0UL; _fuseiter_5268 < 64UL; _fuseiter_5268 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 / 16UL) * 1024UL) + (_fuseiter_5268 + ((fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5263___fuseiter_5264_1854___fuseiter_5265_1855 % 16UL) * 64UL) + _fuseiter_5268))]);
    }
  }
  return true;
}

static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 += 1UL) {
    for (uint64_t _fuseiter_5274 = 0UL; _fuseiter_5274 < 64UL; _fuseiter_5274 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 % 16UL) * 64UL)) + _fuseiter_5274)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1856____itr_2_1857____itr_3_1858____itr_4_1859 % 16UL) * 64UL)) + _fuseiter_5274)]);
    }
  }
  return true;
}

static bool reorder__522(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 = 0UL; fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 < 16UL; fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 += 1UL) {
    for (uint64_t _fuseiter_5281 = 0UL; _fuseiter_5281 < 64UL; _fuseiter_5281 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 / 16UL) * 1024UL) + (_fuseiter_5281 + ((fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5276___fuseiter_5277_1860___fuseiter_5278_1861 % 16UL) * 64UL) + _fuseiter_5281))]);
    }
  }
  return true;
}

static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 += 1UL) {
    for (uint64_t _fuseiter_5287 = 0UL; _fuseiter_5287 < 64UL; _fuseiter_5287 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 % 16UL) * 64UL)) + _fuseiter_5287)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1862____itr_2_1863____itr_3_1864____itr_4_1865 % 16UL) * 64UL)) + _fuseiter_5287)]);
    }
  }
  return true;
}

static bool reorder__531(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 = 0UL; fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 < 16UL; fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 += 1UL) {
    for (uint64_t _fuseiter_5294 = 0UL; _fuseiter_5294 < 64UL; _fuseiter_5294 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 / 16UL) * 1024UL) + (_fuseiter_5294 + ((fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 % 16UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 / 16UL) * 1024UL) + (((fused_0fused_0_fuseiter_5289___fuseiter_5290_1866___fuseiter_5291_1867 % 16UL) * 64UL) + _fuseiter_5294))]);
    }
  }
  return true;
}

static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 < 16UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 += 1UL) {
    for (uint64_t _fuseiter_5300 = 0UL; _fuseiter_5300 < 64UL; _fuseiter_5300 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 % 16UL) * 64UL)) + _fuseiter_5300)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 / 16UL) * 1024UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1868____itr_2_1869____itr_3_1870____itr_4_1871 % 16UL) * 64UL)) + _fuseiter_5300)]);
    }
  }
  return true;
}

static bool reorder__534(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 += 1UL) {
    for (uint64_t _fuseiter_5307 = 0UL; _fuseiter_5307 < 512UL; _fuseiter_5307 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 / 4UL) * 2048UL) + (_fuseiter_5307 + ((fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5302___fuseiter_5303_1872___fuseiter_5304_1873___fuseiter_5305_1874___fuseiter_5306_1875 % 4UL) * 512UL) + _fuseiter_5307))]);
    }
  }
  return true;
}

static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 += 1UL) {
    for (uint64_t _fuseiter_5313 = 0UL; _fuseiter_5313 < 512UL; _fuseiter_5313 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 % 4UL) * 512UL)) + _fuseiter_5313)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1876____itr_2_1877____itr_3_1878____itr_4_1879 % 4UL) * 512UL)) + _fuseiter_5313)]);
    }
  }
  return true;
}

static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 += 1UL) {
    for (uint64_t _fuseiter_5320 = 0UL; _fuseiter_5320 < 512UL; _fuseiter_5320 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 / 4UL) * 2048UL) + (_fuseiter_5320 + ((fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5315___fuseiter_5316_1880___fuseiter_5317_1881___fuseiter_5318_1882___fuseiter_5319_1883 % 4UL) * 512UL) + _fuseiter_5320))]);
    }
  }
  return true;
}

static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 += 1UL) {
    for (uint64_t _fuseiter_5326 = 0UL; _fuseiter_5326 < 512UL; _fuseiter_5326 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 % 4UL) * 512UL)) + _fuseiter_5326)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1884____itr_2_1885____itr_3_1886____itr_4_1887 % 4UL) * 512UL)) + _fuseiter_5326)]);
    }
  }
  return true;
}

static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 += 1UL) {
    for (uint64_t _fuseiter_5333 = 0UL; _fuseiter_5333 < 512UL; _fuseiter_5333 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 / 4UL) * 2048UL) + (_fuseiter_5333 + ((fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5328___fuseiter_5329_1888___fuseiter_5330_1889___fuseiter_5331_1890___fuseiter_5332_1891 % 4UL) * 512UL) + _fuseiter_5333))]);
    }
  }
  return true;
}

static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 += 1UL) {
    for (uint64_t _fuseiter_5339 = 0UL; _fuseiter_5339 < 512UL; _fuseiter_5339 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 % 4UL) * 512UL)) + _fuseiter_5339)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1892____itr_2_1893____itr_3_1894____itr_4_1895 % 4UL) * 512UL)) + _fuseiter_5339)]);
    }
  }
  return true;
}

static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 < 4UL; fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 += 1UL) {
    for (uint64_t _fuseiter_5346 = 0UL; _fuseiter_5346 < 512UL; _fuseiter_5346 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 / 4UL) * 2048UL) + (_fuseiter_5346 + ((fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 / 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5341___fuseiter_5342_1896___fuseiter_5343_1897___fuseiter_5344_1898___fuseiter_5345_1899 % 4UL) * 512UL) + _fuseiter_5346))]);
    }
  }
  return true;
}

static bool mul__673(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 = 0UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 < 4UL; fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 += 1UL) {
    for (uint64_t _fuseiter_5352 = 0UL; _fuseiter_5352 < 512UL; _fuseiter_5352 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 % 4UL) * 512UL)) + _fuseiter_5352)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 / 4UL) * 2048UL) + ((fused_0fused_0fused_0fused_0__itr_0____itr_1_1900____itr_2_1901____itr_3_1902____itr_4_1903 % 4UL) * 512UL)) + _fuseiter_5352)]);
    }
  }
  return true;
}

static bool mul__110(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 = 0UL; fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 < 4096UL; fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1904____itr_2_1905 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__111(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 = 0UL; fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 < 4096UL; fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1906____itr_2_1907 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__421(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 < 16UL; fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 += 1UL) {
    for (uint64_t _fuseiter_5369 = 0UL; _fuseiter_5369 < 64UL; _fuseiter_5369 += 1UL) {
      for (uint64_t _fuseiter_5370 = 0UL; _fuseiter_5370 < 4UL; _fuseiter_5370 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5369 + ((fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 / 16UL) * 64UL)) * 64UL) + (_fuseiter_5370 + ((fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5364___fuseiter_5365_1908___fuseiter_5366_1909___fuseiter_5367_1910___fuseiter_5368_1911 % 16UL) * 256UL) + ((_fuseiter_5369 * 4UL) + _fuseiter_5370)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__107(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 = 0UL; fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 < 16384UL; fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1912____itr_2_1913 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__108(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 = 0UL; fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 < 16384UL; fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1914____itr_2_1915 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__418(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 += 1UL) {
    for (uint64_t _fuseiter_5386 = 0UL; _fuseiter_5386 < 64UL; _fuseiter_5386 += 1UL) {
      for (uint64_t _fuseiter_5387 = 0UL; _fuseiter_5387 < 4UL; _fuseiter_5387 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5386 + ((fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 / 16UL) * 64UL)) * 64UL) + (_fuseiter_5387 + ((fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5381___fuseiter_5382_1916___fuseiter_5383_1917___fuseiter_5384_1918___fuseiter_5385_1919 % 16UL) * 256UL) + ((_fuseiter_5386 * 4UL) + _fuseiter_5387)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__116(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 = 0UL; fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 < 16384UL; fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1920____itr_2_1921 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__117(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 = 0UL; fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 < 16384UL; fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1922____itr_2_1923 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 += 1UL) {
    for (uint64_t _fuseiter_5403 = 0UL; _fuseiter_5403 < 64UL; _fuseiter_5403 += 1UL) {
      for (uint64_t _fuseiter_5404 = 0UL; _fuseiter_5404 < 4UL; _fuseiter_5404 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5403 + ((fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 / 16UL) * 64UL)) * 64UL) + (_fuseiter_5404 + ((fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5398___fuseiter_5399_1924___fuseiter_5400_1925___fuseiter_5401_1926___fuseiter_5402_1927 % 16UL) * 256UL) + ((_fuseiter_5403 * 4UL) + _fuseiter_5404)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__125(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 = 0UL; fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 < 16384UL; fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1928____itr_2_1929 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__126(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 = 0UL; fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 < 16384UL; fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1930____itr_2_1931 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__428(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 += 1UL) {
    for (uint64_t _fuseiter_5420 = 0UL; _fuseiter_5420 < 64UL; _fuseiter_5420 += 1UL) {
      for (uint64_t _fuseiter_5421 = 0UL; _fuseiter_5421 < 4UL; _fuseiter_5421 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5420 + ((fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 / 16UL) * 64UL)) * 64UL) + (_fuseiter_5421 + ((fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5415___fuseiter_5416_1932___fuseiter_5417_1933___fuseiter_5418_1934___fuseiter_5419_1935 % 16UL) * 256UL) + ((_fuseiter_5420 * 4UL) + _fuseiter_5421)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__134(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 = 0UL; fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 < 16384UL; fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1936____itr_2_1937 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__135(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 = 0UL; fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 < 16384UL; fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_1938____itr_2_1939 % 64UL))] = __cached_1;
  }
  return true;
}

static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 += 1UL) {
    for (uint64_t _fuseiter_5437 = 0UL; _fuseiter_5437 < 64UL; _fuseiter_5437 += 1UL) {
      for (uint64_t _fuseiter_5438 = 0UL; _fuseiter_5438 < 4UL; _fuseiter_5438 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5437 + ((fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 / 16UL) * 64UL)) * 64UL) + (_fuseiter_5438 + ((fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5432___fuseiter_5433_1940___fuseiter_5434_1941___fuseiter_5435_1942___fuseiter_5436_1943 % 16UL) * 256UL) + ((_fuseiter_5437 * 4UL) + _fuseiter_5438)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__119(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 = 0UL; fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 < 16384UL; fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1944____itr_2_1945 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__120(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 = 0UL; fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 < 16384UL; fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1946____itr_2_1947 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 += 1UL) {
    for (uint64_t _fuseiter_5454 = 0UL; _fuseiter_5454 < 64UL; _fuseiter_5454 += 1UL) {
      for (uint64_t _fuseiter_5455 = 0UL; _fuseiter_5455 < 4UL; _fuseiter_5455 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5454 + ((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_5455 + ((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5449___fuseiter_5450_1948___fuseiter_5451_1949___fuseiter_5452_1950___fuseiter_5453_1951 % 16UL) * 256UL) + ((_fuseiter_5454 * 4UL) + _fuseiter_5455))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__128(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 = 0UL; fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 < 16384UL; fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1952____itr_2_1953 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__129(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 = 0UL; fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 < 16384UL; fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1954____itr_2_1955 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__431(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 += 1UL) {
    for (uint64_t _fuseiter_5471 = 0UL; _fuseiter_5471 < 64UL; _fuseiter_5471 += 1UL) {
      for (uint64_t _fuseiter_5472 = 0UL; _fuseiter_5472 < 4UL; _fuseiter_5472 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5471 + ((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_5472 + ((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5466___fuseiter_5467_1956___fuseiter_5468_1957___fuseiter_5469_1958___fuseiter_5470_1959 % 16UL) * 256UL) + ((_fuseiter_5471 * 4UL) + _fuseiter_5472))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__140(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 = 0UL; fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 < 32768UL; fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1960____itr_2_1961 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__141(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 = 0UL; fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 < 32768UL; fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_1962____itr_2_1963 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 += 1UL) {
    for (uint64_t _fuseiter_5488 = 0UL; _fuseiter_5488 < 64UL; _fuseiter_5488 += 1UL) {
      for (uint64_t _fuseiter_5489 = 0UL; _fuseiter_5489 < 4UL; _fuseiter_5489 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5488 + ((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_5489 + ((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 / 16UL) % 4UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5483___fuseiter_5484_1964___fuseiter_5485_1965___fuseiter_5486_1966___fuseiter_5487_1967 % 16UL) * 256UL) + ((_fuseiter_5488 * 4UL) + _fuseiter_5489))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__113(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 = 0UL; fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 < 12288UL; fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 += 1UL) {
    for (uint64_t _fuseiter_5494 = 0UL; _fuseiter_5494 < 3UL; _fuseiter_5494 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 % 3UL) * 3UL))) + _fuseiter_5494)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1968____itr_2_1969 % 3UL) * 3UL))) + _fuseiter_5494)] = __cached_2;
    }
  }
  return true;
}

static bool cast__114(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 = 0UL; fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 < 12288UL; fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 += 1UL) {
    for (uint64_t _fuseiter5499 = 0UL; _fuseiter5499 < 3UL; _fuseiter5499 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 % 3UL) * 3UL))) + _fuseiter5499)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1970____itr_2_1971 % 3UL) * 3UL))) + _fuseiter5499)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 += 1UL) {
    for (uint64_t _fuseiter_5505 = 0UL; _fuseiter_5505 < 64UL; _fuseiter_5505 += 1UL) {
      for (uint64_t _fuseiter_5506 = 0UL; _fuseiter_5506 < 4UL; _fuseiter_5506 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5505 + ((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_5506 + ((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5500___fuseiter_5501_1972___fuseiter_5502_1973___fuseiter_5503_1974___fuseiter_5504_1975 % 16UL) * 256UL) + ((_fuseiter_5505 * 4UL) + _fuseiter_5506)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__122(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 = 0UL; fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 < 12288UL; fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 += 1UL) {
    for (uint64_t _fuseiter_5511 = 0UL; _fuseiter_5511 < 3UL; _fuseiter_5511 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 % 3UL) * 3UL))) + _fuseiter_5511)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1976____itr_2_1977 % 3UL) * 3UL))) + _fuseiter_5511)] = __cached_2;
    }
  }
  return true;
}

static bool cast__123(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 = 0UL; fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 < 12288UL; fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 += 1UL) {
    for (uint64_t _fuseiter5516 = 0UL; _fuseiter5516 < 3UL; _fuseiter5516 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 % 3UL) * 3UL))) + _fuseiter5516)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1978____itr_2_1979 % 3UL) * 3UL))) + _fuseiter5516)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__427(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 += 1UL) {
    for (uint64_t _fuseiter_5522 = 0UL; _fuseiter_5522 < 64UL; _fuseiter_5522 += 1UL) {
      for (uint64_t _fuseiter_5523 = 0UL; _fuseiter_5523 < 4UL; _fuseiter_5523 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5522 + ((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_5523 + ((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5517___fuseiter_5518_1980___fuseiter_5519_1981___fuseiter_5520_1982___fuseiter_5521_1983 % 16UL) * 256UL) + ((_fuseiter_5522 * 4UL) + _fuseiter_5523)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__131(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 = 0UL; fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 < 12288UL; fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 += 1UL) {
    for (uint64_t _fuseiter_5528 = 0UL; _fuseiter_5528 < 3UL; _fuseiter_5528 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 % 3UL) * 3UL))) + _fuseiter_5528)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1984____itr_2_1985 % 3UL) * 3UL))) + _fuseiter_5528)] = __cached_2;
    }
  }
  return true;
}

static bool cast__132(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 = 0UL; fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 < 12288UL; fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 += 1UL) {
    for (uint64_t _fuseiter5533 = 0UL; _fuseiter5533 < 3UL; _fuseiter5533 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 % 3UL) * 3UL))) + _fuseiter5533)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_1986____itr_2_1987 % 3UL) * 3UL))) + _fuseiter5533)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__432(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 < 144UL; fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 += 1UL) {
    for (uint64_t _fuseiter_5539 = 0UL; _fuseiter_5539 < 64UL; _fuseiter_5539 += 1UL) {
      for (uint64_t _fuseiter_5540 = 0UL; _fuseiter_5540 < 4UL; _fuseiter_5540 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_5539 + ((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 144UL) * 64UL)) * 576UL) + (((_fuseiter_5540 + ((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 % 16UL) * 4UL)) * 9UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 48UL) % 3UL) * 3UL) + ((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 16UL) % 3UL))))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 144UL) * 36864UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 48UL) % 3UL) * 12288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 / 16UL) % 3UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_5534___fuseiter_5535_1988___fuseiter_5536_1989___fuseiter_5537_1990___fuseiter_5538_1991 % 16UL) * 256UL) + ((_fuseiter_5539 * 4UL) + _fuseiter_5540)))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__146(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 = 0UL; fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 < 65536UL; fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1992____itr_2_1993 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__147(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 = 0UL; fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 < 65536UL; fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1994____itr_2_1995 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5551___fuseiter_5552_1996 = 0UL; fused_0_fuseiter_5551___fuseiter_5552_1996 < 16UL; fused_0_fuseiter_5551___fuseiter_5552_1996 += 1UL) {
    for (uint64_t _fuseiter_5555 = 0UL; _fuseiter_5555 < 16UL; _fuseiter_5555 += 1UL) {
      for (uint64_t _fuseiter_5556 = 0UL; _fuseiter_5556 < 64UL; _fuseiter_5556 += 1UL) {
        for (uint64_t _fuseiter_5557 = 0UL; _fuseiter_5557 < 4UL; _fuseiter_5557 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5556 + ((fused_0_fuseiter_5551___fuseiter_5552_1996 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_5557 + (_fuseiter_5555 * 4UL)) + ((fused_0_fuseiter_5551___fuseiter_5552_1996 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5551___fuseiter_5552_1996 / 2UL) * 8192UL) + (((fused_0_fuseiter_5551___fuseiter_5552_1996 % 2UL) * 4096UL) + ((_fuseiter_5555 * 256UL) + ((_fuseiter_5556 * 4UL) + _fuseiter_5557))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__155(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 = 0UL; fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 < 65536UL; fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1997____itr_2_1998 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__156(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 = 0UL; fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 < 65536UL; fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_1999____itr_2_2000 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5568___fuseiter_5569_2001 = 0UL; fused_0_fuseiter_5568___fuseiter_5569_2001 < 16UL; fused_0_fuseiter_5568___fuseiter_5569_2001 += 1UL) {
    for (uint64_t _fuseiter_5572 = 0UL; _fuseiter_5572 < 16UL; _fuseiter_5572 += 1UL) {
      for (uint64_t _fuseiter_5573 = 0UL; _fuseiter_5573 < 64UL; _fuseiter_5573 += 1UL) {
        for (uint64_t _fuseiter_5574 = 0UL; _fuseiter_5574 < 4UL; _fuseiter_5574 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5573 + ((fused_0_fuseiter_5568___fuseiter_5569_2001 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_5574 + (_fuseiter_5572 * 4UL)) + ((fused_0_fuseiter_5568___fuseiter_5569_2001 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5568___fuseiter_5569_2001 / 2UL) * 8192UL) + (((fused_0_fuseiter_5568___fuseiter_5569_2001 % 2UL) * 4096UL) + ((_fuseiter_5572 * 256UL) + ((_fuseiter_5573 * 4UL) + _fuseiter_5574))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__164(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 = 0UL; fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 < 65536UL; fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2002____itr_2_2003 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__165(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 = 0UL; fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 < 65536UL; fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2004____itr_2_2005 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__463(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5585___fuseiter_5586_2006 = 0UL; fused_0_fuseiter_5585___fuseiter_5586_2006 < 16UL; fused_0_fuseiter_5585___fuseiter_5586_2006 += 1UL) {
    for (uint64_t _fuseiter_5589 = 0UL; _fuseiter_5589 < 16UL; _fuseiter_5589 += 1UL) {
      for (uint64_t _fuseiter_5590 = 0UL; _fuseiter_5590 < 64UL; _fuseiter_5590 += 1UL) {
        for (uint64_t _fuseiter_5591 = 0UL; _fuseiter_5591 < 4UL; _fuseiter_5591 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5590 + ((fused_0_fuseiter_5585___fuseiter_5586_2006 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_5591 + (_fuseiter_5589 * 4UL)) + ((fused_0_fuseiter_5585___fuseiter_5586_2006 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5585___fuseiter_5586_2006 / 2UL) * 8192UL) + (((fused_0_fuseiter_5585___fuseiter_5586_2006 % 2UL) * 4096UL) + ((_fuseiter_5589 * 256UL) + ((_fuseiter_5590 * 4UL) + _fuseiter_5591))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__173(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 = 0UL; fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 < 65536UL; fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2007____itr_2_2008 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__174(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 = 0UL; fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 < 65536UL; fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_2009____itr_2_2010 % 128UL))] = __cached_1;
  }
  return true;
}

static bool reorder__472(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5602___fuseiter_5603_2011 = 0UL; fused_0_fuseiter_5602___fuseiter_5603_2011 < 16UL; fused_0_fuseiter_5602___fuseiter_5603_2011 += 1UL) {
    for (uint64_t _fuseiter_5606 = 0UL; _fuseiter_5606 < 16UL; _fuseiter_5606 += 1UL) {
      for (uint64_t _fuseiter_5607 = 0UL; _fuseiter_5607 < 64UL; _fuseiter_5607 += 1UL) {
        for (uint64_t _fuseiter_5608 = 0UL; _fuseiter_5608 < 4UL; _fuseiter_5608 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5607 + ((fused_0_fuseiter_5602___fuseiter_5603_2011 / 2UL) * 64UL)) * 128UL) + ((_fuseiter_5608 + (_fuseiter_5606 * 4UL)) + ((fused_0_fuseiter_5602___fuseiter_5603_2011 % 2UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5602___fuseiter_5603_2011 / 2UL) * 8192UL) + (((fused_0_fuseiter_5602___fuseiter_5603_2011 % 2UL) * 4096UL) + ((_fuseiter_5606 * 256UL) + ((_fuseiter_5607 * 4UL) + _fuseiter_5608))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__149(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 = 0UL; fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 < 65536UL; fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2012____itr_2_2013 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__150(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 = 0UL; fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 < 65536UL; fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2014____itr_2_2015 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5619___fuseiter_5620_2016 = 0UL; fused_0_fuseiter_5619___fuseiter_5620_2016 < 16UL; fused_0_fuseiter_5619___fuseiter_5620_2016 += 1UL) {
    for (uint64_t _fuseiter_5623 = 0UL; _fuseiter_5623 < 16UL; _fuseiter_5623 += 1UL) {
      for (uint64_t _fuseiter_5624 = 0UL; _fuseiter_5624 < 64UL; _fuseiter_5624 += 1UL) {
        for (uint64_t _fuseiter_5625 = 0UL; _fuseiter_5625 < 4UL; _fuseiter_5625 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5624 + ((fused_0_fuseiter_5619___fuseiter_5620_2016 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_5625 + (_fuseiter_5623 * 4UL)) + ((fused_0_fuseiter_5619___fuseiter_5620_2016 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5619___fuseiter_5620_2016 / 8UL) * 32768UL) + (((fused_0_fuseiter_5619___fuseiter_5620_2016 % 8UL) * 4096UL) + ((_fuseiter_5623 * 256UL) + ((_fuseiter_5624 * 4UL) + _fuseiter_5625))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__158(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 = 0UL; fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 < 65536UL; fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2017____itr_2_2018 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__159(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 = 0UL; fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 < 65536UL; fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2019____itr_2_2020 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5636___fuseiter_5637_2021 = 0UL; fused_0_fuseiter_5636___fuseiter_5637_2021 < 16UL; fused_0_fuseiter_5636___fuseiter_5637_2021 += 1UL) {
    for (uint64_t _fuseiter_5640 = 0UL; _fuseiter_5640 < 16UL; _fuseiter_5640 += 1UL) {
      for (uint64_t _fuseiter_5641 = 0UL; _fuseiter_5641 < 64UL; _fuseiter_5641 += 1UL) {
        for (uint64_t _fuseiter_5642 = 0UL; _fuseiter_5642 < 4UL; _fuseiter_5642 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5641 + ((fused_0_fuseiter_5636___fuseiter_5637_2021 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_5642 + (_fuseiter_5640 * 4UL)) + ((fused_0_fuseiter_5636___fuseiter_5637_2021 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5636___fuseiter_5637_2021 / 8UL) * 32768UL) + (((fused_0_fuseiter_5636___fuseiter_5637_2021 % 8UL) * 4096UL) + ((_fuseiter_5640 * 256UL) + ((_fuseiter_5641 * 4UL) + _fuseiter_5642))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__167(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 = 0UL; fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 < 65536UL; fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2022____itr_2_2023 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__168(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 = 0UL; fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 < 65536UL; fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2024____itr_2_2025 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__466(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5653___fuseiter_5654_2026 = 0UL; fused_0_fuseiter_5653___fuseiter_5654_2026 < 16UL; fused_0_fuseiter_5653___fuseiter_5654_2026 += 1UL) {
    for (uint64_t _fuseiter_5657 = 0UL; _fuseiter_5657 < 16UL; _fuseiter_5657 += 1UL) {
      for (uint64_t _fuseiter_5658 = 0UL; _fuseiter_5658 < 64UL; _fuseiter_5658 += 1UL) {
        for (uint64_t _fuseiter_5659 = 0UL; _fuseiter_5659 < 4UL; _fuseiter_5659 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5658 + ((fused_0_fuseiter_5653___fuseiter_5654_2026 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_5659 + (_fuseiter_5657 * 4UL)) + ((fused_0_fuseiter_5653___fuseiter_5654_2026 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5653___fuseiter_5654_2026 / 8UL) * 32768UL) + (((fused_0_fuseiter_5653___fuseiter_5654_2026 % 8UL) * 4096UL) + ((_fuseiter_5657 * 256UL) + ((_fuseiter_5658 * 4UL) + _fuseiter_5659))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__137(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 = 0UL; fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 < 131072UL; fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2027____itr_2_2028 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__138(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 = 0UL; fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 < 131072UL; fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2029____itr_2_2030 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5670___fuseiter_5671_2031 = 0UL; fused_0_fuseiter_5670___fuseiter_5671_2031 < 32UL; fused_0_fuseiter_5670___fuseiter_5671_2031 += 1UL) {
    for (uint64_t _fuseiter_5674 = 0UL; _fuseiter_5674 < 16UL; _fuseiter_5674 += 1UL) {
      for (uint64_t _fuseiter_5675 = 0UL; _fuseiter_5675 < 64UL; _fuseiter_5675 += 1UL) {
        for (uint64_t _fuseiter_5676 = 0UL; _fuseiter_5676 < 4UL; _fuseiter_5676 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5675 + ((fused_0_fuseiter_5670___fuseiter_5671_2031 / 4UL) * 64UL)) * 256UL) + ((_fuseiter_5676 + (_fuseiter_5674 * 4UL)) + ((fused_0_fuseiter_5670___fuseiter_5671_2031 % 4UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5670___fuseiter_5671_2031 / 4UL) * 16384UL) + (((fused_0_fuseiter_5670___fuseiter_5671_2031 % 4UL) * 4096UL) + ((_fuseiter_5674 * 256UL) + ((_fuseiter_5675 * 4UL) + _fuseiter_5676))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__179(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 = 0UL; fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 < 131072UL; fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2032____itr_2_2033 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__180(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 = 0UL; fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 < 131072UL; fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2034____itr_2_2035 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__478(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5687___fuseiter_5688_2036 = 0UL; fused_0_fuseiter_5687___fuseiter_5688_2036 < 32UL; fused_0_fuseiter_5687___fuseiter_5688_2036 += 1UL) {
    for (uint64_t _fuseiter_5691 = 0UL; _fuseiter_5691 < 16UL; _fuseiter_5691 += 1UL) {
      for (uint64_t _fuseiter_5692 = 0UL; _fuseiter_5692 < 64UL; _fuseiter_5692 += 1UL) {
        for (uint64_t _fuseiter_5693 = 0UL; _fuseiter_5693 < 4UL; _fuseiter_5693 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5692 + ((fused_0_fuseiter_5687___fuseiter_5688_2036 / 8UL) * 64UL)) * 512UL) + ((_fuseiter_5693 + (_fuseiter_5691 * 4UL)) + ((fused_0_fuseiter_5687___fuseiter_5688_2036 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5687___fuseiter_5688_2036 / 8UL) * 32768UL) + (((fused_0_fuseiter_5687___fuseiter_5688_2036 % 8UL) * 4096UL) + ((_fuseiter_5691 * 256UL) + ((_fuseiter_5692 * 4UL) + _fuseiter_5693))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__143(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 = 0UL; fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 < 49152UL; fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 += 1UL) {
    for (uint64_t _fuseiter_5698 = 0UL; _fuseiter_5698 < 3UL; _fuseiter_5698 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 % 3UL) * 3UL))) + _fuseiter_5698)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2037____itr_2_2038 % 3UL) * 3UL))) + _fuseiter_5698)] = __cached_2;
    }
  }
  return true;
}

static bool cast__144(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 = 0UL; fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 < 49152UL; fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 += 1UL) {
    for (uint64_t _fuseiter5703 = 0UL; _fuseiter5703 < 3UL; _fuseiter5703 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 % 3UL) * 3UL))) + _fuseiter5703)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2039____itr_2_2040 % 3UL) * 3UL))) + _fuseiter5703)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 = 0UL; fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 < 12UL; fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 += 1UL) {
    for (uint64_t _fuseiter_5707 = 0UL; _fuseiter_5707 < 3UL; _fuseiter_5707 += 1UL) {
      for (uint64_t _fuseiter_5708 = 0UL; _fuseiter_5708 < 16UL; _fuseiter_5708 += 1UL) {
        for (uint64_t _fuseiter_5709 = 0UL; _fuseiter_5709 < 64UL; _fuseiter_5709 += 1UL) {
          for (uint64_t _fuseiter_5710 = 0UL; _fuseiter_5710 < 4UL; _fuseiter_5710 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5709 + ((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_5710 + (_fuseiter_5708 * 4UL)) + (((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 % 3UL) * 3UL) + _fuseiter_5707)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_5704___fuseiter_5705_2041___fuseiter_5706_2042 % 3UL) * 12288UL) + ((_fuseiter_5707 * 4096UL) + ((_fuseiter_5708 * 256UL) + ((_fuseiter_5709 * 4UL) + _fuseiter_5710))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__152(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 = 0UL; fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 < 49152UL; fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 += 1UL) {
    for (uint64_t _fuseiter_5715 = 0UL; _fuseiter_5715 < 3UL; _fuseiter_5715 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 % 3UL) * 3UL))) + _fuseiter_5715)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2043____itr_2_2044 % 3UL) * 3UL))) + _fuseiter_5715)] = __cached_2;
    }
  }
  return true;
}

static bool cast__153(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 = 0UL; fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 < 49152UL; fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 += 1UL) {
    for (uint64_t _fuseiter5720 = 0UL; _fuseiter5720 < 3UL; _fuseiter5720 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 % 3UL) * 3UL))) + _fuseiter5720)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2045____itr_2_2046 % 3UL) * 3UL))) + _fuseiter5720)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 = 0UL; fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 < 12UL; fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 += 1UL) {
    for (uint64_t _fuseiter_5724 = 0UL; _fuseiter_5724 < 3UL; _fuseiter_5724 += 1UL) {
      for (uint64_t _fuseiter_5725 = 0UL; _fuseiter_5725 < 16UL; _fuseiter_5725 += 1UL) {
        for (uint64_t _fuseiter_5726 = 0UL; _fuseiter_5726 < 64UL; _fuseiter_5726 += 1UL) {
          for (uint64_t _fuseiter_5727 = 0UL; _fuseiter_5727 < 4UL; _fuseiter_5727 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5726 + ((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_5727 + (_fuseiter_5725 * 4UL)) + (((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 % 3UL) * 3UL) + _fuseiter_5724)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_5721___fuseiter_5722_2047___fuseiter_5723_2048 % 3UL) * 12288UL) + ((_fuseiter_5724 * 4096UL) + ((_fuseiter_5725 * 256UL) + ((_fuseiter_5726 * 4UL) + _fuseiter_5727))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__161(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 = 0UL; fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 < 49152UL; fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 += 1UL) {
    for (uint64_t _fuseiter_5732 = 0UL; _fuseiter_5732 < 3UL; _fuseiter_5732 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 % 3UL) * 3UL))) + _fuseiter_5732)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2049____itr_2_2050 % 3UL) * 3UL))) + _fuseiter_5732)] = __cached_2;
    }
  }
  return true;
}

static bool cast__162(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 = 0UL; fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 < 49152UL; fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 += 1UL) {
    for (uint64_t _fuseiter5737 = 0UL; _fuseiter5737 < 3UL; _fuseiter5737 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 % 3UL) * 3UL))) + _fuseiter5737)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2051____itr_2_2052 % 3UL) * 3UL))) + _fuseiter5737)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 = 0UL; fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 < 12UL; fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 += 1UL) {
    for (uint64_t _fuseiter_5741 = 0UL; _fuseiter_5741 < 3UL; _fuseiter_5741 += 1UL) {
      for (uint64_t _fuseiter_5742 = 0UL; _fuseiter_5742 < 16UL; _fuseiter_5742 += 1UL) {
        for (uint64_t _fuseiter_5743 = 0UL; _fuseiter_5743 < 64UL; _fuseiter_5743 += 1UL) {
          for (uint64_t _fuseiter_5744 = 0UL; _fuseiter_5744 < 4UL; _fuseiter_5744 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5743 + ((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_5744 + (_fuseiter_5742 * 4UL)) + (((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 % 3UL) * 3UL) + _fuseiter_5741)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_5738___fuseiter_5739_2053___fuseiter_5740_2054 % 3UL) * 12288UL) + ((_fuseiter_5741 * 4096UL) + ((_fuseiter_5742 * 256UL) + ((_fuseiter_5743 * 4UL) + _fuseiter_5744))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__170(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 = 0UL; fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 < 49152UL; fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 += 1UL) {
    for (uint64_t _fuseiter_5749 = 0UL; _fuseiter_5749 < 3UL; _fuseiter_5749 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 % 3UL) * 3UL))) + _fuseiter_5749)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2055____itr_2_2056 % 3UL) * 3UL))) + _fuseiter_5749)] = __cached_2;
    }
  }
  return true;
}

static bool cast__171(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 = 0UL; fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 < 49152UL; fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 += 1UL) {
    for (uint64_t _fuseiter5754 = 0UL; _fuseiter5754 < 3UL; _fuseiter5754 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 % 3UL) * 3UL))) + _fuseiter5754)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2057____itr_2_2058 % 3UL) * 3UL))) + _fuseiter5754)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__469(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 = 0UL; fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 < 12UL; fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 += 1UL) {
    for (uint64_t _fuseiter_5758 = 0UL; _fuseiter_5758 < 3UL; _fuseiter_5758 += 1UL) {
      for (uint64_t _fuseiter_5759 = 0UL; _fuseiter_5759 < 16UL; _fuseiter_5759 += 1UL) {
        for (uint64_t _fuseiter_5760 = 0UL; _fuseiter_5760 < 64UL; _fuseiter_5760 += 1UL) {
          for (uint64_t _fuseiter_5761 = 0UL; _fuseiter_5761 < 4UL; _fuseiter_5761 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5760 + ((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 / 6UL) * 64UL)) * 1152UL) + ((((_fuseiter_5761 + (_fuseiter_5759 * 4UL)) + (((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 % 3UL) * 3UL) + _fuseiter_5758)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 / 6UL) * 73728UL) + ((((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 / 3UL) % 2UL) * 36864UL) + (((fused_0fused_0_fuseiter_5755___fuseiter_5756_2059___fuseiter_5757_2060 % 3UL) * 12288UL) + ((_fuseiter_5758 * 4096UL) + ((_fuseiter_5759 * 256UL) + ((_fuseiter_5760 * 4UL) + _fuseiter_5761))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__185(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 = 0UL; fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 < 262144UL; fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2061____itr_2_2062 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__186(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 = 0UL; fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 < 262144UL; fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2063____itr_2_2064 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5772 = 0UL; _fuseiter_5772 < 16UL; _fuseiter_5772 += 1UL) {
    for (uint64_t _fuseiter_5773 = 0UL; _fuseiter_5773 < 4UL; _fuseiter_5773 += 1UL) {
      for (uint64_t _fuseiter_5776 = 0UL; _fuseiter_5776 < 16UL; _fuseiter_5776 += 1UL) {
        for (uint64_t _fuseiter_5777 = 0UL; _fuseiter_5777 < 64UL; _fuseiter_5777 += 1UL) {
          for (uint64_t _fuseiter_5778 = 0UL; _fuseiter_5778 < 4UL; _fuseiter_5778 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5777 + (_fuseiter_5772 * 64UL)) * 256UL) + ((_fuseiter_5778 + (_fuseiter_5776 * 4UL)) + (_fuseiter_5773 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5772 * 16384UL) + ((_fuseiter_5773 * 4096UL) + ((_fuseiter_5776 * 256UL) + ((_fuseiter_5777 * 4UL) + _fuseiter_5778))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__194(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 = 0UL; fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 < 262144UL; fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2065____itr_2_2066 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__195(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 = 0UL; fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 < 262144UL; fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2067____itr_2_2068 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5789 = 0UL; _fuseiter_5789 < 16UL; _fuseiter_5789 += 1UL) {
    for (uint64_t _fuseiter_5790 = 0UL; _fuseiter_5790 < 4UL; _fuseiter_5790 += 1UL) {
      for (uint64_t _fuseiter_5793 = 0UL; _fuseiter_5793 < 16UL; _fuseiter_5793 += 1UL) {
        for (uint64_t _fuseiter_5794 = 0UL; _fuseiter_5794 < 64UL; _fuseiter_5794 += 1UL) {
          for (uint64_t _fuseiter_5795 = 0UL; _fuseiter_5795 < 4UL; _fuseiter_5795 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5794 + (_fuseiter_5789 * 64UL)) * 256UL) + ((_fuseiter_5795 + (_fuseiter_5793 * 4UL)) + (_fuseiter_5790 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5789 * 16384UL) + ((_fuseiter_5790 * 4096UL) + ((_fuseiter_5793 * 256UL) + ((_fuseiter_5794 * 4UL) + _fuseiter_5795))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__203(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 = 0UL; fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 < 262144UL; fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2069____itr_2_2070 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__204(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 = 0UL; fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 < 262144UL; fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2071____itr_2_2072 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__502(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5806 = 0UL; _fuseiter_5806 < 16UL; _fuseiter_5806 += 1UL) {
    for (uint64_t _fuseiter_5807 = 0UL; _fuseiter_5807 < 4UL; _fuseiter_5807 += 1UL) {
      for (uint64_t _fuseiter_5810 = 0UL; _fuseiter_5810 < 16UL; _fuseiter_5810 += 1UL) {
        for (uint64_t _fuseiter_5811 = 0UL; _fuseiter_5811 < 64UL; _fuseiter_5811 += 1UL) {
          for (uint64_t _fuseiter_5812 = 0UL; _fuseiter_5812 < 4UL; _fuseiter_5812 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5811 + (_fuseiter_5806 * 64UL)) * 256UL) + ((_fuseiter_5812 + (_fuseiter_5810 * 4UL)) + (_fuseiter_5807 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5806 * 16384UL) + ((_fuseiter_5807 * 4096UL) + ((_fuseiter_5810 * 256UL) + ((_fuseiter_5811 * 4UL) + _fuseiter_5812))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__212(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 = 0UL; fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 < 262144UL; fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2073____itr_2_2074 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__213(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 = 0UL; fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 < 262144UL; fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2075____itr_2_2076 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__511(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5823 = 0UL; _fuseiter_5823 < 16UL; _fuseiter_5823 += 1UL) {
    for (uint64_t _fuseiter_5824 = 0UL; _fuseiter_5824 < 4UL; _fuseiter_5824 += 1UL) {
      for (uint64_t _fuseiter_5827 = 0UL; _fuseiter_5827 < 16UL; _fuseiter_5827 += 1UL) {
        for (uint64_t _fuseiter_5828 = 0UL; _fuseiter_5828 < 64UL; _fuseiter_5828 += 1UL) {
          for (uint64_t _fuseiter_5829 = 0UL; _fuseiter_5829 < 4UL; _fuseiter_5829 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5828 + (_fuseiter_5823 * 64UL)) * 256UL) + ((_fuseiter_5829 + (_fuseiter_5827 * 4UL)) + (_fuseiter_5824 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5823 * 16384UL) + ((_fuseiter_5824 * 4096UL) + ((_fuseiter_5827 * 256UL) + ((_fuseiter_5828 * 4UL) + _fuseiter_5829))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__221(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 = 0UL; fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 < 262144UL; fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2077____itr_2_2078 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__222(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 = 0UL; fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 < 262144UL; fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2079____itr_2_2080 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__520(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5840 = 0UL; _fuseiter_5840 < 16UL; _fuseiter_5840 += 1UL) {
    for (uint64_t _fuseiter_5841 = 0UL; _fuseiter_5841 < 4UL; _fuseiter_5841 += 1UL) {
      for (uint64_t _fuseiter_5844 = 0UL; _fuseiter_5844 < 16UL; _fuseiter_5844 += 1UL) {
        for (uint64_t _fuseiter_5845 = 0UL; _fuseiter_5845 < 64UL; _fuseiter_5845 += 1UL) {
          for (uint64_t _fuseiter_5846 = 0UL; _fuseiter_5846 < 4UL; _fuseiter_5846 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5845 + (_fuseiter_5840 * 64UL)) * 256UL) + ((_fuseiter_5846 + (_fuseiter_5844 * 4UL)) + (_fuseiter_5841 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5840 * 16384UL) + ((_fuseiter_5841 * 4096UL) + ((_fuseiter_5844 * 256UL) + ((_fuseiter_5845 * 4UL) + _fuseiter_5846))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__230(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 = 0UL; fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 < 262144UL; fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2081____itr_2_2082 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__231(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 = 0UL; fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 < 262144UL; fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_2083____itr_2_2084 % 256UL))] = __cached_1;
  }
  return true;
}

static bool reorder__529(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5857 = 0UL; _fuseiter_5857 < 16UL; _fuseiter_5857 += 1UL) {
    for (uint64_t _fuseiter_5858 = 0UL; _fuseiter_5858 < 4UL; _fuseiter_5858 += 1UL) {
      for (uint64_t _fuseiter_5861 = 0UL; _fuseiter_5861 < 16UL; _fuseiter_5861 += 1UL) {
        for (uint64_t _fuseiter_5862 = 0UL; _fuseiter_5862 < 64UL; _fuseiter_5862 += 1UL) {
          for (uint64_t _fuseiter_5863 = 0UL; _fuseiter_5863 < 4UL; _fuseiter_5863 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5862 + (_fuseiter_5857 * 64UL)) * 256UL) + ((_fuseiter_5863 + (_fuseiter_5861 * 4UL)) + (_fuseiter_5858 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5857 * 16384UL) + ((_fuseiter_5858 * 4096UL) + ((_fuseiter_5861 * 256UL) + ((_fuseiter_5862 * 4UL) + _fuseiter_5863))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__188(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 = 0UL; fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 < 262144UL; fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2085____itr_2_2086 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__189(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 = 0UL; fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 < 262144UL; fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2087____itr_2_2088 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5874___fuseiter_5875_2089 = 0UL; fused_0_fuseiter_5874___fuseiter_5875_2089 < 64UL; fused_0_fuseiter_5874___fuseiter_5875_2089 += 1UL) {
    for (uint64_t _fuseiter_5878 = 0UL; _fuseiter_5878 < 16UL; _fuseiter_5878 += 1UL) {
      for (uint64_t _fuseiter_5879 = 0UL; _fuseiter_5879 < 64UL; _fuseiter_5879 += 1UL) {
        for (uint64_t _fuseiter_5880 = 0UL; _fuseiter_5880 < 4UL; _fuseiter_5880 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5879 + ((fused_0_fuseiter_5874___fuseiter_5875_2089 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_5880 + (_fuseiter_5878 * 4UL)) + ((fused_0_fuseiter_5874___fuseiter_5875_2089 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5874___fuseiter_5875_2089 / 16UL) * 65536UL) + (((fused_0_fuseiter_5874___fuseiter_5875_2089 % 16UL) * 4096UL) + ((_fuseiter_5878 * 256UL) + ((_fuseiter_5879 * 4UL) + _fuseiter_5880))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__197(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 = 0UL; fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 < 262144UL; fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2090____itr_2_2091 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__198(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 = 0UL; fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 < 262144UL; fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2092____itr_2_2093 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5891___fuseiter_5892_2094 = 0UL; fused_0_fuseiter_5891___fuseiter_5892_2094 < 64UL; fused_0_fuseiter_5891___fuseiter_5892_2094 += 1UL) {
    for (uint64_t _fuseiter_5895 = 0UL; _fuseiter_5895 < 16UL; _fuseiter_5895 += 1UL) {
      for (uint64_t _fuseiter_5896 = 0UL; _fuseiter_5896 < 64UL; _fuseiter_5896 += 1UL) {
        for (uint64_t _fuseiter_5897 = 0UL; _fuseiter_5897 < 4UL; _fuseiter_5897 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5896 + ((fused_0_fuseiter_5891___fuseiter_5892_2094 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_5897 + (_fuseiter_5895 * 4UL)) + ((fused_0_fuseiter_5891___fuseiter_5892_2094 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5891___fuseiter_5892_2094 / 16UL) * 65536UL) + (((fused_0_fuseiter_5891___fuseiter_5892_2094 % 16UL) * 4096UL) + ((_fuseiter_5895 * 256UL) + ((_fuseiter_5896 * 4UL) + _fuseiter_5897))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__206(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 = 0UL; fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 < 262144UL; fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2095____itr_2_2096 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__207(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 = 0UL; fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 < 262144UL; fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2097____itr_2_2098 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__505(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5908___fuseiter_5909_2099 = 0UL; fused_0_fuseiter_5908___fuseiter_5909_2099 < 64UL; fused_0_fuseiter_5908___fuseiter_5909_2099 += 1UL) {
    for (uint64_t _fuseiter_5912 = 0UL; _fuseiter_5912 < 16UL; _fuseiter_5912 += 1UL) {
      for (uint64_t _fuseiter_5913 = 0UL; _fuseiter_5913 < 64UL; _fuseiter_5913 += 1UL) {
        for (uint64_t _fuseiter_5914 = 0UL; _fuseiter_5914 < 4UL; _fuseiter_5914 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5913 + ((fused_0_fuseiter_5908___fuseiter_5909_2099 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_5914 + (_fuseiter_5912 * 4UL)) + ((fused_0_fuseiter_5908___fuseiter_5909_2099 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5908___fuseiter_5909_2099 / 16UL) * 65536UL) + (((fused_0_fuseiter_5908___fuseiter_5909_2099 % 16UL) * 4096UL) + ((_fuseiter_5912 * 256UL) + ((_fuseiter_5913 * 4UL) + _fuseiter_5914))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__215(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 = 0UL; fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 < 262144UL; fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2100____itr_2_2101 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__216(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 = 0UL; fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 < 262144UL; fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2102____itr_2_2103 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__514(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5925___fuseiter_5926_2104 = 0UL; fused_0_fuseiter_5925___fuseiter_5926_2104 < 64UL; fused_0_fuseiter_5925___fuseiter_5926_2104 += 1UL) {
    for (uint64_t _fuseiter_5929 = 0UL; _fuseiter_5929 < 16UL; _fuseiter_5929 += 1UL) {
      for (uint64_t _fuseiter_5930 = 0UL; _fuseiter_5930 < 64UL; _fuseiter_5930 += 1UL) {
        for (uint64_t _fuseiter_5931 = 0UL; _fuseiter_5931 < 4UL; _fuseiter_5931 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5930 + ((fused_0_fuseiter_5925___fuseiter_5926_2104 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_5931 + (_fuseiter_5929 * 4UL)) + ((fused_0_fuseiter_5925___fuseiter_5926_2104 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5925___fuseiter_5926_2104 / 16UL) * 65536UL) + (((fused_0_fuseiter_5925___fuseiter_5926_2104 % 16UL) * 4096UL) + ((_fuseiter_5929 * 256UL) + ((_fuseiter_5930 * 4UL) + _fuseiter_5931))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__224(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 = 0UL; fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 < 262144UL; fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2105____itr_2_2106 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__225(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 = 0UL; fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 < 262144UL; fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2107____itr_2_2108 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__523(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5942___fuseiter_5943_2109 = 0UL; fused_0_fuseiter_5942___fuseiter_5943_2109 < 64UL; fused_0_fuseiter_5942___fuseiter_5943_2109 += 1UL) {
    for (uint64_t _fuseiter_5946 = 0UL; _fuseiter_5946 < 16UL; _fuseiter_5946 += 1UL) {
      for (uint64_t _fuseiter_5947 = 0UL; _fuseiter_5947 < 64UL; _fuseiter_5947 += 1UL) {
        for (uint64_t _fuseiter_5948 = 0UL; _fuseiter_5948 < 4UL; _fuseiter_5948 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5947 + ((fused_0_fuseiter_5942___fuseiter_5943_2109 / 16UL) * 64UL)) * 1024UL) + ((_fuseiter_5948 + (_fuseiter_5946 * 4UL)) + ((fused_0_fuseiter_5942___fuseiter_5943_2109 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5942___fuseiter_5943_2109 / 16UL) * 65536UL) + (((fused_0_fuseiter_5942___fuseiter_5943_2109 % 16UL) * 4096UL) + ((_fuseiter_5946 * 256UL) + ((_fuseiter_5947 * 4UL) + _fuseiter_5948))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__176(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 = 0UL; fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 < 524288UL; fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2110____itr_2_2111 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__177(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 = 0UL; fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 < 524288UL; fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2112____itr_2_2113 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__475(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_5959 = 0UL; _fuseiter_5959 < 16UL; _fuseiter_5959 += 1UL) {
    for (uint64_t _fuseiter_5960 = 0UL; _fuseiter_5960 < 8UL; _fuseiter_5960 += 1UL) {
      for (uint64_t _fuseiter_5963 = 0UL; _fuseiter_5963 < 16UL; _fuseiter_5963 += 1UL) {
        for (uint64_t _fuseiter_5964 = 0UL; _fuseiter_5964 < 64UL; _fuseiter_5964 += 1UL) {
          for (uint64_t _fuseiter_5965 = 0UL; _fuseiter_5965 < 4UL; _fuseiter_5965 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_5964 + (_fuseiter_5959 * 64UL)) * 512UL) + ((_fuseiter_5965 + (_fuseiter_5963 * 4UL)) + (_fuseiter_5960 * 64UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_5959 * 32768UL) + ((_fuseiter_5960 * 4096UL) + ((_fuseiter_5963 * 256UL) + ((_fuseiter_5964 * 4UL) + _fuseiter_5965))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__236(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 = 0UL; fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 < 524288UL; fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2114____itr_2_2115 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__237(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 = 0UL; fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 < 524288UL; fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2116____itr_2_2117 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__535(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5976___fuseiter_5977_2118 = 0UL; fused_0_fuseiter_5976___fuseiter_5977_2118 < 16UL; fused_0_fuseiter_5976___fuseiter_5977_2118 += 1UL) {
    for (uint64_t _fuseiter_5980 = 0UL; _fuseiter_5980 < 16UL; _fuseiter_5980 += 1UL) {
      for (uint64_t _fuseiter_5981 = 0UL; _fuseiter_5981 < 512UL; _fuseiter_5981 += 1UL) {
        for (uint64_t _fuseiter_5982 = 0UL; _fuseiter_5982 < 4UL; _fuseiter_5982 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_5981 + ((fused_0_fuseiter_5976___fuseiter_5977_2118 / 16UL) * 512UL)) * 1024UL) + ((_fuseiter_5982 + (_fuseiter_5980 * 4UL)) + ((fused_0_fuseiter_5976___fuseiter_5977_2118 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_5976___fuseiter_5977_2118 / 16UL) * 524288UL) + (((fused_0_fuseiter_5976___fuseiter_5977_2118 % 16UL) * 32768UL) + ((_fuseiter_5980 * 2048UL) + ((_fuseiter_5981 * 4UL) + _fuseiter_5982))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__182(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 = 0UL; fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 < 196608UL; fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 += 1UL) {
    for (uint64_t _fuseiter_5987 = 0UL; _fuseiter_5987 < 3UL; _fuseiter_5987 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 % 3UL) * 3UL))) + _fuseiter_5987)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2119____itr_2_2120 % 3UL) * 3UL))) + _fuseiter_5987)] = __cached_2;
    }
  }
  return true;
}

static bool cast__183(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 = 0UL; fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 < 196608UL; fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 += 1UL) {
    for (uint64_t _fuseiter5992 = 0UL; _fuseiter5992 < 3UL; _fuseiter5992 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 % 3UL) * 3UL))) + _fuseiter5992)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2121____itr_2_2122 % 3UL) * 3UL))) + _fuseiter5992)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__481(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_5993___fuseiter_5994_2123 = 0UL; fused_0_fuseiter_5993___fuseiter_5994_2123 < 16UL; fused_0_fuseiter_5993___fuseiter_5994_2123 += 1UL) {
    for (uint64_t _fuseiter_5995 = 0UL; _fuseiter_5995 < 3UL; _fuseiter_5995 += 1UL) {
      for (uint64_t _fuseiter_5996 = 0UL; _fuseiter_5996 < 3UL; _fuseiter_5996 += 1UL) {
        for (uint64_t _fuseiter_5997 = 0UL; _fuseiter_5997 < 16UL; _fuseiter_5997 += 1UL) {
          for (uint64_t _fuseiter_5998 = 0UL; _fuseiter_5998 < 64UL; _fuseiter_5998 += 1UL) {
            for (uint64_t _fuseiter_5999 = 0UL; _fuseiter_5999 < 4UL; _fuseiter_5999 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_5998 + ((fused_0_fuseiter_5993___fuseiter_5994_2123 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_5999 + (_fuseiter_5997 * 4UL)) + ((fused_0_fuseiter_5993___fuseiter_5994_2123 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_5995 * 3UL) + _fuseiter_5996)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_5993___fuseiter_5994_2123 / 4UL) * 147456UL) + (((fused_0_fuseiter_5993___fuseiter_5994_2123 % 4UL) * 36864UL) + ((_fuseiter_5995 * 12288UL) + ((_fuseiter_5996 * 4096UL) + ((_fuseiter_5997 * 256UL) + ((_fuseiter_5998 * 4UL) + _fuseiter_5999))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__191(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 = 0UL; fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 < 196608UL; fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 += 1UL) {
    for (uint64_t _fuseiter_6004 = 0UL; _fuseiter_6004 < 3UL; _fuseiter_6004 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 % 3UL) * 3UL))) + _fuseiter_6004)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2124____itr_2_2125 % 3UL) * 3UL))) + _fuseiter_6004)] = __cached_2;
    }
  }
  return true;
}

static bool cast__192(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 = 0UL; fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 < 196608UL; fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 += 1UL) {
    for (uint64_t _fuseiter6009 = 0UL; _fuseiter6009 < 3UL; _fuseiter6009 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 % 3UL) * 3UL))) + _fuseiter6009)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2126____itr_2_2127 % 3UL) * 3UL))) + _fuseiter6009)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6010___fuseiter_6011_2128 = 0UL; fused_0_fuseiter_6010___fuseiter_6011_2128 < 16UL; fused_0_fuseiter_6010___fuseiter_6011_2128 += 1UL) {
    for (uint64_t _fuseiter_6012 = 0UL; _fuseiter_6012 < 3UL; _fuseiter_6012 += 1UL) {
      for (uint64_t _fuseiter_6013 = 0UL; _fuseiter_6013 < 3UL; _fuseiter_6013 += 1UL) {
        for (uint64_t _fuseiter_6014 = 0UL; _fuseiter_6014 < 16UL; _fuseiter_6014 += 1UL) {
          for (uint64_t _fuseiter_6015 = 0UL; _fuseiter_6015 < 64UL; _fuseiter_6015 += 1UL) {
            for (uint64_t _fuseiter_6016 = 0UL; _fuseiter_6016 < 4UL; _fuseiter_6016 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6015 + ((fused_0_fuseiter_6010___fuseiter_6011_2128 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_6016 + (_fuseiter_6014 * 4UL)) + ((fused_0_fuseiter_6010___fuseiter_6011_2128 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_6012 * 3UL) + _fuseiter_6013)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6010___fuseiter_6011_2128 / 4UL) * 147456UL) + (((fused_0_fuseiter_6010___fuseiter_6011_2128 % 4UL) * 36864UL) + ((_fuseiter_6012 * 12288UL) + ((_fuseiter_6013 * 4096UL) + ((_fuseiter_6014 * 256UL) + ((_fuseiter_6015 * 4UL) + _fuseiter_6016))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__200(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 = 0UL; fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 < 196608UL; fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 += 1UL) {
    for (uint64_t _fuseiter_6021 = 0UL; _fuseiter_6021 < 3UL; _fuseiter_6021 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 % 3UL) * 3UL))) + _fuseiter_6021)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2129____itr_2_2130 % 3UL) * 3UL))) + _fuseiter_6021)] = __cached_2;
    }
  }
  return true;
}

static bool cast__201(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 = 0UL; fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 < 196608UL; fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 += 1UL) {
    for (uint64_t _fuseiter6026 = 0UL; _fuseiter6026 < 3UL; _fuseiter6026 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 % 3UL) * 3UL))) + _fuseiter6026)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2131____itr_2_2132 % 3UL) * 3UL))) + _fuseiter6026)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__499(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6027___fuseiter_6028_2133 = 0UL; fused_0_fuseiter_6027___fuseiter_6028_2133 < 16UL; fused_0_fuseiter_6027___fuseiter_6028_2133 += 1UL) {
    for (uint64_t _fuseiter_6029 = 0UL; _fuseiter_6029 < 3UL; _fuseiter_6029 += 1UL) {
      for (uint64_t _fuseiter_6030 = 0UL; _fuseiter_6030 < 3UL; _fuseiter_6030 += 1UL) {
        for (uint64_t _fuseiter_6031 = 0UL; _fuseiter_6031 < 16UL; _fuseiter_6031 += 1UL) {
          for (uint64_t _fuseiter_6032 = 0UL; _fuseiter_6032 < 64UL; _fuseiter_6032 += 1UL) {
            for (uint64_t _fuseiter_6033 = 0UL; _fuseiter_6033 < 4UL; _fuseiter_6033 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6032 + ((fused_0_fuseiter_6027___fuseiter_6028_2133 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_6033 + (_fuseiter_6031 * 4UL)) + ((fused_0_fuseiter_6027___fuseiter_6028_2133 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_6029 * 3UL) + _fuseiter_6030)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6027___fuseiter_6028_2133 / 4UL) * 147456UL) + (((fused_0_fuseiter_6027___fuseiter_6028_2133 % 4UL) * 36864UL) + ((_fuseiter_6029 * 12288UL) + ((_fuseiter_6030 * 4096UL) + ((_fuseiter_6031 * 256UL) + ((_fuseiter_6032 * 4UL) + _fuseiter_6033))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__209(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 = 0UL; fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 < 196608UL; fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 += 1UL) {
    for (uint64_t _fuseiter_6038 = 0UL; _fuseiter_6038 < 3UL; _fuseiter_6038 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 % 3UL) * 3UL))) + _fuseiter_6038)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2134____itr_2_2135 % 3UL) * 3UL))) + _fuseiter_6038)] = __cached_2;
    }
  }
  return true;
}

static bool cast__210(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 = 0UL; fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 < 196608UL; fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 += 1UL) {
    for (uint64_t _fuseiter6043 = 0UL; _fuseiter6043 < 3UL; _fuseiter6043 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 % 3UL) * 3UL))) + _fuseiter6043)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2136____itr_2_2137 % 3UL) * 3UL))) + _fuseiter6043)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__508(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6044___fuseiter_6045_2138 = 0UL; fused_0_fuseiter_6044___fuseiter_6045_2138 < 16UL; fused_0_fuseiter_6044___fuseiter_6045_2138 += 1UL) {
    for (uint64_t _fuseiter_6046 = 0UL; _fuseiter_6046 < 3UL; _fuseiter_6046 += 1UL) {
      for (uint64_t _fuseiter_6047 = 0UL; _fuseiter_6047 < 3UL; _fuseiter_6047 += 1UL) {
        for (uint64_t _fuseiter_6048 = 0UL; _fuseiter_6048 < 16UL; _fuseiter_6048 += 1UL) {
          for (uint64_t _fuseiter_6049 = 0UL; _fuseiter_6049 < 64UL; _fuseiter_6049 += 1UL) {
            for (uint64_t _fuseiter_6050 = 0UL; _fuseiter_6050 < 4UL; _fuseiter_6050 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6049 + ((fused_0_fuseiter_6044___fuseiter_6045_2138 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_6050 + (_fuseiter_6048 * 4UL)) + ((fused_0_fuseiter_6044___fuseiter_6045_2138 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_6046 * 3UL) + _fuseiter_6047)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6044___fuseiter_6045_2138 / 4UL) * 147456UL) + (((fused_0_fuseiter_6044___fuseiter_6045_2138 % 4UL) * 36864UL) + ((_fuseiter_6046 * 12288UL) + ((_fuseiter_6047 * 4096UL) + ((_fuseiter_6048 * 256UL) + ((_fuseiter_6049 * 4UL) + _fuseiter_6050))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__218(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 = 0UL; fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 < 196608UL; fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 += 1UL) {
    for (uint64_t _fuseiter_6055 = 0UL; _fuseiter_6055 < 3UL; _fuseiter_6055 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 % 3UL) * 3UL))) + _fuseiter_6055)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2139____itr_2_2140 % 3UL) * 3UL))) + _fuseiter_6055)] = __cached_2;
    }
  }
  return true;
}

static bool cast__219(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 = 0UL; fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 < 196608UL; fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 += 1UL) {
    for (uint64_t _fuseiter6060 = 0UL; _fuseiter6060 < 3UL; _fuseiter6060 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 % 3UL) * 3UL))) + _fuseiter6060)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2141____itr_2_2142 % 3UL) * 3UL))) + _fuseiter6060)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__517(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6061___fuseiter_6062_2143 = 0UL; fused_0_fuseiter_6061___fuseiter_6062_2143 < 16UL; fused_0_fuseiter_6061___fuseiter_6062_2143 += 1UL) {
    for (uint64_t _fuseiter_6063 = 0UL; _fuseiter_6063 < 3UL; _fuseiter_6063 += 1UL) {
      for (uint64_t _fuseiter_6064 = 0UL; _fuseiter_6064 < 3UL; _fuseiter_6064 += 1UL) {
        for (uint64_t _fuseiter_6065 = 0UL; _fuseiter_6065 < 16UL; _fuseiter_6065 += 1UL) {
          for (uint64_t _fuseiter_6066 = 0UL; _fuseiter_6066 < 64UL; _fuseiter_6066 += 1UL) {
            for (uint64_t _fuseiter_6067 = 0UL; _fuseiter_6067 < 4UL; _fuseiter_6067 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6066 + ((fused_0_fuseiter_6061___fuseiter_6062_2143 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_6067 + (_fuseiter_6065 * 4UL)) + ((fused_0_fuseiter_6061___fuseiter_6062_2143 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_6063 * 3UL) + _fuseiter_6064)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6061___fuseiter_6062_2143 / 4UL) * 147456UL) + (((fused_0_fuseiter_6061___fuseiter_6062_2143 % 4UL) * 36864UL) + ((_fuseiter_6063 * 12288UL) + ((_fuseiter_6064 * 4096UL) + ((_fuseiter_6065 * 256UL) + ((_fuseiter_6066 * 4UL) + _fuseiter_6067))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__227(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 = 0UL; fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 < 196608UL; fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 += 1UL) {
    for (uint64_t _fuseiter_6072 = 0UL; _fuseiter_6072 < 3UL; _fuseiter_6072 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 % 3UL) * 3UL))) + _fuseiter_6072)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2144____itr_2_2145 % 3UL) * 3UL))) + _fuseiter_6072)] = __cached_2;
    }
  }
  return true;
}

static bool cast__228(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 = 0UL; fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 < 196608UL; fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 += 1UL) {
    for (uint64_t _fuseiter6077 = 0UL; _fuseiter6077 < 3UL; _fuseiter6077 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 % 3UL) * 3UL))) + _fuseiter6077)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2146____itr_2_2147 % 3UL) * 3UL))) + _fuseiter6077)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__526(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6078___fuseiter_6079_2148 = 0UL; fused_0_fuseiter_6078___fuseiter_6079_2148 < 16UL; fused_0_fuseiter_6078___fuseiter_6079_2148 += 1UL) {
    for (uint64_t _fuseiter_6080 = 0UL; _fuseiter_6080 < 3UL; _fuseiter_6080 += 1UL) {
      for (uint64_t _fuseiter_6081 = 0UL; _fuseiter_6081 < 3UL; _fuseiter_6081 += 1UL) {
        for (uint64_t _fuseiter_6082 = 0UL; _fuseiter_6082 < 16UL; _fuseiter_6082 += 1UL) {
          for (uint64_t _fuseiter_6083 = 0UL; _fuseiter_6083 < 64UL; _fuseiter_6083 += 1UL) {
            for (uint64_t _fuseiter_6084 = 0UL; _fuseiter_6084 < 4UL; _fuseiter_6084 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6083 + ((fused_0_fuseiter_6078___fuseiter_6079_2148 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_6084 + (_fuseiter_6082 * 4UL)) + ((fused_0_fuseiter_6078___fuseiter_6079_2148 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_6080 * 3UL) + _fuseiter_6081)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6078___fuseiter_6079_2148 / 4UL) * 147456UL) + (((fused_0_fuseiter_6078___fuseiter_6079_2148 % 4UL) * 36864UL) + ((_fuseiter_6080 * 12288UL) + ((_fuseiter_6081 * 4096UL) + ((_fuseiter_6082 * 256UL) + ((_fuseiter_6083 * 4UL) + _fuseiter_6084))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__242(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 = 0UL; fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 < 1048576UL; fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2149____itr_2_2150 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__243(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 = 0UL; fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 < 1048576UL; fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2151____itr_2_2152 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__539(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 += 1UL) {
    for (uint64_t _fuseiter_6100 = 0UL; _fuseiter_6100 < 512UL; _fuseiter_6100 += 1UL) {
      for (uint64_t _fuseiter_6101 = 0UL; _fuseiter_6101 < 4UL; _fuseiter_6101 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_6100 + ((fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 / 128UL) * 512UL)) * 512UL) + (_fuseiter_6101 + ((fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_6095___fuseiter_6096_2153___fuseiter_6097_2154___fuseiter_6098_2155___fuseiter_6099_2156 % 128UL) * 2048UL) + ((_fuseiter_6100 * 4UL) + _fuseiter_6101)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__251(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 = 0UL; fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 < 1048576UL; fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2157____itr_2_2158 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__252(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 = 0UL; fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 < 1048576UL; fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2159____itr_2_2160 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__548(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 += 1UL) {
    for (uint64_t _fuseiter_6117 = 0UL; _fuseiter_6117 < 512UL; _fuseiter_6117 += 1UL) {
      for (uint64_t _fuseiter_6118 = 0UL; _fuseiter_6118 < 4UL; _fuseiter_6118 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_6117 + ((fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 / 128UL) * 512UL)) * 512UL) + (_fuseiter_6118 + ((fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_6112___fuseiter_6113_2161___fuseiter_6114_2162___fuseiter_6115_2163___fuseiter_6116_2164 % 128UL) * 2048UL) + ((_fuseiter_6117 * 4UL) + _fuseiter_6118)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__260(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 = 0UL; fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 < 1048576UL; fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2165____itr_2_2166 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__261(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 = 0UL; fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 < 1048576UL; fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_2167____itr_2_2168 % 512UL))] = __cached_1;
  }
  return true;
}

static bool reorder__557(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6129___fuseiter_6130_2169 = 0UL; fused_0_fuseiter_6129___fuseiter_6130_2169 < 32UL; fused_0_fuseiter_6129___fuseiter_6130_2169 += 1UL) {
    for (uint64_t _fuseiter_6133 = 0UL; _fuseiter_6133 < 16UL; _fuseiter_6133 += 1UL) {
      for (uint64_t _fuseiter_6134 = 0UL; _fuseiter_6134 < 512UL; _fuseiter_6134 += 1UL) {
        for (uint64_t _fuseiter_6135 = 0UL; _fuseiter_6135 < 4UL; _fuseiter_6135 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_6134 + ((fused_0_fuseiter_6129___fuseiter_6130_2169 / 8UL) * 512UL)) * 512UL) + ((_fuseiter_6135 + (_fuseiter_6133 * 4UL)) + ((fused_0_fuseiter_6129___fuseiter_6130_2169 % 8UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_6129___fuseiter_6130_2169 / 8UL) * 262144UL) + (((fused_0_fuseiter_6129___fuseiter_6130_2169 % 8UL) * 32768UL) + ((_fuseiter_6133 * 2048UL) + ((_fuseiter_6134 * 4UL) + _fuseiter_6135))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__245(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 = 0UL; fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 < 1048576UL; fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2170____itr_2_2171 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__246(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 = 0UL; fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 < 1048576UL; fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2172____itr_2_2173 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool reorder__542(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6146___fuseiter_6147_2174 = 0UL; fused_0_fuseiter_6146___fuseiter_6147_2174 < 32UL; fused_0_fuseiter_6146___fuseiter_6147_2174 += 1UL) {
    for (uint64_t _fuseiter_6150 = 0UL; _fuseiter_6150 < 128UL; _fuseiter_6150 += 1UL) {
      for (uint64_t _fuseiter_6151 = 0UL; _fuseiter_6151 < 64UL; _fuseiter_6151 += 1UL) {
        for (uint64_t _fuseiter_6152 = 0UL; _fuseiter_6152 < 4UL; _fuseiter_6152 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_6151 + ((fused_0_fuseiter_6146___fuseiter_6147_2174 / 4UL) * 64UL)) * 2048UL) + ((_fuseiter_6152 + (_fuseiter_6150 * 4UL)) + ((fused_0_fuseiter_6146___fuseiter_6147_2174 % 4UL) * 512UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_6146___fuseiter_6147_2174 / 4UL) * 131072UL) + (((fused_0_fuseiter_6146___fuseiter_6147_2174 % 4UL) * 32768UL) + ((_fuseiter_6150 * 256UL) + ((_fuseiter_6151 * 4UL) + _fuseiter_6152))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__254(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 = 0UL; fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 < 1048576UL; fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2175____itr_2_2176 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__255(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 = 0UL; fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 < 1048576UL; fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_2177____itr_2_2178 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool reorder__551(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 < 1024UL; fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 += 1UL) {
    for (uint64_t _fuseiter_6168 = 0UL; _fuseiter_6168 < 256UL; _fuseiter_6168 += 1UL) {
      for (uint64_t _fuseiter_6169 = 0UL; _fuseiter_6169 < 4UL; _fuseiter_6169 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_6168 + ((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 / 512UL) * 256UL)) * 2048UL) + ((_fuseiter_6169 + ((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 % 128UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 / 128UL) % 4UL) * 512UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 / 512UL) * 524288UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 / 128UL) % 4UL) * 131072UL) + (((fused_0fused_0fused_0fused_0_fuseiter_6163___fuseiter_6164_2179___fuseiter_6165_2180___fuseiter_6166_2181___fuseiter_6167_2182 % 128UL) * 1024UL) + ((_fuseiter_6168 * 4UL) + _fuseiter_6169))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__233(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 = 0UL; fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 < 2097152UL; fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2183____itr_2_2184 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__234(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 = 0UL; fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 < 2097152UL; fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_2185____itr_2_2186 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool reorder__532(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6180___fuseiter_6181_2187 = 0UL; fused_0_fuseiter_6180___fuseiter_6181_2187 < 64UL; fused_0_fuseiter_6180___fuseiter_6181_2187 += 1UL) {
    for (uint64_t _fuseiter_6184 = 0UL; _fuseiter_6184 < 16UL; _fuseiter_6184 += 1UL) {
      for (uint64_t _fuseiter_6185 = 0UL; _fuseiter_6185 < 512UL; _fuseiter_6185 += 1UL) {
        for (uint64_t _fuseiter_6186 = 0UL; _fuseiter_6186 < 4UL; _fuseiter_6186 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_6185 + ((fused_0_fuseiter_6180___fuseiter_6181_2187 / 16UL) * 512UL)) * 1024UL) + ((_fuseiter_6186 + (_fuseiter_6184 * 4UL)) + ((fused_0_fuseiter_6180___fuseiter_6181_2187 % 16UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_6180___fuseiter_6181_2187 / 16UL) * 524288UL) + (((fused_0_fuseiter_6180___fuseiter_6181_2187 % 16UL) * 32768UL) + ((_fuseiter_6184 * 2048UL) + ((_fuseiter_6185 * 4UL) + _fuseiter_6186))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__239(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 = 0UL; fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 < 786432UL; fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 += 1UL) {
    for (uint64_t _fuseiter_6191 = 0UL; _fuseiter_6191 < 3UL; _fuseiter_6191 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 % 3UL) * 3UL))) + _fuseiter_6191)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2188____itr_2_2189 % 3UL) * 3UL))) + _fuseiter_6191)] = __cached_2;
    }
  }
  return true;
}

static bool cast__240(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 = 0UL; fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 < 786432UL; fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 += 1UL) {
    for (uint64_t _fuseiter6196 = 0UL; _fuseiter6196 < 3UL; _fuseiter6196 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 % 3UL) * 3UL))) + _fuseiter6196)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2190____itr_2_2191 % 3UL) * 3UL))) + _fuseiter6196)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__536(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6197___fuseiter_6198_2192 = 0UL; fused_0_fuseiter_6197___fuseiter_6198_2192 < 16UL; fused_0_fuseiter_6197___fuseiter_6198_2192 += 1UL) {
    for (uint64_t _fuseiter_6199 = 0UL; _fuseiter_6199 < 3UL; _fuseiter_6199 += 1UL) {
      for (uint64_t _fuseiter_6200 = 0UL; _fuseiter_6200 < 3UL; _fuseiter_6200 += 1UL) {
        for (uint64_t _fuseiter_6201 = 0UL; _fuseiter_6201 < 16UL; _fuseiter_6201 += 1UL) {
          for (uint64_t _fuseiter_6202 = 0UL; _fuseiter_6202 < 256UL; _fuseiter_6202 += 1UL) {
            for (uint64_t _fuseiter_6203 = 0UL; _fuseiter_6203 < 4UL; _fuseiter_6203 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6202 + ((fused_0_fuseiter_6197___fuseiter_6198_2192 / 8UL) * 256UL)) * 4608UL) + ((((_fuseiter_6203 + (_fuseiter_6201 * 4UL)) + ((fused_0_fuseiter_6197___fuseiter_6198_2192 % 8UL) * 64UL)) * 9UL) + ((_fuseiter_6199 * 3UL) + _fuseiter_6200)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6197___fuseiter_6198_2192 / 8UL) * 1179648UL) + (((fused_0_fuseiter_6197___fuseiter_6198_2192 % 8UL) * 147456UL) + ((_fuseiter_6199 * 49152UL) + ((_fuseiter_6200 * 16384UL) + ((_fuseiter_6201 * 1024UL) + ((_fuseiter_6202 * 4UL) + _fuseiter_6203))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__248(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 = 0UL; fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 < 786432UL; fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 += 1UL) {
    for (uint64_t _fuseiter_6208 = 0UL; _fuseiter_6208 < 3UL; _fuseiter_6208 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 % 3UL) * 3UL))) + _fuseiter_6208)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2193____itr_2_2194 % 3UL) * 3UL))) + _fuseiter_6208)] = __cached_2;
    }
  }
  return true;
}

static bool cast__249(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 = 0UL; fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 < 786432UL; fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 += 1UL) {
    for (uint64_t _fuseiter6213 = 0UL; _fuseiter6213 < 3UL; _fuseiter6213 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 % 3UL) * 3UL))) + _fuseiter6213)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2195____itr_2_2196 % 3UL) * 3UL))) + _fuseiter6213)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__545(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_6214___fuseiter_6215_2197 = 0UL; fused_0_fuseiter_6214___fuseiter_6215_2197 < 32UL; fused_0_fuseiter_6214___fuseiter_6215_2197 += 1UL) {
    for (uint64_t _fuseiter_6216 = 0UL; _fuseiter_6216 < 3UL; _fuseiter_6216 += 1UL) {
      for (uint64_t _fuseiter_6217 = 0UL; _fuseiter_6217 < 3UL; _fuseiter_6217 += 1UL) {
        for (uint64_t _fuseiter_6218 = 0UL; _fuseiter_6218 < 16UL; _fuseiter_6218 += 1UL) {
          for (uint64_t _fuseiter_6219 = 0UL; _fuseiter_6219 < 128UL; _fuseiter_6219 += 1UL) {
            for (uint64_t _fuseiter_6220 = 0UL; _fuseiter_6220 < 4UL; _fuseiter_6220 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_6219 + ((fused_0_fuseiter_6214___fuseiter_6215_2197 / 8UL) * 128UL)) * 4608UL) + ((((_fuseiter_6220 + (_fuseiter_6218 * 4UL)) + ((fused_0_fuseiter_6214___fuseiter_6215_2197 % 8UL) * 64UL)) * 9UL) + ((_fuseiter_6216 * 3UL) + _fuseiter_6217)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_6214___fuseiter_6215_2197 / 8UL) * 589824UL) + (((fused_0_fuseiter_6214___fuseiter_6215_2197 % 8UL) * 73728UL) + ((_fuseiter_6216 * 24576UL) + ((_fuseiter_6217 * 8192UL) + ((_fuseiter_6218 * 512UL) + ((_fuseiter_6219 * 4UL) + _fuseiter_6220))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__257(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 = 0UL; fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 < 786432UL; fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 += 1UL) {
    for (uint64_t _fuseiter_6225 = 0UL; _fuseiter_6225 < 3UL; _fuseiter_6225 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 % 3UL) * 3UL))) + _fuseiter_6225)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2198____itr_2_2199 % 3UL) * 3UL))) + _fuseiter_6225)] = __cached_2;
    }
  }
  return true;
}

static bool cast__258(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 = 0UL; fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 < 786432UL; fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 += 1UL) {
    for (uint64_t _fuseiter6230 = 0UL; _fuseiter6230 < 3UL; _fuseiter6230 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 % 3UL) * 3UL))) + _fuseiter6230)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_2200____itr_2_2201 % 3UL) * 3UL))) + _fuseiter6230)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__554(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 = 0UL; fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 < 24UL; fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 += 1UL) {
    for (uint64_t _fuseiter_6234 = 0UL; _fuseiter_6234 < 3UL; _fuseiter_6234 += 1UL) {
      for (uint64_t _fuseiter_6235 = 0UL; _fuseiter_6235 < 128UL; _fuseiter_6235 += 1UL) {
        for (uint64_t _fuseiter_6236 = 0UL; _fuseiter_6236 < 64UL; _fuseiter_6236 += 1UL) {
          for (uint64_t _fuseiter_6237 = 0UL; _fuseiter_6237 < 4UL; _fuseiter_6237 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_6236 + ((fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 / 3UL) * 64UL)) * 4608UL) + (((_fuseiter_6237 + (_fuseiter_6235 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 % 3UL) * 3UL) + _fuseiter_6234)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 / 3UL) * 294912UL) + (((fused_0fused_0_fuseiter_6231___fuseiter_6232_2202___fuseiter_6233_2203 % 3UL) * 98304UL) + ((_fuseiter_6234 * 32768UL) + ((_fuseiter_6235 * 256UL) + ((_fuseiter_6236 * 4UL) + _fuseiter_6237)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool batchwise_8_fused_res2a_conv_b_cast_mul_add_cast_res2a_conv_0_cast_mul_add_cast_relu_res2a_conv_1_cast_mul_add_cast_relu_res2a_conv_2_cast_mul_add_cast_add_relu_res2b_conv_0_cast_mul_add_cast_relu_res2b_conv_1_cast_mul_add_cast_relu_res2b_conv_2_cast_mul_add_cast_add_relu_res2c_conv_0_cast_mul_add_cast_relu_res2c_conv_1_cast_mul_add_cast_relu_res2c_conv_2_cast_mul_add_cast_add_relu_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_cast_relu_res3a_conv_1_cast_mul_add_cast_relu_res3a_conv_2_cast_mul_add_cast_add_relu_res3b_conv_0_cast_mul_add_cast_relu_res3b_conv_1_cast_mul_add_cast_relu_res3b_conv_2_cast_mul_add_cast_add_relu_res3c_conv_0_cast_mul_add_cast_relu_res3c_conv_1_cast_mul_add_cast_relu_res3c_conv_2_cast_mul_add_cast_add_relu_res3d_conv_0_cast_mul_add_cast_relu_res3d_conv_1_cast_mul_add_cast_relu_res3d_conv_2_cast_mul_add_cast_add_relu__684(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 8UL; __batchwise_iter_0 += 1UL) {
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
  for (uint64_t fused_0fused_0n__n_i_2204__k_2205 = 0UL; fused_0fused_0n__n_i_2204__k_2205 < 4UL; fused_0fused_0n__n_i_2204__k_2205 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_1560_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2204__k_2205 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0fused_0n__n_i_2204__k_2205 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_0 = &__origouts_1560_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_1560_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter6242 = 0UL; _fuseiter6242 < 56UL; _fuseiter6242 += 1UL) {
        for (uint64_t _fuseiter6243 = 0UL; _fuseiter6243 < 64UL; _fuseiter6243 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1560_shr[((_fuseiter6242 * 64UL) + _fuseiter6243)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2204__k_2205 / 4UL) * 256UL) + ((fused_0fused_0n__n_i_2204__k_2205 % 4UL) * 64UL)) + _fuseiter6243)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2204__k_2205 % 4UL) * 64UL) + _fuseiter6243)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[((((fused_0fused_0n__n_i_2204__k_2205 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_2204__k_2205 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter6242 * 64UL) + _fuseiter6243))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1560_shr);
    }
  }
  return true;
}

static bool res2a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_55 = *(void**)(__module_data + 16);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
    int32_t* __origouts_1570_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[(p_o * 7168UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[0UL];
    B_list[0UL] = __cached_1;
    void* _arg_cache_1 = &__origouts_1570_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_55, A_list, B_list, &__origouts_1570_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter6269 = 0UL; _fuseiter6269 < 2UL; _fuseiter6269 += 1UL) {
      for (uint64_t _fuseiter6270 = 0UL; _fuseiter6270 < 56UL; _fuseiter6270 += 1UL) {
        for (uint64_t _fuseiter6271 = 0UL; _fuseiter6271 < 64UL; _fuseiter6271 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1570_shr[((_fuseiter6269 * 3584UL) + ((_fuseiter6270 * 64UL) + _fuseiter6271))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6271]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6271]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((p_o * 2UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter6269 * 3712UL) + ((_fuseiter6270 * 64UL) + _fuseiter6271)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1570_shr);
  }
  return true;
}

static bool res2a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_57 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_1580_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
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
    void* _arg_cache_2 = &__origouts_1580_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_57, A_list, B_list, &__origouts_1580_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter6305 = 0UL; _fuseiter6305 < 56UL; _fuseiter6305 += 1UL) {
      for (uint64_t _fuseiter6306 = 0UL; _fuseiter6306 < 64UL; _fuseiter6306 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_1580_shr[((_fuseiter6305 * 64UL) + _fuseiter6306)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6306]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6306]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter6305 * 64UL) + _fuseiter6306))]);
      }
    }
    sc_aligned_free(__stream, __origouts_1580_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_2210__n_i_2211 = 0UL; fused_0fused_0k__n_2210__n_i_2211 < 4UL; fused_0fused_0k__n_2210__n_i_2211 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_1590_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0fused_0k__n_2210__n_i_2211 * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_3 = &__origouts_1590_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_1590_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter6340 = 0UL; _fuseiter6340 < 56UL; _fuseiter6340 += 1UL) {
        for (uint64_t _fuseiter6341 = 0UL; _fuseiter6341 < 64UL; _fuseiter6341 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1590_shr[((_fuseiter6340 * 64UL) + _fuseiter6341)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_2210__n_i_2211 * 64UL) + _fuseiter6341)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_2210__n_i_2211 * 64UL) + _fuseiter6341)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_2210__n_i_2211 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter6340 * 64UL) + _fuseiter6341))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_2210__n_i_2211 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter6340 * 64UL) + _fuseiter6341))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1590_shr);
    }
  }
  return true;
}

static bool res2b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_59 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    int32_t* __origouts_1600_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
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
    void* _arg_cache_4 = &__origouts_1600_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_59, A_list, B_list, &__origouts_1600_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter6382 = 0UL; _fuseiter6382 < 56UL; _fuseiter6382 += 1UL) {
      for (uint64_t _fuseiter6383 = 0UL; _fuseiter6383 < 64UL; _fuseiter6383 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_1600_shr[((_fuseiter6382 * 64UL) + _fuseiter6383)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6383]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6383]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((((p_o + 1UL) * 3712UL) + 64UL) + ((_fuseiter6382 * 64UL) + _fuseiter6383))]);
      }
    }
    sc_aligned_free(__stream, __origouts_1600_shr);
  }
  return true;
}

static bool res2b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_57 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_1610_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
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
    void* _arg_cache_5 = &__origouts_1610_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_57, A_list, B_list, &__origouts_1610_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter6417 = 0UL; _fuseiter6417 < 56UL; _fuseiter6417 += 1UL) {
      for (uint64_t _fuseiter6418 = 0UL; _fuseiter6418 < 64UL; _fuseiter6418 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_1610_shr[((_fuseiter6417 * 64UL) + _fuseiter6418)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6418]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6418]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter6417 * 64UL) + _fuseiter6418))]);
      }
    }
    sc_aligned_free(__stream, __origouts_1610_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_2216__n_i_2217 = 0UL; fused_0fused_0k__n_2216__n_i_2217 < 4UL; fused_0fused_0k__n_2216__n_i_2217 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_1620_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0fused_0k__n_2216__n_i_2217 * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_6 = &__origouts_1620_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_1620_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter6452 = 0UL; _fuseiter6452 < 56UL; _fuseiter6452 += 1UL) {
        for (uint64_t _fuseiter6453 = 0UL; _fuseiter6453 < 64UL; _fuseiter6453 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1620_shr[((_fuseiter6452 * 64UL) + _fuseiter6453)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_2216__n_i_2217 * 64UL) + _fuseiter6453)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_2216__n_i_2217 * 64UL) + _fuseiter6453)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_2216__n_i_2217 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter6452 * 64UL) + _fuseiter6453))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_2216__n_i_2217 * 200704UL) + (p_o * 3584UL)) + ((_fuseiter6452 * 64UL) + _fuseiter6453))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1620_shr);
    }
  }
  return true;
}

static bool res2c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_59 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    int32_t* __origouts_1630_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
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
    void* _arg_cache_7 = &__origouts_1630_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_59, A_list, B_list, &__origouts_1630_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter6494 = 0UL; _fuseiter6494 < 56UL; _fuseiter6494 += 1UL) {
      for (uint64_t _fuseiter6495 = 0UL; _fuseiter6495 < 64UL; _fuseiter6495 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_1630_shr[((_fuseiter6494 * 64UL) + _fuseiter6495)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6495]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6495]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((((p_o + 1UL) * 3712UL) + 64UL) + ((_fuseiter6494 * 64UL) + _fuseiter6495))]);
      }
    }
    sc_aligned_free(__stream, __origouts_1630_shr);
  }
  return true;
}

static bool res2c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_57 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[128UL];
    int32_t* __origouts_1640_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
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
    void* _arg_cache_8 = &__origouts_1640_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_57, A_list, B_list, &__origouts_1640_shr[0UL], 1, 64, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter6529 = 0UL; _fuseiter6529 < 56UL; _fuseiter6529 += 1UL) {
      for (uint64_t _fuseiter6530 = 0UL; _fuseiter6530 < 64UL; _fuseiter6530 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_1640_shr[((_fuseiter6529 * 64UL) + _fuseiter6530)]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter6530]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter6530]);
        __cached_3 = (__cached_3 + __cached_5);
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = sc_max(__cached_6, vec_s8x16(0));
        vec_s8x16::store(__cached_7, &__outs_0[((p_o * 3584UL) + ((_fuseiter6529 * 64UL) + _fuseiter6530))]);
      }
    }
    sc_aligned_free(__stream, __origouts_1640_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2222__k_2223 = 0UL; fused_0fused_0n__n_i_2222__k_2223 < 4UL; fused_0fused_0n__n_i_2222__k_2223 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_1650_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2222__k_2223 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0fused_0n__n_i_2222__k_2223 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_9 = &__origouts_1650_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_1650_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter6564 = 0UL; _fuseiter6564 < 56UL; _fuseiter6564 += 1UL) {
        for (uint64_t _fuseiter6565 = 0UL; _fuseiter6565 < 64UL; _fuseiter6565 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1650_shr[((_fuseiter6564 * 64UL) + _fuseiter6565)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2222__k_2223 / 4UL) * 256UL) + ((fused_0fused_0n__n_i_2222__k_2223 % 4UL) * 64UL)) + _fuseiter6565)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2222__k_2223 % 4UL) * 64UL) + _fuseiter6565)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2222__k_2223 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_2222__k_2223 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter6564 * 64UL) + _fuseiter6565))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2222__k_2223 / 4UL) * 802816UL) + (((fused_0fused_0n__n_i_2222__k_2223 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter6564 * 64UL) + _fuseiter6565))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1650_shr);
    }
  }
  return true;
}

static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_59 = *(void**)(__module_data + 32);
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
  for (uint64_t fused_0fused_0n__n_i_2224__k_2225 = 0UL; fused_0fused_0n__n_i_2224__k_2225 < 8UL; fused_0fused_0n__n_i_2224__k_2225 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_1660_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[(((fused_0fused_0n__n_i_2224__k_2225 / 8UL) * 200704UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[(((fused_0fused_0n__n_i_2224__k_2225 % 8UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_10 = &__origouts_1660_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_59, A_list, B_list, &__origouts_1660_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter6605 = 0UL; _fuseiter6605 < 2UL; _fuseiter6605 += 1UL) {
        for (uint64_t _fuseiter6606 = 0UL; _fuseiter6606 < 28UL; _fuseiter6606 += 1UL) {
          for (uint64_t _fuseiter6607 = 0UL; _fuseiter6607 < 64UL; _fuseiter6607 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_1660_shr[((_fuseiter6605 * 1792UL) + ((_fuseiter6606 * 64UL) + _fuseiter6607))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2224__k_2225 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_2224__k_2225 % 8UL) * 64UL)) + _fuseiter6607)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2224__k_2225 % 8UL) * 64UL) + _fuseiter6607)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2224__k_2225 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2224__k_2225 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter6605 * 1792UL) + ((_fuseiter6606 * 64UL) + _fuseiter6607)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1660_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res3a_conv_0_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 215296UL)], 0, 3712UL);
    for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
      memset(&__outs_0[((k * 215296UL) + ((p1 + 1UL) * 3712UL))], 0, 64UL);
      memset(&__outs_0[(((k * 215296UL) + ((p1 + 1UL) * 3712UL)) + 3648UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 215296UL) + 211584UL)], 0, 3712UL);
  }
  for (uint64_t fused_0fused_0n__n_i_2226__k_2227 = 0UL; fused_0fused_0n__n_i_2226__k_2227 < 2UL; fused_0fused_0n__n_i_2226__k_2227 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      for (uint64_t q_i = 0UL; q_i < 2UL; q_i += 1UL) {
        int32_t* __origouts_1670_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
        void** A_list = (void**)&__rescheduled_0[0UL];
        void** B_list = (void**)&__rescheduled_0[64UL];
        for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2226__k_2227 / 2UL) * 802816UL) + ((c * 200704UL) + ((p_o * 3584UL) + (q_i * 1792UL))))];
          A_list[c] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2226__k_2227 % 2UL) * 16384UL) + (c * 4096UL))];
          B_list[c] = __cached_1;
        }
        void* _arg_cache_11 = &__origouts_1670_shr[0UL];
        dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1670_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
        for (uint64_t _fuseiter6634 = 0UL; _fuseiter6634 < 28UL; _fuseiter6634 += 1UL) {
          for (uint64_t _fuseiter6635 = 0UL; _fuseiter6635 < 64UL; _fuseiter6635 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1670_shr[((_fuseiter6634 * 64UL) + _fuseiter6635)]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2226__k_2227 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_2226__k_2227 % 2UL) * 64UL)) + _fuseiter6635)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2226__k_2227 % 2UL) * 64UL) + _fuseiter6635)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[((((fused_0fused_0n__n_i_2226__k_2227 / 2UL) * 430592UL) + (((fused_0fused_0n__n_i_2226__k_2227 % 2UL) * 215296UL) + (((p_o + 1UL) * 3712UL) + (((q_i * 28UL) + 1UL) * 64UL)))) + ((_fuseiter6634 * 64UL) + _fuseiter6635))]);
          }
        }
        sc_aligned_free(__stream, __origouts_1670_shr);
      }
    }
  }
  return true;
}

static bool res3a_conv_1_cast_mul_add_cast_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_63 = *(void**)(__module_data + 48);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  for (uint64_t fused_0fused_0k_o__n_2228__n_i_2229 = 0UL; fused_0fused_0k_o__n_2228__n_i_2229 < 2UL; fused_0fused_0k_o__n_2228__n_i_2229 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[192UL];
      int32_t* __origouts_1680_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[((c_o * 215296UL) + ((((p_o * 2UL) + r) * 3712UL) + (s * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[((fused_0fused_0k_o__n_2228__n_i_2229 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_12 = &__origouts_1680_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_63, A_list, B_list, &__origouts_1680_shr[0UL], 1, 64, 4096, 18, 7, 7, __stream);
      for (uint64_t _fuseiter6669 = 0UL; _fuseiter6669 < 28UL; _fuseiter6669 += 1UL) {
        for (uint64_t _fuseiter6670 = 0UL; _fuseiter6670 < 64UL; _fuseiter6670 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1680_shr[((_fuseiter6669 * 64UL) + _fuseiter6670)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_2228__n_i_2229 * 64UL) + _fuseiter6670)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_2228__n_i_2229 * 64UL) + _fuseiter6670)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0k_o__n_2228__n_i_2229 * 50176UL) + (p_o * 1792UL)) + ((_fuseiter6669 * 64UL) + _fuseiter6670))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1680_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3a_conv_2_cast_mul_add_cast_add_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_65 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_2230__n_i_2231 = 0UL; fused_0fused_0k__n_2230__n_i_2231 < 8UL; fused_0fused_0k__n_2230__n_i_2231 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_1690_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 50176UL) + (p_o * 3584UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0fused_0k__n_2230__n_i_2231 * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_13 = &__origouts_1690_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_65, A_list, B_list, &__origouts_1690_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter6703 = 0UL; _fuseiter6703 < 2UL; _fuseiter6703 += 1UL) {
        for (uint64_t _fuseiter6704 = 0UL; _fuseiter6704 < 28UL; _fuseiter6704 += 1UL) {
          for (uint64_t _fuseiter6705 = 0UL; _fuseiter6705 < 64UL; _fuseiter6705 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1690_shr[((_fuseiter6703 * 1792UL) + ((_fuseiter6704 * 64UL) + _fuseiter6705))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0fused_0k__n_2230__n_i_2231 * 64UL) + _fuseiter6705)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0fused_0k__n_2230__n_i_2231 * 64UL) + _fuseiter6705)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0fused_0k__n_2230__n_i_2231 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter6703 * 1792UL) + ((_fuseiter6704 * 64UL) + _fuseiter6705)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k__n_2230__n_i_2231 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter6703 * 1792UL) + ((_fuseiter6704 * 64UL) + _fuseiter6705)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1690_shr);
    }
  }
  return true;
}

static bool res3b_conv_0_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_67 = *(void**)(__module_data + 64);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_2232__k_2233 = 0UL; fused_0fused_0n__n_i_2232__k_2233 < 2UL; fused_0fused_0n__n_i_2232__k_2233 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_1700_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2232__k_2233 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2232__k_2233 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_14 = &__origouts_1700_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_67, A_list, B_list, &__origouts_1700_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter6746 = 0UL; _fuseiter6746 < 28UL; _fuseiter6746 += 1UL) {
        for (uint64_t _fuseiter6747 = 0UL; _fuseiter6747 < 64UL; _fuseiter6747 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1700_shr[((_fuseiter6746 * 64UL) + _fuseiter6747)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2232__k_2233 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_2232__k_2233 % 2UL) * 64UL)) + _fuseiter6747)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2232__k_2233 % 2UL) * 64UL) + _fuseiter6747)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2232__k_2233 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_2232__k_2233 % 2UL) * 57600UL) + ((p_o + 1UL) * 1920UL))) + 64UL) + ((_fuseiter6746 * 64UL) + _fuseiter6747))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1700_shr);
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
  for (uint64_t fused_0fused_0k_o__n_2234__n_i_2235 = 0UL; fused_0fused_0k_o__n_2234__n_i_2235 < 2UL; fused_0fused_0k_o__n_2234__n_i_2235 += 1UL) {
    int32_t* __origouts_1710_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
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
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_2234__n_i_2235 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_15 = &__origouts_1710_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_1710_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter6780 = 0UL; _fuseiter6780 < 28UL; _fuseiter6780 += 1UL) {
      for (uint64_t _fuseiter6781 = 0UL; _fuseiter6781 < 28UL; _fuseiter6781 += 1UL) {
        for (uint64_t _fuseiter6782 = 0UL; _fuseiter6782 < 64UL; _fuseiter6782 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_1710_shr[((_fuseiter6780 * 1792UL) + ((_fuseiter6781 * 64UL) + _fuseiter6782))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_2234__n_i_2235 * 64UL) + _fuseiter6782)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_2234__n_i_2235 * 64UL) + _fuseiter6782)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_2234__n_i_2235 * 50176UL) + ((_fuseiter6780 * 1792UL) + ((_fuseiter6781 * 64UL) + _fuseiter6782)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1710_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3b_conv_2_cast_mul_add_cast_add_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_70 = *(void**)(__module_data + 72);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2236__k_2237 = 0UL; fused_0fused_0n__n_i_2236__k_2237 < 8UL; fused_0fused_0n__n_i_2236__k_2237 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_1720_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2236__k_2237 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2236__k_2237 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_16 = &__origouts_1720_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_70, A_list, B_list, &__origouts_1720_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter6816 = 0UL; _fuseiter6816 < 28UL; _fuseiter6816 += 1UL) {
        for (uint64_t _fuseiter6817 = 0UL; _fuseiter6817 < 64UL; _fuseiter6817 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1720_shr[((_fuseiter6816 * 64UL) + _fuseiter6817)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2236__k_2237 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_2236__k_2237 % 8UL) * 64UL)) + _fuseiter6817)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2236__k_2237 % 8UL) * 64UL) + _fuseiter6817)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2236__k_2237 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2236__k_2237 % 8UL) * 50176UL) + (p_o * 1792UL))) + ((_fuseiter6816 * 64UL) + _fuseiter6817))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2236__k_2237 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2236__k_2237 % 8UL) * 50176UL) + (p_o * 1792UL))) + ((_fuseiter6816 * 64UL) + _fuseiter6817))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1720_shr);
    }
  }
  return true;
}

static bool res3c_conv_0_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_72 = *(void**)(__module_data + 80);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_2238__k_2239 = 0UL; fused_0fused_0n__n_i_2238__k_2239 < 2UL; fused_0fused_0n__n_i_2238__k_2239 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_1730_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2238__k_2239 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 25088UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2238__k_2239 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_17 = &__origouts_1730_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_72, A_list, B_list, &__origouts_1730_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter6857 = 0UL; _fuseiter6857 < 14UL; _fuseiter6857 += 1UL) {
        for (uint64_t _fuseiter6858 = 0UL; _fuseiter6858 < 28UL; _fuseiter6858 += 1UL) {
          for (uint64_t _fuseiter6859 = 0UL; _fuseiter6859 < 64UL; _fuseiter6859 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1730_shr[((_fuseiter6857 * 1792UL) + ((_fuseiter6858 * 64UL) + _fuseiter6859))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2238__k_2239 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_2238__k_2239 % 2UL) * 64UL)) + _fuseiter6859)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2238__k_2239 % 2UL) * 64UL) + _fuseiter6859)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2238__k_2239 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_2238__k_2239 % 2UL) * 57600UL) + (((p_o * 14UL) + 1UL) * 1920UL))) + 64UL) + ((_fuseiter6857 * 1920UL) + ((_fuseiter6858 * 64UL) + _fuseiter6859)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1730_shr);
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
  for (uint64_t fused_0fused_0k_o__n_2240__n_i_2241 = 0UL; fused_0fused_0k_o__n_2240__n_i_2241 < 2UL; fused_0fused_0k_o__n_2240__n_i_2241 += 1UL) {
    int32_t* __origouts_1740_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
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
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_2240__n_i_2241 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_18 = &__origouts_1740_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_1740_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter6892 = 0UL; _fuseiter6892 < 28UL; _fuseiter6892 += 1UL) {
      for (uint64_t _fuseiter6893 = 0UL; _fuseiter6893 < 28UL; _fuseiter6893 += 1UL) {
        for (uint64_t _fuseiter6894 = 0UL; _fuseiter6894 < 64UL; _fuseiter6894 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_1740_shr[((_fuseiter6892 * 1792UL) + ((_fuseiter6893 * 64UL) + _fuseiter6894))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_2240__n_i_2241 * 64UL) + _fuseiter6894)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_2240__n_i_2241 * 64UL) + _fuseiter6894)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_2240__n_i_2241 * 50176UL) + ((_fuseiter6892 * 1792UL) + ((_fuseiter6893 * 64UL) + _fuseiter6894)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1740_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3c_conv_2_cast_mul_add_cast_add_relu__80(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_74 = *(void**)(__module_data + 88);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2242__k_2243 = 0UL; fused_0fused_0n__n_i_2242__k_2243 < 8UL; fused_0fused_0n__n_i_2242__k_2243 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1750_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2242__k_2243 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 7168UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2242__k_2243 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_19 = &__origouts_1750_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_74, A_list, B_list, &__origouts_1750_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter6927 = 0UL; _fuseiter6927 < 4UL; _fuseiter6927 += 1UL) {
        for (uint64_t _fuseiter6928 = 0UL; _fuseiter6928 < 28UL; _fuseiter6928 += 1UL) {
          for (uint64_t _fuseiter6929 = 0UL; _fuseiter6929 < 64UL; _fuseiter6929 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1750_shr[((_fuseiter6927 * 1792UL) + ((_fuseiter6928 * 64UL) + _fuseiter6929))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2242__k_2243 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_2242__k_2243 % 8UL) * 64UL)) + _fuseiter6929)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2242__k_2243 % 8UL) * 64UL) + _fuseiter6929)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2242__k_2243 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2242__k_2243 % 8UL) * 50176UL) + (p_o * 7168UL))) + ((_fuseiter6927 * 1792UL) + ((_fuseiter6928 * 64UL) + _fuseiter6929)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2242__k_2243 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2242__k_2243 % 8UL) * 50176UL) + (p_o * 7168UL))) + ((_fuseiter6927 * 1792UL) + ((_fuseiter6928 * 64UL) + _fuseiter6929)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1750_shr);
    }
  }
  return true;
}

static bool res3d_conv_0_cast_mul_add_cast_relu__84(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_76 = *(void**)(__module_data + 96);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t k = 0UL; k < 2UL; k += 1UL) {
    memset(&__outs_0[(k * 57600UL)], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[((k * 57600UL) + ((p1 + 1UL) * 1920UL))], 0, 64UL);
      memset(&__outs_0[(((k * 57600UL) + ((p1 + 1UL) * 1920UL)) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((k * 57600UL) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0fused_0n__n_i_2244__k_2245 = 0UL; fused_0fused_0n__n_i_2244__k_2245 < 2UL; fused_0fused_0n__n_i_2244__k_2245 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_1760_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2244__k_2245 / 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2244__k_2245 % 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_20 = &__origouts_1760_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_76, A_list, B_list, &__origouts_1760_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter6969 = 0UL; _fuseiter6969 < 2UL; _fuseiter6969 += 1UL) {
        for (uint64_t _fuseiter6970 = 0UL; _fuseiter6970 < 28UL; _fuseiter6970 += 1UL) {
          for (uint64_t _fuseiter6971 = 0UL; _fuseiter6971 < 64UL; _fuseiter6971 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1760_shr[((_fuseiter6969 * 1792UL) + ((_fuseiter6970 * 64UL) + _fuseiter6971))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2244__k_2245 / 2UL) * 128UL) + ((fused_0fused_0n__n_i_2244__k_2245 % 2UL) * 64UL)) + _fuseiter6971)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2244__k_2245 % 2UL) * 64UL) + _fuseiter6971)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2244__k_2245 / 2UL) * 115200UL) + (((fused_0fused_0n__n_i_2244__k_2245 % 2UL) * 57600UL) + (((p_o * 2UL) + 1UL) * 1920UL))) + 64UL) + ((_fuseiter6969 * 1920UL) + ((_fuseiter6970 * 64UL) + _fuseiter6971)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1760_shr);
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
  for (uint64_t fused_0fused_0k_o__n_2246__n_i_2247 = 0UL; fused_0fused_0k_o__n_2246__n_i_2247 < 2UL; fused_0fused_0k_o__n_2246__n_i_2247 += 1UL) {
    int32_t* __origouts_1770_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
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
            __cached_3 = &__ins_1[((fused_0fused_0k_o__n_2246__n_i_2247 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_3;
          }
        }
      }
      int32_t __cached_4;
      __cached_4 = conv_os_acc_size[o_o];
      void* _arg_cache_21 = &__origouts_1770_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))];
      dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_1770_shr[(uint64_t)(((__cached_4 / 28) * 1792) + ((__cached_4 % 28) * 64))], 1, 64, 4096, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter7004 = 0UL; _fuseiter7004 < 28UL; _fuseiter7004 += 1UL) {
      for (uint64_t _fuseiter7005 = 0UL; _fuseiter7005 < 28UL; _fuseiter7005 += 1UL) {
        for (uint64_t _fuseiter7006 = 0UL; _fuseiter7006 < 64UL; _fuseiter7006 += 16UL) {
          vec_s32x16 __cached_5;
          __cached_5 = vec_s32x16::load(&__origouts_1770_shr[((_fuseiter7004 * 1792UL) + ((_fuseiter7005 * 64UL) + _fuseiter7006))]);
          vec_f32x16 __cached_6;
          __cached_6 = (vec_f32x16)(__cached_5);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_2[((fused_0fused_0k_o__n_2246__n_i_2247 * 64UL) + _fuseiter7006)]);
          __cached_6 = (__cached_6 * __cached_7);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_3[((fused_0fused_0k_o__n_2246__n_i_2247 * 64UL) + _fuseiter7006)]);
          __cached_6 = (__cached_6 + __cached_8);
          vec_s8x16 __cached_9;
          __cached_9 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_6));
          vec_s8x16 __cached_10;
          __cached_10 = sc_max(__cached_9, vec_s8x16(0));
          vec_s8x16::store(__cached_10, &__outs_0[((fused_0fused_0k_o__n_2246__n_i_2247 * 50176UL) + ((_fuseiter7004 * 1792UL) + ((_fuseiter7005 * 64UL) + _fuseiter7006)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1770_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3d_conv_2_cast_mul_add_cast_add_relu__93(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_65 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2248__k_2249 = 0UL; fused_0fused_0n__n_i_2248__k_2249 < 8UL; fused_0fused_0n__n_i_2248__k_2249 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_1780_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2248__k_2249 / 8UL) * 100352UL) + ((c * 50176UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2248__k_2249 % 8UL) * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_22 = &__origouts_1780_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_65, A_list, B_list, &__origouts_1780_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter7039 = 0UL; _fuseiter7039 < 2UL; _fuseiter7039 += 1UL) {
        for (uint64_t _fuseiter7040 = 0UL; _fuseiter7040 < 28UL; _fuseiter7040 += 1UL) {
          for (uint64_t _fuseiter7041 = 0UL; _fuseiter7041 < 64UL; _fuseiter7041 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1780_shr[((_fuseiter7039 * 1792UL) + ((_fuseiter7040 * 64UL) + _fuseiter7041))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0fused_0n__n_i_2248__k_2249 / 8UL) * 512UL) + ((fused_0fused_0n__n_i_2248__k_2249 % 8UL) * 64UL)) + _fuseiter7041)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2248__k_2249 % 8UL) * 64UL) + _fuseiter7041)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2248__k_2249 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2248__k_2249 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter7039 * 1792UL) + ((_fuseiter7040 * 64UL) + _fuseiter7041)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2248__k_2249 / 8UL) * 401408UL) + (((fused_0fused_0n__n_i_2248__k_2249 % 8UL) * 50176UL) + (p_o * 3584UL))) + ((_fuseiter7039 * 1792UL) + ((_fuseiter7040 * 64UL) + _fuseiter7041)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1780_shr);
    }
  }
  return true;
}

static bool batchwise_4_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 4UL; __batchwise_iter_0 += 1UL) {
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

extern "C" void main_entry_53(int8_t* __restrict__ buffer_61, int8_t* __restrict__ buffer_57, int8_t* __restrict__ buffer_56, float* __restrict__ buffer_55, float* __restrict__ buffer_54, int8_t* __restrict__ buffer_53, float* __restrict__ buffer_52, float* __restrict__ buffer_51, int8_t* __restrict__ buffer_50, float* __restrict__ buffer_49, float* __restrict__ buffer_48, int8_t* __restrict__ buffer_47, float* __restrict__ buffer_46, float* __restrict__ buffer_45, int8_t* __restrict__ buffer_44, float* __restrict__ buffer_43, float* __restrict__ buffer_42, int8_t* __restrict__ buffer_41, float* __restrict__ buffer_40, float* __restrict__ buffer_39, int8_t* __restrict__ buffer_38, float* __restrict__ buffer_37, float* __restrict__ buffer_36, int8_t* __restrict__ buffer_35, float* __restrict__ buffer_34, float* __restrict__ buffer_33, int8_t* __restrict__ buffer_32, float* __restrict__ buffer_31, float* __restrict__ buffer_30, int8_t* __restrict__ buffer_29, float* __restrict__ buffer_28, float* __restrict__ buffer_27, int8_t* __restrict__ buffer_26, float* __restrict__ buffer_25, float* __restrict__ buffer_24, int8_t* __restrict__ buffer_23, float* __restrict__ buffer_22, float* __restrict__ buffer_21, int8_t* __restrict__ buffer_20, float* __restrict__ buffer_19, float* __restrict__ buffer_18, int8_t* __restrict__ buffer_17, float* __restrict__ buffer_16, float* __restrict__ buffer_15, int8_t* __restrict__ buffer_14, float* __restrict__ buffer_13, float* __restrict__ buffer_12, int8_t* __restrict__ buffer_11, float* __restrict__ buffer_10, float* __restrict__ buffer_9, int8_t* __restrict__ buffer_8, float* __restrict__ buffer_7, float* __restrict__ buffer_6, int8_t* __restrict__ buffer_5, float* __restrict__ buffer_4, float* __restrict__ buffer_3, int8_t* __restrict__ buffer_2, float* __restrict__ buffer_1, float* __restrict__ buffer_0) noexcept{
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
  void*& __sc_kernel_cache_67 = *(void**)(__module_data + 64);
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
  for (uint64_t fused_0fused_0k__n_2250__n_i_2251 = 0UL; fused_0fused_0k__n_2250__n_i_2251 < 32UL; fused_0fused_0k__n_2250__n_i_2251 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1790_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[(((fused_0fused_0k__n_2250__n_i_2251 % 2UL) * 100352UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[(((fused_0fused_0k__n_2250__n_i_2251 / 2UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_23 = &__origouts_1790_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_67, A_list, B_list, &__origouts_1790_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter7081 = 0UL; _fuseiter7081 < 2UL; _fuseiter7081 += 1UL) {
        for (uint64_t _fuseiter7082 = 0UL; _fuseiter7082 < 14UL; _fuseiter7082 += 1UL) {
          for (uint64_t _fuseiter7083 = 0UL; _fuseiter7083 < 64UL; _fuseiter7083 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_1790_shr[((_fuseiter7081 * 896UL) + ((_fuseiter7082 * 64UL) + _fuseiter7083))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_2250__n_i_2251 / 2UL) * 64UL) + _fuseiter7083)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_2250__n_i_2251 / 2UL) * 64UL) + _fuseiter7083)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_2250__n_i_2251 % 2UL) * 200704UL) + (((fused_0fused_0k__n_2250__n_i_2251 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter7081 * 896UL) + ((_fuseiter7082 * 64UL) + _fuseiter7083)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1790_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res4a_conv_0_cast_mul_add_cast_relu__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_76 = *(void**)(__module_data + 96);
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
  for (uint64_t fused_0fused_0n__n_i_2252__k_2253 = 0UL; fused_0fused_0n__n_i_2252__k_2253 < 8UL; fused_0fused_0n__n_i_2252__k_2253 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_1800_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2252__k_2253 / 8UL) * 802816UL) + ((((fused_0fused_0n__n_i_2252__k_2253 / 4UL) % 2UL) * 401408UL) + ((c * 50176UL) + (p_o * 3584UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2252__k_2253 % 4UL) * 32768UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_24 = &__origouts_1800_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_76, A_list, B_list, &__origouts_1800_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
      for (uint64_t _fuseiter7109 = 0UL; _fuseiter7109 < 2UL; _fuseiter7109 += 1UL) {
        for (uint64_t _fuseiter7110 = 0UL; _fuseiter7110 < 28UL; _fuseiter7110 += 1UL) {
          for (uint64_t _fuseiter7111 = 0UL; _fuseiter7111 < 64UL; _fuseiter7111 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1800_shr[((_fuseiter7109 * 1792UL) + ((_fuseiter7110 * 64UL) + _fuseiter7111))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2252__k_2253 % 4UL) * 64UL) + _fuseiter7111)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2252__k_2253 % 4UL) * 64UL) + _fuseiter7111)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2252__k_2253 / 8UL) * 460800UL) + ((((fused_0fused_0n__n_i_2252__k_2253 / 4UL) % 2UL) * 230400UL) + (((fused_0fused_0n__n_i_2252__k_2253 % 4UL) * 57600UL) + (((p_o * 2UL) + 1UL) * 1920UL)))) + 64UL) + ((_fuseiter7109 * 1920UL) + ((_fuseiter7110 * 64UL) + _fuseiter7111)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1800_shr);
    }
  }
  return true;
}

static bool res4a_conv_1_cast_mul_add_cast_relu__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_78 = *(void**)(__module_data + 104);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  for (uint64_t fused_0fused_0n__n_i_2254__k_o_2255 = 0UL; fused_0fused_0n__n_i_2254__k_o_2255 < 8UL; fused_0fused_0n__n_i_2254__k_o_2255 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[320UL];
      for (uint64_t p_i = 0UL; p_i < 2UL; p_i += 1UL) {
        int32_t* __origouts_1810_shr = (int32_t*)sc_aligned_malloc(__stream, 3584UL);
        for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
          for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
            for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
              void* __cached_0;
              __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2254__k_o_2255 / 8UL) * 460800UL) + ((((fused_0fused_0n__n_i_2254__k_o_2255 / 4UL) % 2UL) * 230400UL) + ((c_o * 57600UL) + ((((((p_o * 2UL) + p_i) * 2UL) + r) * 1920UL) + (s * 64UL)))))];
              A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
              void* __cached_1;
              __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2254__k_o_2255 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
              B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
            }
          }
        }
        void* _arg_cache_25 = &__origouts_1810_shr[0UL];
        dnnl_brgemm_list_call(__sc_kernel_cache_78, A_list, B_list, &__origouts_1810_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
        for (uint64_t _fuseiter7145 = 0UL; _fuseiter7145 < 14UL; _fuseiter7145 += 1UL) {
          for (uint64_t _fuseiter7146 = 0UL; _fuseiter7146 < 64UL; _fuseiter7146 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1810_shr[((_fuseiter7145 * 64UL) + _fuseiter7146)]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2254__k_o_2255 % 4UL) * 64UL) + _fuseiter7146)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2254__k_o_2255 % 4UL) * 64UL) + _fuseiter7146)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[((((fused_0fused_0n__n_i_2254__k_o_2255 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_2254__k_o_2255 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_2254__k_o_2255 % 4UL) * 12544UL) + (((p_o * 2UL) + p_i) * 896UL)))) + ((_fuseiter7145 * 64UL) + _fuseiter7146))]);
          }
        }
        sc_aligned_free(__stream, __origouts_1810_shr);
      }
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4a_conv_2_cast_mul_add_cast_add_relu__16(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_2256__n_i_2257 = 0UL; fused_0fused_0k__n_2256__n_i_2257 < 32UL; fused_0fused_0k__n_2256__n_i_2257 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1820_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_2256__n_i_2257 % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_2256__n_i_2257 / 2UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_26 = &__origouts_1820_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1820_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7179 = 0UL; _fuseiter7179 < 2UL; _fuseiter7179 += 1UL) {
        for (uint64_t _fuseiter7180 = 0UL; _fuseiter7180 < 14UL; _fuseiter7180 += 1UL) {
          for (uint64_t _fuseiter7181 = 0UL; _fuseiter7181 < 64UL; _fuseiter7181 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1820_shr[((_fuseiter7179 * 896UL) + ((_fuseiter7180 * 64UL) + _fuseiter7181))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_2256__n_i_2257 / 2UL) * 64UL) + _fuseiter7181)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_2256__n_i_2257 / 2UL) * 64UL) + _fuseiter7181)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_2256__n_i_2257 % 2UL) * 200704UL) + (((fused_0fused_0k__n_2256__n_i_2257 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter7179 * 896UL) + ((_fuseiter7180 * 64UL) + _fuseiter7181)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_2256__n_i_2257 % 2UL) * 200704UL) + (((fused_0fused_0k__n_2256__n_i_2257 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter7179 * 896UL) + ((_fuseiter7180 * 64UL) + _fuseiter7181)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1820_shr);
    }
  }
  return true;
}

static bool res4b_conv_0_cast_mul_add_cast_relu__20(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
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
  for (uint64_t fused_0fused_0n__n_i_2258__k_2259 = 0UL; fused_0fused_0n__n_i_2258__k_2259 < 8UL; fused_0fused_0n__n_i_2258__k_2259 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1830_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2258__k_2259 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_2258__k_2259 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2258__k_2259 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_27 = &__origouts_1830_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_1830_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7221 = 0UL; _fuseiter7221 < 2UL; _fuseiter7221 += 1UL) {
        for (uint64_t _fuseiter7222 = 0UL; _fuseiter7222 < 14UL; _fuseiter7222 += 1UL) {
          for (uint64_t _fuseiter7223 = 0UL; _fuseiter7223 < 64UL; _fuseiter7223 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1830_shr[((_fuseiter7221 * 896UL) + ((_fuseiter7222 * 64UL) + _fuseiter7223))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2258__k_2259 % 4UL) * 64UL) + _fuseiter7223)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2258__k_2259 % 4UL) * 64UL) + _fuseiter7223)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2258__k_2259 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2258__k_2259 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_2258__k_2259 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter7221 * 1024UL) + ((_fuseiter7222 * 64UL) + _fuseiter7223)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1830_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_1_cast_mul_add_cast_relu__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_2260__k_o_2261 = 0UL; fused_0fused_0n__n_i_2260__k_o_2261 < 8UL; fused_0fused_0n__n_i_2260__k_o_2261 += 1UL) {
    int32_t* __origouts_1840_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_2260__k_o_2261 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2260__k_o_2261 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_2260__k_o_2261 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_28 = &__origouts_1840_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_1840_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter7256 = 0UL; _fuseiter7256 < 14UL; _fuseiter7256 += 1UL) {
      for (uint64_t _fuseiter7257 = 0UL; _fuseiter7257 < 14UL; _fuseiter7257 += 1UL) {
        for (uint64_t _fuseiter7258 = 0UL; _fuseiter7258 < 64UL; _fuseiter7258 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1840_shr[((_fuseiter7256 * 896UL) + ((_fuseiter7257 * 64UL) + _fuseiter7258))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2260__k_o_2261 % 4UL) * 64UL) + _fuseiter7258)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2260__k_o_2261 % 4UL) * 64UL) + _fuseiter7258)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_2260__k_o_2261 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_2260__k_o_2261 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_2260__k_o_2261 % 4UL) * 12544UL) + ((_fuseiter7256 * 896UL) + ((_fuseiter7257 * 64UL) + _fuseiter7258)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1840_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_2_cast_mul_add_cast_add_relu__28(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2262__k_2263 = 0UL; fused_0fused_0n__n_i_2262__k_2263 < 32UL; fused_0fused_0n__n_i_2262__k_2263 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1850_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2262__k_2263 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_2262__k_2263 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2262__k_2263 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_29 = &__origouts_1850_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1850_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7291 = 0UL; _fuseiter7291 < 2UL; _fuseiter7291 += 1UL) {
        for (uint64_t _fuseiter7292 = 0UL; _fuseiter7292 < 14UL; _fuseiter7292 += 1UL) {
          for (uint64_t _fuseiter7293 = 0UL; _fuseiter7293 < 64UL; _fuseiter7293 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1850_shr[((_fuseiter7291 * 896UL) + ((_fuseiter7292 * 64UL) + _fuseiter7293))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2262__k_2263 % 16UL) * 64UL) + _fuseiter7293)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2262__k_2263 % 16UL) * 64UL) + _fuseiter7293)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2262__k_2263 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2262__k_2263 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2262__k_2263 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7291 * 896UL) + ((_fuseiter7292 * 64UL) + _fuseiter7293)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2262__k_2263 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2262__k_2263 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2262__k_2263 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7291 * 896UL) + ((_fuseiter7292 * 64UL) + _fuseiter7293)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1850_shr);
    }
  }
  return true;
}

static bool res4c_conv_0_cast_mul_add_cast_relu__32(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
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
  for (uint64_t fused_0fused_0n__n_i_2264__k_2265 = 0UL; fused_0fused_0n__n_i_2264__k_2265 < 8UL; fused_0fused_0n__n_i_2264__k_2265 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1860_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2264__k_2265 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_2264__k_2265 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2264__k_2265 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_30 = &__origouts_1860_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_1860_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7333 = 0UL; _fuseiter7333 < 2UL; _fuseiter7333 += 1UL) {
        for (uint64_t _fuseiter7334 = 0UL; _fuseiter7334 < 14UL; _fuseiter7334 += 1UL) {
          for (uint64_t _fuseiter7335 = 0UL; _fuseiter7335 < 64UL; _fuseiter7335 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1860_shr[((_fuseiter7333 * 896UL) + ((_fuseiter7334 * 64UL) + _fuseiter7335))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2264__k_2265 % 4UL) * 64UL) + _fuseiter7335)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2264__k_2265 % 4UL) * 64UL) + _fuseiter7335)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2264__k_2265 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2264__k_2265 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_2264__k_2265 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter7333 * 1024UL) + ((_fuseiter7334 * 64UL) + _fuseiter7335)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1860_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_1_cast_mul_add_cast_relu__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_2266__k_o_2267 = 0UL; fused_0fused_0n__n_i_2266__k_o_2267 < 8UL; fused_0fused_0n__n_i_2266__k_o_2267 += 1UL) {
    int32_t* __origouts_1870_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_2266__k_o_2267 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2266__k_o_2267 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_2266__k_o_2267 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_31 = &__origouts_1870_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_1870_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter7368 = 0UL; _fuseiter7368 < 14UL; _fuseiter7368 += 1UL) {
      for (uint64_t _fuseiter7369 = 0UL; _fuseiter7369 < 14UL; _fuseiter7369 += 1UL) {
        for (uint64_t _fuseiter7370 = 0UL; _fuseiter7370 < 64UL; _fuseiter7370 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1870_shr[((_fuseiter7368 * 896UL) + ((_fuseiter7369 * 64UL) + _fuseiter7370))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2266__k_o_2267 % 4UL) * 64UL) + _fuseiter7370)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2266__k_o_2267 % 4UL) * 64UL) + _fuseiter7370)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_2266__k_o_2267 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_2266__k_o_2267 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_2266__k_o_2267 % 4UL) * 12544UL) + ((_fuseiter7368 * 896UL) + ((_fuseiter7369 * 64UL) + _fuseiter7370)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1870_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_2_cast_mul_add_cast_add_relu__40(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2268__k_2269 = 0UL; fused_0fused_0n__n_i_2268__k_2269 < 32UL; fused_0fused_0n__n_i_2268__k_2269 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1880_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2268__k_2269 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_2268__k_2269 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2268__k_2269 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_32 = &__origouts_1880_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1880_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7403 = 0UL; _fuseiter7403 < 2UL; _fuseiter7403 += 1UL) {
        for (uint64_t _fuseiter7404 = 0UL; _fuseiter7404 < 14UL; _fuseiter7404 += 1UL) {
          for (uint64_t _fuseiter7405 = 0UL; _fuseiter7405 < 64UL; _fuseiter7405 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1880_shr[((_fuseiter7403 * 896UL) + ((_fuseiter7404 * 64UL) + _fuseiter7405))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2268__k_2269 % 16UL) * 64UL) + _fuseiter7405)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2268__k_2269 % 16UL) * 64UL) + _fuseiter7405)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2268__k_2269 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2268__k_2269 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2268__k_2269 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7403 * 896UL) + ((_fuseiter7404 * 64UL) + _fuseiter7405)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2268__k_2269 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2268__k_2269 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2268__k_2269 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7403 * 896UL) + ((_fuseiter7404 * 64UL) + _fuseiter7405)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1880_shr);
    }
  }
  return true;
}

static bool res4d_conv_0_cast_mul_add_cast_relu__44(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
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
  for (uint64_t fused_0fused_0n__n_i_2270__k_2271 = 0UL; fused_0fused_0n__n_i_2270__k_2271 < 8UL; fused_0fused_0n__n_i_2270__k_2271 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1890_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2270__k_2271 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_2270__k_2271 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2270__k_2271 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_33 = &__origouts_1890_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_1890_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7445 = 0UL; _fuseiter7445 < 2UL; _fuseiter7445 += 1UL) {
        for (uint64_t _fuseiter7446 = 0UL; _fuseiter7446 < 14UL; _fuseiter7446 += 1UL) {
          for (uint64_t _fuseiter7447 = 0UL; _fuseiter7447 < 64UL; _fuseiter7447 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1890_shr[((_fuseiter7445 * 896UL) + ((_fuseiter7446 * 64UL) + _fuseiter7447))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2270__k_2271 % 4UL) * 64UL) + _fuseiter7447)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2270__k_2271 % 4UL) * 64UL) + _fuseiter7447)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2270__k_2271 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2270__k_2271 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_2270__k_2271 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter7445 * 1024UL) + ((_fuseiter7446 * 64UL) + _fuseiter7447)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1890_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_1_cast_mul_add_cast_relu__48(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_2272__k_o_2273 = 0UL; fused_0fused_0n__n_i_2272__k_o_2273 < 8UL; fused_0fused_0n__n_i_2272__k_o_2273 += 1UL) {
    int32_t* __origouts_1900_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_2272__k_o_2273 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2272__k_o_2273 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_2272__k_o_2273 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_34 = &__origouts_1900_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_1900_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter7480 = 0UL; _fuseiter7480 < 14UL; _fuseiter7480 += 1UL) {
      for (uint64_t _fuseiter7481 = 0UL; _fuseiter7481 < 14UL; _fuseiter7481 += 1UL) {
        for (uint64_t _fuseiter7482 = 0UL; _fuseiter7482 < 64UL; _fuseiter7482 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1900_shr[((_fuseiter7480 * 896UL) + ((_fuseiter7481 * 64UL) + _fuseiter7482))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2272__k_o_2273 % 4UL) * 64UL) + _fuseiter7482)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2272__k_o_2273 % 4UL) * 64UL) + _fuseiter7482)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_2272__k_o_2273 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_2272__k_o_2273 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_2272__k_o_2273 % 4UL) * 12544UL) + ((_fuseiter7480 * 896UL) + ((_fuseiter7481 * 64UL) + _fuseiter7482)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1900_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_2_cast_mul_add_cast_add_relu__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2274__k_2275 = 0UL; fused_0fused_0n__n_i_2274__k_2275 < 32UL; fused_0fused_0n__n_i_2274__k_2275 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1910_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2274__k_2275 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_2274__k_2275 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2274__k_2275 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_35 = &__origouts_1910_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1910_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7515 = 0UL; _fuseiter7515 < 2UL; _fuseiter7515 += 1UL) {
        for (uint64_t _fuseiter7516 = 0UL; _fuseiter7516 < 14UL; _fuseiter7516 += 1UL) {
          for (uint64_t _fuseiter7517 = 0UL; _fuseiter7517 < 64UL; _fuseiter7517 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1910_shr[((_fuseiter7515 * 896UL) + ((_fuseiter7516 * 64UL) + _fuseiter7517))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2274__k_2275 % 16UL) * 64UL) + _fuseiter7517)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2274__k_2275 % 16UL) * 64UL) + _fuseiter7517)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2274__k_2275 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2274__k_2275 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2274__k_2275 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7515 * 896UL) + ((_fuseiter7516 * 64UL) + _fuseiter7517)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2274__k_2275 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2274__k_2275 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2274__k_2275 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7515 * 896UL) + ((_fuseiter7516 * 64UL) + _fuseiter7517)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1910_shr);
    }
  }
  return true;
}

static bool res4e_conv_0_cast_mul_add_cast_relu__56(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
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
  for (uint64_t fused_0fused_0n__n_i_2276__k_2277 = 0UL; fused_0fused_0n__n_i_2276__k_2277 < 8UL; fused_0fused_0n__n_i_2276__k_2277 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1920_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2276__k_2277 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_2276__k_2277 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2276__k_2277 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_36 = &__origouts_1920_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_1920_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7557 = 0UL; _fuseiter7557 < 2UL; _fuseiter7557 += 1UL) {
        for (uint64_t _fuseiter7558 = 0UL; _fuseiter7558 < 14UL; _fuseiter7558 += 1UL) {
          for (uint64_t _fuseiter7559 = 0UL; _fuseiter7559 < 64UL; _fuseiter7559 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1920_shr[((_fuseiter7557 * 896UL) + ((_fuseiter7558 * 64UL) + _fuseiter7559))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2276__k_2277 % 4UL) * 64UL) + _fuseiter7559)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2276__k_2277 % 4UL) * 64UL) + _fuseiter7559)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2276__k_2277 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2276__k_2277 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_2276__k_2277 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter7557 * 1024UL) + ((_fuseiter7558 * 64UL) + _fuseiter7559)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1920_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_1_cast_mul_add_cast_relu__60(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0n__n_i_2278__k_o_2279 = 0UL; fused_0fused_0n__n_i_2278__k_o_2279 < 8UL; fused_0fused_0n__n_i_2278__k_o_2279 += 1UL) {
    int32_t* __origouts_1930_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0n__n_i_2278__k_o_2279 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2278__k_o_2279 / 4UL) % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0n__n_i_2278__k_o_2279 % 4UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_37 = &__origouts_1930_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_1930_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter7592 = 0UL; _fuseiter7592 < 14UL; _fuseiter7592 += 1UL) {
      for (uint64_t _fuseiter7593 = 0UL; _fuseiter7593 < 14UL; _fuseiter7593 += 1UL) {
        for (uint64_t _fuseiter7594 = 0UL; _fuseiter7594 < 64UL; _fuseiter7594 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1930_shr[((_fuseiter7592 * 896UL) + ((_fuseiter7593 * 64UL) + _fuseiter7594))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2278__k_o_2279 % 4UL) * 64UL) + _fuseiter7594)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2278__k_o_2279 % 4UL) * 64UL) + _fuseiter7594)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_2278__k_o_2279 / 8UL) * 100352UL) + ((((fused_0fused_0n__n_i_2278__k_o_2279 / 4UL) % 2UL) * 50176UL) + (((fused_0fused_0n__n_i_2278__k_o_2279 % 4UL) * 12544UL) + ((_fuseiter7592 * 896UL) + ((_fuseiter7593 * 64UL) + _fuseiter7594)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1930_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_2_cast_mul_add_cast_add_relu__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0n__n_i_2280__k_2281 = 0UL; fused_0fused_0n__n_i_2280__k_2281 < 32UL; fused_0fused_0n__n_i_2280__k_2281 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1940_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2280__k_2281 / 32UL) * 100352UL) + ((((fused_0fused_0n__n_i_2280__k_2281 / 16UL) % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2280__k_2281 % 16UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_38 = &__origouts_1940_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1940_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7627 = 0UL; _fuseiter7627 < 2UL; _fuseiter7627 += 1UL) {
        for (uint64_t _fuseiter7628 = 0UL; _fuseiter7628 < 14UL; _fuseiter7628 += 1UL) {
          for (uint64_t _fuseiter7629 = 0UL; _fuseiter7629 < 64UL; _fuseiter7629 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1940_shr[((_fuseiter7627 * 896UL) + ((_fuseiter7628 * 64UL) + _fuseiter7629))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2280__k_2281 % 16UL) * 64UL) + _fuseiter7629)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2280__k_2281 % 16UL) * 64UL) + _fuseiter7629)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2280__k_2281 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2280__k_2281 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2280__k_2281 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7627 * 896UL) + ((_fuseiter7628 * 64UL) + _fuseiter7629)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2280__k_2281 / 32UL) * 401408UL) + ((((fused_0fused_0n__n_i_2280__k_2281 / 16UL) % 2UL) * 200704UL) + (((fused_0fused_0n__n_i_2280__k_2281 % 16UL) * 12544UL) + (p_o * 1792UL)))) + ((_fuseiter7627 * 896UL) + ((_fuseiter7628 * 64UL) + _fuseiter7629)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1940_shr);
    }
  }
  return true;
}

static bool res4f_conv_0_cast_mul_add_cast_relu__68(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
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
  for (uint64_t fused_0fused_0n__n_i_2282__k_2283 = 0UL; fused_0fused_0n__n_i_2282__k_2283 < 8UL; fused_0fused_0n__n_i_2282__k_2283 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1950_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2282__k_2283 / 8UL) * 401408UL) + ((((fused_0fused_0n__n_i_2282__k_2283 / 4UL) % 2UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL))))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2282__k_2283 % 4UL) * 65536UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_39 = &__origouts_1950_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_1950_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7669 = 0UL; _fuseiter7669 < 2UL; _fuseiter7669 += 1UL) {
        for (uint64_t _fuseiter7670 = 0UL; _fuseiter7670 < 14UL; _fuseiter7670 += 1UL) {
          for (uint64_t _fuseiter7671 = 0UL; _fuseiter7671 < 64UL; _fuseiter7671 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1950_shr[((_fuseiter7669 * 896UL) + ((_fuseiter7670 * 64UL) + _fuseiter7671))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2282__k_2283 % 4UL) * 64UL) + _fuseiter7671)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2282__k_2283 % 4UL) * 64UL) + _fuseiter7671)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2282__k_2283 / 8UL) * 131072UL) + ((((fused_0fused_0n__n_i_2282__k_2283 / 4UL) % 2UL) * 65536UL) + (((fused_0fused_0n__n_i_2282__k_2283 % 4UL) * 16384UL) + (((p_o * 2UL) + 1UL) * 1024UL)))) + 64UL) + ((_fuseiter7669 * 1024UL) + ((_fuseiter7670 * 64UL) + _fuseiter7671)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1950_shr);
    }
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_1_cast_mul_add_cast_relu__72(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 640UL);
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0k_o__n_2284__n_i_2285 = 0UL; fused_0fused_0k_o__n_2284__n_i_2285 < 8UL; fused_0fused_0k_o__n_2284__n_i_2285 += 1UL) {
    int32_t* __origouts_1960_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[(((fused_0fused_0k_o__n_2284__n_i_2285 % 2UL) * 65536UL) + ((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[(((fused_0fused_0k_o__n_2284__n_i_2285 / 2UL) * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_40 = &__origouts_1960_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_1960_shr[0UL], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter7704 = 0UL; _fuseiter7704 < 14UL; _fuseiter7704 += 1UL) {
      for (uint64_t _fuseiter7705 = 0UL; _fuseiter7705 < 14UL; _fuseiter7705 += 1UL) {
        for (uint64_t _fuseiter7706 = 0UL; _fuseiter7706 < 64UL; _fuseiter7706 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_1960_shr[((_fuseiter7704 * 896UL) + ((_fuseiter7705 * 64UL) + _fuseiter7706))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k_o__n_2284__n_i_2285 / 2UL) * 64UL) + _fuseiter7706)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k_o__n_2284__n_i_2285 / 2UL) * 64UL) + _fuseiter7706)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0k_o__n_2284__n_i_2285 % 2UL) * 50176UL) + (((fused_0fused_0k_o__n_2284__n_i_2285 / 2UL) * 12544UL) + ((_fuseiter7704 * 896UL) + ((_fuseiter7705 * 64UL) + _fuseiter7706))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1960_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_2_cast_mul_add_cast_add_relu__77(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0fused_0k__n_2286__n_i_2287 = 0UL; fused_0fused_0k__n_2286__n_i_2287 < 32UL; fused_0fused_0k__n_2286__n_i_2287 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1970_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_2286__n_i_2287 % 2UL) * 50176UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_2286__n_i_2287 / 2UL) * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_41 = &__origouts_1970_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_61, A_list, B_list, &__origouts_1970_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter7739 = 0UL; _fuseiter7739 < 2UL; _fuseiter7739 += 1UL) {
        for (uint64_t _fuseiter7740 = 0UL; _fuseiter7740 < 14UL; _fuseiter7740 += 1UL) {
          for (uint64_t _fuseiter7741 = 0UL; _fuseiter7741 < 64UL; _fuseiter7741 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1970_shr[((_fuseiter7739 * 896UL) + ((_fuseiter7740 * 64UL) + _fuseiter7741))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_2286__n_i_2287 / 2UL) * 64UL) + _fuseiter7741)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_2286__n_i_2287 / 2UL) * 64UL) + _fuseiter7741)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_2286__n_i_2287 % 2UL) * 200704UL) + (((fused_0fused_0k__n_2286__n_i_2287 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter7739 * 896UL) + ((_fuseiter7740 * 64UL) + _fuseiter7741)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_2286__n_i_2287 % 2UL) * 200704UL) + (((fused_0fused_0k__n_2286__n_i_2287 / 2UL) * 12544UL) + (p_o * 1792UL))) + ((_fuseiter7739 * 896UL) + ((_fuseiter7740 * 64UL) + _fuseiter7741)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1970_shr);
    }
  }
  return true;
}

static bool res5a_conv_b_cast_mul_add_cast__683(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_86 = *(void**)(__module_data + 120);
  int8_t* input_tmp = (int8_t*)sc_aligned_malloc(__stream, 401408UL);
  for (uint64_t n = 0UL; n < 8UL; n += 1UL) {
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
  for (uint64_t fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 = 0UL; fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 < 32UL; fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_1980_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[128UL];
    for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
      void* __cached_2;
      __cached_2 = &input_tmp[(((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 % 8UL) * 50176UL) + (c * 3136UL))];
      A_list[c] = __cached_2;
      void* __cached_3;
      __cached_3 = &__ins_1[(((((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 8UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 16UL) * 2UL)) * 524288UL) + (c * 32768UL))];
      B_list[c] = __cached_3;
    }
    void* _arg_cache_42 = &__origouts_1980_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_86, A_list, B_list, &__origouts_1980_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
    for (uint64_t _fuseiter7781 = 0UL; _fuseiter7781 < 7UL; _fuseiter7781 += 1UL) {
      for (uint64_t _fuseiter7782 = 0UL; _fuseiter7782 < 7UL; _fuseiter7782 += 1UL) {
        for (uint64_t _fuseiter7783 = 0UL; _fuseiter7783 < 512UL; _fuseiter7783 += 16UL) {
          vec_s32x16 __cached_4;
          __cached_4 = vec_s32x16::load(&__origouts_1980_shr[((_fuseiter7781 * 3584UL) + ((_fuseiter7782 * 512UL) + _fuseiter7783))]);
          vec_f32x16 __cached_5;
          __cached_5 = (vec_f32x16)(__cached_4);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_2[(((((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 8UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 16UL) * 2UL)) * 512UL) + _fuseiter7783)]);
          __cached_5 = (__cached_5 * __cached_6);
          vec_f32x16 __cached_7;
          __cached_7 = vec_f32x16::load(&__ins_3[(((((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 8UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 16UL) * 2UL)) * 512UL) + _fuseiter7783)]);
          __cached_5 = (__cached_5 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 % 8UL) * 100352UL) + ((((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 8UL) % 2UL) + ((fused_0fused_0fused_0oc_i__k_2288__n_2289__n_i_2290 / 16UL) * 2UL)) * 25088UL)) + ((_fuseiter7781 * 3584UL) + ((_fuseiter7782 * 512UL) + _fuseiter7783)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1980_shr);
    sc_aligned_free(__stream, __rescheduled_2);
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res5a_conv_0_cast_mul_add_cast_relu_reorder__682(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_88 = *(void**)(__module_data + 128);
  for (uint64_t n = 0UL; n < 8UL; n += 1UL) {
    for (uint64_t k = 0UL; k < 8UL; k += 1UL) {
      memset(&__outs_0[((n * 131072UL) + (k * 16384UL))], 0, 1024UL);
      for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
        memset(&__outs_0[((n * 131072UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
        memset(&__outs_0[(((n * 131072UL) + ((k * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n * 131072UL) + (k * 16384UL)) + 15360UL)], 0, 1024UL);
    }
  }
  for (uint64_t fused_0fused_0k__n_2291__n_i_2292 = 0UL; fused_0fused_0k__n_2291__n_i_2292 < 8UL; fused_0fused_0k__n_2291__n_i_2292 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1990_shr = (int32_t*)sc_aligned_malloc(__stream, 57344UL);
      void** A_list = (void**)&__rescheduled_2[0UL];
      void** B_list = (void**)&__rescheduled_2[128UL];
      for (uint64_t c = 0UL; c < 16UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0fused_0k__n_2291__n_i_2292 % 8UL) * 200704UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0fused_0k__n_2291__n_i_2292 / 8UL) * 524288UL) + (c * 32768UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_43 = &__origouts_1990_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_88, A_list, B_list, &__origouts_1990_shr[0UL], 1, 1, 1, 16, 7, 7, __stream);
      for (uint64_t _fuseiter7809 = 0UL; _fuseiter7809 < 2UL; _fuseiter7809 += 1UL) {
        for (uint64_t _fuseiter7810 = 0UL; _fuseiter7810 < 14UL; _fuseiter7810 += 1UL) {
          for (uint64_t _fuseiter7811 = 0UL; _fuseiter7811 < 512UL; _fuseiter7811 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_1990_shr[((_fuseiter7809 * 7168UL) + ((_fuseiter7810 * 512UL) + _fuseiter7811))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_2291__n_i_2292 / 8UL) * 512UL) + _fuseiter7811)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_2291__n_i_2292 / 8UL) * 512UL) + _fuseiter7811)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            __cached_6 = sc_max(__cached_6, vec_s8x16(0));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0k__n_2291__n_i_2292 % 8UL) * 131072UL) + ((((_fuseiter7811 + ((fused_0fused_0k__n_2291__n_i_2292 / 8UL) * 512UL)) / 64UL) * 16384UL) + ((((_fuseiter7809 + (p_o * 2UL)) + 1UL) * 1024UL) + (((_fuseiter7810 + 1UL) * 64UL) + ((_fuseiter7811 + ((fused_0fused_0k__n_2291__n_i_2292 / 8UL) * 512UL)) % 64UL)))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_1990_shr);
    }
    sc_aligned_free(__stream, __rescheduled_2);
  }
  return true;
}

static bool res5a_conv_1_cast_mul_add_cast_relu_reorder__681(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_90 = *(void**)(__module_data + 136);
  for (uint64_t fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 = 0UL; fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 < 16UL; fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 1152UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[576UL];
    for (uint64_t p_i = 0UL; p_i < 7UL; p_i += 1UL) {
      int32_t* __origouts_2000_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      for (uint64_t c_o = 0UL; c_o < 8UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[(((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 % 8UL) * 131072UL) + ((c_o * 16384UL) + ((((p_i * 2UL) + r) * 1024UL) + (s * 64UL))))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[(((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 / 8UL) * 1179648UL) + ((c_o * 147456UL) + ((r * 49152UL) + (s * 16384UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_44 = &__origouts_2000_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_90, A_list, B_list, &__origouts_2000_shr[0UL], 1, 64, 16384, 72, 7, 7, __stream);
      for (uint64_t _fuseiter7851 = 0UL; _fuseiter7851 < 7UL; _fuseiter7851 += 1UL) {
        for (uint64_t _fuseiter7852 = 0UL; _fuseiter7852 < 256UL; _fuseiter7852 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2000_shr[((_fuseiter7851 * 256UL) + _fuseiter7852)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 / 8UL) * 256UL) + _fuseiter7852)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 / 8UL) * 256UL) + _fuseiter7852)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 % 8UL) * 25088UL) + ((((_fuseiter7852 + ((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 / 8UL) * 256UL)) / 512UL) * 25088UL) + ((p_i * 3584UL) + ((_fuseiter7851 * 512UL) + ((_fuseiter7852 + ((fused_0fused_0fused_0oc_i__n_2293__n_i_2294__k_o_2295 / 8UL) * 256UL)) % 512UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_2000_shr);
    }
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5a_conv_2_cast_mul_add_cast_add_relu__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_92 = *(void**)(__module_data + 144);
  for (uint64_t fused_0fused_0n__n_i_2296__k_2297 = 0UL; fused_0fused_0n__n_i_2296__k_2297 < 32UL; fused_0fused_0n__n_i_2296__k_2297 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_2010_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0fused_0n__n_i_2296__k_2297 / 4UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0fused_0n__n_i_2296__k_2297 % 4UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_45 = &__origouts_2010_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_92, A_list, B_list, &__origouts_2010_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter7891 = 0UL; _fuseiter7891 < 7UL; _fuseiter7891 += 1UL) {
      for (uint64_t _fuseiter7892 = 0UL; _fuseiter7892 < 7UL; _fuseiter7892 += 1UL) {
        for (uint64_t _fuseiter7893 = 0UL; _fuseiter7893 < 512UL; _fuseiter7893 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2010_shr[((_fuseiter7891 * 3584UL) + ((_fuseiter7892 * 512UL) + _fuseiter7893))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2296__k_2297 % 4UL) * 512UL) + _fuseiter7893)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2296__k_2297 % 4UL) * 512UL) + _fuseiter7893)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2296__k_2297 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_2296__k_2297 % 4UL) * 25088UL)) + ((_fuseiter7891 * 3584UL) + ((_fuseiter7892 * 512UL) + _fuseiter7893)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0n__n_i_2296__k_2297 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_2296__k_2297 % 4UL) * 25088UL)) + ((_fuseiter7891 * 3584UL) + ((_fuseiter7892 * 512UL) + _fuseiter7893)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2010_shr);
  }
  return true;
}

static bool res5b_conv_0_cast_mul_add_cast_relu__679(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_94 = *(void**)(__module_data + 152);
  for (uint64_t n = 0UL; n < 8UL; n += 1UL) {
    for (uint64_t k = 0UL; k < 8UL; k += 1UL) {
      memset(&__outs_0[((n * 41472UL) + (k * 5184UL))], 0, 576UL);
      for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
        memset(&__outs_0[((n * 41472UL) + ((k * 5184UL) + ((p1 + 1UL) * 576UL)))], 0, 64UL);
        memset(&__outs_0[(((n * 41472UL) + ((k * 5184UL) + ((p1 + 1UL) * 576UL))) + 512UL)], 0, 64UL);
      }
      memset(&__outs_0[(((n * 41472UL) + (k * 5184UL)) + 4608UL)], 0, 576UL);
    }
  }
  for (uint64_t fused_0fused_0n__n_i_2298__k_2299 = 0UL; fused_0fused_0n__n_i_2298__k_2299 < 64UL; fused_0fused_0n__n_i_2298__k_2299 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_2020_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2298__k_2299 / 8UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2298__k_2299 % 8UL) * 131072UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_46 = &__origouts_2020_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_94, A_list, B_list, &__origouts_2020_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter7933 = 0UL; _fuseiter7933 < 7UL; _fuseiter7933 += 1UL) {
      for (uint64_t _fuseiter7934 = 0UL; _fuseiter7934 < 7UL; _fuseiter7934 += 1UL) {
        for (uint64_t _fuseiter7935 = 0UL; _fuseiter7935 < 64UL; _fuseiter7935 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2020_shr[((_fuseiter7933 * 448UL) + ((_fuseiter7934 * 64UL) + _fuseiter7935))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2298__k_2299 % 8UL) * 64UL) + _fuseiter7935)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2298__k_2299 % 8UL) * 64UL) + _fuseiter7935)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_7, &__outs_0[(((((fused_0fused_0n__n_i_2298__k_2299 / 8UL) * 41472UL) + ((fused_0fused_0n__n_i_2298__k_2299 % 8UL) * 5184UL)) + 640UL) + ((_fuseiter7933 * 576UL) + ((_fuseiter7934 * 64UL) + _fuseiter7935)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2020_shr);
  }
  return true;
}

static bool res5b_conv_1_cast_mul_add_cast_relu_reorder__678(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_98 = (void**)&__uninitialized_data[23657528UL];
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 = 0UL; fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 < 32UL; fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 1152UL);
    int32_t* __origouts_2030_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[576UL];
    for (uint64_t c_o = 0UL; c_o < 8UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_1;
          __cached_1 = &__ins_0[((((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 2UL) % 8UL) * 41472UL) + ((c_o * 5184UL) + ((r * 576UL) + (s * 64UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          void* __cached_2;
          __cached_2 = &__ins_1[((((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 % 2UL) + ((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 16UL) * 2UL)) * 589824UL) + ((c_o * 73728UL) + ((r * 24576UL) + (s * 8192UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_2;
        }
      }
    }
    void* _arg_cache_47 = &__origouts_2030_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_98[0UL], A_list, B_list, &__origouts_2030_shr[0UL], 1, 64, 8192, 72, 7, 7, __stream);
    for (uint64_t _fuseiter7968 = 0UL; _fuseiter7968 < 7UL; _fuseiter7968 += 1UL) {
      for (uint64_t _fuseiter7969 = 0UL; _fuseiter7969 < 7UL; _fuseiter7969 += 1UL) {
        for (uint64_t _fuseiter7970 = 0UL; _fuseiter7970 < 128UL; _fuseiter7970 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_2030_shr[((_fuseiter7968 * 896UL) + ((_fuseiter7969 * 128UL) + _fuseiter7970))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 % 2UL) + ((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 16UL) * 2UL)) * 128UL) + _fuseiter7970)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 % 2UL) + ((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 16UL) * 2UL)) * 128UL) + _fuseiter7970)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          __cached_7 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16 __cached_8;
          __cached_8 = __cached_7;
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 2UL) % 8UL) * 25088UL) + ((((_fuseiter7970 + (((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 % 2UL) + ((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 16UL) * 2UL)) * 128UL)) / 512UL) * 25088UL) + ((_fuseiter7968 * 3584UL) + ((_fuseiter7969 * 512UL) + ((_fuseiter7970 + (((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 % 2UL) + ((fused_0fused_0fused_0oc_i__n_2300__n_i_2301__k_o_2302 / 16UL) * 2UL)) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2030_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5b_conv_2_cast_mul_add_cast_add_relu__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_92 = *(void**)(__module_data + 144);
  for (uint64_t fused_0fused_0k__n_2303__n_i_2304 = 0UL; fused_0fused_0k__n_2303__n_i_2304 < 32UL; fused_0fused_0k__n_2303__n_i_2304 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_2040_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0fused_0k__n_2303__n_i_2304 % 8UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0fused_0k__n_2303__n_i_2304 / 8UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_48 = &__origouts_2040_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_92, A_list, B_list, &__origouts_2040_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter8009 = 0UL; _fuseiter8009 < 7UL; _fuseiter8009 += 1UL) {
      for (uint64_t _fuseiter8010 = 0UL; _fuseiter8010 < 7UL; _fuseiter8010 += 1UL) {
        for (uint64_t _fuseiter8011 = 0UL; _fuseiter8011 < 512UL; _fuseiter8011 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2040_shr[((_fuseiter8009 * 3584UL) + ((_fuseiter8010 * 512UL) + _fuseiter8011))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0k__n_2303__n_i_2304 / 8UL) * 512UL) + _fuseiter8011)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0k__n_2303__n_i_2304 / 8UL) * 512UL) + _fuseiter8011)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0k__n_2303__n_i_2304 % 8UL) * 100352UL) + ((fused_0fused_0k__n_2303__n_i_2304 / 8UL) * 25088UL)) + ((_fuseiter8009 * 3584UL) + ((_fuseiter8010 * 512UL) + _fuseiter8011)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0k__n_2303__n_i_2304 % 8UL) * 100352UL) + ((fused_0fused_0k__n_2303__n_i_2304 / 8UL) * 25088UL)) + ((_fuseiter8009 * 3584UL) + ((_fuseiter8010 * 512UL) + _fuseiter8011)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2040_shr);
  }
  return true;
}

static bool res5c_conv_0_cast_mul_add_cast_relu_reorder__676(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_100 = *(void**)(__module_data + 160);
  for (uint64_t n = 0UL; n < 8UL; n += 1UL) {
    memset(&__outs_0[(n * 41472UL)], 0, 4608UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[((n * 41472UL) + ((p1 + 1UL) * 4608UL))], 0, 512UL);
      memset(&__outs_0[(((n * 41472UL) + ((p1 + 1UL) * 4608UL)) + 4096UL)], 0, 512UL);
    }
    memset(&__outs_0[((n * 41472UL) + 36864UL)], 0, 4608UL);
  }
  for (uint64_t fused_0fused_0n__n_i_2305__k_2306 = 0UL; fused_0fused_0n__n_i_2305__k_2306 < 16UL; fused_0fused_0n__n_i_2305__k_2306 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_2050_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2305__k_2306 / 2UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2305__k_2306 % 2UL) * 524288UL) + (c * 131072UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_49 = &__origouts_2050_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_100, A_list, B_list, &__origouts_2050_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter8051 = 0UL; _fuseiter8051 < 7UL; _fuseiter8051 += 1UL) {
      for (uint64_t _fuseiter8052 = 0UL; _fuseiter8052 < 7UL; _fuseiter8052 += 1UL) {
        for (uint64_t _fuseiter8053 = 0UL; _fuseiter8053 < 256UL; _fuseiter8053 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2050_shr[((_fuseiter8051 * 1792UL) + ((_fuseiter8052 * 256UL) + _fuseiter8053))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2305__k_2306 % 2UL) * 256UL) + _fuseiter8053)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2305__k_2306 % 2UL) * 256UL) + _fuseiter8053)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0fused_0n__n_i_2305__k_2306 / 2UL) * 41472UL) + ((((_fuseiter8053 + ((fused_0fused_0n__n_i_2305__k_2306 % 2UL) * 256UL)) / 512UL) * 41472UL) + (((_fuseiter8051 + 1UL) * 4608UL) + (((_fuseiter8052 + 1UL) * 512UL) + ((_fuseiter8053 + ((fused_0fused_0n__n_i_2305__k_2306 % 2UL) * 256UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2050_shr);
  }
  return true;
}

static bool res5c_conv_1_cast_mul_add_cast_relu__675(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_102 = (void**)&__uninitialized_data[23657536UL];
  int32_t __cached_0;
  __cached_0 = 0;
  for (uint64_t fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 = 0UL; fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 < 64UL; fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_2060_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[128UL];
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_1;
        __cached_1 = &__ins_0[((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 4UL) % 8UL) * 41472UL) + ((r * 4608UL) + (s * 512UL)))];
        A_list[((r * 3UL) + s)] = __cached_1;
        void* __cached_2;
        __cached_2 = &__ins_1[((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 % 4UL) + ((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 32UL) * 4UL)) * 294912UL) + ((r * 98304UL) + (s * 32768UL)))];
        B_list[((r * 3UL) + s)] = __cached_2;
      }
    }
    void* _arg_cache_50 = &__origouts_2060_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_102[0UL], A_list, B_list, &__origouts_2060_shr[0UL], 1, 512, 32768, 9, 7, 7, __stream);
    for (uint64_t _fuseiter8092 = 0UL; _fuseiter8092 < 7UL; _fuseiter8092 += 1UL) {
      for (uint64_t _fuseiter8093 = 0UL; _fuseiter8093 < 7UL; _fuseiter8093 += 1UL) {
        for (uint64_t _fuseiter8094 = 0UL; _fuseiter8094 < 64UL; _fuseiter8094 += 16UL) {
          vec_s32x16 __cached_3;
          __cached_3 = vec_s32x16::load(&__origouts_2060_shr[((_fuseiter8092 * 448UL) + ((_fuseiter8093 * 64UL) + _fuseiter8094))]);
          vec_f32x16 __cached_4;
          __cached_4 = (vec_f32x16)(__cached_3);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_2[((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 % 4UL) + ((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 32UL) * 4UL)) * 64UL) + _fuseiter8094)]);
          __cached_4 = (__cached_4 * __cached_5);
          vec_f32x16 __cached_6;
          __cached_6 = vec_f32x16::load(&__ins_3[((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 % 4UL) + ((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 32UL) * 4UL)) * 64UL) + _fuseiter8094)]);
          __cached_4 = (__cached_4 + __cached_6);
          vec_s8x16 __cached_7;
          __cached_7 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_4));
          vec_s8x16 __cached_8;
          __cached_8 = sc_max(__cached_7, vec_s8x16(0));
          vec_s8x16::store(__cached_8, &__outs_0[((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 4UL) % 8UL) * 25088UL) + ((((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 % 4UL) + ((fused_0fused_0fused_0oc_i__n_2307__n_i_2308__k_o_2309 / 32UL) * 4UL)) * 3136UL) + ((_fuseiter8092 * 448UL) + ((_fuseiter8093 * 64UL) + _fuseiter8094))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2060_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_104 = *(void**)(__module_data + 168);
  for (uint64_t fused_0fused_0n__n_i_2310__k_2311 = 0UL; fused_0fused_0n__n_i_2310__k_2311 < 32UL; fused_0fused_0n__n_i_2310__k_2311 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_2070_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0fused_0n__n_i_2310__k_2311 / 4UL) * 25088UL) + (c * 3136UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0fused_0n__n_i_2310__k_2311 % 4UL) * 262144UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_51 = &__origouts_2070_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_104, A_list, B_list, &__origouts_2070_shr[0UL], 1, 1, 1, 8, 7, 7, __stream);
    for (uint64_t _fuseiter8127 = 0UL; _fuseiter8127 < 7UL; _fuseiter8127 += 1UL) {
      for (uint64_t _fuseiter8128 = 0UL; _fuseiter8128 < 7UL; _fuseiter8128 += 1UL) {
        for (uint64_t _fuseiter8129 = 0UL; _fuseiter8129 < 512UL; _fuseiter8129 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_2070_shr[((_fuseiter8127 * 3584UL) + ((_fuseiter8128 * 512UL) + _fuseiter8129))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0fused_0n__n_i_2310__k_2311 % 4UL) * 512UL) + _fuseiter8129)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0fused_0n__n_i_2310__k_2311 % 4UL) * 512UL) + _fuseiter8129)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0fused_0n__n_i_2310__k_2311 / 4UL) * 100352UL) + ((fused_0fused_0n__n_i_2310__k_2311 % 4UL) * 25088UL)) + ((_fuseiter8127 * 3584UL) + ((_fuseiter8128 * 512UL) + _fuseiter8129)))]);
          __cached_6 = (__cached_6 + __cached_7);
          __cached_6 = sc_max(__cached_6, vec_s8x16(0));
          vec_s8x16 __cached_8;
          __cached_8 = __cached_6;
          vec_s8x16::store(__cached_8, &__outs_0[(((fused_0fused_0n__n_i_2310__k_2311 / 4UL) * 100352UL) + ((_fuseiter8127 * 14336UL) + ((_fuseiter8128 * 2048UL) + (_fuseiter8129 + ((fused_0fused_0n__n_i_2310__k_2311 % 4UL) * 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_2070_shr);
  }
  return true;
}

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  float* folded_const_103 = (float*)&__module_data[111680UL];
  float* folded_const_156 = (float*)&__uninitialized_data[0UL];
  float* folded_const_101 = (float*)&__module_data[111376UL];
  float* folded_const_157 = (float*)&__uninitialized_data[1024UL];
  float* folded_const_102 = (float*)&__module_data[111424UL];
  float* folded_const_81 = (float*)&__module_data[111296UL];
  float* folded_const_158 = (float*)&__uninitialized_data[1280UL];
  float* folded_const_80 = (float*)&__module_data[111040UL];
  float* folded_const_79 = (float*)&__module_data[110976UL];
  float* folded_const_78 = (float*)&__module_data[109952UL];
  float* folded_const_159 = (float*)&__uninitialized_data[1536UL];
  float* folded_const_100 = (float*)&__module_data[111372UL];
  float* folded_const_160 = (float*)&__uninitialized_data[2560UL];
  float* folded_const_77 = (float*)&__module_data[109696UL];
  float* folded_const_76 = (float*)&__module_data[109632UL];
  float* folded_const_161 = (float*)&__uninitialized_data[2816UL];
  float* folded_const_75 = (float*)&__module_data[109376UL];
  float* folded_const_74 = (float*)&__module_data[109312UL];
  float* folded_const_73 = (float*)&__module_data[108288UL];
  float* folded_const_162 = (float*)&__uninitialized_data[3072UL];
  float* folded_const_99 = (float*)&__module_data[111368UL];
  float* folded_const_163 = (float*)&__uninitialized_data[4096UL];
  float* folded_const_72 = (float*)&__module_data[108032UL];
  float* folded_const_71 = (float*)&__module_data[107968UL];
  float* folded_const_164 = (float*)&__uninitialized_data[4352UL];
  float* folded_const_70 = (float*)&__module_data[107712UL];
  float* folded_const_69 = (float*)&__module_data[107648UL];
  float* folded_const_68 = (float*)&__module_data[106624UL];
  float* folded_const_165 = (float*)&__uninitialized_data[4608UL];
  float* folded_const_98 = (float*)&__module_data[111364UL];
  float* folded_const_67 = (float*)&__module_data[104576UL];
  float* folded_const_166 = (float*)&__uninitialized_data[5632UL];
  float* folded_const_97 = (float*)&__module_data[111360UL];
  float* folded_const_66 = (float*)&__module_data[104064UL];
  float* folded_const_167 = (float*)&__uninitialized_data[7680UL];
  float* folded_const_65 = (float*)&__module_data[104000UL];
  float* folded_const_64 = (float*)&__module_data[103488UL];
  float* folded_const_168 = (float*)&__uninitialized_data[8192UL];
  float* folded_const_63 = (float*)&__module_data[103424UL];
  float* folded_const_62 = (float*)&__module_data[101376UL];
  float* folded_const_169 = (float*)&__uninitialized_data[8704UL];
  float* folded_const_96 = (float*)&__module_data[111356UL];
  float* folded_const_61 = (float*)&__module_data[100864UL];
  float* folded_const_170 = (float*)&__uninitialized_data[10752UL];
  float* folded_const_60 = (float*)&__module_data[100800UL];
  float* folded_const_59 = (float*)&__module_data[100288UL];
  float* folded_const_171 = (float*)&__uninitialized_data[11264UL];
  float* folded_const_58 = (float*)&__module_data[100224UL];
  float* folded_const_57 = (float*)&__module_data[98176UL];
  float* folded_const_172 = (float*)&__uninitialized_data[11776UL];
  float* folded_const_95 = (float*)&__module_data[111352UL];
  float* folded_const_56 = (float*)&__module_data[97664UL];
  float* folded_const_173 = (float*)&__uninitialized_data[13824UL];
  float* folded_const_55 = (float*)&__module_data[97600UL];
  float* folded_const_54 = (float*)&__module_data[97088UL];
  float* folded_const_174 = (float*)&__uninitialized_data[14336UL];
  float* folded_const_53 = (float*)&__module_data[97024UL];
  float* folded_const_52 = (float*)&__module_data[94976UL];
  float* folded_const_175 = (float*)&__uninitialized_data[14848UL];
  float* folded_const_94 = (float*)&__module_data[111348UL];
  float* folded_const_51 = (float*)&__module_data[94464UL];
  float* folded_const_176 = (float*)&__uninitialized_data[16896UL];
  float* folded_const_50 = (float*)&__module_data[94400UL];
  float* folded_const_49 = (float*)&__module_data[93888UL];
  float* folded_const_177 = (float*)&__uninitialized_data[17408UL];
  float* folded_const_48 = (float*)&__module_data[93824UL];
  float* folded_const_47 = (float*)&__module_data[91776UL];
  float* folded_const_178 = (float*)&__uninitialized_data[17920UL];
  float* folded_const_93 = (float*)&__module_data[111344UL];
  float* folded_const_46 = (float*)&__module_data[87680UL];
  float* folded_const_179 = (float*)&__uninitialized_data[19968UL];
  float* folded_const_92 = (float*)&__module_data[111340UL];
  float* folded_const_45 = (float*)&__module_data[86656UL];
  float* folded_const_180 = (float*)&__uninitialized_data[24064UL];
  float* folded_const_44 = (float*)&__module_data[86592UL];
  float* folded_const_43 = (float*)&__module_data[85568UL];
  float* folded_const_181 = (float*)&__uninitialized_data[25088UL];
  float* folded_const_42 = (float*)&__module_data[85504UL];
  float* folded_const_41 = (float*)&__module_data[81408UL];
  float* folded_const_182 = (float*)&__uninitialized_data[26112UL];
  float* folded_const_91 = (float*)&__module_data[111336UL];
  float* folded_const_40 = (float*)&__module_data[80384UL];
  float* folded_const_183 = (float*)&__uninitialized_data[30208UL];
  float* folded_const_39 = (float*)&__module_data[80320UL];
  float* folded_const_38 = (float*)&__module_data[79296UL];
  float* folded_const_184 = (float*)&__uninitialized_data[31232UL];
  float* folded_const_37 = (float*)&__module_data[79232UL];
  float* folded_const_36 = (float*)&__module_data[75136UL];
  float* folded_const_185 = (float*)&__uninitialized_data[32256UL];
  float* folded_const_90 = (float*)&__module_data[111332UL];
  float* folded_const_35 = (float*)&__module_data[74112UL];
  float* folded_const_186 = (float*)&__uninitialized_data[36352UL];
  float* folded_const_34 = (float*)&__module_data[74048UL];
  float* folded_const_33 = (float*)&__module_data[73024UL];
  float* folded_const_187 = (float*)&__uninitialized_data[37376UL];
  float* folded_const_32 = (float*)&__module_data[72960UL];
  float* folded_const_31 = (float*)&__module_data[68864UL];
  float* folded_const_188 = (float*)&__uninitialized_data[38400UL];
  float* folded_const_89 = (float*)&__module_data[111328UL];
  float* folded_const_30 = (float*)&__module_data[67840UL];
  float* folded_const_189 = (float*)&__uninitialized_data[42496UL];
  float* folded_const_29 = (float*)&__module_data[67776UL];
  float* folded_const_28 = (float*)&__module_data[66752UL];
  float* folded_const_190 = (float*)&__uninitialized_data[43520UL];
  float* folded_const_27 = (float*)&__module_data[66688UL];
  float* folded_const_26 = (float*)&__module_data[62592UL];
  float* folded_const_191 = (float*)&__uninitialized_data[44544UL];
  float* folded_const_88 = (float*)&__module_data[111324UL];
  float* folded_const_25 = (float*)&__module_data[61568UL];
  float* folded_const_192 = (float*)&__uninitialized_data[48640UL];
  float* folded_const_24 = (float*)&__module_data[61504UL];
  float* folded_const_23 = (float*)&__module_data[60480UL];
  float* folded_const_193 = (float*)&__uninitialized_data[49664UL];
  float* folded_const_22 = (float*)&__module_data[60416UL];
  float* folded_const_21 = (float*)&__module_data[56320UL];
  float* folded_const_194 = (float*)&__uninitialized_data[50688UL];
  float* folded_const_87 = (float*)&__module_data[111320UL];
  float* folded_const_20 = (float*)&__module_data[55296UL];
  float* folded_const_195 = (float*)&__uninitialized_data[54784UL];
  float* folded_const_19 = (float*)&__module_data[55232UL];
  float* folded_const_18 = (float*)&__module_data[54208UL];
  float* folded_const_196 = (float*)&__uninitialized_data[55808UL];
  float* folded_const_17 = (float*)&__module_data[54144UL];
  float* folded_const_16 = (float*)&__module_data[50048UL];
  float* folded_const_197 = (float*)&__uninitialized_data[56832UL];
  float* folded_const_86 = (float*)&__module_data[111316UL];
  float* folded_const_15 = (float*)&__module_data[41856UL];
  float* folded_const_198 = (float*)&__uninitialized_data[60928UL];
  float* folded_const_85 = (float*)&__module_data[111312UL];
  float* folded_const_199 = (float*)&__uninitialized_data[69120UL];
  float* folded_const_14 = (float*)&__module_data[39808UL];
  float* folded_const_13 = (float*)&__module_data[39744UL];
  float* folded_const_12 = (float*)&__module_data[37696UL];
  float* folded_const_200 = (float*)&__uninitialized_data[71168UL];
  float* folded_const_11 = (float*)&__module_data[37632UL];
  float* folded_const_10 = (float*)&__module_data[29440UL];
  float* folded_const_201 = (float*)&__uninitialized_data[73216UL];
  float* folded_const_84 = (float*)&__module_data[111308UL];
  float* folded_const_9 = (float*)&__module_data[27392UL];
  float* folded_const_202 = (float*)&__uninitialized_data[81408UL];
  float* folded_const_8 = (float*)&__module_data[27328UL];
  float* folded_const_7 = (float*)&__module_data[25280UL];
  float* folded_const_203 = (float*)&__uninitialized_data[83456UL];
  float* folded_const_6 = (float*)&__module_data[25216UL];
  float* folded_const_5 = (float*)&__module_data[17024UL];
  float* folded_const_204 = (float*)&__uninitialized_data[85504UL];
  float* folded_const_83 = (float*)&__module_data[111304UL];
  float* folded_const_4 = (float*)&__module_data[14976UL];
  float* folded_const_205 = (float*)&__uninitialized_data[93696UL];
  float* folded_const_3 = (float*)&__module_data[14912UL];
  float* folded_const_2 = (float*)&__module_data[12864UL];
  float* folded_const_206 = (float*)&__uninitialized_data[95744UL];
  float* folded_const_1 = (float*)&__module_data[12800UL];
  float* folded_const_0 = (float*)&__module_data[4608UL];
  float* folded_const_207 = (float*)&__uninitialized_data[97792UL];
  float* folded_const_82 = (float*)&__module_data[111300UL];
  float* folded_const_208 = (float*)&__uninitialized_data[105984UL];
  float* folded_const_209 = (float*)&__uninitialized_data[106240UL];
  float* folded_const_210 = (float*)&__uninitialized_data[106496UL];
  float* folded_const_211 = (float*)&__uninitialized_data[106752UL];
  float* folded_const_212 = (float*)&__uninitialized_data[107008UL];
  float* folded_const_213 = (float*)&__uninitialized_data[107264UL];
  float* folded_const_214 = (float*)&__uninitialized_data[107520UL];
  float* folded_const_215 = (float*)&__uninitialized_data[108032UL];
  float* folded_const_216 = (float*)&__uninitialized_data[108544UL];
  float* folded_const_217 = (float*)&__uninitialized_data[109056UL];
  float* folded_const_218 = (float*)&__uninitialized_data[109568UL];
  float* folded_const_219 = (float*)&__uninitialized_data[110080UL];
  float* folded_const_220 = (float*)&__uninitialized_data[110592UL];
  float* folded_const_221 = (float*)&__uninitialized_data[111104UL];
  float* folded_const_222 = (float*)&__uninitialized_data[111616UL];
  float* folded_const_223 = (float*)&__uninitialized_data[112640UL];
  float* folded_const_224 = (float*)&__uninitialized_data[113664UL];
  float* folded_const_225 = (float*)&__uninitialized_data[114688UL];
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
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_261 = (float*)&__rescheduled_0[0UL];
  reorder__419(buffer_261, folded_const_103);
  mul__570(folded_const_156, buffer_261, folded_const_101);
  mul__572(folded_const_157, &folded_const_102[0UL], folded_const_81);
  mul__574(folded_const_158, &folded_const_80[0UL], folded_const_79);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_265 = (float*)&__rescheduled_0[0UL];
  reorder__424(buffer_265, folded_const_78);
  mul__576(folded_const_159, buffer_265, folded_const_100);
  mul__578(folded_const_160, &folded_const_77[0UL], folded_const_76);
  mul__580(folded_const_161, &folded_const_75[0UL], folded_const_74);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_269 = (float*)&__rescheduled_0[0UL];
  reorder__429(buffer_269, folded_const_73);
  mul__582(folded_const_162, buffer_269, folded_const_99);
  mul__584(folded_const_163, &folded_const_72[0UL], folded_const_71);
  mul__586(folded_const_164, &folded_const_70[0UL], folded_const_69);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_273 = (float*)&__rescheduled_0[0UL];
  reorder__434(buffer_273, folded_const_68);
  mul__588(folded_const_165, buffer_273, folded_const_98);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_275 = (float*)&__rescheduled_0[0UL];
  reorder__437(buffer_275, folded_const_67);
  mul__590(folded_const_166, buffer_275, folded_const_97);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_277 = (float*)&__rescheduled_0[0UL];
  reorder__440(buffer_277, folded_const_66);
  mul__592(folded_const_167, buffer_277, folded_const_65);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_279 = (float*)&__rescheduled_0[0UL];
  reorder__443(buffer_279, folded_const_64);
  mul__594(folded_const_168, buffer_279, folded_const_63);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_281 = (float*)&__rescheduled_0[0UL];
  reorder__446(buffer_281, folded_const_62);
  mul__596(folded_const_169, buffer_281, folded_const_96);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_283 = (float*)&__rescheduled_0[0UL];
  reorder__449(buffer_283, folded_const_61);
  mul__598(folded_const_170, buffer_283, folded_const_60);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_285 = (float*)&__rescheduled_0[0UL];
  reorder__452(buffer_285, folded_const_59);
  mul__600(folded_const_171, buffer_285, folded_const_58);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_287 = (float*)&__rescheduled_0[0UL];
  reorder__455(buffer_287, folded_const_57);
  mul__602(folded_const_172, buffer_287, folded_const_95);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_289 = (float*)&__rescheduled_0[0UL];
  reorder__458(buffer_289, folded_const_56);
  mul__604(folded_const_173, buffer_289, folded_const_55);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_291 = (float*)&__rescheduled_0[0UL];
  reorder__461(buffer_291, folded_const_54);
  mul__606(folded_const_174, buffer_291, folded_const_53);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_293 = (float*)&__rescheduled_0[0UL];
  reorder__464(buffer_293, folded_const_52);
  mul__608(folded_const_175, buffer_293, folded_const_94);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_295 = (float*)&__rescheduled_0[0UL];
  reorder__467(buffer_295, folded_const_51);
  mul__610(folded_const_176, buffer_295, folded_const_50);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_297 = (float*)&__rescheduled_0[0UL];
  reorder__470(buffer_297, folded_const_49);
  mul__612(folded_const_177, buffer_297, folded_const_48);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_299 = (float*)&__rescheduled_0[0UL];
  reorder__473(buffer_299, folded_const_47);
  mul__614(folded_const_178, buffer_299, folded_const_93);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_301 = (float*)&__rescheduled_0[0UL];
  reorder__476(buffer_301, folded_const_46);
  mul__616(folded_const_179, buffer_301, folded_const_92);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_303 = (float*)&__rescheduled_0[0UL];
  reorder__479(buffer_303, folded_const_45);
  mul__618(folded_const_180, buffer_303, folded_const_44);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_305 = (float*)&__rescheduled_0[0UL];
  reorder__482(buffer_305, folded_const_43);
  mul__620(folded_const_181, buffer_305, folded_const_42);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_307 = (float*)&__rescheduled_0[0UL];
  reorder__485(buffer_307, folded_const_41);
  mul__622(folded_const_182, buffer_307, folded_const_91);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_309 = (float*)&__rescheduled_0[0UL];
  reorder__488(buffer_309, folded_const_40);
  mul__624(folded_const_183, buffer_309, folded_const_39);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_311 = (float*)&__rescheduled_0[0UL];
  reorder__491(buffer_311, folded_const_38);
  mul__626(folded_const_184, buffer_311, folded_const_37);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_313 = (float*)&__rescheduled_0[0UL];
  reorder__494(buffer_313, folded_const_36);
  mul__628(folded_const_185, buffer_313, folded_const_90);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_315 = (float*)&__rescheduled_0[0UL];
  reorder__497(buffer_315, folded_const_35);
  mul__630(folded_const_186, buffer_315, folded_const_34);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_317 = (float*)&__rescheduled_0[0UL];
  reorder__500(buffer_317, folded_const_33);
  mul__632(folded_const_187, buffer_317, folded_const_32);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_319 = (float*)&__rescheduled_0[0UL];
  reorder__503(buffer_319, folded_const_31);
  mul__634(folded_const_188, buffer_319, folded_const_89);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_321 = (float*)&__rescheduled_0[0UL];
  reorder__506(buffer_321, folded_const_30);
  mul__636(folded_const_189, buffer_321, folded_const_29);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_323 = (float*)&__rescheduled_0[0UL];
  reorder__509(buffer_323, folded_const_28);
  mul__638(folded_const_190, buffer_323, folded_const_27);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_325 = (float*)&__rescheduled_0[0UL];
  reorder__512(buffer_325, folded_const_26);
  mul__640(folded_const_191, buffer_325, folded_const_88);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_327 = (float*)&__rescheduled_0[0UL];
  reorder__515(buffer_327, folded_const_25);
  mul__642(folded_const_192, buffer_327, folded_const_24);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_329 = (float*)&__rescheduled_0[0UL];
  reorder__518(buffer_329, folded_const_23);
  mul__644(folded_const_193, buffer_329, folded_const_22);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_331 = (float*)&__rescheduled_0[0UL];
  reorder__521(buffer_331, folded_const_21);
  mul__646(folded_const_194, buffer_331, folded_const_87);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_333 = (float*)&__rescheduled_0[0UL];
  reorder__524(buffer_333, folded_const_20);
  mul__648(folded_const_195, buffer_333, folded_const_19);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_335 = (float*)&__rescheduled_0[0UL];
  reorder__527(buffer_335, folded_const_18);
  mul__650(folded_const_196, buffer_335, folded_const_17);
  // [f32 [1, 1, 16, 1, 1, 64] @ A1aBCD64b]
  float* buffer_337 = (float*)&__rescheduled_0[0UL];
  reorder__530(buffer_337, folded_const_16);
  mul__652(folded_const_197, buffer_337, folded_const_86);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_339 = (float*)&__rescheduled_0[0UL];
  reorder__533(buffer_339, folded_const_15);
  mul__654(folded_const_198, buffer_339, folded_const_85);
  mul__656(folded_const_199, &folded_const_14[0UL], folded_const_13);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_342 = (float*)&__rescheduled_0[0UL];
  reorder__537(buffer_342, folded_const_12);
  mul__658(folded_const_200, buffer_342, folded_const_11);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_344 = (float*)&__rescheduled_0[0UL];
  reorder__540(buffer_344, folded_const_10);
  mul__660(folded_const_201, buffer_344, folded_const_84);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_346 = (float*)&__rescheduled_0[0UL];
  reorder__543(buffer_346, folded_const_9);
  mul__662(folded_const_202, buffer_346, folded_const_8);
  // [f32 [1, 1, 4, 1, 1, 128] @ A1aBCD128b]
  float* buffer_348 = (float*)&__rescheduled_0[0UL];
  reorder__546(buffer_348, folded_const_7);
  mul__664(folded_const_203, buffer_348, folded_const_6);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_350 = (float*)&__rescheduled_0[0UL];
  reorder__549(buffer_350, folded_const_5);
  mul__666(folded_const_204, buffer_350, folded_const_83);
  // [f32 [1, 1, 2, 1, 1, 256] @ A1aBCD256b]
  float* buffer_352 = (float*)&__rescheduled_0[0UL];
  reorder__552(buffer_352, folded_const_4);
  mul__668(folded_const_205, buffer_352, folded_const_3);
  // [f32 [1, 1, 8, 1, 1, 64] @ A1aBCD64b]
  float* buffer_354 = (float*)&__rescheduled_0[0UL];
  reorder__555(buffer_354, folded_const_2);
  mul__670(folded_const_206, buffer_354, folded_const_1);
  // [f32 [1, 1, 4, 1, 1, 512] @ A1aBCD512b]
  float* buffer_356 = (float*)&__rescheduled_0[0UL];
  reorder__558(buffer_356, folded_const_0);
  mul__672(folded_const_207, buffer_356, folded_const_82);
  mul__573(folded_const_208, &res2a_bias_0[0], folded_const_81);
  mul__575(folded_const_209, &res2a_bias_1[0], folded_const_79);
  mul__579(folded_const_210, &res2b_bias_0[0], folded_const_76);
  mul__581(folded_const_211, &res2b_bias_1[0], folded_const_74);
  mul__585(folded_const_212, &res2c_bias_0[0], folded_const_71);
  mul__587(folded_const_213, &res2c_bias_1[0], folded_const_69);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_364 = (float*)&__rescheduled_0[0UL];
  reorder__441(buffer_364, &res3a_bias_0[0]);
  mul__593(folded_const_214, buffer_364, folded_const_65);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_366 = (float*)&__rescheduled_0[0UL];
  reorder__444(buffer_366, &res3a_bias_1[0]);
  mul__595(folded_const_215, buffer_366, folded_const_63);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_368 = (float*)&__rescheduled_0[0UL];
  reorder__450(buffer_368, &res3b_bias_0[0]);
  mul__599(folded_const_216, buffer_368, folded_const_60);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_370 = (float*)&__rescheduled_0[0UL];
  reorder__453(buffer_370, &res3b_bias_1[0]);
  mul__601(folded_const_217, buffer_370, folded_const_58);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_372 = (float*)&__rescheduled_0[0UL];
  reorder__459(buffer_372, &res3c_bias_0[0]);
  mul__605(folded_const_218, buffer_372, folded_const_55);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_374 = (float*)&__rescheduled_0[0UL];
  reorder__462(buffer_374, &res3c_bias_1[0]);
  mul__607(folded_const_219, buffer_374, folded_const_53);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_376 = (float*)&__rescheduled_0[0UL];
  reorder__468(buffer_376, &res3d_bias_0[0]);
  mul__611(folded_const_220, buffer_376, folded_const_50);
  // [f32 [1, 1, 2, 1, 1, 64] @ A1aBCD64b]
  float* buffer_378 = (float*)&__rescheduled_0[0UL];
  reorder__471(buffer_378, &res3d_bias_1[0]);
  mul__613(folded_const_221, buffer_378, folded_const_48);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_380 = (float*)&__rescheduled_0[0UL];
  reorder__420(buffer_380, &res2a_bias_b[0]);
  mul__571(folded_const_222, buffer_380, folded_const_101);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_382 = (float*)&__rescheduled_0[0UL];
  reorder__425(buffer_382, &res2a_bias_2[0]);
  mul__577(folded_const_223, buffer_382, folded_const_100);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_384 = (float*)&__rescheduled_0[0UL];
  reorder__430(buffer_384, &res2b_bias_2[0]);
  mul__583(folded_const_224, buffer_384, folded_const_99);
  // [f32 [1, 1, 4, 1, 1, 64] @ A1aBCD64b]
  float* buffer_386 = (float*)&__rescheduled_0[0UL];
  reorder__435(buffer_386, &res2c_bias_2[0]);
  mul__589(folded_const_225, buffer_386, folded_const_98);
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

extern "C" void sc_init_rn50_backbone_bs8() {
  bool& is_init = *(bool*)(__module_data + 0);
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  uint8_t* __brgemm_attrs = (uint8_t*)&__module_data[192UL];
  void*& __sc_kernel_cache_55 = *(void**)(__module_data + 16);
  uint8_t* __brgemm_attrs_54 = (uint8_t*)&__module_data[320UL];
  void*& __sc_kernel_cache_57 = *(void**)(__module_data + 24);
  uint8_t* __brgemm_attrs_56 = (uint8_t*)&__module_data[448UL];
  void*& __sc_kernel_cache_59 = *(void**)(__module_data + 32);
  uint8_t* __brgemm_attrs_58 = (uint8_t*)&__module_data[576UL];
  void*& __sc_kernel_cache_61 = *(void**)(__module_data + 40);
  uint8_t* __brgemm_attrs_60 = (uint8_t*)&__module_data[704UL];
  void*& __sc_kernel_cache_63 = *(void**)(__module_data + 48);
  uint8_t* __brgemm_attrs_62 = (uint8_t*)&__module_data[832UL];
  void*& __sc_kernel_cache_65 = *(void**)(__module_data + 56);
  uint8_t* __brgemm_attrs_64 = (uint8_t*)&__module_data[960UL];
  void*& __sc_kernel_cache_67 = *(void**)(__module_data + 64);
  uint8_t* __brgemm_attrs_66 = (uint8_t*)&__module_data[1088UL];
  void*& __sc_kernel_cache_70 = *(void**)(__module_data + 72);
  uint8_t* __brgemm_attrs_69 = (uint8_t*)&__module_data[2240UL];
  void*& __sc_kernel_cache_72 = *(void**)(__module_data + 80);
  uint8_t* __brgemm_attrs_71 = (uint8_t*)&__module_data[2368UL];
  void*& __sc_kernel_cache_74 = *(void**)(__module_data + 88);
  uint8_t* __brgemm_attrs_73 = (uint8_t*)&__module_data[2496UL];
  void*& __sc_kernel_cache_76 = *(void**)(__module_data + 96);
  uint8_t* __brgemm_attrs_75 = (uint8_t*)&__module_data[2624UL];
  void*& __sc_kernel_cache_78 = *(void**)(__module_data + 104);
  uint8_t* __brgemm_attrs_77 = (uint8_t*)&__module_data[2752UL];
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 112);
  uint8_t* __brgemm_attrs_79 = (uint8_t*)&__module_data[2880UL];
  void*& __sc_kernel_cache_86 = *(void**)(__module_data + 120);
  uint8_t* __brgemm_attrs_85 = (uint8_t*)&__module_data[3392UL];
  void*& __sc_kernel_cache_88 = *(void**)(__module_data + 128);
  uint8_t* __brgemm_attrs_87 = (uint8_t*)&__module_data[3520UL];
  void*& __sc_kernel_cache_90 = *(void**)(__module_data + 136);
  uint8_t* __brgemm_attrs_89 = (uint8_t*)&__module_data[3648UL];
  void*& __sc_kernel_cache_92 = *(void**)(__module_data + 144);
  uint8_t* __brgemm_attrs_91 = (uint8_t*)&__module_data[3776UL];
  void*& __sc_kernel_cache_94 = *(void**)(__module_data + 152);
  uint8_t* __brgemm_attrs_93 = (uint8_t*)&__module_data[3904UL];
  void*& __sc_kernel_cache_100 = *(void**)(__module_data + 160);
  uint8_t* __brgemm_attrs_99 = (uint8_t*)&__module_data[4224UL];
  void*& __sc_kernel_cache_104 = *(void**)(__module_data + 168);
  uint8_t* __brgemm_attrs_103 = (uint8_t*)&__module_data[4480UL];
  void** __brgemm_bd_mask_arr = (void**)&__uninitialized_data[23657472UL];
  uint8_t* __brgemm_full_bd_mask = (uint8_t*)&__module_data[1344UL];
  void** __brgemm_bd_mask_arr_83 = (void**)&__uninitialized_data[23657504UL];
  uint8_t* __brgemm_full_bd_mask_82 = (uint8_t*)&__module_data[3136UL];
  void** __brgemm_bd_mask_arr_97 = (void**)&__uninitialized_data[23657520UL];
  uint8_t* __brgemm_full_bd_mask_96 = (uint8_t*)&__module_data[4152UL];
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  uint8_t* __brgemm_attrs_68 = (uint8_t*)&__module_data[1216UL];
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657512UL];
  uint8_t* __brgemm_attrs_81 = (uint8_t*)&__module_data[3008UL];
  void** __sc_kernel_cache_arr_98 = (void**)&__uninitialized_data[23657528UL];
  uint8_t* __brgemm_attrs_95 = (uint8_t*)&__module_data[4032UL];
  void** __sc_kernel_cache_arr_102 = (void**)&__uninitialized_data[23657536UL];
  uint8_t* __brgemm_attrs_101 = (uint8_t*)&__module_data[4352UL];
  is_init = false;
  __sc_kernel_cache = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs, ((void*)0), ((void*)0));
  __sc_kernel_cache_55 = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_54, ((void*)0), ((void*)0));
  __sc_kernel_cache_57 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_56, ((void*)0), ((void*)0));
  __sc_kernel_cache_59 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_58, ((void*)0), ((void*)0));
  __sc_kernel_cache_61 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_60, ((void*)0), ((void*)0));
  __sc_kernel_cache_63 = dnnl_brgemm_list_func(28, 64, 64, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_62, ((void*)0), ((void*)0));
  __sc_kernel_cache_65 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_64, ((void*)0), ((void*)0));
  __sc_kernel_cache_67 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_66, ((void*)0), ((void*)0));
  __sc_kernel_cache_70 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_69, ((void*)0), ((void*)0));
  __sc_kernel_cache_72 = dnnl_brgemm_list_func(392, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_71, ((void*)0), ((void*)0));
  __sc_kernel_cache_74 = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_73, ((void*)0), ((void*)0));
  __sc_kernel_cache_76 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_75, ((void*)0), ((void*)0));
  __sc_kernel_cache_78 = dnnl_brgemm_list_func(14, 64, 64, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_77, ((void*)0), ((void*)0));
  __sc_kernel_cache_80 = dnnl_brgemm_list_func(28, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_79, ((void*)0), ((void*)0));
  __sc_kernel_cache_86 = dnnl_brgemm_list_func(49, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_85, ((void*)0), ((void*)0));
  __sc_kernel_cache_88 = dnnl_brgemm_list_func(28, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_87, ((void*)0), ((void*)0));
  __sc_kernel_cache_90 = dnnl_brgemm_list_func(7, 256, 64, 128, 256, 256, 0.f, 7, 7, __brgemm_attrs_89, ((void*)0), ((void*)0));
  __sc_kernel_cache_92 = dnnl_brgemm_list_func(49, 512, 512, 512, 512, 512, 0.f, 7, 7, __brgemm_attrs_91, ((void*)0), ((void*)0));
  __sc_kernel_cache_94 = dnnl_brgemm_list_func(49, 64, 512, 512, 64, 64, 0.f, 7, 7, __brgemm_attrs_93, ((void*)0), ((void*)0));
  __sc_kernel_cache_100 = dnnl_brgemm_list_func(49, 256, 512, 512, 256, 256, 0.f, 7, 7, __brgemm_attrs_99, ((void*)0), ((void*)0));
  __sc_kernel_cache_104 = dnnl_brgemm_list_func(49, 512, 64, 64, 512, 512, 0.f, 7, 7, __brgemm_attrs_103, ((void*)0), ((void*)0));
  __brgemm_bd_mask_arr[0] = &__brgemm_full_bd_mask[(0 * 419)];
  __brgemm_bd_mask_arr[1] = &__brgemm_full_bd_mask[(1 * 419)];
  __brgemm_bd_mask_arr_83[0] = &__brgemm_full_bd_mask_82[(0 * 222)];
  __brgemm_bd_mask_arr_97[0] = &__brgemm_full_bd_mask_96[(0 * 61)];
  __sc_kernel_cache_arr[0] = dnnl_brgemm_list_func(419, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_68, __brgemm_bd_mask_arr[0], ((void*)0));
  __sc_kernel_cache_arr[1] = dnnl_brgemm_list_func(419, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_68, __brgemm_bd_mask_arr[1], ((void*)0));
  __sc_kernel_cache_arr_84[0] = dnnl_brgemm_list_func(222, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_81, __brgemm_bd_mask_arr_83[0], ((void*)0));
  __sc_kernel_cache_arr_98[0] = dnnl_brgemm_list_func(61, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_95, __brgemm_bd_mask_arr_97[0], ((void*)0));
  __sc_kernel_cache_arr_102[0] = dnnl_brgemm_list_func(61, 64, 512, 512, 64, 64, 0.f, 7, 7, __brgemm_attrs_101, __brgemm_bd_mask_arr_97[0], ((void*)0));
}
