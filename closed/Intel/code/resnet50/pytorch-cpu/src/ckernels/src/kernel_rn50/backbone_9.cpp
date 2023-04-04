
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t rn50_backbone_bs9_data[222400];
static constexpr int8_t* __module_data = rn50_backbone_bs9_data;
alignas(64) static int8_t __uninitialized_data[23657592UL];

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106)));
static bool batchwise_9_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69, int8_t* __restrict__ __ins_70, float* __restrict__ __ins_71, float* __restrict__ __ins_72, int8_t* __restrict__ __ins_73, float* __restrict__ __ins_74, float* __restrict__ __ins_75, int8_t* __restrict__ __ins_76, float* __restrict__ __ins_77, float* __restrict__ __ins_78, int8_t* __restrict__ __ins_79, float* __restrict__ __ins_80, float* __restrict__ __ins_81, int8_t* __restrict__ __ins_82, float* __restrict__ __ins_83, float* __restrict__ __ins_84, int8_t* __restrict__ __ins_85, float* __restrict__ __ins_86, float* __restrict__ __ins_87, int8_t* __restrict__ __ins_88, float* __restrict__ __ins_89, float* __restrict__ __ins_90, int8_t* __restrict__ __ins_91, float* __restrict__ __ins_92, float* __restrict__ __ins_93, int8_t* __restrict__ __ins_94, float* __restrict__ __ins_95, float* __restrict__ __ins_96, int8_t* __restrict__ __ins_97, float* __restrict__ __ins_98, float* __restrict__ __ins_99, int8_t* __restrict__ __ins_100, float* __restrict__ __ins_101, float* __restrict__ __ins_102, int8_t* __restrict__ __ins_103, float* __restrict__ __ins_104, float* __restrict__ __ins_105, int8_t* __restrict__ __ins_106, float* __restrict__ __ins_107, float* __restrict__ __ins_108, int8_t* __restrict__ __ins_109, float* __restrict__ __ins_110, float* __restrict__ __ins_111, int8_t* __restrict__ __ins_112, float* __restrict__ __ins_113, float* __restrict__ __ins_114, int8_t* __restrict__ __ins_115, float* __restrict__ __ins_116, float* __restrict__ __ins_117, int8_t* __restrict__ __ins_118, float* __restrict__ __ins_119, float* __restrict__ __ins_120, int8_t* __restrict__ __ins_121, float* __restrict__ __ins_122, float* __restrict__ __ins_123, int8_t* __restrict__ __ins_124, float* __restrict__ __ins_125, float* __restrict__ __ins_126) noexcept __attribute__((nonnull (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128)));
static bool res5a_conv_0_cast_mul_add_relu_cast_reorder__681(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_1_cast_mul_add_relu_cast_reorder__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_b_cast_mul_add_cast_reorder__682(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5a_conv_2_cast_mul_add_cast_add_cast__679(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5b_conv_0_cast_mul_add_relu_cast_reorder__678(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_1_cast_mul_add_relu_cast_reorder__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res5c_conv_0_cast_mul_add_relu_cast_reorder__675(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_1_cast_mul_add_relu_cast_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool reorder__105(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool res2a_conv_0_cast_mul_add_relu_cast__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_1_cast_mul_add_relu_cast__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_b_cast_mul_add_cast_reorder__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2b_conv_0_cast_mul_add_relu_cast__20(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_1_cast_mul_add_relu_cast_reorder__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res2c_conv_0_cast_mul_add_relu_cast_reorder__32(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_1_cast_mul_add_relu_cast_reorder__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3a_conv_0_cast_mul_add_relu_cast_reorder__48(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_1_cast_mul_add_relu_cast__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3b_conv_0_cast_mul_add_relu_cast_reorder__60(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_1_cast_mul_add_relu_cast_reorder__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3c_conv_0_cast_mul_add_relu_cast_reorder__72(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_1_cast_mul_add_relu_cast_reorder__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res3d_conv_0_cast_mul_add_relu_cast_reorder__84(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_1_cast_mul_add_relu_cast__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res3d_conv_2_cast_mul_add_cast_add_cast__92(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4a_conv_0_cast_mul_add_relu_cast__100(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_1_cast_mul_add_relu_cast_reorder__104(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_b_cast_mul_add_cast_reorder__96(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4b_conv_0_cast_mul_add_relu_cast_reorder__112(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_1_cast_mul_add_relu_cast_reorder__116(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4c_conv_0_cast_mul_add_relu_cast_reorder__124(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_1_cast_mul_add_relu_cast_reorder__128(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4c_conv_2_cast_mul_add_cast_add_cast__132(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4d_conv_0_cast_mul_add_relu_cast__136(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_1_cast_mul_add_relu_cast_reorder__140(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4d_conv_2_cast_mul_add_cast_add_cast__144(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
static bool res4e_conv_0_cast_mul_add_relu_cast_reorder__148(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_1_cast_mul_add_relu_cast_reorder__152(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __outs_1, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6,7)));
static bool reorder__157(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool res4f_conv_0_cast_mul_add_relu_cast_reorder__161(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_1_cast_mul_add_relu_cast_reorder__165(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept __attribute__((nonnull (1,2,3,4,5)));
static bool res4f_conv_2_cast_mul_add_cast_add_cast__170(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept __attribute__((nonnull (1,2,3,4,5,6)));
extern "C" void* memset(void* ptr, int32_t v, uint64_t len) noexcept;
static bool reorder__481(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__520(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__529(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__427(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__431(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__466(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__469(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__472(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__475(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__478(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__517(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__523(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__526(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__532(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__535(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__432(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__421(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__428(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__499(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__502(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__508(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__511(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__463(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__536(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__539(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__545(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__548(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__554(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__557(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__505(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__514(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__542(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__551(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__560(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__111(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__112(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__108(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__109(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__117(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__118(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__126(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__127(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__135(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__136(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__120(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__121(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__129(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__130(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__141(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__142(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__114(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__115(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__123(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__124(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__132(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__133(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__147(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__148(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__156(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__157(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__165(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__166(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__174(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__175(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__150(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__151(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__159(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__160(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__168(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__169(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__138(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__139(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__180(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__181(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__144(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__145(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__153(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__154(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__162(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__163(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__171(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__172(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__186(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__187(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__195(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__196(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__204(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__205(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__213(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__214(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__222(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__223(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__231(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__232(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__189(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__190(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__198(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__199(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__207(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__208(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__216(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__217(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__225(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__226(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__177(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__178(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__237(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__238(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__183(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__184(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__192(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__193(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__201(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__202(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__210(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__211(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__219(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__220(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__228(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__229(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__525(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__569(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__522(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__515(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__506(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__497(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__528(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__519(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__512(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__503(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__516(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__509(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__500(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__480(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__474(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__465(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__483(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__471(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__464(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__477(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__468(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__461(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__430(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__429(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__419(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__534(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__243(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__244(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__252(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__253(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__261(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__262(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__246(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__247(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__255(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__256(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__234(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__235(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__531(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__240(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__241(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder__537(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__540(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__543(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__249(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__250(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__258(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool cast__259(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__546(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__549(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__552(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__555(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept __attribute__((nonnull (1,2,3)));
static bool reorder__558(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));


extern "C" void rn50_backbone_bs9(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[8608768UL];
  float* folded_const_212 = (float*)&__uninitialized_data[699392UL];
  float* folded_const_211 = (float*)&__uninitialized_data[698368UL];
  int8_t* folded_const_224 = (int8_t*)&__uninitialized_data[3584000UL];
  float* folded_const_252 = (float*)&__uninitialized_data[3599104UL];
  float* folded_const_251 = (float*)&__uninitialized_data[3598848UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[8457216UL];
  float* folded_const_250 = (float*)&__uninitialized_data[3598592UL];
  float* folded_const_249 = (float*)&__uninitialized_data[3598336UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[8592384UL];
  float* folded_const_210 = (float*)&__uninitialized_data[697344UL];
  float* folded_const_209 = (float*)&__uninitialized_data[696320UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[8543232UL];
  float* folded_const_248 = (float*)&__uninitialized_data[3598080UL];
  float* folded_const_247 = (float*)&__uninitialized_data[3597824UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[8420352UL];
  float* folded_const_246 = (float*)&__uninitialized_data[3597568UL];
  float* folded_const_245 = (float*)&__uninitialized_data[3597312UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[8576000UL];
  float* folded_const_208 = (float*)&__uninitialized_data[695296UL];
  float* folded_const_207 = (float*)&__uninitialized_data[694272UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[8526848UL];
  float* folded_const_244 = (float*)&__uninitialized_data[3597056UL];
  float* folded_const_243 = (float*)&__uninitialized_data[3596800UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[8383488UL];
  float* folded_const_242 = (float*)&__uninitialized_data[3596544UL];
  float* folded_const_241 = (float*)&__uninitialized_data[3596288UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[8559616UL];
  float* folded_const_206 = (float*)&__uninitialized_data[693248UL];
  float* folded_const_205 = (float*)&__uninitialized_data[692224UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[7793664UL];
  float* folded_const_180 = (float*)&__uninitialized_data[665600UL];
  float* folded_const_179 = (float*)&__uninitialized_data[663552UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[8494080UL];
  float* folded_const_240 = (float*)&__uninitialized_data[3595776UL];
  float* folded_const_239 = (float*)&__uninitialized_data[3595264UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[7515136UL];
  float* folded_const_238 = (float*)&__uninitialized_data[3594752UL];
  float* folded_const_237 = (float*)&__uninitialized_data[3594240UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[8317952UL];
  float* folded_const_178 = (float*)&__uninitialized_data[661504UL];
  float* folded_const_177 = (float*)&__uninitialized_data[659456UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[8055808UL];
  float* folded_const_236 = (float*)&__uninitialized_data[3593728UL];
  float* folded_const_235 = (float*)&__uninitialized_data[3593216UL];
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[7367680UL];
  float* folded_const_234 = (float*)&__uninitialized_data[3592704UL];
  float* folded_const_233 = (float*)&__uninitialized_data[3592192UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[8252416UL];
  float* folded_const_176 = (float*)&__uninitialized_data[657408UL];
  float* folded_const_175 = (float*)&__uninitialized_data[655360UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[7990272UL];
  float* folded_const_232 = (float*)&__uninitialized_data[3591680UL];
  float* folded_const_231 = (float*)&__uninitialized_data[3591168UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[7220224UL];
  float* folded_const_230 = (float*)&__uninitialized_data[3590656UL];
  float* folded_const_229 = (float*)&__uninitialized_data[3590144UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[8186880UL];
  float* folded_const_174 = (float*)&__uninitialized_data[653312UL];
  float* folded_const_173 = (float*)&__uninitialized_data[651264UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[7924736UL];
  float* folded_const_228 = (float*)&__uninitialized_data[3589632UL];
  float* folded_const_227 = (float*)&__uninitialized_data[3589120UL];
  int8_t* folded_const_259 = (int8_t*)&__uninitialized_data[7072768UL];
  float* folded_const_226 = (float*)&__uninitialized_data[3588608UL];
  float* folded_const_225 = (float*)&__uninitialized_data[3588096UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[8121344UL];
  float* folded_const_172 = (float*)&__uninitialized_data[649216UL];
  float* folded_const_171 = (float*)&__uninitialized_data[647168UL];
  int8_t* folded_const_258 = (int8_t*)&__uninitialized_data[6548480UL];
  float* folded_const_170 = (float*)&__uninitialized_data[643072UL];
  float* folded_const_169 = (float*)&__uninitialized_data[638976UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[7662592UL];
  float* folded_const_204 = (float*)&__uninitialized_data[691200UL];
  float* folded_const_203 = (float*)&__uninitialized_data[690176UL];
  int8_t* folded_const_257 = (int8_t*)&__uninitialized_data[5958656UL];
  float* folded_const_202 = (float*)&__uninitialized_data[689152UL];
  float* folded_const_201 = (float*)&__uninitialized_data[688128UL];
  int8_t* folded_const_223 = (int8_t*)&__uninitialized_data[3321856UL];
  float* folded_const_168 = (float*)&__uninitialized_data[634880UL];
  float* folded_const_167 = (float*)&__uninitialized_data[630784UL];
  int8_t* folded_const_217 = (int8_t*)&__uninitialized_data[1748992UL];
  float* folded_const_200 = (float*)&__uninitialized_data[687104UL];
  float* folded_const_199 = (float*)&__uninitialized_data[686080UL];
  int8_t* folded_const_256 = (int8_t*)&__uninitialized_data[5368832UL];
  float* folded_const_198 = (float*)&__uninitialized_data[685056UL];
  float* folded_const_197 = (float*)&__uninitialized_data[684032UL];
  int8_t* folded_const_222 = (int8_t*)&__uninitialized_data[3059712UL];
  float* folded_const_166 = (float*)&__uninitialized_data[626688UL];
  float* folded_const_165 = (float*)&__uninitialized_data[622592UL];
  int8_t* folded_const_216 = (int8_t*)&__uninitialized_data[1486848UL];
  float* folded_const_196 = (float*)&__uninitialized_data[683008UL];
  float* folded_const_195 = (float*)&__uninitialized_data[681984UL];
  int8_t* folded_const_255 = (int8_t*)&__uninitialized_data[4779008UL];
  float* folded_const_194 = (float*)&__uninitialized_data[680960UL];
  float* folded_const_193 = (float*)&__uninitialized_data[679936UL];
  int8_t* folded_const_221 = (int8_t*)&__uninitialized_data[2797568UL];
  float* folded_const_164 = (float*)&__uninitialized_data[618496UL];
  float* folded_const_163 = (float*)&__uninitialized_data[614400UL];
  int8_t* folded_const_215 = (int8_t*)&__uninitialized_data[1224704UL];
  float* folded_const_192 = (float*)&__uninitialized_data[678912UL];
  float* folded_const_191 = (float*)&__uninitialized_data[677888UL];
  int8_t* folded_const_254 = (int8_t*)&__uninitialized_data[4189184UL];
  float* folded_const_190 = (float*)&__uninitialized_data[676864UL];
  float* folded_const_189 = (float*)&__uninitialized_data[675840UL];
  int8_t* folded_const_220 = (int8_t*)&__uninitialized_data[2535424UL];
  float* folded_const_162 = (float*)&__uninitialized_data[610304UL];
  float* folded_const_161 = (float*)&__uninitialized_data[606208UL];
  int8_t* folded_const_214 = (int8_t*)&__uninitialized_data[962560UL];
  float* folded_const_188 = (float*)&__uninitialized_data[674816UL];
  float* folded_const_187 = (float*)&__uninitialized_data[673792UL];
  int8_t* folded_const_253 = (int8_t*)&__uninitialized_data[3599360UL];
  float* folded_const_186 = (float*)&__uninitialized_data[672768UL];
  float* folded_const_185 = (float*)&__uninitialized_data[671744UL];
  int8_t* folded_const_219 = (int8_t*)&__uninitialized_data[2273280UL];
  float* folded_const_160 = (float*)&__uninitialized_data[602112UL];
  float* folded_const_159 = (float*)&__uninitialized_data[598016UL];
  int8_t* folded_const_213 = (int8_t*)&__uninitialized_data[700416UL];
  float* folded_const_184 = (float*)&__uninitialized_data[670720UL];
  float* folded_const_183 = (float*)&__uninitialized_data[669696UL];
  int8_t* folded_const_156 = (int8_t*)&__uninitialized_data[0UL];
  float* folded_const_182 = (float*)&__uninitialized_data[668672UL];
  float* folded_const_181 = (float*)&__uninitialized_data[667648UL];
  int8_t* folded_const_218 = (int8_t*)&__uninitialized_data[2011136UL];
  float* folded_const_158 = (float*)&__uninitialized_data[593920UL];
  float* folded_const_157 = (float*)&__uninitialized_data[589824UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[8629248UL];
  float* folded_const_283 = (float*)&__uninitialized_data[8627200UL];
  float* folded_const_282 = (float*)&__uninitialized_data[8625152UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[11267072UL];
  float* folded_const_290 = (float*)&__uninitialized_data[13628416UL];
  float* folded_const_289 = (float*)&__uninitialized_data[13626368UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[9153536UL];
  float* folded_const_287 = (float*)&__uninitialized_data[11258880UL];
  float* folded_const_286 = (float*)&__uninitialized_data[11250688UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[13646848UL];
  float* folded_const_292 = (float*)&__uninitialized_data[13638656UL];
  float* folded_const_291 = (float*)&__uninitialized_data[13630464UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[14699520UL];
  float* folded_const_295 = (float*)&__uninitialized_data[14697472UL];
  float* folded_const_294 = (float*)&__uninitialized_data[14695424UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[15752192UL];
  float* folded_const_298 = (float*)&__uninitialized_data[15750144UL];
  float* folded_const_297 = (float*)&__uninitialized_data[15748096UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[18127872UL];
  float* folded_const_301 = (float*)&__uninitialized_data[18119680UL];
  float* folded_const_300 = (float*)&__uninitialized_data[18111488UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[19180544UL];
  float* folded_const_304 = (float*)&__uninitialized_data[19178496UL];
  float* folded_const_303 = (float*)&__uninitialized_data[19176448UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[20233216UL];
  float* folded_const_307 = (float*)&__uninitialized_data[20231168UL];
  float* folded_const_306 = (float*)&__uninitialized_data[20229120UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[22608896UL];
  float* folded_const_310 = (float*)&__uninitialized_data[22600704UL];
  float* folded_const_309 = (float*)&__uninitialized_data[22592512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 3211776UL);
  if (!is_init) {
    __init_const_globals(backbone_output, backbone_input, res2a_weight_b, res2a_bias_b, res2a_weight_0, res2a_bias_0, res2a_weight_1, res2a_bias_1, res2a_weight_2, res2a_bias_2, res2b_weight_0, res2b_bias_0, res2b_weight_1, res2b_bias_1, res2b_weight_2, res2b_bias_2, res2c_weight_0, res2c_bias_0, res2c_weight_1, res2c_bias_1, res2c_weight_2, res2c_bias_2, res3a_weight_b, res3a_bias_b, res3a_weight_0, res3a_bias_0, res3a_weight_1, res3a_bias_1, res3a_weight_2, res3a_bias_2, res3b_weight_0, res3b_bias_0, res3b_weight_1, res3b_bias_1, res3b_weight_2, res3b_bias_2, res3c_weight_0, res3c_bias_0, res3c_weight_1, res3c_bias_1, res3c_weight_2, res3c_bias_2, res3d_weight_0, res3d_bias_0, res3d_weight_1, res3d_bias_1, res3d_weight_2, res3d_bias_2, res4a_weight_b, res4a_bias_b, res4a_weight_0, res4a_bias_0, res4a_weight_1, res4a_bias_1, res4a_weight_2, res4a_bias_2, res4b_weight_0, res4b_bias_0, res4b_weight_1, res4b_bias_1, res4b_weight_2, res4b_bias_2, res4c_weight_0, res4c_bias_0, res4c_weight_1, res4c_bias_1, res4c_weight_2, res4c_bias_2, res4d_weight_0, res4d_bias_0, res4d_weight_1, res4d_bias_1, res4d_weight_2, res4d_bias_2, res4e_weight_0, res4e_bias_0, res4e_weight_1, res4e_bias_1, res4e_weight_2, res4e_bias_2, res4f_weight_0, res4f_bias_0, res4f_weight_1, res4f_bias_1, res4f_weight_2, res4f_bias_2, res5a_weight_b, res5a_bias_b, res5a_weight_0, res5a_bias_0, res5a_weight_1, res5a_bias_1, res5a_weight_2, res5a_bias_2, res5b_weight_0, res5b_bias_0, res5b_weight_1, res5b_bias_1, res5b_weight_2, res5b_bias_2, res5c_weight_0, res5c_bias_0, res5c_weight_1, res5c_bias_1, res5c_weight_2, res5c_bias_2);
  }
  // [u8 [9, 4, 14, 14, 256] @ ABCD256b]
  uint8_t* buffer_578 = (uint8_t*)&__rescheduled_0[0UL];
  batchwise_9_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(buffer_578, &backbone_input[0UL], folded_const_281, folded_const_212, folded_const_211, folded_const_224, folded_const_252, folded_const_251, folded_const_274, folded_const_250, folded_const_249, folded_const_280, folded_const_210, folded_const_209, folded_const_277, folded_const_248, folded_const_247, folded_const_273, folded_const_246, folded_const_245, folded_const_279, folded_const_208, folded_const_207, folded_const_276, folded_const_244, folded_const_243, folded_const_272, folded_const_242, folded_const_241, folded_const_278, folded_const_206, folded_const_205, folded_const_264, folded_const_180, folded_const_179, folded_const_275, folded_const_240, folded_const_239, folded_const_262, folded_const_238, folded_const_237, folded_const_271, folded_const_178, folded_const_177, folded_const_267, folded_const_236, folded_const_235, folded_const_261, folded_const_234, folded_const_233, folded_const_270, folded_const_176, folded_const_175, folded_const_266, folded_const_232, folded_const_231, folded_const_260, folded_const_230, folded_const_229, folded_const_269, folded_const_174, folded_const_173, folded_const_265, folded_const_228, folded_const_227, folded_const_259, folded_const_226, folded_const_225, folded_const_268, folded_const_172, folded_const_171, folded_const_258, folded_const_170, folded_const_169, folded_const_263, folded_const_204, folded_const_203, folded_const_257, folded_const_202, folded_const_201, folded_const_223, folded_const_168, folded_const_167, folded_const_217, folded_const_200, folded_const_199, folded_const_256, folded_const_198, folded_const_197, folded_const_222, folded_const_166, folded_const_165, folded_const_216, folded_const_196, folded_const_195, folded_const_255, folded_const_194, folded_const_193, folded_const_221, folded_const_164, folded_const_163, folded_const_215, folded_const_192, folded_const_191, folded_const_254, folded_const_190, folded_const_189, folded_const_220, folded_const_162, folded_const_161, folded_const_214, folded_const_188, folded_const_187, folded_const_253, folded_const_186, folded_const_185, folded_const_219, folded_const_160, folded_const_159, folded_const_213, folded_const_184, folded_const_183, folded_const_156, folded_const_182, folded_const_181, folded_const_218, folded_const_158, folded_const_157);
  // [s8 [9, 2, 16, 16, 256] @ ABCD256b]
  int8_t* buffer_585 = (int8_t*)&__rescheduled_0[1806336UL];
  res5a_conv_0_cast_mul_add_relu_cast_reorder__681(buffer_585, buffer_578, folded_const_284, folded_const_283, folded_const_282);
  // [s8 [9, 2, 7, 7, 256] @ ABCD256b]
  int8_t* buffer_588 = (int8_t*)&__rescheduled_0[2985984UL];
  res5a_conv_1_cast_mul_add_relu_cast_reorder__680(buffer_588, buffer_585, folded_const_288, folded_const_290, folded_const_289);
  // [s8 [9, 32, 7, 7, 64] @ ABCD64b]
  int8_t* buffer_589 = (int8_t*)&__rescheduled_0[1806336UL];
  res5a_conv_b_cast_mul_add_cast_reorder__682(buffer_589, buffer_578, folded_const_285, folded_const_287, folded_const_286);
  // [u8 [9, 32, 7, 7, 64] @ ABCD64b]
  uint8_t* buffer_603 = (uint8_t*)&__rescheduled_0[0UL];
  res5a_conv_2_cast_mul_add_cast_add_cast__679(buffer_603, buffer_588, folded_const_293, folded_const_292, folded_const_291, buffer_589);
  // [s8 [9, 1, 9, 9, 512] @ ABCD512b]
  int8_t* buffer_604 = (int8_t*)&__rescheduled_0[903168UL];
  res5b_conv_0_cast_mul_add_relu_cast_reorder__678(buffer_604, buffer_603, folded_const_296, folded_const_295, folded_const_294);
  // [s8 [9, 2, 7, 7, 256] @ ABCD256b]
  int8_t* buffer_605 = (int8_t*)&__rescheduled_0[1276416UL];
  res5b_conv_1_cast_mul_add_relu_cast_reorder__677(buffer_605, buffer_604, folded_const_299, folded_const_298, folded_const_297);
  // [u8 [9, 4, 7, 7, 512] @ ABCD512b]
  uint8_t* buffer_615 = (uint8_t*)&__rescheduled_0[1502208UL];
  res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(buffer_615, buffer_605, folded_const_302, folded_const_301, folded_const_300, buffer_603);
  // [s8 [9, 2, 9, 9, 256] @ ABCD256b]
  int8_t* buffer_616 = (int8_t*)&__rescheduled_0[2405376UL];
  res5c_conv_0_cast_mul_add_relu_cast_reorder__675(buffer_616, buffer_615, folded_const_305, folded_const_304, folded_const_303);
  // [s8 [9, 1, 7, 7, 512] @ ABCD512b]
  int8_t* buffer_617 = (int8_t*)&__rescheduled_0[2778624UL];
  res5c_conv_1_cast_mul_add_relu_cast_reorder__674(buffer_617, buffer_616, folded_const_308, folded_const_307, folded_const_306);
  // [s8 [9, 4, 7, 7, 512] @ ABCD512b]
  int8_t* buffer_621 = (int8_t*)&__rescheduled_0[0UL];
  res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(buffer_621, buffer_617, folded_const_311, folded_const_310, folded_const_309, buffer_615);
  reorder__105(backbone_output, buffer_621);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool reorder__481(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_0___fuseiter_1_19 = 0UL; fused_0_fuseiter_0___fuseiter_1_19 < 32UL; fused_0_fuseiter_0___fuseiter_1_19 += 1UL) {
    for (uint64_t _fuseiter_4 = 0UL; _fuseiter_4 < 32UL; _fuseiter_4 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_0___fuseiter_1_19 / 32UL) * 1024UL) + (_fuseiter_4 + ((fused_0_fuseiter_0___fuseiter_1_19 % 32UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_0___fuseiter_1_19 / 32UL) * 1024UL) + (((fused_0_fuseiter_0___fuseiter_1_19 % 32UL) * 32UL) + _fuseiter_4))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__488(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 = 0UL; fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 < 8UL; fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 += 1UL) {
    for (uint64_t _fuseiter_9 = 0UL; _fuseiter_9 < 128UL; _fuseiter_9 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 / 8UL) * 1024UL) + (_fuseiter_9 + ((fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_5___fuseiter_6_20___fuseiter_7_21___fuseiter_8_22 % 8UL) * 128UL) + _fuseiter_9))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__504(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 = 0UL; fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 < 8UL; fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 += 1UL) {
    for (uint64_t _fuseiter_14 = 0UL; _fuseiter_14 < 128UL; _fuseiter_14 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 / 8UL) * 1024UL) + (_fuseiter_14 + ((fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_10___fuseiter_11_23___fuseiter_12_24___fuseiter_13_25 % 8UL) * 128UL) + _fuseiter_14))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__513(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 = 0UL; fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 < 8UL; fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 += 1UL) {
    for (uint64_t _fuseiter_19 = 0UL; _fuseiter_19 < 128UL; _fuseiter_19 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 / 8UL) * 1024UL) + (_fuseiter_19 + ((fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_15___fuseiter_16_26___fuseiter_17_27___fuseiter_18_28 % 8UL) * 128UL) + _fuseiter_19))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__520(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 = 0UL; fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 < 8UL; fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 += 1UL) {
    for (uint64_t _fuseiter_24 = 0UL; _fuseiter_24 < 128UL; _fuseiter_24 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 / 8UL) * 1024UL) + (_fuseiter_24 + ((fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 % 8UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_20___fuseiter_21_29___fuseiter_22_30___fuseiter_23_31 % 8UL) * 128UL) + _fuseiter_24))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__529(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 = 0UL; fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 < 4UL; fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 += 1UL) {
    for (uint64_t _fuseiter_29 = 0UL; _fuseiter_29 < 256UL; _fuseiter_29 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 / 4UL) * 1024UL) + (_fuseiter_29 + ((fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 % 4UL) * 256UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 / 4UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_25___fuseiter_26_32___fuseiter_27_33___fuseiter_28_34 % 4UL) * 256UL) + _fuseiter_29))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__420(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_30___fuseiter_31_35 = 0UL; fused_0_fuseiter_30___fuseiter_31_35 < 16UL; fused_0_fuseiter_30___fuseiter_31_35 += 1UL) {
    for (uint64_t _fuseiter_34 = 0UL; _fuseiter_34 < 16UL; _fuseiter_34 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_30___fuseiter_31_35 / 16UL) * 256UL) + (_fuseiter_34 + ((fused_0_fuseiter_30___fuseiter_31_35 % 16UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_30___fuseiter_31_35 / 16UL) * 256UL) + (((fused_0_fuseiter_30___fuseiter_31_35 % 16UL) * 16UL) + _fuseiter_34))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__424(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 = 0UL; fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 < 2UL; fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 += 1UL) {
    for (uint64_t _fuseiter_39 = 0UL; _fuseiter_39 < 32UL; _fuseiter_39 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 / 2UL) * 64UL) + (_fuseiter_39 + ((fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_35___fuseiter_36_36___fuseiter_37_37___fuseiter_38_38 % 2UL) * 32UL) + _fuseiter_39))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__427(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 = 0UL; fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 < 4UL; fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 += 1UL) {
    for (uint64_t _fuseiter_44 = 0UL; _fuseiter_44 < 64UL; _fuseiter_44 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 / 4UL) * 256UL) + (_fuseiter_44 + ((fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_40___fuseiter_41_39___fuseiter_42_40___fuseiter_43_41 % 4UL) * 64UL) + _fuseiter_44))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__431(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 = 0UL; fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 < 4UL; fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 += 1UL) {
    for (uint64_t _fuseiter_49 = 0UL; _fuseiter_49 < 16UL; _fuseiter_49 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 / 4UL) * 64UL) + (_fuseiter_49 + ((fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 % 4UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 / 4UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_45___fuseiter_46_42___fuseiter_47_43___fuseiter_48_44 % 4UL) * 16UL) + _fuseiter_49))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__434(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 = 0UL; fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 < 8UL; fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 += 1UL) {
    for (uint64_t _fuseiter_54 = 0UL; _fuseiter_54 < 32UL; _fuseiter_54 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 / 8UL) * 256UL) + (_fuseiter_54 + ((fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_50___fuseiter_51_45___fuseiter_52_46___fuseiter_53_47 % 8UL) * 32UL) + _fuseiter_54))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__437(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 = 0UL; fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 < 2UL; fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 += 1UL) {
    for (uint64_t _fuseiter_59 = 0UL; _fuseiter_59 < 32UL; _fuseiter_59 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 / 2UL) * 64UL) + (_fuseiter_59 + ((fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_55___fuseiter_56_48___fuseiter_57_49___fuseiter_58_50 % 2UL) * 32UL) + _fuseiter_59))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__440(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 = 0UL; fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 < 2UL; fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 += 1UL) {
    for (uint64_t _fuseiter_64 = 0UL; _fuseiter_64 < 32UL; _fuseiter_64 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 / 2UL) * 64UL) + (_fuseiter_64 + ((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 % 2UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_51___fuseiter_62_52___fuseiter_63_53 % 2UL) * 32UL) + _fuseiter_64))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__443(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 = 0UL; fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 < 4UL; fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 += 1UL) {
    for (uint64_t _fuseiter_69 = 0UL; _fuseiter_69 < 64UL; _fuseiter_69 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 / 4UL) * 256UL) + (_fuseiter_69 + ((fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_65___fuseiter_66_54___fuseiter_67_55___fuseiter_68_56 % 4UL) * 64UL) + _fuseiter_69))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__446(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_70___fuseiter_71_57 = 0UL; fused_0_fuseiter_70___fuseiter_71_57 < 16UL; fused_0_fuseiter_70___fuseiter_71_57 += 1UL) {
    for (uint64_t _fuseiter_74 = 0UL; _fuseiter_74 < 32UL; _fuseiter_74 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_70___fuseiter_71_57 / 16UL) * 512UL) + (_fuseiter_74 + ((fused_0_fuseiter_70___fuseiter_71_57 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_70___fuseiter_71_57 / 16UL) * 512UL) + (((fused_0_fuseiter_70___fuseiter_71_57 % 16UL) * 32UL) + _fuseiter_74))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__449(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 = 0UL; fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 < 2UL; fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 += 1UL) {
    for (uint64_t _fuseiter_79 = 0UL; _fuseiter_79 < 64UL; _fuseiter_79 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 / 2UL) * 128UL) + (_fuseiter_79 + ((fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_75___fuseiter_76_58___fuseiter_77_59___fuseiter_78_60 % 2UL) * 64UL) + _fuseiter_79))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__452(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 = 0UL; fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 < 4UL; fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 += 1UL) {
    for (uint64_t _fuseiter_84 = 0UL; _fuseiter_84 < 32UL; _fuseiter_84 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 / 4UL) * 128UL) + (_fuseiter_84 + ((fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_80___fuseiter_81_61___fuseiter_82_62___fuseiter_83_63 % 4UL) * 32UL) + _fuseiter_84))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__455(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_85___fuseiter_86_64 = 0UL; fused_0_fuseiter_85___fuseiter_86_64 < 16UL; fused_0_fuseiter_85___fuseiter_86_64 += 1UL) {
    for (uint64_t _fuseiter_89 = 0UL; _fuseiter_89 < 32UL; _fuseiter_89 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_85___fuseiter_86_64 / 16UL) * 512UL) + (_fuseiter_89 + ((fused_0_fuseiter_85___fuseiter_86_64 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_85___fuseiter_86_64 / 16UL) * 512UL) + (((fused_0_fuseiter_85___fuseiter_86_64 % 16UL) * 32UL) + _fuseiter_89))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__458(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 = 0UL; fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 < 4UL; fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 += 1UL) {
    for (uint64_t _fuseiter_94 = 0UL; _fuseiter_94 < 32UL; _fuseiter_94 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 / 4UL) * 128UL) + (_fuseiter_94 + ((fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_90___fuseiter_91_65___fuseiter_92_66___fuseiter_93_67 % 4UL) * 32UL) + _fuseiter_94))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__462(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 = 0UL; fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 < 4UL; fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 += 1UL) {
    for (uint64_t _fuseiter_99 = 0UL; _fuseiter_99 < 128UL; _fuseiter_99 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 / 4UL) * 512UL) + (_fuseiter_99 + ((fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_95___fuseiter_96_68___fuseiter_97_69___fuseiter_98_70 % 4UL) * 128UL) + _fuseiter_99))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__466(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 = 0UL; fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 < 4UL; fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 += 1UL) {
    for (uint64_t _fuseiter_104 = 0UL; _fuseiter_104 < 32UL; _fuseiter_104 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 / 4UL) * 128UL) + (_fuseiter_104 + ((fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_100___fuseiter_101_71___fuseiter_102_72___fuseiter_103_73 % 4UL) * 32UL) + _fuseiter_104))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__469(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 = 0UL; fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 < 8UL; fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 += 1UL) {
    for (uint64_t _fuseiter_109 = 0UL; _fuseiter_109 < 64UL; _fuseiter_109 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 / 8UL) * 512UL) + (_fuseiter_109 + ((fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_105___fuseiter_106_74___fuseiter_107_75___fuseiter_108_76 % 8UL) * 64UL) + _fuseiter_109))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__472(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 = 0UL; fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 < 2UL; fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 += 1UL) {
    for (uint64_t _fuseiter_114 = 0UL; _fuseiter_114 < 64UL; _fuseiter_114 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 / 2UL) * 128UL) + (_fuseiter_114 + ((fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 % 2UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_110___fuseiter_111_77___fuseiter_112_78___fuseiter_113_79 % 2UL) * 64UL) + _fuseiter_114))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__475(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 = 0UL; fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 < 4UL; fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 += 1UL) {
    for (uint64_t _fuseiter_119 = 0UL; _fuseiter_119 < 32UL; _fuseiter_119 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 / 4UL) * 128UL) + (_fuseiter_119 + ((fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 % 4UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_115___fuseiter_116_80___fuseiter_117_81___fuseiter_118_82 % 4UL) * 32UL) + _fuseiter_119))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__478(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 = 0UL; fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 < 4UL; fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 += 1UL) {
    for (uint64_t _fuseiter_124 = 0UL; _fuseiter_124 < 128UL; _fuseiter_124 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 / 4UL) * 512UL) + (_fuseiter_124 + ((fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_120___fuseiter_121_83___fuseiter_122_84___fuseiter_123_85 % 4UL) * 128UL) + _fuseiter_124))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__485(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 = 0UL; fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 < 2UL; fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 += 1UL) {
    for (uint64_t _fuseiter_129 = 0UL; _fuseiter_129 < 128UL; _fuseiter_129 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 / 2UL) * 256UL) + (_fuseiter_129 + ((fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 % 2UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 / 2UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_125___fuseiter_126_86___fuseiter_127_87___fuseiter_128_88 % 2UL) * 128UL) + _fuseiter_129))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__491(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 = 0UL; fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 < 4UL; fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 += 1UL) {
    for (uint64_t _fuseiter_134 = 0UL; _fuseiter_134 < 64UL; _fuseiter_134 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 / 4UL) * 256UL) + (_fuseiter_134 + ((fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_130___fuseiter_131_89___fuseiter_132_90___fuseiter_133_91 % 4UL) * 64UL) + _fuseiter_134))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__494(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_135___fuseiter_136_92 = 0UL; fused_0_fuseiter_135___fuseiter_136_92 < 16UL; fused_0_fuseiter_135___fuseiter_136_92 += 1UL) {
    for (uint64_t _fuseiter_139 = 0UL; _fuseiter_139 < 16UL; _fuseiter_139 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_135___fuseiter_136_92 / 16UL) * 256UL) + (_fuseiter_139 + ((fused_0_fuseiter_135___fuseiter_136_92 % 16UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_135___fuseiter_136_92 / 16UL) * 256UL) + (((fused_0_fuseiter_135___fuseiter_136_92 % 16UL) * 16UL) + _fuseiter_139))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__498(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 = 0UL; fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 < 8UL; fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 += 1UL) {
    for (uint64_t _fuseiter_144 = 0UL; _fuseiter_144 < 32UL; _fuseiter_144 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 / 8UL) * 256UL) + (_fuseiter_144 + ((fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_140___fuseiter_141_93___fuseiter_142_94___fuseiter_143_95 % 8UL) * 32UL) + _fuseiter_144))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__501(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 = 0UL; fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 < 8UL; fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 += 1UL) {
    for (uint64_t _fuseiter_149 = 0UL; _fuseiter_149 < 32UL; _fuseiter_149 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 / 8UL) * 256UL) + (_fuseiter_149 + ((fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_145___fuseiter_146_96___fuseiter_147_97___fuseiter_148_98 % 8UL) * 32UL) + _fuseiter_149))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__507(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 = 0UL; fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 < 4UL; fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 += 1UL) {
    for (uint64_t _fuseiter_154 = 0UL; _fuseiter_154 < 64UL; _fuseiter_154 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 / 4UL) * 256UL) + (_fuseiter_154 + ((fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_150___fuseiter_151_99___fuseiter_152_100___fuseiter_153_101 % 4UL) * 64UL) + _fuseiter_154))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__510(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 = 0UL; fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 < 4UL; fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 += 1UL) {
    for (uint64_t _fuseiter_159 = 0UL; _fuseiter_159 < 64UL; _fuseiter_159 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 / 4UL) * 256UL) + (_fuseiter_159 + ((fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_155___fuseiter_156_102___fuseiter_157_103___fuseiter_158_104 % 4UL) * 64UL) + _fuseiter_159))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__517(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 = 0UL; fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 < 8UL; fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 += 1UL) {
    for (uint64_t _fuseiter_164 = 0UL; _fuseiter_164 < 32UL; _fuseiter_164 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 / 8UL) * 256UL) + (_fuseiter_164 + ((fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_160___fuseiter_161_105___fuseiter_162_106___fuseiter_163_107 % 8UL) * 32UL) + _fuseiter_164))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__523(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 = 0UL; fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 < 8UL; fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 += 1UL) {
    for (uint64_t _fuseiter_169 = 0UL; _fuseiter_169 < 32UL; _fuseiter_169 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 / 8UL) * 256UL) + (_fuseiter_169 + ((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 % 8UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_108___fuseiter_167_109___fuseiter_168_110 % 8UL) * 32UL) + _fuseiter_169))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__526(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 = 0UL; fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 < 4UL; fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 += 1UL) {
    for (uint64_t _fuseiter_174 = 0UL; _fuseiter_174 < 64UL; _fuseiter_174 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 / 4UL) * 256UL) + (_fuseiter_174 + ((fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 % 4UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_170___fuseiter_171_111___fuseiter_172_112___fuseiter_173_113 % 4UL) * 64UL) + _fuseiter_174))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__532(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_175___fuseiter_176_114 = 0UL; fused_0_fuseiter_175___fuseiter_176_114 < 128UL; fused_0_fuseiter_175___fuseiter_176_114 += 1UL) {
    for (uint64_t _fuseiter_179 = 0UL; _fuseiter_179 < 16UL; _fuseiter_179 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_175___fuseiter_176_114 / 128UL) * 2048UL) + (_fuseiter_179 + ((fused_0_fuseiter_175___fuseiter_176_114 % 128UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_175___fuseiter_176_114 / 128UL) * 2048UL) + (((fused_0_fuseiter_175___fuseiter_176_114 % 128UL) * 16UL) + _fuseiter_179))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__535(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 = 0UL; fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 < 8UL; fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 += 1UL) {
    for (uint64_t _fuseiter_184 = 0UL; _fuseiter_184 < 64UL; _fuseiter_184 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 / 8UL) * 512UL) + (_fuseiter_184 + ((fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 % 8UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_180___fuseiter_181_115___fuseiter_182_116___fuseiter_183_117 % 8UL) * 64UL) + _fuseiter_184))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__538(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 = 0UL; fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 < 4UL; fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 += 1UL) {
    for (uint64_t _fuseiter_189 = 0UL; _fuseiter_189 < 128UL; _fuseiter_189 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 / 4UL) * 512UL) + (_fuseiter_189 + ((fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_185___fuseiter_186_118___fuseiter_187_119___fuseiter_188_120 % 4UL) * 128UL) + _fuseiter_189))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__541(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_190___fuseiter_191_121 = 0UL; fused_0_fuseiter_190___fuseiter_191_121 < 32UL; fused_0_fuseiter_190___fuseiter_191_121 += 1UL) {
    for (uint64_t _fuseiter_194 = 0UL; _fuseiter_194 < 64UL; _fuseiter_194 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_190___fuseiter_191_121 / 32UL) * 2048UL) + (_fuseiter_194 + ((fused_0_fuseiter_190___fuseiter_191_121 % 32UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_190___fuseiter_191_121 / 32UL) * 2048UL) + (((fused_0_fuseiter_190___fuseiter_191_121 % 32UL) * 64UL) + _fuseiter_194))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__544(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 = 0UL; fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 < 4UL; fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 += 1UL) {
    for (uint64_t _fuseiter_199 = 0UL; _fuseiter_199 < 128UL; _fuseiter_199 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 / 4UL) * 512UL) + (_fuseiter_199 + ((fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_195___fuseiter_196_122___fuseiter_197_123___fuseiter_198_124 % 4UL) * 128UL) + _fuseiter_199))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__547(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_200___fuseiter_201_125 = 0UL; fused_0_fuseiter_200___fuseiter_201_125 < 32UL; fused_0_fuseiter_200___fuseiter_201_125 += 1UL) {
    for (uint64_t _fuseiter_204 = 0UL; _fuseiter_204 < 16UL; _fuseiter_204 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_200___fuseiter_201_125 / 32UL) * 512UL) + (_fuseiter_204 + ((fused_0_fuseiter_200___fuseiter_201_125 % 32UL) * 16UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_200___fuseiter_201_125 / 32UL) * 512UL) + (((fused_0_fuseiter_200___fuseiter_201_125 % 32UL) * 16UL) + _fuseiter_204))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__550(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_205___fuseiter_206_126 = 0UL; fused_0_fuseiter_205___fuseiter_206_126 < 32UL; fused_0_fuseiter_205___fuseiter_206_126 += 1UL) {
    for (uint64_t _fuseiter_209 = 0UL; _fuseiter_209 < 64UL; _fuseiter_209 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_205___fuseiter_206_126 / 32UL) * 2048UL) + (_fuseiter_209 + ((fused_0_fuseiter_205___fuseiter_206_126 % 32UL) * 64UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_205___fuseiter_206_126 / 32UL) * 2048UL) + (((fused_0_fuseiter_205___fuseiter_206_126 % 32UL) * 64UL) + _fuseiter_209))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__553(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_210___fuseiter_211_127 = 0UL; fused_0_fuseiter_210___fuseiter_211_127 < 16UL; fused_0_fuseiter_210___fuseiter_211_127 += 1UL) {
    for (uint64_t _fuseiter_214 = 0UL; _fuseiter_214 < 32UL; _fuseiter_214 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_210___fuseiter_211_127 / 16UL) * 512UL) + (_fuseiter_214 + ((fused_0_fuseiter_210___fuseiter_211_127 % 16UL) * 32UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_210___fuseiter_211_127 / 16UL) * 512UL) + (((fused_0_fuseiter_210___fuseiter_211_127 % 16UL) * 32UL) + _fuseiter_214))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__556(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 = 0UL; fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 < 4UL; fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 += 1UL) {
    for (uint64_t _fuseiter_219 = 0UL; _fuseiter_219 < 128UL; _fuseiter_219 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 / 4UL) * 512UL) + (_fuseiter_219 + ((fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 % 4UL) * 128UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_215___fuseiter_216_128___fuseiter_217_129___fuseiter_218_130 % 4UL) * 128UL) + _fuseiter_219))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__559(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 = 0UL; fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 < 4UL; fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 += 1UL) {
    for (uint64_t _fuseiter_224 = 0UL; _fuseiter_224 < 512UL; _fuseiter_224 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[(((fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 / 4UL) * 2048UL) + (_fuseiter_224 + ((fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 % 4UL) * 512UL)))];
      float __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 / 4UL) * 2048UL) + (((fused_0fused_0fused_0_fuseiter_220___fuseiter_221_131___fuseiter_222_132___fuseiter_223_133 % 4UL) * 512UL) + _fuseiter_224))] = __cached_1;
    }
  }
  return true;
}

static bool reorder__425(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 = 0UL; fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 < 2UL; fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 += 1UL) {
    for (uint64_t _fuseiter_229 = 0UL; _fuseiter_229 < 32UL; _fuseiter_229 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 / 2UL) * 64UL) + (_fuseiter_229 + ((fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_225___fuseiter_226_134___fuseiter_227_135___fuseiter_228_136 % 2UL) * 32UL) + _fuseiter_229))]);
    }
  }
  return true;
}

static bool reorder__432(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 = 0UL; fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 < 4UL; fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 / 4UL) * 64UL) + ((fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 % 4UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 / 4UL) * 64UL) + ((fused_0fused_0fused_0_fuseiter_230___fuseiter_231_137___fuseiter_232_138___fuseiter_233_139 % 4UL) * 16UL))]);
  }
  return true;
}

static bool reorder__438(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 = 0UL; fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 < 2UL; fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 += 1UL) {
    for (uint64_t _fuseiter_239 = 0UL; _fuseiter_239 < 32UL; _fuseiter_239 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 / 2UL) * 64UL) + (_fuseiter_239 + ((fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_235___fuseiter_236_140___fuseiter_237_141___fuseiter_238_142 % 2UL) * 32UL) + _fuseiter_239))]);
    }
  }
  return true;
}

static bool reorder__441(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 = 0UL; fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 < 2UL; fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 += 1UL) {
    for (uint64_t _fuseiter_244 = 0UL; _fuseiter_244 < 32UL; _fuseiter_244 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 / 2UL) * 64UL) + (_fuseiter_244 + ((fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 % 2UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 / 2UL) * 64UL) + (((fused_0fused_0fused_0_fuseiter_240___fuseiter_241_143___fuseiter_242_144___fuseiter_243_145 % 2UL) * 32UL) + _fuseiter_244))]);
    }
  }
  return true;
}

static bool reorder__450(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 = 0UL; fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 < 2UL; fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 += 1UL) {
    for (uint64_t _fuseiter_249 = 0UL; _fuseiter_249 < 64UL; _fuseiter_249 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 / 2UL) * 128UL) + (_fuseiter_249 + ((fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_245___fuseiter_246_146___fuseiter_247_147___fuseiter_248_148 % 2UL) * 64UL) + _fuseiter_249))]);
    }
  }
  return true;
}

static bool reorder__453(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 = 0UL; fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 < 4UL; fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 += 1UL) {
    for (uint64_t _fuseiter_254 = 0UL; _fuseiter_254 < 32UL; _fuseiter_254 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 / 4UL) * 128UL) + (_fuseiter_254 + ((fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_250___fuseiter_251_149___fuseiter_252_150___fuseiter_253_151 % 4UL) * 32UL) + _fuseiter_254))]);
    }
  }
  return true;
}

static bool reorder__459(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 = 0UL; fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 < 4UL; fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 += 1UL) {
    for (uint64_t _fuseiter_259 = 0UL; _fuseiter_259 < 32UL; _fuseiter_259 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 / 4UL) * 128UL) + (_fuseiter_259 + ((fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_255___fuseiter_256_152___fuseiter_257_153___fuseiter_258_154 % 4UL) * 32UL) + _fuseiter_259))]);
    }
  }
  return true;
}

static bool reorder__467(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 = 0UL; fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 < 4UL; fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 += 1UL) {
    for (uint64_t _fuseiter_264 = 0UL; _fuseiter_264 < 32UL; _fuseiter_264 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 / 4UL) * 128UL) + (_fuseiter_264 + ((fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_260___fuseiter_261_155___fuseiter_262_156___fuseiter_263_157 % 4UL) * 32UL) + _fuseiter_264))]);
    }
  }
  return true;
}

static bool reorder__473(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 = 0UL; fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 < 2UL; fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 += 1UL) {
    for (uint64_t _fuseiter_269 = 0UL; _fuseiter_269 < 64UL; _fuseiter_269 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 / 2UL) * 128UL) + (_fuseiter_269 + ((fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 % 2UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 / 2UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_265___fuseiter_266_158___fuseiter_267_159___fuseiter_268_160 % 2UL) * 64UL) + _fuseiter_269))]);
    }
  }
  return true;
}

static bool reorder__476(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 = 0UL; fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 < 4UL; fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 += 1UL) {
    for (uint64_t _fuseiter_274 = 0UL; _fuseiter_274 < 32UL; _fuseiter_274 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 / 4UL) * 128UL) + (_fuseiter_274 + ((fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 % 4UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 / 4UL) * 128UL) + (((fused_0fused_0fused_0_fuseiter_270___fuseiter_271_161___fuseiter_272_162___fuseiter_273_163 % 4UL) * 32UL) + _fuseiter_274))]);
    }
  }
  return true;
}

static bool reorder__421(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_275___fuseiter_276_164 = 0UL; fused_0_fuseiter_275___fuseiter_276_164 < 16UL; fused_0_fuseiter_275___fuseiter_276_164 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_275___fuseiter_276_164 / 16UL) * 256UL) + ((fused_0_fuseiter_275___fuseiter_276_164 % 16UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_275___fuseiter_276_164 / 16UL) * 256UL) + ((fused_0_fuseiter_275___fuseiter_276_164 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__428(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 = 0UL; fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 < 4UL; fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 += 1UL) {
    for (uint64_t _fuseiter_284 = 0UL; _fuseiter_284 < 64UL; _fuseiter_284 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 / 4UL) * 256UL) + (_fuseiter_284 + ((fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_280___fuseiter_281_165___fuseiter_282_166___fuseiter_283_167 % 4UL) * 64UL) + _fuseiter_284))]);
    }
  }
  return true;
}

static bool reorder__435(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 = 0UL; fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 < 8UL; fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 += 1UL) {
    for (uint64_t _fuseiter_289 = 0UL; _fuseiter_289 < 32UL; _fuseiter_289 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 / 8UL) * 256UL) + (_fuseiter_289 + ((fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_285___fuseiter_286_168___fuseiter_287_169___fuseiter_288_170 % 8UL) * 32UL) + _fuseiter_289))]);
    }
  }
  return true;
}

static bool reorder__444(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 = 0UL; fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 < 4UL; fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 += 1UL) {
    for (uint64_t _fuseiter_294 = 0UL; _fuseiter_294 < 64UL; _fuseiter_294 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 / 4UL) * 256UL) + (_fuseiter_294 + ((fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_290___fuseiter_291_171___fuseiter_292_172___fuseiter_293_173 % 4UL) * 64UL) + _fuseiter_294))]);
    }
  }
  return true;
}

static bool reorder__486(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 = 0UL; fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 < 2UL; fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 += 1UL) {
    for (uint64_t _fuseiter_299 = 0UL; _fuseiter_299 < 128UL; _fuseiter_299 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 / 2UL) * 256UL) + (_fuseiter_299 + ((fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 % 2UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 / 2UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_295___fuseiter_296_174___fuseiter_297_175___fuseiter_298_176 % 2UL) * 128UL) + _fuseiter_299))]);
    }
  }
  return true;
}

static bool reorder__492(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 = 0UL; fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 < 4UL; fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 += 1UL) {
    for (uint64_t _fuseiter_304 = 0UL; _fuseiter_304 < 64UL; _fuseiter_304 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 / 4UL) * 256UL) + (_fuseiter_304 + ((fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_300___fuseiter_301_177___fuseiter_302_178___fuseiter_303_179 % 4UL) * 64UL) + _fuseiter_304))]);
    }
  }
  return true;
}

static bool reorder__495(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_305___fuseiter_306_180 = 0UL; fused_0_fuseiter_305___fuseiter_306_180 < 16UL; fused_0_fuseiter_305___fuseiter_306_180 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_305___fuseiter_306_180 / 16UL) * 256UL) + ((fused_0_fuseiter_305___fuseiter_306_180 % 16UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_305___fuseiter_306_180 / 16UL) * 256UL) + ((fused_0_fuseiter_305___fuseiter_306_180 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__499(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 = 0UL; fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 < 8UL; fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 += 1UL) {
    for (uint64_t _fuseiter_314 = 0UL; _fuseiter_314 < 32UL; _fuseiter_314 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 / 8UL) * 256UL) + (_fuseiter_314 + ((fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_310___fuseiter_311_181___fuseiter_312_182___fuseiter_313_183 % 8UL) * 32UL) + _fuseiter_314))]);
    }
  }
  return true;
}

static bool reorder__502(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 = 0UL; fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 < 8UL; fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 += 1UL) {
    for (uint64_t _fuseiter_319 = 0UL; _fuseiter_319 < 32UL; _fuseiter_319 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 / 8UL) * 256UL) + (_fuseiter_319 + ((fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_315___fuseiter_316_184___fuseiter_317_185___fuseiter_318_186 % 8UL) * 32UL) + _fuseiter_319))]);
    }
  }
  return true;
}

static bool reorder__508(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 = 0UL; fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 < 4UL; fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 += 1UL) {
    for (uint64_t _fuseiter_324 = 0UL; _fuseiter_324 < 64UL; _fuseiter_324 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 / 4UL) * 256UL) + (_fuseiter_324 + ((fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_320___fuseiter_321_187___fuseiter_322_188___fuseiter_323_189 % 4UL) * 64UL) + _fuseiter_324))]);
    }
  }
  return true;
}

static bool reorder__511(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 = 0UL; fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 < 4UL; fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 += 1UL) {
    for (uint64_t _fuseiter_329 = 0UL; _fuseiter_329 < 64UL; _fuseiter_329 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 / 4UL) * 256UL) + (_fuseiter_329 + ((fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_325___fuseiter_326_190___fuseiter_327_191___fuseiter_328_192 % 4UL) * 64UL) + _fuseiter_329))]);
    }
  }
  return true;
}

static bool reorder__518(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 = 0UL; fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 < 8UL; fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 += 1UL) {
    for (uint64_t _fuseiter_334 = 0UL; _fuseiter_334 < 32UL; _fuseiter_334 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 / 8UL) * 256UL) + (_fuseiter_334 + ((fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_330___fuseiter_331_193___fuseiter_332_194___fuseiter_333_195 % 8UL) * 32UL) + _fuseiter_334))]);
    }
  }
  return true;
}

static bool reorder__524(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 = 0UL; fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 < 8UL; fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 += 1UL) {
    for (uint64_t _fuseiter_339 = 0UL; _fuseiter_339 < 32UL; _fuseiter_339 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 / 8UL) * 256UL) + (_fuseiter_339 + ((fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 % 8UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 / 8UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_335___fuseiter_336_196___fuseiter_337_197___fuseiter_338_198 % 8UL) * 32UL) + _fuseiter_339))]);
    }
  }
  return true;
}

static bool reorder__527(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 = 0UL; fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 < 4UL; fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 += 1UL) {
    for (uint64_t _fuseiter_344 = 0UL; _fuseiter_344 < 64UL; _fuseiter_344 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 / 4UL) * 256UL) + (_fuseiter_344 + ((fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 % 4UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 / 4UL) * 256UL) + (((fused_0fused_0fused_0_fuseiter_340___fuseiter_341_199___fuseiter_342_200___fuseiter_343_201 % 4UL) * 64UL) + _fuseiter_344))]);
    }
  }
  return true;
}

static bool reorder__447(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_345___fuseiter_346_202 = 0UL; fused_0_fuseiter_345___fuseiter_346_202 < 16UL; fused_0_fuseiter_345___fuseiter_346_202 += 1UL) {
    for (uint64_t _fuseiter_349 = 0UL; _fuseiter_349 < 32UL; _fuseiter_349 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_345___fuseiter_346_202 / 16UL) * 512UL) + (_fuseiter_349 + ((fused_0_fuseiter_345___fuseiter_346_202 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_345___fuseiter_346_202 / 16UL) * 512UL) + (((fused_0_fuseiter_345___fuseiter_346_202 % 16UL) * 32UL) + _fuseiter_349))]);
    }
  }
  return true;
}

static bool reorder__456(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_350___fuseiter_351_203 = 0UL; fused_0_fuseiter_350___fuseiter_351_203 < 16UL; fused_0_fuseiter_350___fuseiter_351_203 += 1UL) {
    for (uint64_t _fuseiter_354 = 0UL; _fuseiter_354 < 32UL; _fuseiter_354 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_350___fuseiter_351_203 / 16UL) * 512UL) + (_fuseiter_354 + ((fused_0_fuseiter_350___fuseiter_351_203 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_350___fuseiter_351_203 / 16UL) * 512UL) + (((fused_0_fuseiter_350___fuseiter_351_203 % 16UL) * 32UL) + _fuseiter_354))]);
    }
  }
  return true;
}

static bool reorder__463(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 = 0UL; fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 < 4UL; fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 += 1UL) {
    for (uint64_t _fuseiter_359 = 0UL; _fuseiter_359 < 128UL; _fuseiter_359 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 / 4UL) * 512UL) + (_fuseiter_359 + ((fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_355___fuseiter_356_204___fuseiter_357_205___fuseiter_358_206 % 4UL) * 128UL) + _fuseiter_359))]);
    }
  }
  return true;
}

static bool reorder__470(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 = 0UL; fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 < 8UL; fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 += 1UL) {
    for (uint64_t _fuseiter_364 = 0UL; _fuseiter_364 < 64UL; _fuseiter_364 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 / 8UL) * 512UL) + (_fuseiter_364 + ((fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_360___fuseiter_361_207___fuseiter_362_208___fuseiter_363_209 % 8UL) * 64UL) + _fuseiter_364))]);
    }
  }
  return true;
}

static bool reorder__479(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 = 0UL; fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 < 4UL; fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 += 1UL) {
    for (uint64_t _fuseiter_369 = 0UL; _fuseiter_369 < 128UL; _fuseiter_369 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 / 4UL) * 512UL) + (_fuseiter_369 + ((fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_365___fuseiter_366_210___fuseiter_367_211___fuseiter_368_212 % 4UL) * 128UL) + _fuseiter_369))]);
    }
  }
  return true;
}

static bool reorder__536(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 = 0UL; fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 < 8UL; fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 += 1UL) {
    for (uint64_t _fuseiter_374 = 0UL; _fuseiter_374 < 64UL; _fuseiter_374 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 / 8UL) * 512UL) + (_fuseiter_374 + ((fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 % 8UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 / 8UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_370___fuseiter_371_213___fuseiter_372_214___fuseiter_373_215 % 8UL) * 64UL) + _fuseiter_374))]);
    }
  }
  return true;
}

static bool reorder__539(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 = 0UL; fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 < 4UL; fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 += 1UL) {
    for (uint64_t _fuseiter_379 = 0UL; _fuseiter_379 < 128UL; _fuseiter_379 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 / 4UL) * 512UL) + (_fuseiter_379 + ((fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_375___fuseiter_376_216___fuseiter_377_217___fuseiter_378_218 % 4UL) * 128UL) + _fuseiter_379))]);
    }
  }
  return true;
}

static bool reorder__545(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 = 0UL; fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 < 4UL; fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 += 1UL) {
    for (uint64_t _fuseiter_384 = 0UL; _fuseiter_384 < 128UL; _fuseiter_384 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 / 4UL) * 512UL) + (_fuseiter_384 + ((fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_380___fuseiter_381_219___fuseiter_382_220___fuseiter_383_221 % 4UL) * 128UL) + _fuseiter_384))]);
    }
  }
  return true;
}

static bool reorder__548(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_385___fuseiter_386_222 = 0UL; fused_0_fuseiter_385___fuseiter_386_222 < 32UL; fused_0_fuseiter_385___fuseiter_386_222 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_385___fuseiter_386_222 / 32UL) * 512UL) + ((fused_0_fuseiter_385___fuseiter_386_222 % 32UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_385___fuseiter_386_222 / 32UL) * 512UL) + ((fused_0_fuseiter_385___fuseiter_386_222 % 32UL) * 16UL))]);
  }
  return true;
}

static bool reorder__554(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_390___fuseiter_391_223 = 0UL; fused_0_fuseiter_390___fuseiter_391_223 < 16UL; fused_0_fuseiter_390___fuseiter_391_223 += 1UL) {
    for (uint64_t _fuseiter_394 = 0UL; _fuseiter_394 < 32UL; _fuseiter_394 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_390___fuseiter_391_223 / 16UL) * 512UL) + (_fuseiter_394 + ((fused_0_fuseiter_390___fuseiter_391_223 % 16UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_390___fuseiter_391_223 / 16UL) * 512UL) + (((fused_0_fuseiter_390___fuseiter_391_223 % 16UL) * 32UL) + _fuseiter_394))]);
    }
  }
  return true;
}

static bool reorder__557(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 = 0UL; fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 < 4UL; fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 += 1UL) {
    for (uint64_t _fuseiter_399 = 0UL; _fuseiter_399 < 128UL; _fuseiter_399 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 / 4UL) * 512UL) + (_fuseiter_399 + ((fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 % 4UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 / 4UL) * 512UL) + (((fused_0fused_0fused_0_fuseiter_395___fuseiter_396_224___fuseiter_397_225___fuseiter_398_226 % 4UL) * 128UL) + _fuseiter_399))]);
    }
  }
  return true;
}

static bool reorder__482(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_400___fuseiter_401_227 = 0UL; fused_0_fuseiter_400___fuseiter_401_227 < 32UL; fused_0_fuseiter_400___fuseiter_401_227 += 1UL) {
    for (uint64_t _fuseiter_404 = 0UL; _fuseiter_404 < 32UL; _fuseiter_404 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_400___fuseiter_401_227 / 32UL) * 1024UL) + (_fuseiter_404 + ((fused_0_fuseiter_400___fuseiter_401_227 % 32UL) * 32UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_400___fuseiter_401_227 / 32UL) * 1024UL) + (((fused_0_fuseiter_400___fuseiter_401_227 % 32UL) * 32UL) + _fuseiter_404))]);
    }
  }
  return true;
}

static bool reorder__489(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 = 0UL; fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 < 8UL; fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 += 1UL) {
    for (uint64_t _fuseiter_409 = 0UL; _fuseiter_409 < 128UL; _fuseiter_409 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 / 8UL) * 1024UL) + (_fuseiter_409 + ((fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_405___fuseiter_406_228___fuseiter_407_229___fuseiter_408_230 % 8UL) * 128UL) + _fuseiter_409))]);
    }
  }
  return true;
}

static bool reorder__505(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 = 0UL; fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 < 8UL; fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 += 1UL) {
    for (uint64_t _fuseiter_414 = 0UL; _fuseiter_414 < 128UL; _fuseiter_414 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 / 8UL) * 1024UL) + (_fuseiter_414 + ((fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_410___fuseiter_411_231___fuseiter_412_232___fuseiter_413_233 % 8UL) * 128UL) + _fuseiter_414))]);
    }
  }
  return true;
}

static bool reorder__514(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 = 0UL; fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 < 8UL; fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 += 1UL) {
    for (uint64_t _fuseiter_419 = 0UL; _fuseiter_419 < 128UL; _fuseiter_419 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 / 8UL) * 1024UL) + (_fuseiter_419 + ((fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_415___fuseiter_416_234___fuseiter_417_235___fuseiter_418_236 % 8UL) * 128UL) + _fuseiter_419))]);
    }
  }
  return true;
}

static bool reorder__521(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 = 0UL; fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 < 8UL; fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 += 1UL) {
    for (uint64_t _fuseiter_424 = 0UL; _fuseiter_424 < 128UL; _fuseiter_424 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 / 8UL) * 1024UL) + (_fuseiter_424 + ((fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 % 8UL) * 128UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 / 8UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_420___fuseiter_421_237___fuseiter_422_238___fuseiter_423_239 % 8UL) * 128UL) + _fuseiter_424))]);
    }
  }
  return true;
}

static bool reorder__530(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 = 0UL; fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 < 4UL; fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 += 1UL) {
    for (uint64_t _fuseiter_429 = 0UL; _fuseiter_429 < 256UL; _fuseiter_429 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 / 4UL) * 1024UL) + (_fuseiter_429 + ((fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 % 4UL) * 256UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 / 4UL) * 1024UL) + (((fused_0fused_0fused_0_fuseiter_425___fuseiter_426_240___fuseiter_427_241___fuseiter_428_242 % 4UL) * 256UL) + _fuseiter_429))]);
    }
  }
  return true;
}

static bool reorder__533(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_430___fuseiter_431_243 = 0UL; fused_0_fuseiter_430___fuseiter_431_243 < 128UL; fused_0_fuseiter_430___fuseiter_431_243 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_430___fuseiter_431_243 / 128UL) * 2048UL) + ((fused_0_fuseiter_430___fuseiter_431_243 % 128UL) * 16UL))]);
    vec_f32x16 __cached_1;
    __cached_1 = __cached_0;
    vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_430___fuseiter_431_243 / 128UL) * 2048UL) + ((fused_0_fuseiter_430___fuseiter_431_243 % 128UL) * 16UL))]);
  }
  return true;
}

static bool reorder__542(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_435___fuseiter_436_244 = 0UL; fused_0_fuseiter_435___fuseiter_436_244 < 32UL; fused_0_fuseiter_435___fuseiter_436_244 += 1UL) {
    for (uint64_t _fuseiter_439 = 0UL; _fuseiter_439 < 64UL; _fuseiter_439 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_435___fuseiter_436_244 / 32UL) * 2048UL) + (_fuseiter_439 + ((fused_0_fuseiter_435___fuseiter_436_244 % 32UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_435___fuseiter_436_244 / 32UL) * 2048UL) + (((fused_0_fuseiter_435___fuseiter_436_244 % 32UL) * 64UL) + _fuseiter_439))]);
    }
  }
  return true;
}

static bool reorder__551(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_440___fuseiter_441_245 = 0UL; fused_0_fuseiter_440___fuseiter_441_245 < 32UL; fused_0_fuseiter_440___fuseiter_441_245 += 1UL) {
    for (uint64_t _fuseiter_444 = 0UL; _fuseiter_444 < 64UL; _fuseiter_444 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0_fuseiter_440___fuseiter_441_245 / 32UL) * 2048UL) + (_fuseiter_444 + ((fused_0_fuseiter_440___fuseiter_441_245 % 32UL) * 64UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0_fuseiter_440___fuseiter_441_245 / 32UL) * 2048UL) + (((fused_0_fuseiter_440___fuseiter_441_245 % 32UL) * 64UL) + _fuseiter_444))]);
    }
  }
  return true;
}

static bool reorder__560(float* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 = 0UL; fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 < 4UL; fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 += 1UL) {
    for (uint64_t _fuseiter_449 = 0UL; _fuseiter_449 < 512UL; _fuseiter_449 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 / 4UL) * 2048UL) + (_fuseiter_449 + ((fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 % 4UL) * 512UL)))]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16::store(__cached_1, &__outs_0[(((fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 / 4UL) * 2048UL) + (((fused_0fused_0fused_0_fuseiter_445___fuseiter_446_246___fuseiter_447_247___fuseiter_448_248 % 4UL) * 512UL) + _fuseiter_449))]);
    }
  }
  return true;
}

static bool mul__111(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_249____itr_2_250 = 0UL; fused_0fused_0__itr_0____itr_1_249____itr_2_250 < 4096UL; fused_0fused_0__itr_0____itr_1_249____itr_2_250 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_249____itr_2_250 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_249____itr_2_250 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_249____itr_2_250 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_249____itr_2_250 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_249____itr_2_250 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__112(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_251____itr_2_252 = 0UL; fused_0fused_0__itr_0____itr_1_251____itr_2_252 < 4096UL; fused_0fused_0__itr_0____itr_1_251____itr_2_252 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_251____itr_2_252 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_251____itr_2_252 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_251____itr_2_252 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_251____itr_2_252 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__108(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_253____itr_2_254 = 0UL; fused_0fused_0__itr_0____itr_1_253____itr_2_254 < 16384UL; fused_0fused_0__itr_0____itr_1_253____itr_2_254 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_253____itr_2_254 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_253____itr_2_254 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_253____itr_2_254 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_253____itr_2_254 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_253____itr_2_254 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__109(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_255____itr_2_256 = 0UL; fused_0fused_0__itr_0____itr_1_255____itr_2_256 < 16384UL; fused_0fused_0__itr_0____itr_1_255____itr_2_256 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_255____itr_2_256 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_255____itr_2_256 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_255____itr_2_256 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_255____itr_2_256 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__117(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_257____itr_2_258 = 0UL; fused_0fused_0__itr_0____itr_1_257____itr_2_258 < 16384UL; fused_0fused_0__itr_0____itr_1_257____itr_2_258 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_257____itr_2_258 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_257____itr_2_258 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_257____itr_2_258 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_257____itr_2_258 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_257____itr_2_258 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__118(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_259____itr_2_260 = 0UL; fused_0fused_0__itr_0____itr_1_259____itr_2_260 < 16384UL; fused_0fused_0__itr_0____itr_1_259____itr_2_260 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_259____itr_2_260 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_259____itr_2_260 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_259____itr_2_260 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_259____itr_2_260 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__126(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_261____itr_2_262 = 0UL; fused_0fused_0__itr_0____itr_1_261____itr_2_262 < 16384UL; fused_0fused_0__itr_0____itr_1_261____itr_2_262 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_261____itr_2_262 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_261____itr_2_262 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_261____itr_2_262 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_261____itr_2_262 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_261____itr_2_262 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__127(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_263____itr_2_264 = 0UL; fused_0fused_0__itr_0____itr_1_263____itr_2_264 < 16384UL; fused_0fused_0__itr_0____itr_1_263____itr_2_264 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_263____itr_2_264 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_263____itr_2_264 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_263____itr_2_264 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_263____itr_2_264 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__135(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_265____itr_2_266 = 0UL; fused_0fused_0__itr_0____itr_1_265____itr_2_266 < 16384UL; fused_0fused_0__itr_0____itr_1_265____itr_2_266 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_265____itr_2_266 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_265____itr_2_266 % 64UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_265____itr_2_266 / 64UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_265____itr_2_266 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_265____itr_2_266 % 64UL))] = __cached_2;
  }
  return true;
}

static bool cast__136(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_267____itr_2_268 = 0UL; fused_0fused_0__itr_0____itr_1_267____itr_2_268 < 16384UL; fused_0fused_0__itr_0____itr_1_267____itr_2_268 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_267____itr_2_268 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_267____itr_2_268 % 64UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_267____itr_2_268 / 64UL) * 64UL) + (fused_0fused_0__itr_0____itr_1_267____itr_2_268 % 64UL))] = __cached_1;
  }
  return true;
}

static bool mul__120(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_269____itr_2_270 = 0UL; fused_0fused_0__itr_0____itr_1_269____itr_2_270 < 16384UL; fused_0fused_0__itr_0____itr_1_269____itr_2_270 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_269____itr_2_270 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_269____itr_2_270 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_269____itr_2_270 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_269____itr_2_270 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_269____itr_2_270 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__121(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_271____itr_2_272 = 0UL; fused_0fused_0__itr_0____itr_1_271____itr_2_272 < 16384UL; fused_0fused_0__itr_0____itr_1_271____itr_2_272 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_271____itr_2_272 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_271____itr_2_272 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_271____itr_2_272 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_271____itr_2_272 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__129(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_273____itr_2_274 = 0UL; fused_0fused_0__itr_0____itr_1_273____itr_2_274 < 16384UL; fused_0fused_0__itr_0____itr_1_273____itr_2_274 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_273____itr_2_274 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_273____itr_2_274 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_273____itr_2_274 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_273____itr_2_274 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_273____itr_2_274 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__130(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_275____itr_2_276 = 0UL; fused_0fused_0__itr_0____itr_1_275____itr_2_276 < 16384UL; fused_0fused_0__itr_0____itr_1_275____itr_2_276 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_275____itr_2_276 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_275____itr_2_276 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_275____itr_2_276 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_275____itr_2_276 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__141(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_277____itr_2_278 = 0UL; fused_0fused_0__itr_0____itr_1_277____itr_2_278 < 32768UL; fused_0fused_0__itr_0____itr_1_277____itr_2_278 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_277____itr_2_278 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_277____itr_2_278 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_277____itr_2_278 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_277____itr_2_278 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_277____itr_2_278 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__142(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_279____itr_2_280 = 0UL; fused_0fused_0__itr_0____itr_1_279____itr_2_280 < 32768UL; fused_0fused_0__itr_0____itr_1_279____itr_2_280 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_279____itr_2_280 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_279____itr_2_280 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_279____itr_2_280 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_279____itr_2_280 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__114(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_281____itr_2_282 = 0UL; fused_0fused_0__itr_0____itr_1_281____itr_2_282 < 12288UL; fused_0fused_0__itr_0____itr_1_281____itr_2_282 += 1UL) {
    for (uint64_t _fuseiter_534 = 0UL; _fuseiter_534 < 3UL; _fuseiter_534 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_281____itr_2_282 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_281____itr_2_282 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_281____itr_2_282 % 3UL) * 3UL))) + _fuseiter_534)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_281____itr_2_282 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_281____itr_2_282 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_281____itr_2_282 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_281____itr_2_282 % 3UL) * 3UL))) + _fuseiter_534)] = __cached_2;
    }
  }
  return true;
}

static bool cast__115(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_283____itr_2_284 = 0UL; fused_0fused_0__itr_0____itr_1_283____itr_2_284 < 12288UL; fused_0fused_0__itr_0____itr_1_283____itr_2_284 += 1UL) {
    for (uint64_t _fuseiter539 = 0UL; _fuseiter539 < 3UL; _fuseiter539 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_283____itr_2_284 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_283____itr_2_284 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_283____itr_2_284 % 3UL) * 3UL))) + _fuseiter539)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_283____itr_2_284 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_283____itr_2_284 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_283____itr_2_284 % 3UL) * 3UL))) + _fuseiter539)] = __cached_1;
    }
  }
  return true;
}

static bool mul__123(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_285____itr_2_286 = 0UL; fused_0fused_0__itr_0____itr_1_285____itr_2_286 < 12288UL; fused_0fused_0__itr_0____itr_1_285____itr_2_286 += 1UL) {
    for (uint64_t _fuseiter_544 = 0UL; _fuseiter_544 < 3UL; _fuseiter_544 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_285____itr_2_286 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_285____itr_2_286 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_285____itr_2_286 % 3UL) * 3UL))) + _fuseiter_544)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_285____itr_2_286 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_285____itr_2_286 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_285____itr_2_286 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_285____itr_2_286 % 3UL) * 3UL))) + _fuseiter_544)] = __cached_2;
    }
  }
  return true;
}

static bool cast__124(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_287____itr_2_288 = 0UL; fused_0fused_0__itr_0____itr_1_287____itr_2_288 < 12288UL; fused_0fused_0__itr_0____itr_1_287____itr_2_288 += 1UL) {
    for (uint64_t _fuseiter549 = 0UL; _fuseiter549 < 3UL; _fuseiter549 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_287____itr_2_288 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_287____itr_2_288 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_287____itr_2_288 % 3UL) * 3UL))) + _fuseiter549)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_287____itr_2_288 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_287____itr_2_288 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_287____itr_2_288 % 3UL) * 3UL))) + _fuseiter549)] = __cached_1;
    }
  }
  return true;
}

static bool mul__132(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_289____itr_2_290 = 0UL; fused_0fused_0__itr_0____itr_1_289____itr_2_290 < 12288UL; fused_0fused_0__itr_0____itr_1_289____itr_2_290 += 1UL) {
    for (uint64_t _fuseiter_554 = 0UL; _fuseiter_554 < 3UL; _fuseiter_554 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_289____itr_2_290 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_289____itr_2_290 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_289____itr_2_290 % 3UL) * 3UL))) + _fuseiter_554)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_289____itr_2_290 / 192UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_289____itr_2_290 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_289____itr_2_290 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_289____itr_2_290 % 3UL) * 3UL))) + _fuseiter_554)] = __cached_2;
    }
  }
  return true;
}

static bool cast__133(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_291____itr_2_292 = 0UL; fused_0fused_0__itr_0____itr_1_291____itr_2_292 < 12288UL; fused_0fused_0__itr_0____itr_1_291____itr_2_292 += 1UL) {
    for (uint64_t _fuseiter559 = 0UL; _fuseiter559 < 3UL; _fuseiter559 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_291____itr_2_292 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_291____itr_2_292 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_291____itr_2_292 % 3UL) * 3UL))) + _fuseiter559)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_291____itr_2_292 / 192UL) * 576UL) + ((((fused_0fused_0__itr_0____itr_1_291____itr_2_292 / 3UL) % 64UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_291____itr_2_292 % 3UL) * 3UL))) + _fuseiter559)] = __cached_1;
    }
  }
  return true;
}

static bool mul__147(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_293____itr_2_294 = 0UL; fused_0fused_0__itr_0____itr_1_293____itr_2_294 < 65536UL; fused_0fused_0__itr_0____itr_1_293____itr_2_294 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_293____itr_2_294 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_293____itr_2_294 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_293____itr_2_294 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_293____itr_2_294 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_293____itr_2_294 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__148(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_295____itr_2_296 = 0UL; fused_0fused_0__itr_0____itr_1_295____itr_2_296 < 65536UL; fused_0fused_0__itr_0____itr_1_295____itr_2_296 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_295____itr_2_296 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_295____itr_2_296 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_295____itr_2_296 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_295____itr_2_296 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__156(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_297____itr_2_298 = 0UL; fused_0fused_0__itr_0____itr_1_297____itr_2_298 < 65536UL; fused_0fused_0__itr_0____itr_1_297____itr_2_298 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_297____itr_2_298 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_297____itr_2_298 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_297____itr_2_298 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_297____itr_2_298 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_297____itr_2_298 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__157(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_299____itr_2_300 = 0UL; fused_0fused_0__itr_0____itr_1_299____itr_2_300 < 65536UL; fused_0fused_0__itr_0____itr_1_299____itr_2_300 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_299____itr_2_300 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_299____itr_2_300 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_299____itr_2_300 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_299____itr_2_300 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__165(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_301____itr_2_302 = 0UL; fused_0fused_0__itr_0____itr_1_301____itr_2_302 < 65536UL; fused_0fused_0__itr_0____itr_1_301____itr_2_302 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_301____itr_2_302 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_301____itr_2_302 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_301____itr_2_302 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_301____itr_2_302 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_301____itr_2_302 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__166(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_303____itr_2_304 = 0UL; fused_0fused_0__itr_0____itr_1_303____itr_2_304 < 65536UL; fused_0fused_0__itr_0____itr_1_303____itr_2_304 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_303____itr_2_304 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_303____itr_2_304 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_303____itr_2_304 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_303____itr_2_304 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__174(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_305____itr_2_306 = 0UL; fused_0fused_0__itr_0____itr_1_305____itr_2_306 < 65536UL; fused_0fused_0__itr_0____itr_1_305____itr_2_306 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_305____itr_2_306 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_305____itr_2_306 % 128UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_305____itr_2_306 / 128UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_305____itr_2_306 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_305____itr_2_306 % 128UL))] = __cached_2;
  }
  return true;
}

static bool cast__175(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_307____itr_2_308 = 0UL; fused_0fused_0__itr_0____itr_1_307____itr_2_308 < 65536UL; fused_0fused_0__itr_0____itr_1_307____itr_2_308 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_307____itr_2_308 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_307____itr_2_308 % 128UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_307____itr_2_308 / 128UL) * 128UL) + (fused_0fused_0__itr_0____itr_1_307____itr_2_308 % 128UL))] = __cached_1;
  }
  return true;
}

static bool mul__150(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_309____itr_2_310 = 0UL; fused_0fused_0__itr_0____itr_1_309____itr_2_310 < 65536UL; fused_0fused_0__itr_0____itr_1_309____itr_2_310 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_309____itr_2_310 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_309____itr_2_310 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_309____itr_2_310 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_309____itr_2_310 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_309____itr_2_310 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__151(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_311____itr_2_312 = 0UL; fused_0fused_0__itr_0____itr_1_311____itr_2_312 < 65536UL; fused_0fused_0__itr_0____itr_1_311____itr_2_312 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_311____itr_2_312 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_311____itr_2_312 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_311____itr_2_312 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_311____itr_2_312 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__159(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_313____itr_2_314 = 0UL; fused_0fused_0__itr_0____itr_1_313____itr_2_314 < 65536UL; fused_0fused_0__itr_0____itr_1_313____itr_2_314 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_313____itr_2_314 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_313____itr_2_314 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_313____itr_2_314 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_313____itr_2_314 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_313____itr_2_314 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__160(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_315____itr_2_316 = 0UL; fused_0fused_0__itr_0____itr_1_315____itr_2_316 < 65536UL; fused_0fused_0__itr_0____itr_1_315____itr_2_316 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_315____itr_2_316 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_315____itr_2_316 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_315____itr_2_316 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_315____itr_2_316 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__168(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_317____itr_2_318 = 0UL; fused_0fused_0__itr_0____itr_1_317____itr_2_318 < 65536UL; fused_0fused_0__itr_0____itr_1_317____itr_2_318 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_317____itr_2_318 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_317____itr_2_318 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_317____itr_2_318 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_317____itr_2_318 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_317____itr_2_318 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__169(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_319____itr_2_320 = 0UL; fused_0fused_0__itr_0____itr_1_319____itr_2_320 < 65536UL; fused_0fused_0__itr_0____itr_1_319____itr_2_320 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_319____itr_2_320 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_319____itr_2_320 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_319____itr_2_320 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_319____itr_2_320 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__138(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_321____itr_2_322 = 0UL; fused_0fused_0__itr_0____itr_1_321____itr_2_322 < 131072UL; fused_0fused_0__itr_0____itr_1_321____itr_2_322 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_321____itr_2_322 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_321____itr_2_322 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_321____itr_2_322 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_321____itr_2_322 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_321____itr_2_322 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__139(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_323____itr_2_324 = 0UL; fused_0fused_0__itr_0____itr_1_323____itr_2_324 < 131072UL; fused_0fused_0__itr_0____itr_1_323____itr_2_324 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_323____itr_2_324 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_323____itr_2_324 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_323____itr_2_324 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_323____itr_2_324 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__180(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_325____itr_2_326 = 0UL; fused_0fused_0__itr_0____itr_1_325____itr_2_326 < 131072UL; fused_0fused_0__itr_0____itr_1_325____itr_2_326 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_325____itr_2_326 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_325____itr_2_326 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_325____itr_2_326 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_325____itr_2_326 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_325____itr_2_326 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__181(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_327____itr_2_328 = 0UL; fused_0fused_0__itr_0____itr_1_327____itr_2_328 < 131072UL; fused_0fused_0__itr_0____itr_1_327____itr_2_328 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_327____itr_2_328 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_327____itr_2_328 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_327____itr_2_328 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_327____itr_2_328 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__144(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_329____itr_2_330 = 0UL; fused_0fused_0__itr_0____itr_1_329____itr_2_330 < 49152UL; fused_0fused_0__itr_0____itr_1_329____itr_2_330 += 1UL) {
    for (uint64_t _fuseiter_654 = 0UL; _fuseiter_654 < 3UL; _fuseiter_654 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_329____itr_2_330 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_329____itr_2_330 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_329____itr_2_330 % 3UL) * 3UL))) + _fuseiter_654)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_329____itr_2_330 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_329____itr_2_330 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_329____itr_2_330 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_329____itr_2_330 % 3UL) * 3UL))) + _fuseiter_654)] = __cached_2;
    }
  }
  return true;
}

static bool cast__145(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_331____itr_2_332 = 0UL; fused_0fused_0__itr_0____itr_1_331____itr_2_332 < 49152UL; fused_0fused_0__itr_0____itr_1_331____itr_2_332 += 1UL) {
    for (uint64_t _fuseiter659 = 0UL; _fuseiter659 < 3UL; _fuseiter659 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_331____itr_2_332 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_331____itr_2_332 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_331____itr_2_332 % 3UL) * 3UL))) + _fuseiter659)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_331____itr_2_332 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_331____itr_2_332 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_331____itr_2_332 % 3UL) * 3UL))) + _fuseiter659)] = __cached_1;
    }
  }
  return true;
}

static bool mul__153(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_333____itr_2_334 = 0UL; fused_0fused_0__itr_0____itr_1_333____itr_2_334 < 49152UL; fused_0fused_0__itr_0____itr_1_333____itr_2_334 += 1UL) {
    for (uint64_t _fuseiter_664 = 0UL; _fuseiter_664 < 3UL; _fuseiter_664 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_333____itr_2_334 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_333____itr_2_334 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_333____itr_2_334 % 3UL) * 3UL))) + _fuseiter_664)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_333____itr_2_334 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_333____itr_2_334 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_333____itr_2_334 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_333____itr_2_334 % 3UL) * 3UL))) + _fuseiter_664)] = __cached_2;
    }
  }
  return true;
}

static bool cast__154(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_335____itr_2_336 = 0UL; fused_0fused_0__itr_0____itr_1_335____itr_2_336 < 49152UL; fused_0fused_0__itr_0____itr_1_335____itr_2_336 += 1UL) {
    for (uint64_t _fuseiter669 = 0UL; _fuseiter669 < 3UL; _fuseiter669 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_335____itr_2_336 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_335____itr_2_336 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_335____itr_2_336 % 3UL) * 3UL))) + _fuseiter669)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_335____itr_2_336 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_335____itr_2_336 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_335____itr_2_336 % 3UL) * 3UL))) + _fuseiter669)] = __cached_1;
    }
  }
  return true;
}

static bool mul__162(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_337____itr_2_338 = 0UL; fused_0fused_0__itr_0____itr_1_337____itr_2_338 < 49152UL; fused_0fused_0__itr_0____itr_1_337____itr_2_338 += 1UL) {
    for (uint64_t _fuseiter_674 = 0UL; _fuseiter_674 < 3UL; _fuseiter_674 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_337____itr_2_338 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_337____itr_2_338 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_337____itr_2_338 % 3UL) * 3UL))) + _fuseiter_674)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_337____itr_2_338 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_337____itr_2_338 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_337____itr_2_338 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_337____itr_2_338 % 3UL) * 3UL))) + _fuseiter_674)] = __cached_2;
    }
  }
  return true;
}

static bool cast__163(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_339____itr_2_340 = 0UL; fused_0fused_0__itr_0____itr_1_339____itr_2_340 < 49152UL; fused_0fused_0__itr_0____itr_1_339____itr_2_340 += 1UL) {
    for (uint64_t _fuseiter679 = 0UL; _fuseiter679 < 3UL; _fuseiter679 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_339____itr_2_340 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_339____itr_2_340 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_339____itr_2_340 % 3UL) * 3UL))) + _fuseiter679)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_339____itr_2_340 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_339____itr_2_340 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_339____itr_2_340 % 3UL) * 3UL))) + _fuseiter679)] = __cached_1;
    }
  }
  return true;
}

static bool mul__171(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_341____itr_2_342 = 0UL; fused_0fused_0__itr_0____itr_1_341____itr_2_342 < 49152UL; fused_0fused_0__itr_0____itr_1_341____itr_2_342 += 1UL) {
    for (uint64_t _fuseiter_684 = 0UL; _fuseiter_684 < 3UL; _fuseiter_684 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_341____itr_2_342 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_341____itr_2_342 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_341____itr_2_342 % 3UL) * 3UL))) + _fuseiter_684)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_341____itr_2_342 / 384UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_341____itr_2_342 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_341____itr_2_342 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_341____itr_2_342 % 3UL) * 3UL))) + _fuseiter_684)] = __cached_2;
    }
  }
  return true;
}

static bool cast__172(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_343____itr_2_344 = 0UL; fused_0fused_0__itr_0____itr_1_343____itr_2_344 < 49152UL; fused_0fused_0__itr_0____itr_1_343____itr_2_344 += 1UL) {
    for (uint64_t _fuseiter689 = 0UL; _fuseiter689 < 3UL; _fuseiter689 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_343____itr_2_344 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_343____itr_2_344 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_343____itr_2_344 % 3UL) * 3UL))) + _fuseiter689)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_343____itr_2_344 / 384UL) * 1152UL) + ((((fused_0fused_0__itr_0____itr_1_343____itr_2_344 / 3UL) % 128UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_343____itr_2_344 % 3UL) * 3UL))) + _fuseiter689)] = __cached_1;
    }
  }
  return true;
}

static bool mul__186(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_345____itr_2_346 = 0UL; fused_0fused_0__itr_0____itr_1_345____itr_2_346 < 262144UL; fused_0fused_0__itr_0____itr_1_345____itr_2_346 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_345____itr_2_346 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_345____itr_2_346 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_345____itr_2_346 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_345____itr_2_346 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_345____itr_2_346 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__187(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_347____itr_2_348 = 0UL; fused_0fused_0__itr_0____itr_1_347____itr_2_348 < 262144UL; fused_0fused_0__itr_0____itr_1_347____itr_2_348 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_347____itr_2_348 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_347____itr_2_348 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_347____itr_2_348 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_347____itr_2_348 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__195(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_349____itr_2_350 = 0UL; fused_0fused_0__itr_0____itr_1_349____itr_2_350 < 262144UL; fused_0fused_0__itr_0____itr_1_349____itr_2_350 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_349____itr_2_350 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_349____itr_2_350 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_349____itr_2_350 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_349____itr_2_350 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_349____itr_2_350 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__196(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_351____itr_2_352 = 0UL; fused_0fused_0__itr_0____itr_1_351____itr_2_352 < 262144UL; fused_0fused_0__itr_0____itr_1_351____itr_2_352 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_351____itr_2_352 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_351____itr_2_352 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_351____itr_2_352 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_351____itr_2_352 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__204(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_353____itr_2_354 = 0UL; fused_0fused_0__itr_0____itr_1_353____itr_2_354 < 262144UL; fused_0fused_0__itr_0____itr_1_353____itr_2_354 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_353____itr_2_354 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_353____itr_2_354 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_353____itr_2_354 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_353____itr_2_354 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_353____itr_2_354 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__205(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_355____itr_2_356 = 0UL; fused_0fused_0__itr_0____itr_1_355____itr_2_356 < 262144UL; fused_0fused_0__itr_0____itr_1_355____itr_2_356 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_355____itr_2_356 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_355____itr_2_356 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_355____itr_2_356 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_355____itr_2_356 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__213(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_357____itr_2_358 = 0UL; fused_0fused_0__itr_0____itr_1_357____itr_2_358 < 262144UL; fused_0fused_0__itr_0____itr_1_357____itr_2_358 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_357____itr_2_358 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_357____itr_2_358 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_357____itr_2_358 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_357____itr_2_358 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_357____itr_2_358 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__214(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_359____itr_2_360 = 0UL; fused_0fused_0__itr_0____itr_1_359____itr_2_360 < 262144UL; fused_0fused_0__itr_0____itr_1_359____itr_2_360 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_359____itr_2_360 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_359____itr_2_360 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_359____itr_2_360 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_359____itr_2_360 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__222(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_361____itr_2_362 = 0UL; fused_0fused_0__itr_0____itr_1_361____itr_2_362 < 262144UL; fused_0fused_0__itr_0____itr_1_361____itr_2_362 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_361____itr_2_362 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_361____itr_2_362 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_361____itr_2_362 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_361____itr_2_362 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_361____itr_2_362 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__223(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_363____itr_2_364 = 0UL; fused_0fused_0__itr_0____itr_1_363____itr_2_364 < 262144UL; fused_0fused_0__itr_0____itr_1_363____itr_2_364 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_363____itr_2_364 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_363____itr_2_364 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_363____itr_2_364 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_363____itr_2_364 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__231(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_365____itr_2_366 = 0UL; fused_0fused_0__itr_0____itr_1_365____itr_2_366 < 262144UL; fused_0fused_0__itr_0____itr_1_365____itr_2_366 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_365____itr_2_366 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_365____itr_2_366 % 256UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_365____itr_2_366 / 256UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_365____itr_2_366 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_365____itr_2_366 % 256UL))] = __cached_2;
  }
  return true;
}

static bool cast__232(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_367____itr_2_368 = 0UL; fused_0fused_0__itr_0____itr_1_367____itr_2_368 < 262144UL; fused_0fused_0__itr_0____itr_1_367____itr_2_368 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_367____itr_2_368 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_367____itr_2_368 % 256UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_367____itr_2_368 / 256UL) * 256UL) + (fused_0fused_0__itr_0____itr_1_367____itr_2_368 % 256UL))] = __cached_1;
  }
  return true;
}

static bool mul__189(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_369____itr_2_370 = 0UL; fused_0fused_0__itr_0____itr_1_369____itr_2_370 < 262144UL; fused_0fused_0__itr_0____itr_1_369____itr_2_370 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_369____itr_2_370 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_369____itr_2_370 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_369____itr_2_370 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_369____itr_2_370 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_369____itr_2_370 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__190(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_371____itr_2_372 = 0UL; fused_0fused_0__itr_0____itr_1_371____itr_2_372 < 262144UL; fused_0fused_0__itr_0____itr_1_371____itr_2_372 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_371____itr_2_372 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_371____itr_2_372 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_371____itr_2_372 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_371____itr_2_372 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__198(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_373____itr_2_374 = 0UL; fused_0fused_0__itr_0____itr_1_373____itr_2_374 < 262144UL; fused_0fused_0__itr_0____itr_1_373____itr_2_374 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_373____itr_2_374 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_373____itr_2_374 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_373____itr_2_374 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_373____itr_2_374 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_373____itr_2_374 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__199(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_375____itr_2_376 = 0UL; fused_0fused_0__itr_0____itr_1_375____itr_2_376 < 262144UL; fused_0fused_0__itr_0____itr_1_375____itr_2_376 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_375____itr_2_376 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_375____itr_2_376 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_375____itr_2_376 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_375____itr_2_376 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__207(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_377____itr_2_378 = 0UL; fused_0fused_0__itr_0____itr_1_377____itr_2_378 < 262144UL; fused_0fused_0__itr_0____itr_1_377____itr_2_378 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_377____itr_2_378 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_377____itr_2_378 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_377____itr_2_378 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_377____itr_2_378 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_377____itr_2_378 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__208(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_379____itr_2_380 = 0UL; fused_0fused_0__itr_0____itr_1_379____itr_2_380 < 262144UL; fused_0fused_0__itr_0____itr_1_379____itr_2_380 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_379____itr_2_380 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_379____itr_2_380 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_379____itr_2_380 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_379____itr_2_380 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__216(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_381____itr_2_382 = 0UL; fused_0fused_0__itr_0____itr_1_381____itr_2_382 < 262144UL; fused_0fused_0__itr_0____itr_1_381____itr_2_382 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_381____itr_2_382 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_381____itr_2_382 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_381____itr_2_382 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_381____itr_2_382 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_381____itr_2_382 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__217(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_383____itr_2_384 = 0UL; fused_0fused_0__itr_0____itr_1_383____itr_2_384 < 262144UL; fused_0fused_0__itr_0____itr_1_383____itr_2_384 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_383____itr_2_384 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_383____itr_2_384 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_383____itr_2_384 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_383____itr_2_384 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__225(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_385____itr_2_386 = 0UL; fused_0fused_0__itr_0____itr_1_385____itr_2_386 < 262144UL; fused_0fused_0__itr_0____itr_1_385____itr_2_386 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_385____itr_2_386 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_385____itr_2_386 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_385____itr_2_386 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_385____itr_2_386 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_385____itr_2_386 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__226(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_387____itr_2_388 = 0UL; fused_0fused_0__itr_0____itr_1_387____itr_2_388 < 262144UL; fused_0fused_0__itr_0____itr_1_387____itr_2_388 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_387____itr_2_388 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_387____itr_2_388 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_387____itr_2_388 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_387____itr_2_388 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__177(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_389____itr_2_390 = 0UL; fused_0fused_0__itr_0____itr_1_389____itr_2_390 < 524288UL; fused_0fused_0__itr_0____itr_1_389____itr_2_390 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_389____itr_2_390 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_389____itr_2_390 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_389____itr_2_390 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_389____itr_2_390 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_389____itr_2_390 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__178(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_391____itr_2_392 = 0UL; fused_0fused_0__itr_0____itr_1_391____itr_2_392 < 524288UL; fused_0fused_0__itr_0____itr_1_391____itr_2_392 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_391____itr_2_392 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_391____itr_2_392 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_391____itr_2_392 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_391____itr_2_392 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__237(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_393____itr_2_394 = 0UL; fused_0fused_0__itr_0____itr_1_393____itr_2_394 < 524288UL; fused_0fused_0__itr_0____itr_1_393____itr_2_394 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_393____itr_2_394 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_393____itr_2_394 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_393____itr_2_394 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_393____itr_2_394 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_393____itr_2_394 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__238(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_395____itr_2_396 = 0UL; fused_0fused_0__itr_0____itr_1_395____itr_2_396 < 524288UL; fused_0fused_0__itr_0____itr_1_395____itr_2_396 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_395____itr_2_396 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_395____itr_2_396 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_395____itr_2_396 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_395____itr_2_396 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool mul__183(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_397____itr_2_398 = 0UL; fused_0fused_0__itr_0____itr_1_397____itr_2_398 < 196608UL; fused_0fused_0__itr_0____itr_1_397____itr_2_398 += 1UL) {
    for (uint64_t _fuseiter_824 = 0UL; _fuseiter_824 < 3UL; _fuseiter_824 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_397____itr_2_398 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_397____itr_2_398 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_397____itr_2_398 % 3UL) * 3UL))) + _fuseiter_824)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_397____itr_2_398 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_397____itr_2_398 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_397____itr_2_398 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_397____itr_2_398 % 3UL) * 3UL))) + _fuseiter_824)] = __cached_2;
    }
  }
  return true;
}

static bool cast__184(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_399____itr_2_400 = 0UL; fused_0fused_0__itr_0____itr_1_399____itr_2_400 < 196608UL; fused_0fused_0__itr_0____itr_1_399____itr_2_400 += 1UL) {
    for (uint64_t _fuseiter829 = 0UL; _fuseiter829 < 3UL; _fuseiter829 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_399____itr_2_400 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_399____itr_2_400 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_399____itr_2_400 % 3UL) * 3UL))) + _fuseiter829)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_399____itr_2_400 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_399____itr_2_400 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_399____itr_2_400 % 3UL) * 3UL))) + _fuseiter829)] = __cached_1;
    }
  }
  return true;
}

static bool mul__192(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_401____itr_2_402 = 0UL; fused_0fused_0__itr_0____itr_1_401____itr_2_402 < 196608UL; fused_0fused_0__itr_0____itr_1_401____itr_2_402 += 1UL) {
    for (uint64_t _fuseiter_834 = 0UL; _fuseiter_834 < 3UL; _fuseiter_834 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_401____itr_2_402 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_401____itr_2_402 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_401____itr_2_402 % 3UL) * 3UL))) + _fuseiter_834)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_401____itr_2_402 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_401____itr_2_402 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_401____itr_2_402 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_401____itr_2_402 % 3UL) * 3UL))) + _fuseiter_834)] = __cached_2;
    }
  }
  return true;
}

static bool cast__193(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_403____itr_2_404 = 0UL; fused_0fused_0__itr_0____itr_1_403____itr_2_404 < 196608UL; fused_0fused_0__itr_0____itr_1_403____itr_2_404 += 1UL) {
    for (uint64_t _fuseiter839 = 0UL; _fuseiter839 < 3UL; _fuseiter839 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_403____itr_2_404 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_403____itr_2_404 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_403____itr_2_404 % 3UL) * 3UL))) + _fuseiter839)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_403____itr_2_404 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_403____itr_2_404 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_403____itr_2_404 % 3UL) * 3UL))) + _fuseiter839)] = __cached_1;
    }
  }
  return true;
}

static bool mul__201(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_405____itr_2_406 = 0UL; fused_0fused_0__itr_0____itr_1_405____itr_2_406 < 196608UL; fused_0fused_0__itr_0____itr_1_405____itr_2_406 += 1UL) {
    for (uint64_t _fuseiter_844 = 0UL; _fuseiter_844 < 3UL; _fuseiter_844 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_405____itr_2_406 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_405____itr_2_406 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_405____itr_2_406 % 3UL) * 3UL))) + _fuseiter_844)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_405____itr_2_406 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_405____itr_2_406 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_405____itr_2_406 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_405____itr_2_406 % 3UL) * 3UL))) + _fuseiter_844)] = __cached_2;
    }
  }
  return true;
}

static bool cast__202(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_407____itr_2_408 = 0UL; fused_0fused_0__itr_0____itr_1_407____itr_2_408 < 196608UL; fused_0fused_0__itr_0____itr_1_407____itr_2_408 += 1UL) {
    for (uint64_t _fuseiter849 = 0UL; _fuseiter849 < 3UL; _fuseiter849 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_407____itr_2_408 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_407____itr_2_408 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_407____itr_2_408 % 3UL) * 3UL))) + _fuseiter849)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_407____itr_2_408 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_407____itr_2_408 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_407____itr_2_408 % 3UL) * 3UL))) + _fuseiter849)] = __cached_1;
    }
  }
  return true;
}

static bool mul__210(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_409____itr_2_410 = 0UL; fused_0fused_0__itr_0____itr_1_409____itr_2_410 < 196608UL; fused_0fused_0__itr_0____itr_1_409____itr_2_410 += 1UL) {
    for (uint64_t _fuseiter_854 = 0UL; _fuseiter_854 < 3UL; _fuseiter_854 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_409____itr_2_410 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_409____itr_2_410 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_409____itr_2_410 % 3UL) * 3UL))) + _fuseiter_854)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_409____itr_2_410 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_409____itr_2_410 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_409____itr_2_410 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_409____itr_2_410 % 3UL) * 3UL))) + _fuseiter_854)] = __cached_2;
    }
  }
  return true;
}

static bool cast__211(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_411____itr_2_412 = 0UL; fused_0fused_0__itr_0____itr_1_411____itr_2_412 < 196608UL; fused_0fused_0__itr_0____itr_1_411____itr_2_412 += 1UL) {
    for (uint64_t _fuseiter859 = 0UL; _fuseiter859 < 3UL; _fuseiter859 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_411____itr_2_412 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_411____itr_2_412 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_411____itr_2_412 % 3UL) * 3UL))) + _fuseiter859)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_411____itr_2_412 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_411____itr_2_412 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_411____itr_2_412 % 3UL) * 3UL))) + _fuseiter859)] = __cached_1;
    }
  }
  return true;
}

static bool mul__219(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_413____itr_2_414 = 0UL; fused_0fused_0__itr_0____itr_1_413____itr_2_414 < 196608UL; fused_0fused_0__itr_0____itr_1_413____itr_2_414 += 1UL) {
    for (uint64_t _fuseiter_864 = 0UL; _fuseiter_864 < 3UL; _fuseiter_864 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_413____itr_2_414 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_413____itr_2_414 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_413____itr_2_414 % 3UL) * 3UL))) + _fuseiter_864)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_413____itr_2_414 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_413____itr_2_414 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_413____itr_2_414 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_413____itr_2_414 % 3UL) * 3UL))) + _fuseiter_864)] = __cached_2;
    }
  }
  return true;
}

static bool cast__220(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_415____itr_2_416 = 0UL; fused_0fused_0__itr_0____itr_1_415____itr_2_416 < 196608UL; fused_0fused_0__itr_0____itr_1_415____itr_2_416 += 1UL) {
    for (uint64_t _fuseiter869 = 0UL; _fuseiter869 < 3UL; _fuseiter869 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_415____itr_2_416 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_415____itr_2_416 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_415____itr_2_416 % 3UL) * 3UL))) + _fuseiter869)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_415____itr_2_416 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_415____itr_2_416 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_415____itr_2_416 % 3UL) * 3UL))) + _fuseiter869)] = __cached_1;
    }
  }
  return true;
}

static bool mul__228(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_417____itr_2_418 = 0UL; fused_0fused_0__itr_0____itr_1_417____itr_2_418 < 196608UL; fused_0fused_0__itr_0____itr_1_417____itr_2_418 += 1UL) {
    for (uint64_t _fuseiter_874 = 0UL; _fuseiter_874 < 3UL; _fuseiter_874 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_417____itr_2_418 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_417____itr_2_418 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_417____itr_2_418 % 3UL) * 3UL))) + _fuseiter_874)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_417____itr_2_418 / 768UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_417____itr_2_418 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_417____itr_2_418 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_417____itr_2_418 % 3UL) * 3UL))) + _fuseiter_874)] = __cached_2;
    }
  }
  return true;
}

static bool cast__229(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_419____itr_2_420 = 0UL; fused_0fused_0__itr_0____itr_1_419____itr_2_420 < 196608UL; fused_0fused_0__itr_0____itr_1_419____itr_2_420 += 1UL) {
    for (uint64_t _fuseiter879 = 0UL; _fuseiter879 < 3UL; _fuseiter879 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_419____itr_2_420 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_419____itr_2_420 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_419____itr_2_420 % 3UL) * 3UL))) + _fuseiter879)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_419____itr_2_420 / 768UL) * 2304UL) + ((((fused_0fused_0__itr_0____itr_1_419____itr_2_420 / 3UL) % 256UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_419____itr_2_420 % 3UL) * 3UL))) + _fuseiter879)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__525(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_880___fuseiter_881_421 = 0UL; fused_0_fuseiter_880___fuseiter_881_421 < 16UL; fused_0_fuseiter_880___fuseiter_881_421 += 1UL) {
    for (uint64_t _fuseiter_882 = 0UL; _fuseiter_882 < 3UL; _fuseiter_882 += 1UL) {
      for (uint64_t _fuseiter_883 = 0UL; _fuseiter_883 < 3UL; _fuseiter_883 += 1UL) {
        for (uint64_t _fuseiter_884 = 0UL; _fuseiter_884 < 16UL; _fuseiter_884 += 1UL) {
          for (uint64_t _fuseiter_885 = 0UL; _fuseiter_885 < 64UL; _fuseiter_885 += 1UL) {
            for (uint64_t _fuseiter_886 = 0UL; _fuseiter_886 < 4UL; _fuseiter_886 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_885 + ((fused_0_fuseiter_880___fuseiter_881_421 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_886 + (_fuseiter_884 * 4UL)) + ((fused_0_fuseiter_880___fuseiter_881_421 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_882 * 3UL) + _fuseiter_883)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_880___fuseiter_881_421 / 4UL) * 147456UL) + (((fused_0_fuseiter_880___fuseiter_881_421 % 4UL) * 36864UL) + ((_fuseiter_882 * 12288UL) + ((_fuseiter_883 * 4096UL) + ((_fuseiter_884 * 256UL) + ((_fuseiter_885 * 4UL) + _fuseiter_886))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool mul__652(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 += 1UL) {
    for (uint64_t _fuseiter_891 = 0UL; _fuseiter_891 < 256UL; _fuseiter_891 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 % 4UL) * 256UL)) + _fuseiter_891)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_422____itr_2_423____itr_3_424 % 4UL) * 256UL)) + _fuseiter_891)]);
    }
  }
  return true;
}

static bool mul__651(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 += 1UL) {
    for (uint64_t _fuseiter_897 = 0UL; _fuseiter_897 < 256UL; _fuseiter_897 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 % 4UL) * 256UL)) + _fuseiter_897)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 / 4UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_425____itr_2_426____itr_3_427 % 4UL) * 256UL)) + _fuseiter_897)]);
    }
  }
  return true;
}

static bool mul__646(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 += 1UL) {
    for (uint64_t _fuseiter_903 = 0UL; _fuseiter_903 < 128UL; _fuseiter_903 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 % 8UL) * 128UL)) + _fuseiter_903)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_428____itr_2_429____itr_3_430 % 8UL) * 128UL)) + _fuseiter_903)]);
    }
  }
  return true;
}

static bool mul__645(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 += 1UL) {
    for (uint64_t _fuseiter_909 = 0UL; _fuseiter_909 < 128UL; _fuseiter_909 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 % 8UL) * 128UL)) + _fuseiter_909)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_431____itr_2_432____itr_3_433 % 8UL) * 128UL)) + _fuseiter_909)]);
    }
  }
  return true;
}

static bool mul__640(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 += 1UL) {
    for (uint64_t _fuseiter_915 = 0UL; _fuseiter_915 < 128UL; _fuseiter_915 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 % 8UL) * 128UL)) + _fuseiter_915)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_434____itr_2_435____itr_3_436 % 8UL) * 128UL)) + _fuseiter_915)]);
    }
  }
  return true;
}

static bool mul__639(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 += 1UL) {
    for (uint64_t _fuseiter_921 = 0UL; _fuseiter_921 < 128UL; _fuseiter_921 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 % 8UL) * 128UL)) + _fuseiter_921)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_437____itr_2_438____itr_3_439 % 8UL) * 128UL)) + _fuseiter_921)]);
    }
  }
  return true;
}

static bool mul__634(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 += 1UL) {
    for (uint64_t _fuseiter_927 = 0UL; _fuseiter_927 < 128UL; _fuseiter_927 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 % 8UL) * 128UL)) + _fuseiter_927)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_440____itr_2_441____itr_3_442 % 8UL) * 128UL)) + _fuseiter_927)]);
    }
  }
  return true;
}

static bool mul__633(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 += 1UL) {
    for (uint64_t _fuseiter_933 = 0UL; _fuseiter_933 < 128UL; _fuseiter_933 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 % 8UL) * 128UL)) + _fuseiter_933)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_443____itr_2_444____itr_3_445 % 8UL) * 128UL)) + _fuseiter_933)]);
    }
  }
  return true;
}

static bool mul__628(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_939 = 0UL; _fuseiter_939 < 1024UL; _fuseiter_939 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_939]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_939]);
  }
  return true;
}

static bool mul__627(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_945 = 0UL; _fuseiter_945 < 1024UL; _fuseiter_945 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_945]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_945]);
  }
  return true;
}

static bool mul__622(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 += 1UL) {
    for (uint64_t _fuseiter_951 = 0UL; _fuseiter_951 < 128UL; _fuseiter_951 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 % 8UL) * 128UL)) + _fuseiter_951)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_452____itr_2_453____itr_3_454 % 8UL) * 128UL)) + _fuseiter_951)]);
    }
  }
  return true;
}

static bool mul__621(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 += 1UL) {
    for (uint64_t _fuseiter_957 = 0UL; _fuseiter_957 < 128UL; _fuseiter_957 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 % 8UL) * 128UL)) + _fuseiter_957)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 / 8UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_455____itr_2_456____itr_3_457 % 8UL) * 128UL)) + _fuseiter_957)]);
    }
  }
  return true;
}

static bool mul__616(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 += 1UL) {
    for (uint64_t _fuseiter_963 = 0UL; _fuseiter_963 < 32UL; _fuseiter_963 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 % 32UL) * 32UL)) + _fuseiter_963)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_458____itr_2_459____itr_3_460 % 32UL) * 32UL)) + _fuseiter_963)]);
    }
  }
  return true;
}

static bool mul__615(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 += 1UL) {
    for (uint64_t _fuseiter_969 = 0UL; _fuseiter_969 < 32UL; _fuseiter_969 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 % 32UL) * 32UL)) + _fuseiter_969)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 / 32UL) * 1024UL) + ((fused_0fused_0fused_0__itr_0____itr_1_461____itr_2_462____itr_3_463 % 32UL) * 32UL)) + _fuseiter_969)]);
    }
  }
  return true;
}

static bool mul__614(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 += 1UL) {
    for (uint64_t _fuseiter_975 = 0UL; _fuseiter_975 < 128UL; _fuseiter_975 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 % 4UL) * 128UL)) + _fuseiter_975)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_464____itr_2_465____itr_3_466 % 4UL) * 128UL)) + _fuseiter_975)]);
    }
  }
  return true;
}

static bool mul__613(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 += 1UL) {
    for (uint64_t _fuseiter_981 = 0UL; _fuseiter_981 < 128UL; _fuseiter_981 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 % 4UL) * 128UL)) + _fuseiter_981)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_467____itr_2_468____itr_3_469 % 4UL) * 128UL)) + _fuseiter_981)]);
    }
  }
  return true;
}

static bool mul__608(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 += 1UL) {
    for (uint64_t _fuseiter_987 = 0UL; _fuseiter_987 < 64UL; _fuseiter_987 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 % 8UL) * 64UL)) + _fuseiter_987)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_470____itr_2_471____itr_3_472 % 8UL) * 64UL)) + _fuseiter_987)]);
    }
  }
  return true;
}

static bool mul__607(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 += 1UL) {
    for (uint64_t _fuseiter_993 = 0UL; _fuseiter_993 < 64UL; _fuseiter_993 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 % 8UL) * 64UL)) + _fuseiter_993)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_473____itr_2_474____itr_3_475 % 8UL) * 64UL)) + _fuseiter_993)]);
    }
  }
  return true;
}

static bool mul__602(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 += 1UL) {
    for (uint64_t _fuseiter_999 = 0UL; _fuseiter_999 < 128UL; _fuseiter_999 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 % 4UL) * 128UL)) + _fuseiter_999)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_476____itr_2_477____itr_3_478 % 4UL) * 128UL)) + _fuseiter_999)]);
    }
  }
  return true;
}

static bool mul__601(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 += 1UL) {
    for (uint64_t _fuseiter_1005 = 0UL; _fuseiter_1005 < 128UL; _fuseiter_1005 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 % 4UL) * 128UL)) + _fuseiter_1005)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_479____itr_2_480____itr_3_481 % 4UL) * 128UL)) + _fuseiter_1005)]);
    }
  }
  return true;
}

static bool mul__596(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 += 1UL) {
    for (uint64_t _fuseiter_1011 = 0UL; _fuseiter_1011 < 32UL; _fuseiter_1011 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 % 16UL) * 32UL)) + _fuseiter_1011)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_482____itr_2_483____itr_3_484 % 16UL) * 32UL)) + _fuseiter_1011)]);
    }
  }
  return true;
}

static bool mul__595(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 += 1UL) {
    for (uint64_t _fuseiter_1017 = 0UL; _fuseiter_1017 < 32UL; _fuseiter_1017 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 % 16UL) * 32UL)) + _fuseiter_1017)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_485____itr_2_486____itr_3_487 % 16UL) * 32UL)) + _fuseiter_1017)]);
    }
  }
  return true;
}

static bool mul__590(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 += 1UL) {
    for (uint64_t _fuseiter_1023 = 0UL; _fuseiter_1023 < 32UL; _fuseiter_1023 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 % 16UL) * 32UL)) + _fuseiter_1023)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_488____itr_2_489____itr_3_490 % 16UL) * 32UL)) + _fuseiter_1023)]);
    }
  }
  return true;
}

static bool mul__589(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 += 1UL) {
    for (uint64_t _fuseiter_1029 = 0UL; _fuseiter_1029 < 32UL; _fuseiter_1029 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 % 16UL) * 32UL)) + _fuseiter_1029)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_491____itr_2_492____itr_3_493 % 16UL) * 32UL)) + _fuseiter_1029)]);
    }
  }
  return true;
}

static bool mul__650(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 += 1UL) {
    for (uint64_t _fuseiter_1035 = 0UL; _fuseiter_1035 < 64UL; _fuseiter_1035 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 % 4UL) * 64UL)) + _fuseiter_1035)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_494____itr_2_495____itr_3_496 % 4UL) * 64UL)) + _fuseiter_1035)]);
    }
  }
  return true;
}

static bool mul__649(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 += 1UL) {
    for (uint64_t _fuseiter_1041 = 0UL; _fuseiter_1041 < 64UL; _fuseiter_1041 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 % 4UL) * 64UL)) + _fuseiter_1041)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_497____itr_2_498____itr_3_499 % 4UL) * 64UL)) + _fuseiter_1041)]);
    }
  }
  return true;
}

static bool mul__648(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 += 1UL) {
    for (uint64_t _fuseiter_1047 = 0UL; _fuseiter_1047 < 32UL; _fuseiter_1047 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 % 8UL) * 32UL)) + _fuseiter_1047)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_500____itr_2_501____itr_3_502 % 8UL) * 32UL)) + _fuseiter_1047)]);
    }
  }
  return true;
}

static bool mul__647(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 += 1UL) {
    for (uint64_t _fuseiter_1053 = 0UL; _fuseiter_1053 < 32UL; _fuseiter_1053 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 % 8UL) * 32UL)) + _fuseiter_1053)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_503____itr_2_504____itr_3_505 % 8UL) * 32UL)) + _fuseiter_1053)]);
    }
  }
  return true;
}

static bool mul__644(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 += 1UL) {
    for (uint64_t _fuseiter_1059 = 0UL; _fuseiter_1059 < 32UL; _fuseiter_1059 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 % 8UL) * 32UL)) + _fuseiter_1059)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_506____itr_2_507____itr_3_508 % 8UL) * 32UL)) + _fuseiter_1059)]);
    }
  }
  return true;
}

static bool mul__643(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 += 1UL) {
    for (uint64_t _fuseiter_1065 = 0UL; _fuseiter_1065 < 32UL; _fuseiter_1065 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 % 8UL) * 32UL)) + _fuseiter_1065)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_509____itr_2_510____itr_3_511 % 8UL) * 32UL)) + _fuseiter_1065)]);
    }
  }
  return true;
}

static bool mul__642(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1071 = 0UL; _fuseiter_1071 < 256UL; _fuseiter_1071 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1071]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1071]);
  }
  return true;
}

static bool mul__641(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1077 = 0UL; _fuseiter_1077 < 256UL; _fuseiter_1077 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1077]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1077]);
  }
  return true;
}

static bool mul__638(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 += 1UL) {
    for (uint64_t _fuseiter_1083 = 0UL; _fuseiter_1083 < 64UL; _fuseiter_1083 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 % 4UL) * 64UL)) + _fuseiter_1083)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_518____itr_2_519____itr_3_520 % 4UL) * 64UL)) + _fuseiter_1083)]);
    }
  }
  return true;
}

static bool mul__637(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 += 1UL) {
    for (uint64_t _fuseiter_1089 = 0UL; _fuseiter_1089 < 64UL; _fuseiter_1089 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 % 4UL) * 64UL)) + _fuseiter_1089)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_521____itr_2_522____itr_3_523 % 4UL) * 64UL)) + _fuseiter_1089)]);
    }
  }
  return true;
}

static bool mul__636(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 += 1UL) {
    for (uint64_t _fuseiter_1095 = 0UL; _fuseiter_1095 < 64UL; _fuseiter_1095 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 % 4UL) * 64UL)) + _fuseiter_1095)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_524____itr_2_525____itr_3_526 % 4UL) * 64UL)) + _fuseiter_1095)]);
    }
  }
  return true;
}

static bool mul__635(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 += 1UL) {
    for (uint64_t _fuseiter_1101 = 0UL; _fuseiter_1101 < 64UL; _fuseiter_1101 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 % 4UL) * 64UL)) + _fuseiter_1101)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_527____itr_2_528____itr_3_529 % 4UL) * 64UL)) + _fuseiter_1101)]);
    }
  }
  return true;
}

static bool mul__632(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 += 1UL) {
    for (uint64_t _fuseiter_1107 = 0UL; _fuseiter_1107 < 32UL; _fuseiter_1107 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 % 8UL) * 32UL)) + _fuseiter_1107)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_530____itr_2_531____itr_3_532 % 8UL) * 32UL)) + _fuseiter_1107)]);
    }
  }
  return true;
}

static bool mul__631(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 += 1UL) {
    for (uint64_t _fuseiter_1113 = 0UL; _fuseiter_1113 < 32UL; _fuseiter_1113 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 % 8UL) * 32UL)) + _fuseiter_1113)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_533____itr_2_534____itr_3_535 % 8UL) * 32UL)) + _fuseiter_1113)]);
    }
  }
  return true;
}

static bool mul__630(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 += 1UL) {
    for (uint64_t _fuseiter_1119 = 0UL; _fuseiter_1119 < 32UL; _fuseiter_1119 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 % 8UL) * 32UL)) + _fuseiter_1119)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_536____itr_2_537____itr_3_538 % 8UL) * 32UL)) + _fuseiter_1119)]);
    }
  }
  return true;
}

static bool mul__629(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 += 1UL) {
    for (uint64_t _fuseiter_1125 = 0UL; _fuseiter_1125 < 32UL; _fuseiter_1125 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 % 8UL) * 32UL)) + _fuseiter_1125)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_539____itr_2_540____itr_3_541 % 8UL) * 32UL)) + _fuseiter_1125)]);
    }
  }
  return true;
}

static bool mul__626(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_542____itr_2_543____itr_3_544 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__625(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_545____itr_2_546____itr_3_547 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__624(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 += 1UL) {
    for (uint64_t _fuseiter_1143 = 0UL; _fuseiter_1143 < 64UL; _fuseiter_1143 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 % 4UL) * 64UL)) + _fuseiter_1143)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_548____itr_2_549____itr_3_550 % 4UL) * 64UL)) + _fuseiter_1143)]);
    }
  }
  return true;
}

static bool mul__623(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 += 1UL) {
    for (uint64_t _fuseiter_1149 = 0UL; _fuseiter_1149 < 64UL; _fuseiter_1149 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 % 4UL) * 64UL)) + _fuseiter_1149)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_551____itr_2_552____itr_3_553 % 4UL) * 64UL)) + _fuseiter_1149)]);
    }
  }
  return true;
}

static bool mul__620(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 += 1UL) {
    for (uint64_t _fuseiter_1155 = 0UL; _fuseiter_1155 < 128UL; _fuseiter_1155 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 % 2UL) * 128UL)) + _fuseiter_1155)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_554____itr_2_555____itr_3_556 % 2UL) * 128UL)) + _fuseiter_1155)]);
    }
  }
  return true;
}

static bool mul__619(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 += 1UL) {
    for (uint64_t _fuseiter_1161 = 0UL; _fuseiter_1161 < 128UL; _fuseiter_1161 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 % 2UL) * 128UL)) + _fuseiter_1161)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 / 2UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_557____itr_2_558____itr_3_559 % 2UL) * 128UL)) + _fuseiter_1161)]);
    }
  }
  return true;
}

static bool mul__618(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1167 = 0UL; _fuseiter_1167 < 256UL; _fuseiter_1167 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1167]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1167]);
  }
  return true;
}

static bool mul__617(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1173 = 0UL; _fuseiter_1173 < 256UL; _fuseiter_1173 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1173]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1173]);
  }
  return true;
}

static bool mul__588(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 += 1UL) {
    for (uint64_t _fuseiter_1179 = 0UL; _fuseiter_1179 < 64UL; _fuseiter_1179 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 % 4UL) * 64UL)) + _fuseiter_1179)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_566____itr_2_567____itr_3_568 % 4UL) * 64UL)) + _fuseiter_1179)]);
    }
  }
  return true;
}

static bool mul__587(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 += 1UL) {
    for (uint64_t _fuseiter_1185 = 0UL; _fuseiter_1185 < 64UL; _fuseiter_1185 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 % 4UL) * 64UL)) + _fuseiter_1185)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_569____itr_2_570____itr_3_571 % 4UL) * 64UL)) + _fuseiter_1185)]);
    }
  }
  return true;
}

static bool mul__582(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 += 1UL) {
    for (uint64_t _fuseiter_1191 = 0UL; _fuseiter_1191 < 32UL; _fuseiter_1191 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 % 8UL) * 32UL)) + _fuseiter_1191)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_572____itr_2_573____itr_3_574 % 8UL) * 32UL)) + _fuseiter_1191)]);
    }
  }
  return true;
}

static bool mul__581(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 += 1UL) {
    for (uint64_t _fuseiter_1197 = 0UL; _fuseiter_1197 < 32UL; _fuseiter_1197 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 % 8UL) * 32UL)) + _fuseiter_1197)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 / 8UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_575____itr_2_576____itr_3_577 % 8UL) * 32UL)) + _fuseiter_1197)]);
    }
  }
  return true;
}

static bool mul__576(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 += 1UL) {
    for (uint64_t _fuseiter_1203 = 0UL; _fuseiter_1203 < 64UL; _fuseiter_1203 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 % 4UL) * 64UL)) + _fuseiter_1203)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_578____itr_2_579____itr_3_580 % 4UL) * 64UL)) + _fuseiter_1203)]);
    }
  }
  return true;
}

static bool mul__575(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 += 1UL) {
    for (uint64_t _fuseiter_1209 = 0UL; _fuseiter_1209 < 64UL; _fuseiter_1209 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 % 4UL) * 64UL)) + _fuseiter_1209)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 / 4UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_581____itr_2_582____itr_3_583 % 4UL) * 64UL)) + _fuseiter_1209)]);
    }
  }
  return true;
}

static bool mul__570(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_584____itr_2_585____itr_3_586 % 16UL) * 16UL))]);
  }
  return true;
}

static bool mul__569(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 % 16UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 / 16UL) * 256UL) + ((fused_0fused_0fused_0__itr_0____itr_1_587____itr_2_588____itr_3_589 % 16UL) * 16UL))]);
  }
  return true;
}

static bool reorder__522(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1223___fuseiter_1224_590 = 0UL; fused_0_fuseiter_1223___fuseiter_1224_590 < 16UL; fused_0_fuseiter_1223___fuseiter_1224_590 += 1UL) {
    for (uint64_t _fuseiter_1227 = 0UL; _fuseiter_1227 < 128UL; _fuseiter_1227 += 1UL) {
      for (uint64_t _fuseiter_1228 = 0UL; _fuseiter_1228 < 32UL; _fuseiter_1228 += 1UL) {
        for (uint64_t _fuseiter_1229 = 0UL; _fuseiter_1229 < 4UL; _fuseiter_1229 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1228 + ((fused_0_fuseiter_1223___fuseiter_1224_590 / 2UL) * 32UL)) * 1024UL) + ((_fuseiter_1229 + (_fuseiter_1227 * 4UL)) + ((fused_0_fuseiter_1223___fuseiter_1224_590 % 2UL) * 512UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1223___fuseiter_1224_590 / 2UL) * 32768UL) + (((fused_0_fuseiter_1223___fuseiter_1224_590 % 2UL) * 16384UL) + ((_fuseiter_1227 * 128UL) + ((_fuseiter_1228 * 4UL) + _fuseiter_1229))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__515(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 += 1UL) {
    for (uint64_t _fuseiter_1235 = 0UL; _fuseiter_1235 < 256UL; _fuseiter_1235 += 1UL) {
      for (uint64_t _fuseiter_1236 = 0UL; _fuseiter_1236 < 4UL; _fuseiter_1236 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1235 + ((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 / 256UL) * 256UL)) * 1024UL) + ((_fuseiter_1236 + ((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 / 32UL) % 8UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 / 256UL) * 262144UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 / 32UL) % 8UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1230___fuseiter_1231_591___fuseiter_1232_592___fuseiter_1233_593___fuseiter_1234_594 % 32UL) * 1024UL) + ((_fuseiter_1235 * 4UL) + _fuseiter_1236))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__506(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1237___fuseiter_1238_595 = 0UL; fused_0_fuseiter_1237___fuseiter_1238_595 < 32UL; fused_0_fuseiter_1237___fuseiter_1238_595 += 1UL) {
    for (uint64_t _fuseiter_1241 = 0UL; _fuseiter_1241 < 32UL; _fuseiter_1241 += 1UL) {
      for (uint64_t _fuseiter_1242 = 0UL; _fuseiter_1242 < 64UL; _fuseiter_1242 += 1UL) {
        for (uint64_t _fuseiter_1243 = 0UL; _fuseiter_1243 < 4UL; _fuseiter_1243 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1242 + ((fused_0_fuseiter_1237___fuseiter_1238_595 / 8UL) * 64UL)) * 1024UL) + ((_fuseiter_1243 + (_fuseiter_1241 * 4UL)) + ((fused_0_fuseiter_1237___fuseiter_1238_595 % 8UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1237___fuseiter_1238_595 / 8UL) * 65536UL) + (((fused_0_fuseiter_1237___fuseiter_1238_595 % 8UL) * 8192UL) + ((_fuseiter_1241 * 256UL) + ((_fuseiter_1242 * 4UL) + _fuseiter_1243))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__497(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1244___fuseiter_1245_596 = 0UL; fused_0_fuseiter_1244___fuseiter_1245_596 < 64UL; fused_0_fuseiter_1244___fuseiter_1245_596 += 1UL) {
    for (uint64_t _fuseiter_1248 = 0UL; _fuseiter_1248 < 32UL; _fuseiter_1248 += 1UL) {
      for (uint64_t _fuseiter_1249 = 0UL; _fuseiter_1249 < 32UL; _fuseiter_1249 += 1UL) {
        for (uint64_t _fuseiter_1250 = 0UL; _fuseiter_1250 < 4UL; _fuseiter_1250 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1249 + ((fused_0_fuseiter_1244___fuseiter_1245_596 / 8UL) * 32UL)) * 1024UL) + ((_fuseiter_1250 + (_fuseiter_1248 * 4UL)) + ((fused_0_fuseiter_1244___fuseiter_1245_596 % 8UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1244___fuseiter_1245_596 / 8UL) * 32768UL) + (((fused_0_fuseiter_1244___fuseiter_1245_596 % 8UL) * 4096UL) + ((_fuseiter_1248 * 128UL) + ((_fuseiter_1249 * 4UL) + _fuseiter_1250))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__490(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 < 1024UL; fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 += 1UL) {
    for (uint64_t _fuseiter_1256 = 0UL; _fuseiter_1256 < 64UL; _fuseiter_1256 += 1UL) {
      for (uint64_t _fuseiter_1257 = 0UL; _fuseiter_1257 < 4UL; _fuseiter_1257 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1256 + ((fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 / 256UL) * 64UL)) * 1024UL) + (_fuseiter_1257 + ((fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 % 256UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 / 256UL) * 65536UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1251___fuseiter_1252_597___fuseiter_1253_598___fuseiter_1254_599___fuseiter_1255_600 % 256UL) * 256UL) + ((_fuseiter_1256 * 4UL) + _fuseiter_1257)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__528(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 += 1UL) {
    for (uint64_t _fuseiter_1263 = 0UL; _fuseiter_1263 < 256UL; _fuseiter_1263 += 1UL) {
      for (uint64_t _fuseiter_1264 = 0UL; _fuseiter_1264 < 4UL; _fuseiter_1264 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1263 + ((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 / 64UL) * 256UL)) * 256UL) + ((_fuseiter_1264 + ((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 / 32UL) % 2UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 / 64UL) * 65536UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 / 32UL) % 2UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1258___fuseiter_1259_601___fuseiter_1260_602___fuseiter_1261_603___fuseiter_1262_604 % 32UL) * 1024UL) + ((_fuseiter_1263 * 4UL) + _fuseiter_1264))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__519(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 += 1UL) {
    for (uint64_t _fuseiter_1270 = 0UL; _fuseiter_1270 < 128UL; _fuseiter_1270 += 1UL) {
      for (uint64_t _fuseiter_1271 = 0UL; _fuseiter_1271 < 4UL; _fuseiter_1271 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1270 + ((fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 / 64UL) * 128UL)) * 256UL) + (_fuseiter_1271 + ((fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1265___fuseiter_1266_605___fuseiter_1267_606___fuseiter_1268_607___fuseiter_1269_608 % 64UL) * 512UL) + ((_fuseiter_1270 * 4UL) + _fuseiter_1271)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__512(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1272___fuseiter_1273_609 = 0UL; fused_0_fuseiter_1272___fuseiter_1273_609 < 16UL; fused_0_fuseiter_1272___fuseiter_1273_609 += 1UL) {
    for (uint64_t _fuseiter_1276 = 0UL; _fuseiter_1276 < 32UL; _fuseiter_1276 += 1UL) {
      for (uint64_t _fuseiter_1277 = 0UL; _fuseiter_1277 < 128UL; _fuseiter_1277 += 1UL) {
        for (uint64_t _fuseiter_1278 = 0UL; _fuseiter_1278 < 4UL; _fuseiter_1278 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1277 + ((fused_0_fuseiter_1272___fuseiter_1273_609 / 2UL) * 128UL)) * 256UL) + ((_fuseiter_1278 + (_fuseiter_1276 * 4UL)) + ((fused_0_fuseiter_1272___fuseiter_1273_609 % 2UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1272___fuseiter_1273_609 / 2UL) * 32768UL) + (((fused_0_fuseiter_1272___fuseiter_1273_609 % 2UL) * 16384UL) + ((_fuseiter_1276 * 512UL) + ((_fuseiter_1277 * 4UL) + _fuseiter_1278))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__503(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 += 1UL) {
    for (uint64_t _fuseiter_1284 = 0UL; _fuseiter_1284 < 128UL; _fuseiter_1284 += 1UL) {
      for (uint64_t _fuseiter_1285 = 0UL; _fuseiter_1285 < 4UL; _fuseiter_1285 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1284 + ((fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 / 64UL) * 128UL)) * 256UL) + (_fuseiter_1285 + ((fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1279___fuseiter_1280_610___fuseiter_1281_611___fuseiter_1282_612___fuseiter_1283_613 % 64UL) * 512UL) + ((_fuseiter_1284 * 4UL) + _fuseiter_1285)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__496(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 += 1UL) {
    for (uint64_t _fuseiter_1291 = 0UL; _fuseiter_1291 < 1024UL; _fuseiter_1291 += 1UL) {
      for (uint64_t _fuseiter_1292 = 0UL; _fuseiter_1292 < 4UL; _fuseiter_1292 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1291 + ((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 / 64UL) * 1024UL)) * 256UL) + ((_fuseiter_1292 + ((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 / 64UL) * 262144UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 / 16UL) % 4UL) * 65536UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1286___fuseiter_1287_614___fuseiter_1288_615___fuseiter_1289_616___fuseiter_1290_617 % 16UL) * 4096UL) + ((_fuseiter_1291 * 4UL) + _fuseiter_1292))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__487(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 += 1UL) {
    for (uint64_t _fuseiter_1298 = 0UL; _fuseiter_1298 < 128UL; _fuseiter_1298 += 1UL) {
      for (uint64_t _fuseiter_1299 = 0UL; _fuseiter_1299 < 4UL; _fuseiter_1299 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1298 + ((fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 / 64UL) * 128UL)) * 256UL) + (_fuseiter_1299 + ((fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 % 64UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 / 64UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1293___fuseiter_1294_618___fuseiter_1295_619___fuseiter_1296_620___fuseiter_1297_621 % 64UL) * 512UL) + ((_fuseiter_1298 * 4UL) + _fuseiter_1299)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__422(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 < 16UL; fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 += 1UL) {
    for (uint64_t _fuseiter_1305 = 0UL; _fuseiter_1305 < 64UL; _fuseiter_1305 += 1UL) {
      for (uint64_t _fuseiter_1306 = 0UL; _fuseiter_1306 < 4UL; _fuseiter_1306 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1305 + ((fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1306 + ((fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1300___fuseiter_1301_622___fuseiter_1302_623___fuseiter_1303_624___fuseiter_1304_625 % 16UL) * 256UL) + ((_fuseiter_1305 * 4UL) + _fuseiter_1306)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool mul__612(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 += 1UL) {
    for (uint64_t _fuseiter_1311 = 0UL; _fuseiter_1311 < 32UL; _fuseiter_1311 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 % 4UL) * 32UL)) + _fuseiter_1311)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_626____itr_2_627____itr_3_628 % 4UL) * 32UL)) + _fuseiter_1311)]);
    }
  }
  return true;
}

static bool mul__611(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 += 1UL) {
    for (uint64_t _fuseiter_1317 = 0UL; _fuseiter_1317 < 32UL; _fuseiter_1317 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 % 4UL) * 32UL)) + _fuseiter_1317)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_629____itr_2_630____itr_3_631 % 4UL) * 32UL)) + _fuseiter_1317)]);
    }
  }
  return true;
}

static bool mul__610(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 += 1UL) {
    for (uint64_t _fuseiter_1323 = 0UL; _fuseiter_1323 < 64UL; _fuseiter_1323 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 % 2UL) * 64UL)) + _fuseiter_1323)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_632____itr_2_633____itr_3_634 % 2UL) * 64UL)) + _fuseiter_1323)]);
    }
  }
  return true;
}

static bool mul__609(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 += 1UL) {
    for (uint64_t _fuseiter_1329 = 0UL; _fuseiter_1329 < 64UL; _fuseiter_1329 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 % 2UL) * 64UL)) + _fuseiter_1329)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_635____itr_2_636____itr_3_637 % 2UL) * 64UL)) + _fuseiter_1329)]);
    }
  }
  return true;
}

static bool mul__606(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 += 1UL) {
    for (uint64_t _fuseiter_1335 = 0UL; _fuseiter_1335 < 32UL; _fuseiter_1335 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 % 4UL) * 32UL)) + _fuseiter_1335)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_638____itr_2_639____itr_3_640 % 4UL) * 32UL)) + _fuseiter_1335)]);
    }
  }
  return true;
}

static bool mul__605(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 += 1UL) {
    for (uint64_t _fuseiter_1341 = 0UL; _fuseiter_1341 < 32UL; _fuseiter_1341 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 % 4UL) * 32UL)) + _fuseiter_1341)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_641____itr_2_642____itr_3_643 % 4UL) * 32UL)) + _fuseiter_1341)]);
    }
  }
  return true;
}

static bool mul__604(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1347 = 0UL; _fuseiter_1347 < 128UL; _fuseiter_1347 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1347]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1347]);
  }
  return true;
}

static bool mul__603(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1353 = 0UL; _fuseiter_1353 < 128UL; _fuseiter_1353 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1353]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1353]);
  }
  return true;
}

static bool mul__600(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1359 = 0UL; _fuseiter_1359 < 128UL; _fuseiter_1359 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1359]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1359]);
  }
  return true;
}

static bool mul__599(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1365 = 0UL; _fuseiter_1365 < 128UL; _fuseiter_1365 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1365]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1365]);
  }
  return true;
}

static bool mul__598(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 += 1UL) {
    for (uint64_t _fuseiter_1371 = 0UL; _fuseiter_1371 < 32UL; _fuseiter_1371 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 % 4UL) * 32UL)) + _fuseiter_1371)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_656____itr_2_657____itr_3_658 % 4UL) * 32UL)) + _fuseiter_1371)]);
    }
  }
  return true;
}

static bool mul__597(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 += 1UL) {
    for (uint64_t _fuseiter_1377 = 0UL; _fuseiter_1377 < 32UL; _fuseiter_1377 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 % 4UL) * 32UL)) + _fuseiter_1377)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_659____itr_2_660____itr_3_661 % 4UL) * 32UL)) + _fuseiter_1377)]);
    }
  }
  return true;
}

static bool mul__594(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 += 1UL) {
    for (uint64_t _fuseiter_1383 = 0UL; _fuseiter_1383 < 32UL; _fuseiter_1383 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 % 4UL) * 32UL)) + _fuseiter_1383)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_662____itr_2_663____itr_3_664 % 4UL) * 32UL)) + _fuseiter_1383)]);
    }
  }
  return true;
}

static bool mul__593(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 += 1UL) {
    for (uint64_t _fuseiter_1389 = 0UL; _fuseiter_1389 < 32UL; _fuseiter_1389 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 % 4UL) * 32UL)) + _fuseiter_1389)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 / 4UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_665____itr_2_666____itr_3_667 % 4UL) * 32UL)) + _fuseiter_1389)]);
    }
  }
  return true;
}

static bool mul__592(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 += 1UL) {
    for (uint64_t _fuseiter_1395 = 0UL; _fuseiter_1395 < 64UL; _fuseiter_1395 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 % 2UL) * 64UL)) + _fuseiter_1395)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_668____itr_2_669____itr_3_670 % 2UL) * 64UL)) + _fuseiter_1395)]);
    }
  }
  return true;
}

static bool mul__591(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 += 1UL) {
    for (uint64_t _fuseiter_1401 = 0UL; _fuseiter_1401 < 64UL; _fuseiter_1401 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 % 2UL) * 64UL)) + _fuseiter_1401)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 / 2UL) * 128UL) + ((fused_0fused_0fused_0__itr_0____itr_1_671____itr_2_672____itr_3_673 % 2UL) * 64UL)) + _fuseiter_1401)]);
    }
  }
  return true;
}

static bool mul__586(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 += 1UL) {
    for (uint64_t _fuseiter_1407 = 0UL; _fuseiter_1407 < 32UL; _fuseiter_1407 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 % 2UL) * 32UL)) + _fuseiter_1407)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_674____itr_2_675____itr_3_676 % 2UL) * 32UL)) + _fuseiter_1407)]);
    }
  }
  return true;
}

static bool mul__585(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 += 1UL) {
    for (uint64_t _fuseiter_1413 = 0UL; _fuseiter_1413 < 32UL; _fuseiter_1413 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 % 2UL) * 32UL)) + _fuseiter_1413)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_677____itr_2_678____itr_3_679 % 2UL) * 32UL)) + _fuseiter_1413)]);
    }
  }
  return true;
}

static bool mul__584(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 += 1UL) {
    for (uint64_t _fuseiter_1419 = 0UL; _fuseiter_1419 < 32UL; _fuseiter_1419 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 % 2UL) * 32UL)) + _fuseiter_1419)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_680____itr_2_681____itr_3_682 % 2UL) * 32UL)) + _fuseiter_1419)]);
    }
  }
  return true;
}

static bool mul__583(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 += 1UL) {
    for (uint64_t _fuseiter_1425 = 0UL; _fuseiter_1425 < 32UL; _fuseiter_1425 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 % 2UL) * 32UL)) + _fuseiter_1425)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_683____itr_2_684____itr_3_685 % 2UL) * 32UL)) + _fuseiter_1425)]);
    }
  }
  return true;
}

static bool mul__580(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 % 4UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_686____itr_2_687____itr_3_688 % 4UL) * 16UL))]);
  }
  return true;
}

static bool mul__579(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 % 4UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 / 4UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_689____itr_2_690____itr_3_691 % 4UL) * 16UL))]);
  }
  return true;
}

static bool mul__578(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1443 = 0UL; _fuseiter_1443 < 64UL; _fuseiter_1443 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1443]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1443]);
  }
  return true;
}

static bool mul__577(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1449 = 0UL; _fuseiter_1449 < 64UL; _fuseiter_1449 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1449]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1449]);
  }
  return true;
}

static bool mul__574(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 += 1UL) {
    for (uint64_t _fuseiter_1455 = 0UL; _fuseiter_1455 < 32UL; _fuseiter_1455 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 % 2UL) * 32UL)) + _fuseiter_1455)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_698____itr_2_699____itr_3_700 % 2UL) * 32UL)) + _fuseiter_1455)]);
    }
  }
  return true;
}

static bool mul__573(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 < 2UL; fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 += 1UL) {
    for (uint64_t _fuseiter_1461 = 0UL; _fuseiter_1461 < 32UL; _fuseiter_1461 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 % 2UL) * 32UL)) + _fuseiter_1461)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 / 2UL) * 64UL) + ((fused_0fused_0fused_0__itr_0____itr_1_701____itr_2_702____itr_3_703 % 2UL) * 32UL)) + _fuseiter_1461)]);
    }
  }
  return true;
}

static bool mul__572(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1467 = 0UL; _fuseiter_1467 < 64UL; _fuseiter_1467 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1467]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1467]);
  }
  return true;
}

static bool mul__571(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t _fuseiter_1473 = 0UL; _fuseiter_1473 < 64UL; _fuseiter_1473 += 16UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[_fuseiter_1473]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[_fuseiter_1473]);
  }
  return true;
}

static bool reorder__516(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1475___fuseiter_1476_710 = 0UL; fused_0_fuseiter_1475___fuseiter_1476_710 < 16UL; fused_0_fuseiter_1475___fuseiter_1476_710 += 1UL) {
    for (uint64_t _fuseiter_1477 = 0UL; _fuseiter_1477 < 3UL; _fuseiter_1477 += 1UL) {
      for (uint64_t _fuseiter_1478 = 0UL; _fuseiter_1478 < 3UL; _fuseiter_1478 += 1UL) {
        for (uint64_t _fuseiter_1479 = 0UL; _fuseiter_1479 < 32UL; _fuseiter_1479 += 1UL) {
          for (uint64_t _fuseiter_1480 = 0UL; _fuseiter_1480 < 32UL; _fuseiter_1480 += 1UL) {
            for (uint64_t _fuseiter_1481 = 0UL; _fuseiter_1481 < 4UL; _fuseiter_1481 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1480 + ((fused_0_fuseiter_1475___fuseiter_1476_710 / 2UL) * 32UL)) * 2304UL) + ((((_fuseiter_1481 + (_fuseiter_1479 * 4UL)) + ((fused_0_fuseiter_1475___fuseiter_1476_710 % 2UL) * 128UL)) * 9UL) + ((_fuseiter_1477 * 3UL) + _fuseiter_1478)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1475___fuseiter_1476_710 / 2UL) * 73728UL) + (((fused_0_fuseiter_1475___fuseiter_1476_710 % 2UL) * 36864UL) + ((_fuseiter_1477 * 12288UL) + ((_fuseiter_1478 * 4096UL) + ((_fuseiter_1479 * 128UL) + ((_fuseiter_1480 * 4UL) + _fuseiter_1481))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__509(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1482___fuseiter_1483_711 = 0UL; fused_0_fuseiter_1482___fuseiter_1483_711 < 16UL; fused_0_fuseiter_1482___fuseiter_1483_711 += 1UL) {
    for (uint64_t _fuseiter_1484 = 0UL; _fuseiter_1484 < 3UL; _fuseiter_1484 += 1UL) {
      for (uint64_t _fuseiter_1485 = 0UL; _fuseiter_1485 < 3UL; _fuseiter_1485 += 1UL) {
        for (uint64_t _fuseiter_1486 = 0UL; _fuseiter_1486 < 16UL; _fuseiter_1486 += 1UL) {
          for (uint64_t _fuseiter_1487 = 0UL; _fuseiter_1487 < 64UL; _fuseiter_1487 += 1UL) {
            for (uint64_t _fuseiter_1488 = 0UL; _fuseiter_1488 < 4UL; _fuseiter_1488 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1487 + ((fused_0_fuseiter_1482___fuseiter_1483_711 / 4UL) * 64UL)) * 2304UL) + ((((_fuseiter_1488 + (_fuseiter_1486 * 4UL)) + ((fused_0_fuseiter_1482___fuseiter_1483_711 % 4UL) * 64UL)) * 9UL) + ((_fuseiter_1484 * 3UL) + _fuseiter_1485)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1482___fuseiter_1483_711 / 4UL) * 147456UL) + (((fused_0_fuseiter_1482___fuseiter_1483_711 % 4UL) * 36864UL) + ((_fuseiter_1484 * 12288UL) + ((_fuseiter_1485 * 4096UL) + ((_fuseiter_1486 * 256UL) + ((_fuseiter_1487 * 4UL) + _fuseiter_1488))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__500(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1489___fuseiter_1490_712 = 0UL; fused_0_fuseiter_1489___fuseiter_1490_712 < 16UL; fused_0_fuseiter_1489___fuseiter_1490_712 += 1UL) {
    for (uint64_t _fuseiter_1491 = 0UL; _fuseiter_1491 < 3UL; _fuseiter_1491 += 1UL) {
      for (uint64_t _fuseiter_1492 = 0UL; _fuseiter_1492 < 3UL; _fuseiter_1492 += 1UL) {
        for (uint64_t _fuseiter_1493 = 0UL; _fuseiter_1493 < 32UL; _fuseiter_1493 += 1UL) {
          for (uint64_t _fuseiter_1494 = 0UL; _fuseiter_1494 < 32UL; _fuseiter_1494 += 1UL) {
            for (uint64_t _fuseiter_1495 = 0UL; _fuseiter_1495 < 4UL; _fuseiter_1495 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1494 + ((fused_0_fuseiter_1489___fuseiter_1490_712 / 2UL) * 32UL)) * 2304UL) + ((((_fuseiter_1495 + (_fuseiter_1493 * 4UL)) + ((fused_0_fuseiter_1489___fuseiter_1490_712 % 2UL) * 128UL)) * 9UL) + ((_fuseiter_1491 * 3UL) + _fuseiter_1492)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[(((fused_0_fuseiter_1489___fuseiter_1490_712 / 2UL) * 73728UL) + (((fused_0_fuseiter_1489___fuseiter_1490_712 % 2UL) * 36864UL) + ((_fuseiter_1491 * 12288UL) + ((_fuseiter_1492 * 4096UL) + ((_fuseiter_1493 * 128UL) + ((_fuseiter_1494 * 4UL) + _fuseiter_1495))))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__493(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1496 = 0UL; _fuseiter_1496 < 16UL; _fuseiter_1496 += 1UL) {
    for (uint64_t _fuseiter_1498 = 0UL; _fuseiter_1498 < 3UL; _fuseiter_1498 += 1UL) {
      for (uint64_t _fuseiter_1499 = 0UL; _fuseiter_1499 < 3UL; _fuseiter_1499 += 1UL) {
        for (uint64_t _fuseiter_1500 = 0UL; _fuseiter_1500 < 64UL; _fuseiter_1500 += 1UL) {
          for (uint64_t _fuseiter_1501 = 0UL; _fuseiter_1501 < 16UL; _fuseiter_1501 += 1UL) {
            for (uint64_t _fuseiter_1502 = 0UL; _fuseiter_1502 < 4UL; _fuseiter_1502 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_1501 + (_fuseiter_1496 * 16UL)) * 2304UL) + (((_fuseiter_1502 + (_fuseiter_1500 * 4UL)) * 9UL) + ((_fuseiter_1498 * 3UL) + _fuseiter_1499)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[((_fuseiter_1496 * 36864UL) + ((_fuseiter_1498 * 12288UL) + ((_fuseiter_1499 * 4096UL) + ((_fuseiter_1500 * 64UL) + ((_fuseiter_1501 * 4UL) + _fuseiter_1502)))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__484(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 = 0UL; fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 < 18UL; fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 += 1UL) {
    for (uint64_t _fuseiter_1507 = 0UL; _fuseiter_1507 < 64UL; _fuseiter_1507 += 1UL) {
      for (uint64_t _fuseiter_1508 = 0UL; _fuseiter_1508 < 128UL; _fuseiter_1508 += 1UL) {
        for (uint64_t _fuseiter_1509 = 0UL; _fuseiter_1509 < 4UL; _fuseiter_1509 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1508 + ((fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 / 9UL) * 128UL)) * 2304UL) + (((_fuseiter_1509 + (_fuseiter_1507 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 / 9UL) * 294912UL) + ((((fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 / 3UL) % 3UL) * 98304UL) + (((fused_0fused_0fused_0_fuseiter_1503___fuseiter_1504_713___fuseiter_1505_714___fuseiter_1506_715 % 3UL) * 32768UL) + ((_fuseiter_1507 * 512UL) + ((_fuseiter_1508 * 4UL) + _fuseiter_1509)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__480(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1510 = 0UL; _fuseiter_1510 < 32UL; _fuseiter_1510 += 1UL) {
    for (uint64_t _fuseiter_1511 = 0UL; _fuseiter_1511 < 4UL; _fuseiter_1511 += 1UL) {
      for (uint64_t _fuseiter_1514 = 0UL; _fuseiter_1514 < 32UL; _fuseiter_1514 += 1UL) {
        for (uint64_t _fuseiter_1515 = 0UL; _fuseiter_1515 < 32UL; _fuseiter_1515 += 1UL) {
          for (uint64_t _fuseiter_1516 = 0UL; _fuseiter_1516 < 4UL; _fuseiter_1516 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1515 + (_fuseiter_1510 * 32UL)) * 512UL) + ((_fuseiter_1516 + (_fuseiter_1514 * 4UL)) + (_fuseiter_1511 * 128UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1510 * 16384UL) + ((_fuseiter_1511 * 4096UL) + ((_fuseiter_1514 * 128UL) + ((_fuseiter_1515 * 4UL) + _fuseiter_1516))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__474(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 = 0UL; fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 < 12UL; fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 += 1UL) {
    for (uint64_t _fuseiter_1520 = 0UL; _fuseiter_1520 < 3UL; _fuseiter_1520 += 1UL) {
      for (uint64_t _fuseiter_1521 = 0UL; _fuseiter_1521 < 32UL; _fuseiter_1521 += 1UL) {
        for (uint64_t _fuseiter_1522 = 0UL; _fuseiter_1522 < 32UL; _fuseiter_1522 += 1UL) {
          for (uint64_t _fuseiter_1523 = 0UL; _fuseiter_1523 < 4UL; _fuseiter_1523 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1522 + ((fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 / 3UL) * 32UL)) * 1152UL) + (((_fuseiter_1523 + (_fuseiter_1521 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 % 3UL) * 3UL) + _fuseiter_1520)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 / 3UL) * 36864UL) + (((fused_0fused_0_fuseiter_1517___fuseiter_1518_716___fuseiter_1519_717 % 3UL) * 12288UL) + ((_fuseiter_1520 * 4096UL) + ((_fuseiter_1521 * 128UL) + ((_fuseiter_1522 * 4UL) + _fuseiter_1523)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__465(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 = 0UL; fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 < 24UL; fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 += 1UL) {
    for (uint64_t _fuseiter_1527 = 0UL; _fuseiter_1527 < 3UL; _fuseiter_1527 += 1UL) {
      for (uint64_t _fuseiter_1528 = 0UL; _fuseiter_1528 < 16UL; _fuseiter_1528 += 1UL) {
        for (uint64_t _fuseiter_1529 = 0UL; _fuseiter_1529 < 32UL; _fuseiter_1529 += 1UL) {
          for (uint64_t _fuseiter_1530 = 0UL; _fuseiter_1530 < 4UL; _fuseiter_1530 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1529 + ((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 / 6UL) * 32UL)) * 1152UL) + ((((_fuseiter_1530 + (_fuseiter_1528 * 4UL)) + (((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 / 3UL) % 2UL) * 64UL)) * 9UL) + (((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 % 3UL) * 3UL) + _fuseiter_1527)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 / 6UL) * 36864UL) + ((((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 / 3UL) % 2UL) * 18432UL) + (((fused_0fused_0_fuseiter_1524___fuseiter_1525_718___fuseiter_1526_719 % 3UL) * 6144UL) + ((_fuseiter_1527 * 2048UL) + ((_fuseiter_1528 * 128UL) + ((_fuseiter_1529 * 4UL) + _fuseiter_1530))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__460(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 = 0UL; fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 < 18UL; fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 += 1UL) {
    for (uint64_t _fuseiter_1535 = 0UL; _fuseiter_1535 < 16UL; _fuseiter_1535 += 1UL) {
      for (uint64_t _fuseiter_1536 = 0UL; _fuseiter_1536 < 128UL; _fuseiter_1536 += 1UL) {
        for (uint64_t _fuseiter_1537 = 0UL; _fuseiter_1537 < 4UL; _fuseiter_1537 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1536 + ((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 18UL) * 128UL)) * 1152UL) + ((((_fuseiter_1537 + (_fuseiter_1535 * 4UL)) + (((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 9UL) % 2UL) * 64UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 18UL) * 147456UL) + ((((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 9UL) % 2UL) * 73728UL) + ((((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 / 3UL) % 3UL) * 24576UL) + (((fused_0fused_0fused_0_fuseiter_1531___fuseiter_1532_720___fuseiter_1533_721___fuseiter_1534_722 % 3UL) * 8192UL) + ((_fuseiter_1535 * 512UL) + ((_fuseiter_1536 * 4UL) + _fuseiter_1537))))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__451(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 = 0UL; fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 < 12UL; fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 += 1UL) {
    for (uint64_t _fuseiter_1541 = 0UL; _fuseiter_1541 < 3UL; _fuseiter_1541 += 1UL) {
      for (uint64_t _fuseiter_1542 = 0UL; _fuseiter_1542 < 32UL; _fuseiter_1542 += 1UL) {
        for (uint64_t _fuseiter_1543 = 0UL; _fuseiter_1543 < 32UL; _fuseiter_1543 += 1UL) {
          for (uint64_t _fuseiter_1544 = 0UL; _fuseiter_1544 < 4UL; _fuseiter_1544 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1543 + ((fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 / 3UL) * 32UL)) * 1152UL) + (((_fuseiter_1544 + (_fuseiter_1542 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 % 3UL) * 3UL) + _fuseiter_1541)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 / 3UL) * 36864UL) + (((fused_0fused_0_fuseiter_1538___fuseiter_1539_723___fuseiter_1540_724 % 3UL) * 12288UL) + ((_fuseiter_1541 * 4096UL) + ((_fuseiter_1542 * 128UL) + ((_fuseiter_1543 * 4UL) + _fuseiter_1544)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__483(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 += 1UL) {
    for (uint64_t _fuseiter_1550 = 0UL; _fuseiter_1550 < 256UL; _fuseiter_1550 += 1UL) {
      for (uint64_t _fuseiter_1551 = 0UL; _fuseiter_1551 < 4UL; _fuseiter_1551 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1550 + ((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 / 128UL) * 256UL)) * 512UL) + ((_fuseiter_1551 + ((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 / 32UL) % 4UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 / 128UL) * 131072UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 / 32UL) % 4UL) * 32768UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1545___fuseiter_1546_725___fuseiter_1547_726___fuseiter_1548_727___fuseiter_1549_728 % 32UL) * 1024UL) + ((_fuseiter_1550 * 4UL) + _fuseiter_1551))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__445(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1552 = 0UL; _fuseiter_1552 < 16UL; _fuseiter_1552 += 1UL) {
    for (uint64_t _fuseiter_1553 = 0UL; _fuseiter_1553 < 2UL; _fuseiter_1553 += 1UL) {
      for (uint64_t _fuseiter_1556 = 0UL; _fuseiter_1556 < 32UL; _fuseiter_1556 += 1UL) {
        for (uint64_t _fuseiter_1557 = 0UL; _fuseiter_1557 < 32UL; _fuseiter_1557 += 1UL) {
          for (uint64_t _fuseiter_1558 = 0UL; _fuseiter_1558 < 4UL; _fuseiter_1558 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1557 + (_fuseiter_1552 * 32UL)) * 256UL) + ((_fuseiter_1558 + (_fuseiter_1556 * 4UL)) + (_fuseiter_1553 * 128UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1552 * 8192UL) + ((_fuseiter_1553 * 4096UL) + ((_fuseiter_1556 * 128UL) + ((_fuseiter_1557 * 4UL) + _fuseiter_1558))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__471(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 += 1UL) {
    for (uint64_t _fuseiter_1564 = 0UL; _fuseiter_1564 < 64UL; _fuseiter_1564 += 1UL) {
      for (uint64_t _fuseiter_1565 = 0UL; _fuseiter_1565 < 4UL; _fuseiter_1565 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1564 + ((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 / 128UL) * 64UL)) * 512UL) + ((_fuseiter_1565 + ((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 / 32UL) % 4UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 / 128UL) * 32768UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 / 32UL) % 4UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1559___fuseiter_1560_729___fuseiter_1561_730___fuseiter_1562_731___fuseiter_1563_732 % 32UL) * 256UL) + ((_fuseiter_1564 * 4UL) + _fuseiter_1565))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__464(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 += 1UL) {
    for (uint64_t _fuseiter_1571 = 0UL; _fuseiter_1571 < 128UL; _fuseiter_1571 += 1UL) {
      for (uint64_t _fuseiter_1572 = 0UL; _fuseiter_1572 < 4UL; _fuseiter_1572 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1571 + ((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 / 128UL) * 128UL)) * 512UL) + ((_fuseiter_1572 + ((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 / 16UL) % 8UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 / 128UL) * 65536UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 / 16UL) % 8UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1566___fuseiter_1567_733___fuseiter_1568_734___fuseiter_1569_735___fuseiter_1570_736 % 16UL) * 512UL) + ((_fuseiter_1571 * 4UL) + _fuseiter_1572))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__457(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1573___fuseiter_1574_737 = 0UL; fused_0_fuseiter_1573___fuseiter_1574_737 < 16UL; fused_0_fuseiter_1573___fuseiter_1574_737 += 1UL) {
    for (uint64_t _fuseiter_1577 = 0UL; _fuseiter_1577 < 32UL; _fuseiter_1577 += 1UL) {
      for (uint64_t _fuseiter_1578 = 0UL; _fuseiter_1578 < 32UL; _fuseiter_1578 += 1UL) {
        for (uint64_t _fuseiter_1579 = 0UL; _fuseiter_1579 < 4UL; _fuseiter_1579 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1578 + ((fused_0_fuseiter_1573___fuseiter_1574_737 / 4UL) * 32UL)) * 512UL) + ((_fuseiter_1579 + (_fuseiter_1577 * 4UL)) + ((fused_0_fuseiter_1573___fuseiter_1574_737 % 4UL) * 128UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1573___fuseiter_1574_737 / 4UL) * 16384UL) + (((fused_0_fuseiter_1573___fuseiter_1574_737 % 4UL) * 4096UL) + ((_fuseiter_1577 * 128UL) + ((_fuseiter_1578 * 4UL) + _fuseiter_1579))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__477(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1580___fuseiter_1581_738 = 0UL; fused_0_fuseiter_1580___fuseiter_1581_738 < 16UL; fused_0_fuseiter_1580___fuseiter_1581_738 += 1UL) {
    for (uint64_t _fuseiter_1584 = 0UL; _fuseiter_1584 < 8UL; _fuseiter_1584 += 1UL) {
      for (uint64_t _fuseiter_1585 = 0UL; _fuseiter_1585 < 128UL; _fuseiter_1585 += 1UL) {
        for (uint64_t _fuseiter_1586 = 0UL; _fuseiter_1586 < 4UL; _fuseiter_1586 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1585 + ((fused_0_fuseiter_1580___fuseiter_1581_738 / 4UL) * 128UL)) * 128UL) + ((_fuseiter_1586 + (_fuseiter_1584 * 4UL)) + ((fused_0_fuseiter_1580___fuseiter_1581_738 % 4UL) * 32UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1580___fuseiter_1581_738 / 4UL) * 16384UL) + (((fused_0_fuseiter_1580___fuseiter_1581_738 % 4UL) * 4096UL) + ((_fuseiter_1584 * 512UL) + ((_fuseiter_1585 * 4UL) + _fuseiter_1586))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__468(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 < 256UL; fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 += 1UL) {
    for (uint64_t _fuseiter_1592 = 0UL; _fuseiter_1592 < 64UL; _fuseiter_1592 += 1UL) {
      for (uint64_t _fuseiter_1593 = 0UL; _fuseiter_1593 < 4UL; _fuseiter_1593 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1592 + ((fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 / 32UL) * 64UL)) * 128UL) + (_fuseiter_1593 + ((fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 % 32UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 / 32UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1587___fuseiter_1588_739___fuseiter_1589_740___fuseiter_1590_741___fuseiter_1591_742 % 32UL) * 256UL) + ((_fuseiter_1592 * 4UL) + _fuseiter_1593)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__461(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 += 1UL) {
    for (uint64_t _fuseiter_1599 = 0UL; _fuseiter_1599 < 128UL; _fuseiter_1599 += 1UL) {
      for (uint64_t _fuseiter_1600 = 0UL; _fuseiter_1600 < 4UL; _fuseiter_1600 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1599 + ((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 / 32UL) * 128UL)) * 128UL) + ((_fuseiter_1600 + ((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 / 16UL) % 2UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 / 32UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 / 16UL) % 2UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1594___fuseiter_1595_743___fuseiter_1596_744___fuseiter_1597_745___fuseiter_1598_746 % 16UL) * 512UL) + ((_fuseiter_1599 * 4UL) + _fuseiter_1600))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__454(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1601 = 0UL; _fuseiter_1601 < 16UL; _fuseiter_1601 += 1UL) {
    for (uint64_t _fuseiter_1602 = 0UL; _fuseiter_1602 < 4UL; _fuseiter_1602 += 1UL) {
      for (uint64_t _fuseiter_1605 = 0UL; _fuseiter_1605 < 8UL; _fuseiter_1605 += 1UL) {
        for (uint64_t _fuseiter_1606 = 0UL; _fuseiter_1606 < 32UL; _fuseiter_1606 += 1UL) {
          for (uint64_t _fuseiter_1607 = 0UL; _fuseiter_1607 < 4UL; _fuseiter_1607 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1606 + (_fuseiter_1601 * 32UL)) * 128UL) + ((_fuseiter_1607 + (_fuseiter_1605 * 4UL)) + (_fuseiter_1602 * 32UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_1601 * 4096UL) + ((_fuseiter_1602 * 1024UL) + ((_fuseiter_1605 * 128UL) + ((_fuseiter_1606 * 4UL) + _fuseiter_1607))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__439(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 = 0UL; fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 < 18UL; fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 += 1UL) {
    for (uint64_t _fuseiter_1612 = 0UL; _fuseiter_1612 < 16UL; _fuseiter_1612 += 1UL) {
      for (uint64_t _fuseiter_1613 = 0UL; _fuseiter_1613 < 32UL; _fuseiter_1613 += 1UL) {
        for (uint64_t _fuseiter_1614 = 0UL; _fuseiter_1614 < 4UL; _fuseiter_1614 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1613 + ((fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 / 9UL) * 32UL)) * 576UL) + (((_fuseiter_1614 + (_fuseiter_1612 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 / 9UL) * 18432UL) + ((((fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 / 3UL) % 3UL) * 6144UL) + (((fused_0fused_0fused_0_fuseiter_1608___fuseiter_1609_747___fuseiter_1610_748___fuseiter_1611_749 % 3UL) * 2048UL) + ((_fuseiter_1612 * 128UL) + ((_fuseiter_1613 * 4UL) + _fuseiter_1614)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__430(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 = 0UL; fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 < 12UL; fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 += 1UL) {
    for (uint64_t _fuseiter_1618 = 0UL; _fuseiter_1618 < 3UL; _fuseiter_1618 += 1UL) {
      for (uint64_t _fuseiter_1619 = 0UL; _fuseiter_1619 < 16UL; _fuseiter_1619 += 1UL) {
        for (uint64_t _fuseiter_1620 = 0UL; _fuseiter_1620 < 16UL; _fuseiter_1620 += 1UL) {
          for (uint64_t _fuseiter_1621 = 0UL; _fuseiter_1621 < 4UL; _fuseiter_1621 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_1620 + ((fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 / 3UL) * 16UL)) * 576UL) + (((_fuseiter_1621 + (_fuseiter_1619 * 4UL)) * 9UL) + (((fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 % 3UL) * 3UL) + _fuseiter_1618)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 / 3UL) * 9216UL) + (((fused_0fused_0_fuseiter_1615___fuseiter_1616_750___fuseiter_1617_751 % 3UL) * 3072UL) + ((_fuseiter_1618 * 1024UL) + ((_fuseiter_1619 * 64UL) + ((_fuseiter_1620 * 4UL) + _fuseiter_1621)))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool reorder__423(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 = 0UL; fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 < 18UL; fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 += 1UL) {
    for (uint64_t _fuseiter_1626 = 0UL; _fuseiter_1626 < 16UL; _fuseiter_1626 += 1UL) {
      for (uint64_t _fuseiter_1627 = 0UL; _fuseiter_1627 < 32UL; _fuseiter_1627 += 1UL) {
        for (uint64_t _fuseiter_1628 = 0UL; _fuseiter_1628 < 4UL; _fuseiter_1628 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1627 + ((fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 / 9UL) * 32UL)) * 576UL) + (((_fuseiter_1628 + (_fuseiter_1626 * 4UL)) * 9UL) + ((((fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 / 3UL) % 3UL) * 3UL) + (fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 % 3UL))))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 / 9UL) * 18432UL) + ((((fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 / 3UL) % 3UL) * 6144UL) + (((fused_0fused_0fused_0_fuseiter_1622___fuseiter_1623_752___fuseiter_1624_753___fuseiter_1625_754 % 3UL) * 2048UL) + ((_fuseiter_1626 * 128UL) + ((_fuseiter_1627 * 4UL) + _fuseiter_1628)))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool reorder__448(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 += 1UL) {
    for (uint64_t _fuseiter_1634 = 0UL; _fuseiter_1634 < 64UL; _fuseiter_1634 += 1UL) {
      for (uint64_t _fuseiter_1635 = 0UL; _fuseiter_1635 < 4UL; _fuseiter_1635 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1634 + ((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_1635 + ((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 % 32UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 / 32UL) % 2UL) * 128UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 / 32UL) % 2UL) * 8192UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1629___fuseiter_1630_755___fuseiter_1631_756___fuseiter_1632_757___fuseiter_1633_758 % 32UL) * 256UL) + ((_fuseiter_1634 * 4UL) + _fuseiter_1635))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__436(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 += 1UL) {
    for (uint64_t _fuseiter_1641 = 0UL; _fuseiter_1641 < 32UL; _fuseiter_1641 += 1UL) {
      for (uint64_t _fuseiter_1642 = 0UL; _fuseiter_1642 < 4UL; _fuseiter_1642 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1641 + ((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 / 64UL) * 32UL)) * 256UL) + ((_fuseiter_1642 + ((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 % 16UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 / 16UL) % 4UL) * 64UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 / 64UL) * 8192UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 / 16UL) % 4UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1636___fuseiter_1637_759___fuseiter_1638_760___fuseiter_1639_761___fuseiter_1640_762 % 16UL) * 128UL) + ((_fuseiter_1641 * 4UL) + _fuseiter_1642))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__429(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 += 1UL) {
    for (uint64_t _fuseiter_1648 = 0UL; _fuseiter_1648 < 64UL; _fuseiter_1648 += 1UL) {
      for (uint64_t _fuseiter_1649 = 0UL; _fuseiter_1649 < 4UL; _fuseiter_1649 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1648 + ((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 / 64UL) * 64UL)) * 256UL) + ((_fuseiter_1649 + ((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 % 8UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 / 8UL) % 8UL) * 32UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 / 64UL) * 16384UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 / 8UL) % 8UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1643___fuseiter_1644_763___fuseiter_1645_764___fuseiter_1646_765___fuseiter_1647_766 % 8UL) * 256UL) + ((_fuseiter_1648 * 4UL) + _fuseiter_1649))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__442(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 += 1UL) {
    for (uint64_t _fuseiter_1655 = 0UL; _fuseiter_1655 < 64UL; _fuseiter_1655 += 1UL) {
      for (uint64_t _fuseiter_1656 = 0UL; _fuseiter_1656 < 4UL; _fuseiter_1656 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1655 + ((fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 / 16UL) * 64UL)) * 64UL) + (_fuseiter_1656 + ((fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 / 16UL) * 4096UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1650___fuseiter_1651_767___fuseiter_1652_768___fuseiter_1653_769___fuseiter_1654_770 % 16UL) * 256UL) + ((_fuseiter_1655 * 4UL) + _fuseiter_1656)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__433(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 < 128UL; fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 += 1UL) {
    for (uint64_t _fuseiter_1662 = 0UL; _fuseiter_1662 < 32UL; _fuseiter_1662 += 1UL) {
      for (uint64_t _fuseiter_1663 = 0UL; _fuseiter_1663 < 4UL; _fuseiter_1663 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1662 + ((fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 / 16UL) * 32UL)) * 64UL) + (_fuseiter_1663 + ((fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 % 16UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 / 16UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1657___fuseiter_1658_771___fuseiter_1659_772___fuseiter_1660_773___fuseiter_1661_774 % 16UL) * 128UL) + ((_fuseiter_1662 * 4UL) + _fuseiter_1663)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__426(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 < 64UL; fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 += 1UL) {
    for (uint64_t _fuseiter_1669 = 0UL; _fuseiter_1669 < 64UL; _fuseiter_1669 += 1UL) {
      for (uint64_t _fuseiter_1670 = 0UL; _fuseiter_1670 < 4UL; _fuseiter_1670 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_1669 + ((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 / 16UL) * 64UL)) * 64UL) + ((_fuseiter_1670 + ((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 % 8UL) * 4UL)) + (((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 / 8UL) % 2UL) * 32UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 / 16UL) * 4096UL) + ((((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 / 8UL) % 2UL) * 2048UL) + (((fused_0fused_0fused_0fused_0_fuseiter_1664___fuseiter_1665_775___fuseiter_1666_776___fuseiter_1667_777___fuseiter_1668_778 % 8UL) * 256UL) + ((_fuseiter_1669 * 4UL) + _fuseiter_1670))))] = __cached_1;
      }
    }
  }
  return true;
}

static bool reorder__419(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_1671 = 0UL; _fuseiter_1671 < 16UL; _fuseiter_1671 += 1UL) {
    for (uint64_t _fuseiter_1675 = 0UL; _fuseiter_1675 < 16UL; _fuseiter_1675 += 1UL) {
      for (uint64_t _fuseiter_1676 = 0UL; _fuseiter_1676 < 16UL; _fuseiter_1676 += 1UL) {
        for (uint64_t _fuseiter_1677 = 0UL; _fuseiter_1677 < 4UL; _fuseiter_1677 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1676 + (_fuseiter_1671 * 16UL)) * 64UL) + (_fuseiter_1677 + (_fuseiter_1675 * 4UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[((_fuseiter_1671 * 1024UL) + ((_fuseiter_1675 * 64UL) + ((_fuseiter_1676 * 4UL) + _fuseiter_1677)))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__656(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 += 1UL) {
    for (uint64_t _fuseiter_1682 = 0UL; _fuseiter_1682 < 64UL; _fuseiter_1682 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 % 8UL) * 64UL)) + _fuseiter_1682)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_779____itr_2_780____itr_3_781 % 8UL) * 64UL)) + _fuseiter_1682)]);
    }
  }
  return true;
}

static bool mul__655(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 < 8UL; fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 += 1UL) {
    for (uint64_t _fuseiter_1688 = 0UL; _fuseiter_1688 < 64UL; _fuseiter_1688 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 % 8UL) * 64UL)) + _fuseiter_1688)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 / 8UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_782____itr_2_783____itr_3_784 % 8UL) * 64UL)) + _fuseiter_1688)]);
    }
  }
  return true;
}

static bool reorder__534(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_1690___fuseiter_1691_785 = 0UL; fused_0_fuseiter_1690___fuseiter_1691_785 < 32UL; fused_0_fuseiter_1690___fuseiter_1691_785 += 1UL) {
    for (uint64_t _fuseiter_1694 = 0UL; _fuseiter_1694 < 64UL; _fuseiter_1694 += 1UL) {
      for (uint64_t _fuseiter_1695 = 0UL; _fuseiter_1695 < 64UL; _fuseiter_1695 += 1UL) {
        for (uint64_t _fuseiter_1696 = 0UL; _fuseiter_1696 < 4UL; _fuseiter_1696 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_1695 + ((fused_0_fuseiter_1690___fuseiter_1691_785 / 4UL) * 64UL)) * 1024UL) + ((_fuseiter_1696 + (_fuseiter_1694 * 4UL)) + ((fused_0_fuseiter_1690___fuseiter_1691_785 % 4UL) * 256UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_1690___fuseiter_1691_785 / 4UL) * 65536UL) + (((fused_0_fuseiter_1690___fuseiter_1691_785 % 4UL) * 16384UL) + ((_fuseiter_1694 * 256UL) + ((_fuseiter_1695 * 4UL) + _fuseiter_1696))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__243(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_786____itr_2_787 = 0UL; fused_0fused_0__itr_0____itr_1_786____itr_2_787 < 1048576UL; fused_0fused_0__itr_0____itr_1_786____itr_2_787 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_786____itr_2_787 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_786____itr_2_787 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_786____itr_2_787 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_786____itr_2_787 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_786____itr_2_787 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__244(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_788____itr_2_789 = 0UL; fused_0fused_0__itr_0____itr_1_788____itr_2_789 < 1048576UL; fused_0fused_0__itr_0____itr_1_788____itr_2_789 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_788____itr_2_789 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_788____itr_2_789 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_788____itr_2_789 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_788____itr_2_789 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__252(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_790____itr_2_791 = 0UL; fused_0fused_0__itr_0____itr_1_790____itr_2_791 < 1048576UL; fused_0fused_0__itr_0____itr_1_790____itr_2_791 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_790____itr_2_791 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_790____itr_2_791 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_790____itr_2_791 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_790____itr_2_791 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_790____itr_2_791 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__253(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_792____itr_2_793 = 0UL; fused_0fused_0__itr_0____itr_1_792____itr_2_793 < 1048576UL; fused_0fused_0__itr_0____itr_1_792____itr_2_793 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_792____itr_2_793 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_792____itr_2_793 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_792____itr_2_793 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_792____itr_2_793 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__261(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_794____itr_2_795 = 0UL; fused_0fused_0__itr_0____itr_1_794____itr_2_795 < 1048576UL; fused_0fused_0__itr_0____itr_1_794____itr_2_795 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_794____itr_2_795 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_794____itr_2_795 % 512UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_794____itr_2_795 / 512UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_794____itr_2_795 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_794____itr_2_795 % 512UL))] = __cached_2;
  }
  return true;
}

static bool cast__262(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_796____itr_2_797 = 0UL; fused_0fused_0__itr_0____itr_1_796____itr_2_797 < 1048576UL; fused_0fused_0__itr_0____itr_1_796____itr_2_797 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_796____itr_2_797 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_796____itr_2_797 % 512UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_796____itr_2_797 / 512UL) * 512UL) + (fused_0fused_0__itr_0____itr_1_796____itr_2_797 % 512UL))] = __cached_1;
  }
  return true;
}

static bool mul__246(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_798____itr_2_799 = 0UL; fused_0fused_0__itr_0____itr_1_798____itr_2_799 < 1048576UL; fused_0fused_0__itr_0____itr_1_798____itr_2_799 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_798____itr_2_799 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_798____itr_2_799 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_798____itr_2_799 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_798____itr_2_799 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_798____itr_2_799 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__247(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_800____itr_2_801 = 0UL; fused_0fused_0__itr_0____itr_1_800____itr_2_801 < 1048576UL; fused_0fused_0__itr_0____itr_1_800____itr_2_801 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_800____itr_2_801 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_800____itr_2_801 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_800____itr_2_801 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_800____itr_2_801 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool mul__255(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_802____itr_2_803 = 0UL; fused_0fused_0__itr_0____itr_1_802____itr_2_803 < 1048576UL; fused_0fused_0__itr_0____itr_1_802____itr_2_803 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_802____itr_2_803 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_802____itr_2_803 % 2048UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_802____itr_2_803 / 2048UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_802____itr_2_803 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_802____itr_2_803 % 2048UL))] = __cached_2;
  }
  return true;
}

static bool cast__256(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_804____itr_2_805 = 0UL; fused_0fused_0__itr_0____itr_1_804____itr_2_805 < 1048576UL; fused_0fused_0__itr_0____itr_1_804____itr_2_805 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_804____itr_2_805 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_804____itr_2_805 % 2048UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_804____itr_2_805 / 2048UL) * 2048UL) + (fused_0fused_0__itr_0____itr_1_804____itr_2_805 % 2048UL))] = __cached_1;
  }
  return true;
}

static bool mul__234(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_806____itr_2_807 = 0UL; fused_0fused_0__itr_0____itr_1_806____itr_2_807 < 2097152UL; fused_0fused_0__itr_0____itr_1_806____itr_2_807 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_806____itr_2_807 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_806____itr_2_807 % 1024UL))];
    float __cached_1;
    __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_806____itr_2_807 / 1024UL)];
    float __cached_2;
    __cached_2 = (__cached_0 * __cached_1);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_806____itr_2_807 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_806____itr_2_807 % 1024UL))] = __cached_2;
  }
  return true;
}

static bool cast__235(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_808____itr_2_809 = 0UL; fused_0fused_0__itr_0____itr_1_808____itr_2_809 < 2097152UL; fused_0fused_0__itr_0____itr_1_808____itr_2_809 += 1UL) {
    float __cached_0;
    __cached_0 = __ins_0[(((fused_0fused_0__itr_0____itr_1_808____itr_2_809 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_808____itr_2_809 % 1024UL))];
    int8_t __cached_1;
    __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
    __outs_0[(((fused_0fused_0__itr_0____itr_1_808____itr_2_809 / 1024UL) * 1024UL) + (fused_0fused_0__itr_0____itr_1_808____itr_2_809 % 1024UL))] = __cached_1;
  }
  return true;
}

static bool batchwise_9_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_res2a_conv_2_cast_mul_add_cast_add_cast_reorder_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_reorder_res2c_conv_0_cast_mul_add_relu_cast_reorder_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_reorder_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_reorder_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_res3d_conv_2_cast_mul_add_cast_add_cast_res4a_conv_b_cast_mul_add_cast_reorder_res4a_conv_0_cast_mul_add_relu_cast_res4a_conv_1_cast_mul_add_relu_cast_reorder_res4a_conv_2_cast_mul_add_cast_add_cast_reorder_res4b_conv_0_cast_mul_add_relu_cast_reorder_res4b_conv_1_cast_mul_add_relu_cast_reorder_res4b_conv_2_cast_mul_add_cast_add_cast_reorder_res4c_conv_0_cast_mul_add_relu_cast_reorder_res4c_conv_1_cast_mul_add_relu_cast_reorder_res4c_conv_2_cast_mul_add_cast_add_cast_res4d_conv_0_cast_mul_add_relu_cast_res4d_conv_1_cast_mul_add_relu_cast_reorder_res4d_conv_2_cast_mul_add_cast_add_cast_res4e_conv_0_cast_mul_add_relu_cast_reorder_res4e_conv_1_cast_mul_add_relu_cast_reorder_res4e_conv_2_cast_mul_add_cast_add_cast_reorder_reorder_res4f_conv_0_cast_mul_add_relu_cast_reorder_res4f_conv_1_cast_mul_add_relu_cast_reorder_res4f_conv_2_cast_mul_add_cast_add_cast__683(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69, int8_t* __restrict__ __ins_70, float* __restrict__ __ins_71, float* __restrict__ __ins_72, int8_t* __restrict__ __ins_73, float* __restrict__ __ins_74, float* __restrict__ __ins_75, int8_t* __restrict__ __ins_76, float* __restrict__ __ins_77, float* __restrict__ __ins_78, int8_t* __restrict__ __ins_79, float* __restrict__ __ins_80, float* __restrict__ __ins_81, int8_t* __restrict__ __ins_82, float* __restrict__ __ins_83, float* __restrict__ __ins_84, int8_t* __restrict__ __ins_85, float* __restrict__ __ins_86, float* __restrict__ __ins_87, int8_t* __restrict__ __ins_88, float* __restrict__ __ins_89, float* __restrict__ __ins_90, int8_t* __restrict__ __ins_91, float* __restrict__ __ins_92, float* __restrict__ __ins_93, int8_t* __restrict__ __ins_94, float* __restrict__ __ins_95, float* __restrict__ __ins_96, int8_t* __restrict__ __ins_97, float* __restrict__ __ins_98, float* __restrict__ __ins_99, int8_t* __restrict__ __ins_100, float* __restrict__ __ins_101, float* __restrict__ __ins_102, int8_t* __restrict__ __ins_103, float* __restrict__ __ins_104, float* __restrict__ __ins_105, int8_t* __restrict__ __ins_106, float* __restrict__ __ins_107, float* __restrict__ __ins_108, int8_t* __restrict__ __ins_109, float* __restrict__ __ins_110, float* __restrict__ __ins_111, int8_t* __restrict__ __ins_112, float* __restrict__ __ins_113, float* __restrict__ __ins_114, int8_t* __restrict__ __ins_115, float* __restrict__ __ins_116, float* __restrict__ __ins_117, int8_t* __restrict__ __ins_118, float* __restrict__ __ins_119, float* __restrict__ __ins_120, int8_t* __restrict__ __ins_121, float* __restrict__ __ins_122, float* __restrict__ __ins_123, int8_t* __restrict__ __ins_124, float* __restrict__ __ins_125, float* __restrict__ __ins_126) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 9UL; __batchwise_iter_0 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 2021632UL);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_127 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_0_cast_mul_add_relu_cast__8(buffer_127, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_4[0UL], &__ins_5[0UL], &__ins_6[0UL]);
    // [s8 [1, 2, 56, 56, 32] @ ABCD32b]
    int8_t* buffer_128 = (int8_t*)&__rescheduled_1[802816UL];
    res2a_conv_1_cast_mul_add_relu_cast__12(buffer_128, buffer_127, &__ins_7[0UL], &__ins_8[0UL], &__ins_9[0UL]);
    // [s8 [1, 4, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_129 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_b_cast_mul_add_cast_reorder__4(buffer_129, &__ins_0[(__batchwise_iter_0 * 200704UL)], &__ins_1[0UL], &__ins_2[0UL], &__ins_3[0UL]);
    // [u8 [1, 8, 56, 56, 32] @ ABCD32b]
    uint8_t* buffer_130 = (uint8_t*)&__rescheduled_1[1218816UL];
    res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(buffer_130, buffer_128, &__ins_10[0UL], &__ins_11[0UL], &__ins_12[0UL], buffer_129);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_131 = (int8_t*)&__rescheduled_1[0UL];
    res2b_conv_0_cast_mul_add_relu_cast__20(buffer_131, buffer_130, &__ins_13[0UL], &__ins_14[0UL], &__ins_15[0UL]);
    // [s8 [1, 1, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_132 = (int8_t*)&__rescheduled_1[215296UL];
    res2b_conv_1_cast_mul_add_relu_cast_reorder__24(buffer_132, buffer_131, &__ins_16[0UL], &__ins_17[0UL], &__ins_18[0UL]);
    // [u8 [1, 4, 56, 56, 64] @ ABCD64b]
    uint8_t* buffer_133 = (uint8_t*)&__rescheduled_1[416000UL];
    res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(buffer_133, buffer_132, &__ins_19[0UL], &__ins_20[0UL], &__ins_21[0UL], buffer_130);
    // [s8 [1, 1, 58, 58, 64] @ ABCD64b]
    int8_t* buffer_134 = (int8_t*)&__rescheduled_1[0UL];
    res2c_conv_0_cast_mul_add_relu_cast_reorder__32(buffer_134, buffer_133, &__ins_22[0UL], &__ins_23[0UL], &__ins_24[0UL]);
    // [s8 [1, 1, 56, 56, 64] @ ABCD64b]
    int8_t* buffer_135 = (int8_t*)&__rescheduled_1[215296UL];
    res2c_conv_1_cast_mul_add_relu_cast_reorder__36(buffer_135, buffer_134, &__ins_25[0UL], &__ins_26[0UL], &__ins_27[0UL]);
    // [u8 [1, 2, 56, 56, 128] @ ABCD128b]
    uint8_t* buffer_136 = (uint8_t*)&__rescheduled_1[1218816UL];
    res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(buffer_136, buffer_135, &__ins_28[0UL], &__ins_29[0UL], &__ins_30[0UL], buffer_133);
    // [s8 [1, 1, 58, 58, 128] @ ABCD128b]
    int8_t* buffer_137 = (int8_t*)&__rescheduled_1[0UL];
    res3a_conv_0_cast_mul_add_relu_cast_reorder__48(buffer_137, buffer_136, &__ins_34[0UL], &__ins_35[0UL], &__ins_36[0UL]);
    // [s8 [1, 4, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_138 = (int8_t*)&__rescheduled_1[430592UL];
    res3a_conv_1_cast_mul_add_relu_cast__52(buffer_138, buffer_137, &__ins_37[0UL], &__ins_38[0UL], &__ins_39[0UL]);
    // [s8 [1, 16, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_139 = (int8_t*)&__rescheduled_1[0UL];
    res3a_conv_b_cast_mul_add_cast__44(buffer_139, buffer_136, &__ins_31[0UL], &__ins_32[0UL], &__ins_33[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_140 = (uint8_t*)&__rescheduled_1[530944UL];
    res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(buffer_140, buffer_138, &__ins_40[0UL], &__ins_41[0UL], &__ins_42[0UL], buffer_139);
    // [s8 [1, 2, 30, 30, 64] @ ABCD64b]
    int8_t* buffer_141 = (int8_t*)&__rescheduled_1[0UL];
    res3b_conv_0_cast_mul_add_relu_cast_reorder__60(buffer_141, buffer_140, &__ins_43[0UL], &__ins_44[0UL], &__ins_45[0UL]);
    // [s8 [1, 2, 28, 28, 64] @ ABCD64b]
    int8_t* buffer_142 = (int8_t*)&__rescheduled_1[115200UL];
    res3b_conv_1_cast_mul_add_relu_cast_reorder__64(buffer_142, buffer_141, &__ins_46[0UL], &__ins_47[0UL], &__ins_48[0UL]);
    // [u8 [1, 8, 28, 28, 64] @ ABCD64b]
    uint8_t* buffer_143 = (uint8_t*)&__rescheduled_1[932352UL];
    res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(buffer_143, buffer_142, &__ins_49[0UL], &__ins_50[0UL], &__ins_51[0UL], buffer_140);
    // [s8 [1, 2, 30, 30, 64] @ ABCD64b]
    int8_t* buffer_144 = (int8_t*)&__rescheduled_1[1333760UL];
    res3c_conv_0_cast_mul_add_relu_cast_reorder__72(buffer_144, buffer_143, &__ins_52[0UL], &__ins_53[0UL], &__ins_54[0UL]);
    // [s8 [1, 1, 28, 28, 128] @ ABCD128b]
    int8_t* buffer_145 = (int8_t*)&__rescheduled_1[1448960UL];
    res3c_conv_1_cast_mul_add_relu_cast_reorder__76(buffer_145, buffer_144, &__ins_55[0UL], &__ins_56[0UL], &__ins_57[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_146 = (uint8_t*)&__rescheduled_1[1549312UL];
    res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(buffer_146, buffer_145, &__ins_58[0UL], &__ins_59[0UL], &__ins_60[0UL], buffer_143);
    // [s8 [1, 1, 30, 30, 128] @ ABCD128b]
    int8_t* buffer_147 = (int8_t*)&__rescheduled_1[0UL];
    res3d_conv_0_cast_mul_add_relu_cast_reorder__84(buffer_147, buffer_146, &__ins_61[0UL], &__ins_62[0UL], &__ins_63[0UL]);
    // [s8 [1, 4, 28, 28, 32] @ ABCD32b]
    int8_t* buffer_148 = (int8_t*)&__rescheduled_1[115200UL];
    res3d_conv_1_cast_mul_add_relu_cast__88(buffer_148, buffer_147, &__ins_64[0UL], &__ins_65[0UL], &__ins_66[0UL]);
    // [u8 [1, 4, 28, 28, 128] @ ABCD128b]
    uint8_t* buffer_149 = (uint8_t*)&__rescheduled_1[215552UL];
    res3d_conv_2_cast_mul_add_cast_add_cast__92(buffer_149, buffer_148, &__ins_67[0UL], &__ins_68[0UL], &__ins_69[0UL], buffer_146);
    // [s8 [1, 1, 30, 30, 256] @ ABCD256b]
    int8_t* buffer_150 = (int8_t*)&__rescheduled_1[616960UL];
    res4a_conv_0_cast_mul_add_relu_cast__100(buffer_150, buffer_149, &__ins_73[0UL], &__ins_74[0UL], &__ins_75[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_151 = (int8_t*)&__rescheduled_1[0UL];
    res4a_conv_1_cast_mul_add_relu_cast_reorder__104(buffer_151, buffer_150, &__ins_76[0UL], &__ins_77[0UL], &__ins_78[0UL]);
    // [s8 [1, 8, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_152 = (int8_t*)&__rescheduled_1[616960UL];
    res4a_conv_b_cast_mul_add_cast_reorder__96(buffer_152, buffer_149, &__ins_70[0UL], &__ins_71[0UL], &__ins_72[0UL]);
    // [u8 [1, 1, 14, 14, 1024] @ ABCD1024b]
    uint8_t* buffer_153 = (uint8_t*)&__rescheduled_1[50176UL];
    res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(buffer_153, buffer_151, &__ins_79[0UL], &__ins_80[0UL], &__ins_81[0UL], buffer_152);
    // [s8 [1, 1, 16, 16, 256] @ ABCD256b]
    int8_t* buffer_154 = (int8_t*)&__rescheduled_1[250880UL];
    res4b_conv_0_cast_mul_add_relu_cast_reorder__112(buffer_154, buffer_153, &__ins_82[0UL], &__ins_83[0UL], &__ins_84[0UL]);
    // [s8 [1, 4, 14, 14, 64] @ ABCD64b]
    int8_t* buffer_155 = (int8_t*)&__rescheduled_1[0UL];
    res4b_conv_1_cast_mul_add_relu_cast_reorder__116(buffer_155, buffer_154, &__ins_85[0UL], &__ins_86[0UL], &__ins_87[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_156 = (uint8_t*)&__rescheduled_1[250880UL];
    res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(buffer_156, buffer_155, &__ins_88[0UL], &__ins_89[0UL], &__ins_90[0UL], buffer_153);
    // [s8 [1, 2, 16, 16, 128] @ ABCD128b]
    int8_t* buffer_157 = (int8_t*)&__rescheduled_1[0UL];
    res4c_conv_0_cast_mul_add_relu_cast_reorder__124(buffer_157, buffer_156, &__ins_91[0UL], &__ins_92[0UL], &__ins_93[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_158 = (int8_t*)&__rescheduled_1[65536UL];
    res4c_conv_1_cast_mul_add_relu_cast_reorder__128(buffer_158, buffer_157, &__ins_94[0UL], &__ins_95[0UL], &__ins_96[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_159 = (uint8_t*)&__rescheduled_1[451584UL];
    res4c_conv_2_cast_mul_add_cast_add_cast__132(buffer_159, buffer_158, &__ins_97[0UL], &__ins_98[0UL], &__ins_99[0UL], buffer_156);
    // [s8 [1, 4, 16, 16, 64] @ ABCD64b]
    int8_t* buffer_160 = (int8_t*)&__rescheduled_1[0UL];
    res4d_conv_0_cast_mul_add_relu_cast__136(buffer_160, buffer_159, &__ins_100[0UL], &__ins_101[0UL], &__ins_102[0UL]);
    // [s8 [1, 2, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_161 = (int8_t*)&__rescheduled_1[65536UL];
    res4d_conv_1_cast_mul_add_relu_cast_reorder__140(buffer_161, buffer_160, &__ins_103[0UL], &__ins_104[0UL], &__ins_105[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_162 = (uint8_t*)&__rescheduled_1[115712UL];
    res4d_conv_2_cast_mul_add_cast_add_cast__144(buffer_162, buffer_161, &__ins_106[0UL], &__ins_107[0UL], &__ins_108[0UL], buffer_159);
    // [s8 [1, 2, 16, 16, 128] @ ABCD128b]
    int8_t* buffer_163 = (int8_t*)&__rescheduled_1[0UL];
    res4e_conv_0_cast_mul_add_relu_cast_reorder__148(buffer_163, buffer_162, &__ins_109[0UL], &__ins_110[0UL], &__ins_111[0UL]);
    // [s8 [1, 1, 14, 14, 256] @ ABCD256b]
    int8_t* buffer_164 = (int8_t*)&__rescheduled_1[65536UL];
    res4e_conv_1_cast_mul_add_relu_cast_reorder__152(buffer_164, buffer_163, &__ins_112[0UL], &__ins_113[0UL], &__ins_114[0UL]);
    // [u8 [1, 8, 14, 14, 128] @ ABCD128b]
    uint8_t* buffer_165 = (uint8_t*)&__rescheduled_1[316416UL];
    // [u8 [1, 4, 14, 14, 256] @ ABCD256b]
    uint8_t* buffer_166 = (uint8_t*)&__rescheduled_1[517120UL];
    res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(buffer_165, buffer_166, buffer_164, &__ins_115[0UL], &__ins_116[0UL], &__ins_117[0UL], buffer_162);
    // [u8 [1, 2, 14, 14, 512] @ ABCD512b]
    uint8_t* buffer_167 = (uint8_t*)&__rescheduled_1[0UL];
    reorder__157(buffer_167, buffer_165);
    // [s8 [1, 4, 16, 16, 64] @ ABCD64b]
    int8_t* buffer_168 = (int8_t*)&__rescheduled_1[200704UL];
    res4f_conv_0_cast_mul_add_relu_cast_reorder__161(buffer_168, buffer_167, &__ins_118[0UL], &__ins_119[0UL], &__ins_120[0UL]);
    // [s8 [1, 2, 14, 14, 128] @ ABCD128b]
    int8_t* buffer_169 = (int8_t*)&__rescheduled_1[0UL];
    res4f_conv_1_cast_mul_add_relu_cast_reorder__165(buffer_169, buffer_168, &__ins_121[0UL], &__ins_122[0UL], &__ins_123[0UL]);
    res4f_conv_2_cast_mul_add_cast_add_cast__170(&__outs_0[(__batchwise_iter_0 * 200704UL)], buffer_169, &__ins_124[0UL], &__ins_125[0UL], &__ins_126[0UL], buffer_166);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}


static bool res2a_conv_0_cast_mul_add_relu_cast__8(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
    int32_t* __origouts_520_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[(p_o * 7168UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[0UL];
    B_list[0UL] = __cached_1;
    void* _arg_cache_0 = &__origouts_520_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache, A_list, B_list, &__origouts_520_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter1759 = 0UL; _fuseiter1759 < 2UL; _fuseiter1759 += 1UL) {
      for (uint64_t _fuseiter1760 = 0UL; _fuseiter1760 < 56UL; _fuseiter1760 += 1UL) {
        for (uint64_t _fuseiter1761 = 0UL; _fuseiter1761 < 64UL; _fuseiter1761 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_520_shr[((_fuseiter1759 * 3584UL) + ((_fuseiter1760 * 64UL) + _fuseiter1761))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter1761]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter1761]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 2UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter1759 * 3712UL) + ((_fuseiter1760 * 64UL) + _fuseiter1761)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_520_shr);
  }
  return true;
}

static bool res2a_conv_1_cast_mul_add_relu_cast__12(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_2 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0k_o__n_812 = 0UL; fused_0k_o__n_812 < 2UL; fused_0k_o__n_812 += 1UL) {
    int32_t* __origouts_530_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    for (uint64_t o_o = 0UL; o_o < 224UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_530_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))], 0, 1792UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 14UL) / 56UL) + r) * 3712UL) + ((((o_o * 14UL) % 56UL) + s) * 64UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_812 * 18432UL) + ((r * 6144UL) + (s * 2048UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_1 = &__origouts_530_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_2, A_list, B_list, &__origouts_530_shr[((((o_o * 14UL) / 56UL) * 1792UL) + (((o_o * 14UL) % 56UL) * 32UL))], 1, 64, 2048, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter1789 = 0UL; _fuseiter1789 < 56UL; _fuseiter1789 += 1UL) {
      for (uint64_t _fuseiter1790 = 0UL; _fuseiter1790 < 56UL; _fuseiter1790 += 1UL) {
        for (uint64_t _fuseiter1791 = 0UL; _fuseiter1791 < 32UL; _fuseiter1791 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_530_shr[((_fuseiter1789 * 1792UL) + ((_fuseiter1790 * 32UL) + _fuseiter1791))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_812 * 32UL) + _fuseiter1791)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_812 * 32UL) + _fuseiter1791)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[((fused_0k_o__n_812 * 100352UL) + ((_fuseiter1789 * 1792UL) + ((_fuseiter1790 * 32UL) + _fuseiter1791)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_530_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2a_conv_b_cast_mul_add_cast_reorder__4(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_4 = *(void**)(__module_data + 24);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_813 = 0UL; fused_0n__k_813 < 16UL; fused_0n__k_813 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_540_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_813 / 16UL) * 200704UL) + (p_o * 7168UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_813 % 16UL) * 1024UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_2 = &__origouts_540_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_4, A_list, B_list, &__origouts_540_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter1819 = 0UL; _fuseiter1819 < 2UL; _fuseiter1819 += 1UL) {
        for (uint64_t _fuseiter1820 = 0UL; _fuseiter1820 < 56UL; _fuseiter1820 += 1UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_540_shr[((_fuseiter1819 * 896UL) + (_fuseiter1820 * 16UL))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_813 / 16UL) * 256UL) + ((fused_0n__k_813 % 16UL) * 16UL))]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0n__k_813 % 16UL) * 16UL)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_813 / 16UL) * 802816UL) + (((((fused_0n__k_813 % 16UL) * 16UL) / 64UL) * 200704UL) + (((_fuseiter1819 + (p_o * 2UL)) * 3584UL) + ((_fuseiter1820 * 64UL) + (((fused_0n__k_813 % 16UL) * 16UL) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_540_shr);
    }
  }
  return true;
}

static bool res2a_conv_2_cast_mul_add_cast_add_cast_reorder__16(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_6 = *(void**)(__module_data + 32);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_814 = 0UL; fused_0n__k_814 < 4UL; fused_0n__k_814 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_550_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_814 / 4UL) * 200704UL) + ((c * 100352UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_814 % 4UL) * 4096UL) + (c * 2048UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_3 = &__origouts_550_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_6, A_list, B_list, &__origouts_550_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter1849 = 0UL; _fuseiter1849 < 56UL; _fuseiter1849 += 1UL) {
        for (uint64_t _fuseiter1850 = 0UL; _fuseiter1850 < 64UL; _fuseiter1850 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_550_shr[((_fuseiter1849 * 64UL) + _fuseiter1850)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_814 / 4UL) * 256UL) + ((fused_0n__k_814 % 4UL) * 64UL)) + _fuseiter1850)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_814 % 4UL) * 64UL) + _fuseiter1850)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0n__k_814 / 4UL) * 802816UL) + (((fused_0n__k_814 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter1849 * 64UL) + _fuseiter1850))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_814 / 4UL) * 802816UL) + ((((_fuseiter1850 + ((fused_0n__k_814 % 4UL) * 64UL)) / 32UL) * 100352UL) + ((p_o * 1792UL) + ((_fuseiter1849 * 32UL) + ((_fuseiter1850 + ((fused_0n__k_814 % 4UL) * 64UL)) % 32UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_550_shr);
    }
  }
  return true;
}

static bool res2b_conv_0_cast_mul_add_relu_cast__20(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_8 = *(void**)(__module_data + 40);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t p_o = 0UL; p_o < 8UL; p_o += 1UL) {
    int32_t* __origouts_560_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 2048UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_4 = &__origouts_560_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_8, A_list, B_list, &__origouts_560_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter1889 = 0UL; _fuseiter1889 < 7UL; _fuseiter1889 += 1UL) {
      for (uint64_t _fuseiter1890 = 0UL; _fuseiter1890 < 56UL; _fuseiter1890 += 1UL) {
        for (uint64_t _fuseiter1891 = 0UL; _fuseiter1891 < 64UL; _fuseiter1891 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_560_shr[((_fuseiter1889 * 3584UL) + ((_fuseiter1890 * 64UL) + _fuseiter1891))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter1891]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter1891]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 7UL) + 1UL) * 3712UL) + 64UL) + ((_fuseiter1889 * 3712UL) + ((_fuseiter1890 * 64UL) + _fuseiter1891)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_560_shr);
  }
  return true;
}

static bool res2b_conv_1_cast_mul_add_relu_cast_reorder__24(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_10 = *(void**)(__module_data + 48);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0n__k_o_817 = 0UL; fused_0n__k_o_817 < 4UL; fused_0n__k_o_817 += 1UL) {
    int32_t* __origouts_570_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    for (uint64_t o_o = 0UL; o_o < 112UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_570_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))], 0, 1792UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0n__k_o_817 / 4UL) * 215296UL) + (((((o_o * 28UL) / 56UL) + r) * 3712UL) + ((((o_o * 28UL) % 56UL) + s) * 64UL)))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0n__k_o_817 % 4UL) * 9216UL) + ((r * 3072UL) + (s * 1024UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_5 = &__origouts_570_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_10, A_list, B_list, &__origouts_570_shr[((((o_o * 28UL) / 56UL) * 896UL) + (((o_o * 28UL) % 56UL) * 16UL))], 1, 64, 1024, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter1919 = 0UL; _fuseiter1919 < 56UL; _fuseiter1919 += 1UL) {
      for (uint64_t _fuseiter1920 = 0UL; _fuseiter1920 < 56UL; _fuseiter1920 += 1UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_570_shr[((_fuseiter1919 * 896UL) + (_fuseiter1920 * 16UL))]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_817 / 4UL) * 64UL) + ((fused_0n__k_o_817 % 4UL) * 16UL))]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[((fused_0n__k_o_817 % 4UL) * 16UL)]);
        __cached_3 = (__cached_3 + __cached_5);
        __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = __cached_6;
        vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_o_817 / 4UL) * 200704UL) + (((((fused_0n__k_o_817 % 4UL) * 16UL) / 64UL) * 200704UL) + ((_fuseiter1919 * 3584UL) + ((_fuseiter1920 * 64UL) + (((fused_0n__k_o_817 % 4UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_570_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2b_conv_2_cast_mul_add_cast_add_cast_reorder__28(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_12 = *(void**)(__module_data + 56);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_818 = 0UL; fused_0k__n_818 < 8UL; fused_0k__n_818 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_580_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_818 * 2048UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_6 = &__origouts_580_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_12, A_list, B_list, &__origouts_580_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter1955 = 0UL; _fuseiter1955 < 56UL; _fuseiter1955 += 1UL) {
        for (uint64_t _fuseiter1956 = 0UL; _fuseiter1956 < 32UL; _fuseiter1956 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_580_shr[((_fuseiter1955 * 32UL) + _fuseiter1956)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_818 * 32UL) + _fuseiter1956)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_818 * 32UL) + _fuseiter1956)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_818 * 100352UL) + (p_o * 1792UL)) + ((_fuseiter1955 * 32UL) + _fuseiter1956))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter1956 + (fused_0k__n_818 * 32UL)) / 64UL) * 200704UL) + ((p_o * 3584UL) + ((_fuseiter1955 * 64UL) + ((_fuseiter1956 + (fused_0k__n_818 * 32UL)) % 64UL))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_580_shr);
    }
  }
  return true;
}

static bool res2c_conv_0_cast_mul_add_relu_cast_reorder__32(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_14 = *(void**)(__module_data + 64);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3712UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3712UL)], 0, 64UL);
    memset(&__outs_0[(((p1 + 1UL) * 3712UL) + 3648UL)], 0, 64UL);
  }
  memset(&__outs_0[211584UL], 0, 3712UL);
  for (uint64_t fused_0n__k_820 = 0UL; fused_0n__k_820 < 2UL; fused_0n__k_820 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_590_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_820 / 2UL) * 802816UL) + ((c * 200704UL) + (p_o * 3584UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_820 % 2UL) * 8192UL) + (c * 2048UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_7 = &__origouts_590_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_14, A_list, B_list, &__origouts_590_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter1996 = 0UL; _fuseiter1996 < 56UL; _fuseiter1996 += 1UL) {
        for (uint64_t _fuseiter1997 = 0UL; _fuseiter1997 < 32UL; _fuseiter1997 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_590_shr[((_fuseiter1996 * 32UL) + _fuseiter1997)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_820 / 2UL) * 64UL) + ((fused_0n__k_820 % 2UL) * 32UL)) + _fuseiter1997)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_820 % 2UL) * 32UL) + _fuseiter1997)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_820 / 2UL) * 215296UL) + ((((_fuseiter1997 + ((fused_0n__k_820 % 2UL) * 32UL)) / 64UL) * 215296UL) + (((p_o + 1UL) * 3712UL) + (((_fuseiter1996 + 1UL) * 64UL) + ((_fuseiter1997 + ((fused_0n__k_820 % 2UL) * 32UL)) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_590_shr);
    }
  }
  return true;
}

static bool res2c_conv_1_cast_mul_add_relu_cast_reorder__36(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_16 = *(void**)(__module_data + 72);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0k_o__n_821 = 0UL; fused_0k_o__n_821 < 2UL; fused_0k_o__n_821 += 1UL) {
    int32_t* __origouts_600_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    for (uint64_t o_o = 0UL; o_o < 112UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_600_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 28UL) / 56UL) + r) * 3712UL) + ((((o_o * 28UL) % 56UL) + s) * 64UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_821 * 18432UL) + ((r * 6144UL) + (s * 2048UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_8 = &__origouts_600_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))];
      dnnl_brgemm_list_call(__sc_kernel_cache_16, A_list, B_list, &__origouts_600_shr[((((o_o * 28UL) / 56UL) * 1792UL) + (((o_o * 28UL) % 56UL) * 32UL))], 1, 64, 2048, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2030 = 0UL; _fuseiter2030 < 56UL; _fuseiter2030 += 1UL) {
      for (uint64_t _fuseiter2031 = 0UL; _fuseiter2031 < 56UL; _fuseiter2031 += 1UL) {
        for (uint64_t _fuseiter2032 = 0UL; _fuseiter2032 < 32UL; _fuseiter2032 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_600_shr[((_fuseiter2030 * 1792UL) + ((_fuseiter2031 * 32UL) + _fuseiter2032))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_821 * 32UL) + _fuseiter2032)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_821 * 32UL) + _fuseiter2032)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2032 + (fused_0k_o__n_821 * 32UL)) / 64UL) * 200704UL) + ((_fuseiter2030 * 3584UL) + ((_fuseiter2031 * 64UL) + ((_fuseiter2032 + (fused_0k_o__n_821 * 32UL)) % 64UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_600_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res2c_conv_2_cast_mul_add_cast_add_cast_reorder__40(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_18 = *(void**)(__module_data + 80);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_822 = 0UL; fused_0n__k_822 < 4UL; fused_0n__k_822 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 56UL; p_o += 1UL) {
      int32_t* __origouts_610_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_822 / 4UL) * 200704UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_822 % 4UL) * 4096UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_9 = &__origouts_610_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_18, A_list, B_list, &__origouts_610_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2066 = 0UL; _fuseiter2066 < 56UL; _fuseiter2066 += 1UL) {
        for (uint64_t _fuseiter2067 = 0UL; _fuseiter2067 < 64UL; _fuseiter2067 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_610_shr[((_fuseiter2066 * 64UL) + _fuseiter2067)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_822 / 4UL) * 256UL) + ((fused_0n__k_822 % 4UL) * 64UL)) + _fuseiter2067)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_822 % 4UL) * 64UL) + _fuseiter2067)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_822 / 4UL) * 802816UL) + (((fused_0n__k_822 % 4UL) * 200704UL) + (p_o * 3584UL))) + ((_fuseiter2066 * 64UL) + _fuseiter2067))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_822 / 4UL) * 802816UL) + ((((_fuseiter2067 + ((fused_0n__k_822 % 4UL) * 64UL)) / 128UL) * 401408UL) + ((p_o * 7168UL) + ((_fuseiter2066 * 128UL) + ((_fuseiter2067 + ((fused_0n__k_822 % 4UL) * 64UL)) % 128UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_610_shr);
    }
  }
  return true;
}

static bool res3a_conv_0_cast_mul_add_relu_cast_reorder__48(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_20 = *(void**)(__module_data + 88);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 7424UL);
  for (uint64_t p1 = 0UL; p1 < 56UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 7424UL)], 0, 128UL);
    memset(&__outs_0[(((p1 + 1UL) * 7424UL) + 7296UL)], 0, 128UL);
  }
  memset(&__outs_0[423168UL], 0, 7424UL);
  for (uint64_t fused_0k__n_824 = 0UL; fused_0k__n_824 < 2UL; fused_0k__n_824 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_620_shr = (int32_t*)sc_aligned_malloc(__stream, 28672UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 401408UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_824 * 16384UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_10 = &__origouts_620_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_20, A_list, B_list, &__origouts_620_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter2106 = 0UL; _fuseiter2106 < 2UL; _fuseiter2106 += 1UL) {
        for (uint64_t _fuseiter2107 = 0UL; _fuseiter2107 < 56UL; _fuseiter2107 += 1UL) {
          for (uint64_t _fuseiter2108 = 0UL; _fuseiter2108 < 64UL; _fuseiter2108 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_620_shr[((_fuseiter2106 * 3584UL) + ((_fuseiter2107 * 64UL) + _fuseiter2108))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_824 * 64UL) + _fuseiter2108)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_824 * 64UL) + _fuseiter2108)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2108 + (fused_0k__n_824 * 64UL)) / 128UL) * 430592UL) + ((((_fuseiter2106 + (p_o * 2UL)) + 1UL) * 7424UL) + (((_fuseiter2107 + 1UL) * 128UL) + ((_fuseiter2108 + (fused_0k__n_824 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_620_shr);
    }
  }
  return true;
}

static bool res3a_conv_1_cast_mul_add_relu_cast__52(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_22 = *(void**)(__module_data + 96);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0n__k_o_825 = 0UL; fused_0n__k_o_825 < 4UL; fused_0n__k_o_825 += 1UL) {
    int32_t* __origouts_630_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_630_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((fused_0n__k_o_825 / 4UL) * 430592UL) + ((((((o_o * 28UL) / 28UL) * 2UL) + r) * 7424UL) + (s * 128UL)))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[(((fused_0n__k_o_825 % 4UL) * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_11 = &__origouts_630_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_22, A_list, B_list, &__origouts_630_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 128, 4096, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2141 = 0UL; _fuseiter2141 < 28UL; _fuseiter2141 += 1UL) {
      for (uint64_t _fuseiter2142 = 0UL; _fuseiter2142 < 28UL; _fuseiter2142 += 1UL) {
        for (uint64_t _fuseiter2143 = 0UL; _fuseiter2143 < 32UL; _fuseiter2143 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_630_shr[((_fuseiter2141 * 896UL) + ((_fuseiter2142 * 32UL) + _fuseiter2143))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_825 / 4UL) * 128UL) + (((fused_0n__k_o_825 % 4UL) * 32UL) + _fuseiter2143))]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_o_825 % 4UL) * 32UL) + _fuseiter2143)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((fused_0n__k_o_825 / 4UL) * 100352UL) + (((fused_0n__k_o_825 % 4UL) * 25088UL) + ((_fuseiter2141 * 896UL) + ((_fuseiter2142 * 32UL) + _fuseiter2143))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_630_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3a_conv_b_cast_mul_add_cast__44(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_24 = *(void**)(__module_data + 104);
  alignas(64) int8_t __rescheduled_0[128UL];
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 200704UL);
  for (uint64_t fused_0n__c_o_826 = 0UL; fused_0n__c_o_826 < 2UL; fused_0n__c_o_826 += 1UL) {
    for (uint64_t p = 0UL; p < 28UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 28UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 128UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_826 / 2UL) * 802816UL) + (((fused_0n__c_o_826 % 2UL) * 401408UL) + ((p * 14336UL) + ((q * 256UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_826 / 2UL) * 200704UL) + (((fused_0n__c_o_826 % 2UL) * 100352UL) + ((p * 3584UL) + ((q * 128UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_827 = 0UL; fused_0k__n_827 < 16UL; fused_0k__n_827 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 4UL; p_o += 1UL) {
      int32_t* __origouts_640_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[((c * 100352UL) + (p_o * 25088UL))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[((fused_0k__n_827 * 8192UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_12 = &__origouts_640_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_24, A_list, B_list, &__origouts_640_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter2171 = 0UL; _fuseiter2171 < 7UL; _fuseiter2171 += 1UL) {
        for (uint64_t _fuseiter2172 = 0UL; _fuseiter2172 < 28UL; _fuseiter2172 += 1UL) {
          for (uint64_t _fuseiter2173 = 0UL; _fuseiter2173 < 32UL; _fuseiter2173 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_640_shr[((_fuseiter2171 * 896UL) + ((_fuseiter2172 * 32UL) + _fuseiter2173))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_827 * 32UL) + _fuseiter2173)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_827 * 32UL) + _fuseiter2173)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16::store(__cached_8, &__outs_0[(((fused_0k__n_827 * 25088UL) + (p_o * 6272UL)) + ((_fuseiter2171 * 896UL) + ((_fuseiter2172 * 32UL) + _fuseiter2173)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_640_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res3a_conv_2_cast_mul_add_cast_add_cast_reorder__56(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_26 = *(void**)(__module_data + 112);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_828 = 0UL; fused_0k__n_828 < 16UL; fused_0k__n_828 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_650_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 12544UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_828 * 4096UL) + (c * 1024UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_13 = &__origouts_650_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_26, A_list, B_list, &__origouts_650_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter2195 = 0UL; _fuseiter2195 < 14UL; _fuseiter2195 += 1UL) {
        for (uint64_t _fuseiter2196 = 0UL; _fuseiter2196 < 28UL; _fuseiter2196 += 1UL) {
          for (uint64_t _fuseiter2197 = 0UL; _fuseiter2197 < 32UL; _fuseiter2197 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_650_shr[((_fuseiter2195 * 896UL) + ((_fuseiter2196 * 32UL) + _fuseiter2197))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_828 * 32UL) + _fuseiter2197)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_828 * 32UL) + _fuseiter2197)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0k__n_828 * 25088UL) + (p_o * 12544UL)) + ((_fuseiter2195 * 896UL) + ((_fuseiter2196 * 32UL) + _fuseiter2197)))]);
            __cached_6 = (__cached_6 + __cached_7);
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter2197 + (fused_0k__n_828 * 32UL)) / 128UL) * 100352UL) + (((_fuseiter2195 + (p_o * 14UL)) * 3584UL) + ((_fuseiter2196 * 128UL) + ((_fuseiter2197 + (fused_0k__n_828 * 32UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_650_shr);
    }
  }
  return true;
}

static bool res3b_conv_0_cast_mul_add_relu_cast_reorder__60(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 120);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_829 = 0UL; fused_0n__k_829 < 2UL; fused_0n__k_829 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_829 / 2UL) * 115200UL) + ((fused_0n__k_829 % 2UL) * 57600UL))], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_829 / 2UL) * 115200UL) + (((fused_0n__k_829 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_829 / 2UL) * 115200UL) + (((fused_0n__k_829 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL))) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_829 / 2UL) * 115200UL) + ((fused_0n__k_829 % 2UL) * 57600UL)) + 55680UL)], 0, 1920UL);
  }
  for (uint64_t fused_0k__n_830 = 0UL; fused_0k__n_830 < 4UL; fused_0k__n_830 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_660_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_830 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_14 = &__origouts_660_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_28, A_list, B_list, &__origouts_660_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter2236 = 0UL; _fuseiter2236 < 4UL; _fuseiter2236 += 1UL) {
        for (uint64_t _fuseiter2237 = 0UL; _fuseiter2237 < 28UL; _fuseiter2237 += 1UL) {
          for (uint64_t _fuseiter2238 = 0UL; _fuseiter2238 < 32UL; _fuseiter2238 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_660_shr[((_fuseiter2236 * 896UL) + ((_fuseiter2237 * 32UL) + _fuseiter2238))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_830 * 32UL) + _fuseiter2238)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_830 * 32UL) + _fuseiter2238)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2238 + (fused_0k__n_830 * 32UL)) / 64UL) * 57600UL) + ((((_fuseiter2236 + (p_o * 4UL)) + 1UL) * 1920UL) + (((_fuseiter2237 + 1UL) * 64UL) + ((_fuseiter2238 + (fused_0k__n_830 * 32UL)) % 64UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_660_shr);
    }
  }
  return true;
}

static bool res3b_conv_1_cast_mul_add_relu_cast_reorder__64(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 392;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  int32_t __cached_2;
  __cached_2 = 392;
  conv_os_blk_size[1] = __cached_2;
  int32_t __cached_3;
  __cached_3 = 392;
  conv_os_acc_size[1] = __cached_3;
  int32_t* __origouts_670_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
  for (uint64_t o_o = 0UL; o_o < 2UL; o_o += 1UL) {
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_4;
    __cached_4 = conv_os_acc_size[o_o];
    int32_t __cached_5;
    __cached_5 = conv_os_blk_size[o_o];
    memset(&__origouts_670_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))], 0, ((uint64_t)(__cached_5 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_6;
          __cached_6 = &__ins_0[((c_o * 57600UL) + (((((o_o * 419UL) / 30UL) + r) * 1920UL) + ((((o_o * 419UL) % 30UL) + s) * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_6;
          void* __cached_7;
          __cached_7 = &__ins_1[((c_o * 73728UL) + ((r * 24576UL) + (s * 8192UL)))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_7;
        }
      }
    }
    void* _arg_cache_15 = &__origouts_670_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr[o_o], A_list, B_list, &__origouts_670_shr[(uint64_t)(((__cached_4 / 28) * 3584) + ((__cached_4 % 28) * 128))], 1, 64, 8192, 18, 7, 7, __stream);
  }
  for (uint64_t _fuseiter2271 = 0UL; _fuseiter2271 < 28UL; _fuseiter2271 += 1UL) {
    for (uint64_t _fuseiter2272 = 0UL; _fuseiter2272 < 28UL; _fuseiter2272 += 1UL) {
      for (uint64_t _fuseiter2273 = 0UL; _fuseiter2273 < 128UL; _fuseiter2273 += 16UL) {
        vec_s32x16 __cached_8;
        __cached_8 = vec_s32x16::load(&__origouts_670_shr[((_fuseiter2271 * 3584UL) + ((_fuseiter2272 * 128UL) + _fuseiter2273))]);
        vec_f32x16 __cached_9;
        __cached_9 = (vec_f32x16)(__cached_8);
        vec_f32x16 __cached_10;
        __cached_10 = vec_f32x16::load(&__ins_2[_fuseiter2273]);
        __cached_9 = (__cached_9 * __cached_10);
        vec_f32x16 __cached_11;
        __cached_11 = vec_f32x16::load(&__ins_3[_fuseiter2273]);
        __cached_9 = (__cached_9 + __cached_11);
        __cached_9 = sc_max(__cached_9, vec_f32x16(0.f));
        vec_s8x16 __cached_12;
        __cached_12 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_9));
        vec_s8x16 __cached_13;
        __cached_13 = __cached_12;
        vec_s8x16::store(__cached_13, &__outs_0[(((_fuseiter2273 / 64UL) * 50176UL) + ((_fuseiter2271 * 1792UL) + ((_fuseiter2272 * 64UL) + (_fuseiter2273 % 64UL))))]);
      }
    }
  }
  sc_aligned_free(__stream, __origouts_670_shr);
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3b_conv_2_cast_mul_add_cast_add_cast_reorder__68(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_31 = *(void**)(__module_data + 128);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_832 = 0UL; fused_0n__k_832 < 4UL; fused_0n__k_832 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 28UL; p_o += 1UL) {
      int32_t* __origouts_680_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_832 / 4UL) * 100352UL) + ((c * 50176UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_832 % 4UL) * 16384UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_16 = &__origouts_680_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_31, A_list, B_list, &__origouts_680_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter2307 = 0UL; _fuseiter2307 < 28UL; _fuseiter2307 += 1UL) {
        for (uint64_t _fuseiter2308 = 0UL; _fuseiter2308 < 128UL; _fuseiter2308 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_680_shr[((_fuseiter2307 * 128UL) + _fuseiter2308)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_832 / 4UL) * 512UL) + ((fused_0n__k_832 % 4UL) * 128UL)) + _fuseiter2308)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_832 % 4UL) * 128UL) + _fuseiter2308)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_832 / 4UL) * 401408UL) + (((fused_0n__k_832 % 4UL) * 100352UL) + (p_o * 3584UL))) + ((_fuseiter2307 * 128UL) + _fuseiter2308))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_832 / 4UL) * 401408UL) + ((((_fuseiter2308 + ((fused_0n__k_832 % 4UL) * 128UL)) / 64UL) * 50176UL) + ((p_o * 1792UL) + ((_fuseiter2307 * 64UL) + ((_fuseiter2308 + ((fused_0n__k_832 % 4UL) * 128UL)) % 64UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_680_shr);
    }
  }
  return true;
}

static bool res3c_conv_0_cast_mul_add_relu_cast_reorder__72(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_33 = *(void**)(__module_data + 136);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_833 = 0UL; fused_0n__k_833 < 2UL; fused_0n__k_833 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_833 / 2UL) * 115200UL) + ((fused_0n__k_833 % 2UL) * 57600UL))], 0, 1920UL);
    for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_833 / 2UL) * 115200UL) + (((fused_0n__k_833 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_833 / 2UL) * 115200UL) + (((fused_0n__k_833 % 2UL) * 57600UL) + ((p1 + 1UL) * 1920UL))) + 1856UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_833 / 2UL) * 115200UL) + ((fused_0n__k_833 % 2UL) * 57600UL)) + 55680UL)], 0, 1920UL);
  }
  int32_t* __origouts_690_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
  void** A_list = (void**)&__rescheduled_0[0UL];
  void** B_list = (void**)&__rescheduled_0[64UL];
  for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
    void* __cached_0;
    __cached_0 = &__ins_0[(c * 50176UL)];
    A_list[c] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[(c * 8192UL)];
    B_list[c] = __cached_1;
  }
  void* _arg_cache_17 = &__origouts_690_shr[0UL];
  dnnl_brgemm_list_call(__sc_kernel_cache_33, A_list, B_list, &__origouts_690_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
  for (uint64_t _fuseiter2347 = 0UL; _fuseiter2347 < 28UL; _fuseiter2347 += 1UL) {
    for (uint64_t _fuseiter2348 = 0UL; _fuseiter2348 < 28UL; _fuseiter2348 += 1UL) {
      for (uint64_t _fuseiter2349 = 0UL; _fuseiter2349 < 128UL; _fuseiter2349 += 16UL) {
        vec_s32x16 __cached_2;
        __cached_2 = vec_s32x16::load(&__origouts_690_shr[((_fuseiter2347 * 3584UL) + ((_fuseiter2348 * 128UL) + _fuseiter2349))]);
        vec_f32x16 __cached_3;
        __cached_3 = (vec_f32x16)(__cached_2);
        vec_f32x16 __cached_4;
        __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2349]);
        __cached_3 = (__cached_3 * __cached_4);
        vec_f32x16 __cached_5;
        __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2349]);
        __cached_3 = (__cached_3 + __cached_5);
        __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
        vec_s8x16 __cached_6;
        __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
        vec_s8x16 __cached_7;
        __cached_7 = __cached_6;
        vec_s8x16::store(__cached_7, &__outs_0[(((_fuseiter2349 / 64UL) * 57600UL) + (((_fuseiter2347 + 1UL) * 1920UL) + (((_fuseiter2348 + 1UL) * 64UL) + (_fuseiter2349 % 64UL))))]);
      }
    }
  }
  sc_aligned_free(__stream, __origouts_690_shr);
  return true;
}

static bool res3c_conv_1_cast_mul_add_relu_cast_reorder__76(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_35 = *(void**)(__module_data + 144);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  for (uint64_t fused_0k_o__n_835 = 0UL; fused_0k_o__n_835 < 4UL; fused_0k_o__n_835 += 1UL) {
    int32_t* __origouts_700_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[192UL];
      memset(&__origouts_700_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
        for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
          for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
            void* __cached_0;
            __cached_0 = &__ins_0[((c_o * 57600UL) + (((((o_o * 28UL) / 28UL) + r) * 1920UL) + (s * 64UL)))];
            A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_0;
            void* __cached_1;
            __cached_1 = &__ins_1[((fused_0k_o__n_835 * 36864UL) + ((c_o * 18432UL) + ((r * 6144UL) + (s * 2048UL))))];
            B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_1;
          }
        }
      }
      void* _arg_cache_18 = &__origouts_700_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_35, A_list, B_list, &__origouts_700_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 64, 2048, 18, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2382 = 0UL; _fuseiter2382 < 28UL; _fuseiter2382 += 1UL) {
      for (uint64_t _fuseiter2383 = 0UL; _fuseiter2383 < 28UL; _fuseiter2383 += 1UL) {
        for (uint64_t _fuseiter2384 = 0UL; _fuseiter2384 < 32UL; _fuseiter2384 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_700_shr[((_fuseiter2382 * 896UL) + ((_fuseiter2383 * 32UL) + _fuseiter2384))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_835 * 32UL) + _fuseiter2384)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_835 * 32UL) + _fuseiter2384)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2384 + (fused_0k_o__n_835 * 32UL)) / 128UL) * 100352UL) + ((_fuseiter2382 * 3584UL) + ((_fuseiter2383 * 128UL) + ((_fuseiter2384 + (fused_0k_o__n_835 * 32UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_700_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3c_conv_2_cast_mul_add_cast_add_cast_reorder__80(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_37 = *(void**)(__module_data + 152);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_836 = 0UL; fused_0k__n_836 < 8UL; fused_0k__n_836 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_710_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 7168UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_836 * 8192UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_19 = &__origouts_710_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_37, A_list, B_list, &__origouts_710_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2417 = 0UL; _fuseiter2417 < 2UL; _fuseiter2417 += 1UL) {
        for (uint64_t _fuseiter2418 = 0UL; _fuseiter2418 < 28UL; _fuseiter2418 += 1UL) {
          for (uint64_t _fuseiter2419 = 0UL; _fuseiter2419 < 64UL; _fuseiter2419 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_710_shr[((_fuseiter2417 * 1792UL) + ((_fuseiter2418 * 64UL) + _fuseiter2419))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_836 * 64UL) + _fuseiter2419)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_836 * 64UL) + _fuseiter2419)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_u8x16 __cached_7;
            __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_836 * 50176UL) + (p_o * 3584UL)) + ((_fuseiter2417 * 1792UL) + ((_fuseiter2418 * 64UL) + _fuseiter2419)))]);
            __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter2419 + (fused_0k__n_836 * 64UL)) / 128UL) * 100352UL) + (((_fuseiter2417 + (p_o * 2UL)) * 3584UL) + ((_fuseiter2418 * 128UL) + ((_fuseiter2419 + (fused_0k__n_836 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_710_shr);
    }
  }
  return true;
}

static bool res3d_conv_0_cast_mul_add_relu_cast_reorder__84(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_39 = *(void**)(__module_data + 160);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 3840UL);
  for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 3840UL)], 0, 128UL);
    memset(&__outs_0[(((p1 + 1UL) * 3840UL) + 3712UL)], 0, 128UL);
  }
  memset(&__outs_0[111360UL], 0, 3840UL);
  for (uint64_t fused_0k__n_838 = 0UL; fused_0k__n_838 < 2UL; fused_0k__n_838 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_720_shr = (int32_t*)sc_aligned_malloc(__stream, 14336UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 7168UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_838 * 32768UL) + (c * 8192UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_20 = &__origouts_720_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_39, A_list, B_list, &__origouts_720_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter2458 = 0UL; _fuseiter2458 < 2UL; _fuseiter2458 += 1UL) {
        for (uint64_t _fuseiter2459 = 0UL; _fuseiter2459 < 28UL; _fuseiter2459 += 1UL) {
          for (uint64_t _fuseiter2460 = 0UL; _fuseiter2460 < 64UL; _fuseiter2460 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_720_shr[((_fuseiter2458 * 1792UL) + ((_fuseiter2459 * 64UL) + _fuseiter2460))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_838 * 64UL) + _fuseiter2460)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_838 * 64UL) + _fuseiter2460)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2460 + (fused_0k__n_838 * 64UL)) / 128UL) * 115200UL) + ((((_fuseiter2458 + (p_o * 2UL)) + 1UL) * 3840UL) + (((_fuseiter2459 + 1UL) * 128UL) + ((_fuseiter2460 + (fused_0k__n_838 * 64UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_720_shr);
    }
  }
  return true;
}

static bool res3d_conv_1_cast_mul_add_relu_cast__88(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_40 = *(void**)(__module_data + 168);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
  for (uint64_t fused_0k_o__n_839 = 0UL; fused_0k_o__n_839 < 4UL; fused_0k_o__n_839 += 1UL) {
    int32_t* __origouts_730_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    for (uint64_t o_o = 0UL; o_o < 28UL; o_o += 1UL) {
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[128UL];
      memset(&__origouts_730_shr[(((o_o * 28UL) / 28UL) * 896UL)], 0, 3584UL);
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_0;
          __cached_0 = &__ins_0[(((((o_o * 28UL) / 28UL) + r) * 3840UL) + (s * 128UL))];
          A_list[((r * 3UL) + s)] = __cached_0;
          void* __cached_1;
          __cached_1 = &__ins_1[((fused_0k_o__n_839 * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
          B_list[((r * 3UL) + s)] = __cached_1;
        }
      }
      void* _arg_cache_21 = &__origouts_730_shr[(((o_o * 28UL) / 28UL) * 896UL)];
      dnnl_brgemm_list_call(__sc_kernel_cache_40, A_list, B_list, &__origouts_730_shr[(((o_o * 28UL) / 28UL) * 896UL)], 1, 128, 4096, 9, 7, 7, __stream);
    }
    for (uint64_t _fuseiter2493 = 0UL; _fuseiter2493 < 28UL; _fuseiter2493 += 1UL) {
      for (uint64_t _fuseiter2494 = 0UL; _fuseiter2494 < 28UL; _fuseiter2494 += 1UL) {
        for (uint64_t _fuseiter2495 = 0UL; _fuseiter2495 < 32UL; _fuseiter2495 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_730_shr[((_fuseiter2493 * 896UL) + ((_fuseiter2494 * 32UL) + _fuseiter2495))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_839 * 32UL) + _fuseiter2495)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_839 * 32UL) + _fuseiter2495)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[((fused_0k_o__n_839 * 25088UL) + ((_fuseiter2493 * 896UL) + ((_fuseiter2494 * 32UL) + _fuseiter2495)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_730_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res3d_conv_2_cast_mul_add_cast_add_cast__92(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_42 = *(void**)(__module_data + 176);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_840 = 0UL; fused_0k__n_840 < 4UL; fused_0k__n_840 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 4UL; p_o += 1UL) {
      int32_t* __origouts_740_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 6272UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_840 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_22 = &__origouts_740_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_42, A_list, B_list, &__origouts_740_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
      for (uint64_t _fuseiter2523 = 0UL; _fuseiter2523 < 7UL; _fuseiter2523 += 1UL) {
        for (uint64_t _fuseiter2524 = 0UL; _fuseiter2524 < 28UL; _fuseiter2524 += 1UL) {
          for (uint64_t _fuseiter2525 = 0UL; _fuseiter2525 < 128UL; _fuseiter2525 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_740_shr[((_fuseiter2523 * 3584UL) + ((_fuseiter2524 * 128UL) + _fuseiter2525))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_840 * 128UL) + _fuseiter2525)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_840 * 128UL) + _fuseiter2525)]);
            __cached_3 = (__cached_3 + __cached_5);
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_u8x16 __cached_7;
            __cached_7 = vec_u8x16::load(&__ins_4[(((fused_0k__n_840 * 100352UL) + (p_o * 25088UL)) + ((_fuseiter2523 * 3584UL) + ((_fuseiter2524 * 128UL) + _fuseiter2525)))]);
            __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
            vec_u8x16 __cached_8;
            __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
            vec_u8x16::store(__cached_8, &__outs_0[(((fused_0k__n_840 * 100352UL) + (p_o * 25088UL)) + ((_fuseiter2523 * 3584UL) + ((_fuseiter2524 * 128UL) + _fuseiter2525)))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_740_shr);
    }
  }
  return true;
}

static bool res4a_conv_0_cast_mul_add_relu_cast__100(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_44 = *(void**)(__module_data + 184);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 7680UL);
  for (uint64_t p1 = 0UL; p1 < 28UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 7680UL)], 0, 256UL);
    memset(&__outs_0[(((p1 + 1UL) * 7680UL) + 7424UL)], 0, 256UL);
  }
  memset(&__outs_0[222720UL], 0, 7680UL);
  for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
    int32_t* __origouts_750_shr = (int32_t*)sc_aligned_malloc(__stream, 57344UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 7168UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 32768UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_23 = &__origouts_750_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_44, A_list, B_list, &__origouts_750_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter2559 = 0UL; _fuseiter2559 < 2UL; _fuseiter2559 += 1UL) {
      for (uint64_t _fuseiter2560 = 0UL; _fuseiter2560 < 28UL; _fuseiter2560 += 1UL) {
        for (uint64_t _fuseiter2561 = 0UL; _fuseiter2561 < 256UL; _fuseiter2561 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_750_shr[((_fuseiter2559 * 7168UL) + ((_fuseiter2560 * 256UL) + _fuseiter2561))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2561]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2561]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((((p_o * 2UL) + 1UL) * 7680UL) + 256UL) + ((_fuseiter2559 * 7680UL) + ((_fuseiter2560 * 256UL) + _fuseiter2561)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_750_shr);
  }
  return true;
}

static bool res4a_conv_1_cast_mul_add_relu_cast_reorder__104(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_48 = (void**)&__uninitialized_data[23657512UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_843 = 0UL; fused_0k_o__n_843 < 2UL; fused_0k_o__n_843 += 1UL) {
    int32_t* __origouts_760_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[256UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_760_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[((r * 7680UL) + (s * 256UL))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[((fused_0k_o__n_843 * 294912UL) + ((r * 98304UL) + (s * 32768UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_24 = &__origouts_760_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_48[0UL], A_list, B_list, &__origouts_760_shr[(uint64_t)(((__cached_2 / 14) * 1792) + ((__cached_2 % 14) * 128))], 1, 256, 32768, 9, 7, 7, __stream);
    for (uint64_t _fuseiter2589 = 0UL; _fuseiter2589 < 14UL; _fuseiter2589 += 1UL) {
      for (uint64_t _fuseiter2590 = 0UL; _fuseiter2590 < 14UL; _fuseiter2590 += 1UL) {
        for (uint64_t _fuseiter2591 = 0UL; _fuseiter2591 < 128UL; _fuseiter2591 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_760_shr[((_fuseiter2589 * 1792UL) + ((_fuseiter2590 * 128UL) + _fuseiter2591))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_843 * 128UL) + _fuseiter2591)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_843 * 128UL) + _fuseiter2591)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter2591 + (fused_0k_o__n_843 * 128UL)) / 256UL) * 50176UL) + ((_fuseiter2589 * 3584UL) + ((_fuseiter2590 * 256UL) + ((_fuseiter2591 + (fused_0k_o__n_843 * 128UL)) % 256UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_760_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4a_conv_b_cast_mul_add_cast_reorder__96(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_50 = *(void**)(__module_data + 192);
  alignas(64) int8_t __rescheduled_0[128UL];
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 100352UL);
  for (uint64_t fused_0n__c_o_844 = 0UL; fused_0n__c_o_844 < 4UL; fused_0n__c_o_844 += 1UL) {
    for (uint64_t p = 0UL; p < 14UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 14UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 128UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_844 / 4UL) * 401408UL) + (((fused_0n__c_o_844 % 4UL) * 100352UL) + ((p * 7168UL) + ((q * 256UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_844 / 4UL) * 100352UL) + (((fused_0n__c_o_844 % 4UL) * 25088UL) + ((p * 1792UL) + ((q * 128UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_845 = 0UL; fused_0k__n_845 < 32UL; fused_0k__n_845 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
      int32_t* __origouts_770_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
        void* __cached_2;
        __cached_2 = &input_tmp[((c * 25088UL) + (p_o * 12544UL))];
        A_list[c] = __cached_2;
        void* __cached_3;
        __cached_3 = &__ins_1[((fused_0k__n_845 * 16384UL) + (c * 4096UL))];
        B_list[c] = __cached_3;
      }
      void* _arg_cache_25 = &__origouts_770_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_50, A_list, B_list, &__origouts_770_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
      for (uint64_t _fuseiter2624 = 0UL; _fuseiter2624 < 7UL; _fuseiter2624 += 1UL) {
        for (uint64_t _fuseiter2625 = 0UL; _fuseiter2625 < 14UL; _fuseiter2625 += 1UL) {
          for (uint64_t _fuseiter2626 = 0UL; _fuseiter2626 < 32UL; _fuseiter2626 += 16UL) {
            vec_s32x16 __cached_4;
            __cached_4 = vec_s32x16::load(&__origouts_770_shr[((_fuseiter2624 * 448UL) + ((_fuseiter2625 * 32UL) + _fuseiter2626))]);
            vec_f32x16 __cached_5;
            __cached_5 = (vec_f32x16)(__cached_4);
            vec_f32x16 __cached_6;
            __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_845 * 32UL) + _fuseiter2626)]);
            __cached_5 = (__cached_5 * __cached_6);
            vec_f32x16 __cached_7;
            __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_845 * 32UL) + _fuseiter2626)]);
            __cached_5 = (__cached_5 + __cached_7);
            vec_s8x16 __cached_8;
            __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
            vec_s8x16 __cached_9;
            __cached_9 = __cached_8;
            vec_s8x16::store(__cached_9, &__outs_0[((((_fuseiter2626 + (fused_0k__n_845 * 32UL)) / 128UL) * 25088UL) + (((_fuseiter2624 + (p_o * 7UL)) * 1792UL) + ((_fuseiter2625 * 128UL) + ((_fuseiter2626 + (fused_0k__n_845 * 32UL)) % 128UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_770_shr);
    }
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool res4a_conv_2_cast_mul_add_cast_add_cast_reorder__108(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_52 = *(void**)(__module_data + 200);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_846 = 0UL; fused_0k__n_846 < 8UL; fused_0k__n_846 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_780_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(p_o * 3584UL)];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(fused_0k__n_846 * 32768UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_26 = &__origouts_780_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_52, A_list, B_list, &__origouts_780_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter2654 = 0UL; _fuseiter2654 < 14UL; _fuseiter2654 += 1UL) {
        for (uint64_t _fuseiter2655 = 0UL; _fuseiter2655 < 128UL; _fuseiter2655 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_780_shr[((_fuseiter2654 * 128UL) + _fuseiter2655)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_846 * 128UL) + _fuseiter2655)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_846 * 128UL) + _fuseiter2655)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[(((fused_0k__n_846 * 25088UL) + (p_o * 1792UL)) + ((_fuseiter2654 * 128UL) + _fuseiter2655))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[((((_fuseiter2655 + (fused_0k__n_846 * 128UL)) / 1024UL) * 200704UL) + ((p_o * 14336UL) + ((_fuseiter2654 * 1024UL) + ((_fuseiter2655 + (fused_0k__n_846 * 128UL)) % 1024UL))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_780_shr);
    }
  }
  return true;
}

static bool res4b_conv_0_cast_mul_add_relu_cast_reorder__112(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_54 = *(void**)(__module_data + 208);
  alignas(64) int8_t __rescheduled_0[128UL];
  memset(&__outs_0[0UL], 0, 4096UL);
  for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
    memset(&__outs_0[((p1 + 1UL) * 4096UL)], 0, 256UL);
    memset(&__outs_0[(((p1 + 1UL) * 4096UL) + 3840UL)], 0, 256UL);
  }
  memset(&__outs_0[61440UL], 0, 4096UL);
  for (uint64_t fused_0n__k_848 = 0UL; fused_0n__k_848 < 4UL; fused_0n__k_848 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_790_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_848 / 4UL) * 200704UL) + (p_o * 28672UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_848 % 4UL) * 65536UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_27 = &__origouts_790_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_54, A_list, B_list, &__origouts_790_shr[0UL], 1, 1, 1, 1, 8, 7, __stream);
      for (uint64_t _fuseiter2694 = 0UL; _fuseiter2694 < 2UL; _fuseiter2694 += 1UL) {
        for (uint64_t _fuseiter2695 = 0UL; _fuseiter2695 < 14UL; _fuseiter2695 += 1UL) {
          for (uint64_t _fuseiter2696 = 0UL; _fuseiter2696 < 64UL; _fuseiter2696 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_790_shr[((_fuseiter2694 * 896UL) + ((_fuseiter2695 * 64UL) + _fuseiter2696))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_848 / 4UL) * 256UL) + ((fused_0n__k_848 % 4UL) * 64UL)) + _fuseiter2696)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_848 % 4UL) * 64UL) + _fuseiter2696)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_848 / 4UL) * 65536UL) + ((((_fuseiter2696 + ((fused_0n__k_848 % 4UL) * 64UL)) / 256UL) * 65536UL) + ((((_fuseiter2694 + (p_o * 2UL)) + 1UL) * 4096UL) + (((_fuseiter2695 + 1UL) * 256UL) + ((_fuseiter2696 + ((fused_0n__k_848 % 4UL) * 64UL)) % 256UL)))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_790_shr);
    }
  }
  return true;
}

static bool res4b_conv_1_cast_mul_add_relu_cast_reorder__116(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_58 = (void**)&__uninitialized_data[23657528UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0n__k_o_849 = 0UL; fused_0n__k_o_849 < 16UL; fused_0n__k_o_849 += 1UL) {
    int32_t* __origouts_800_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[256UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_800_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))], 0, ((uint64_t)(__cached_3 * 16) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[(((fused_0n__k_o_849 / 16UL) * 65536UL) + ((r * 4096UL) + (s * 256UL)))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[(((fused_0n__k_o_849 % 16UL) * 36864UL) + ((r * 12288UL) + (s * 4096UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_28 = &__origouts_800_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_58[0UL], A_list, B_list, &__origouts_800_shr[(uint64_t)(((__cached_2 / 14) * 224) + ((__cached_2 % 14) * 16))], 1, 256, 4096, 9, 7, 7, __stream);
    for (uint64_t _fuseiter2729 = 0UL; _fuseiter2729 < 14UL; _fuseiter2729 += 1UL) {
      for (uint64_t _fuseiter2730 = 0UL; _fuseiter2730 < 14UL; _fuseiter2730 += 1UL) {
        vec_s32x16 __cached_6;
        __cached_6 = vec_s32x16::load(&__origouts_800_shr[((_fuseiter2729 * 224UL) + (_fuseiter2730 * 16UL))]);
        vec_f32x16 __cached_7;
        __cached_7 = (vec_f32x16)(__cached_6);
        vec_f32x16 __cached_8;
        __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0n__k_o_849 / 16UL) * 256UL) + ((fused_0n__k_o_849 % 16UL) * 16UL))]);
        __cached_7 = (__cached_7 * __cached_8);
        vec_f32x16 __cached_9;
        __cached_9 = vec_f32x16::load(&__ins_3[((fused_0n__k_o_849 % 16UL) * 16UL)]);
        __cached_7 = (__cached_7 + __cached_9);
        __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
        vec_s8x16 __cached_10;
        __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
        vec_s8x16 __cached_11;
        __cached_11 = __cached_10;
        vec_s8x16::store(__cached_11, &__outs_0[(((fused_0n__k_o_849 / 16UL) * 50176UL) + (((((fused_0n__k_o_849 % 16UL) * 16UL) / 64UL) * 12544UL) + ((_fuseiter2729 * 896UL) + ((_fuseiter2730 * 64UL) + (((fused_0n__k_o_849 % 16UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_800_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4b_conv_2_cast_mul_add_cast_add_cast_reorder__120(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_60 = *(void**)(__module_data + 216);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
    int32_t* __origouts_810_shr = (int32_t*)sc_aligned_malloc(__stream, 401408UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 12544UL) + (p_o * 6272UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 65536UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_29 = &__origouts_810_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_60, A_list, B_list, &__origouts_810_shr[0UL], 1, 1, 1, 4, 7, 7, __stream);
    for (uint64_t _fuseiter2764 = 0UL; _fuseiter2764 < 7UL; _fuseiter2764 += 1UL) {
      for (uint64_t _fuseiter2765 = 0UL; _fuseiter2765 < 14UL; _fuseiter2765 += 1UL) {
        for (uint64_t _fuseiter2766 = 0UL; _fuseiter2766 < 1024UL; _fuseiter2766 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_810_shr[((_fuseiter2764 * 14336UL) + ((_fuseiter2765 * 1024UL) + _fuseiter2766))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter2766]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter2766]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((p_o * 100352UL) + ((_fuseiter2764 * 14336UL) + ((_fuseiter2765 * 1024UL) + _fuseiter2766)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((_fuseiter2766 / 128UL) * 25088UL) + (((_fuseiter2764 + (p_o * 7UL)) * 1792UL) + ((_fuseiter2765 * 128UL) + (_fuseiter2766 % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_810_shr);
  }
  return true;
}

static bool res4c_conv_0_cast_mul_add_relu_cast_reorder__124(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_62 = *(void**)(__module_data + 224);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_851 = 0UL; fused_0n__k_851 < 2UL; fused_0n__k_851 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_851 / 2UL) * 65536UL) + ((fused_0n__k_851 % 2UL) * 32768UL))], 0, 2048UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_851 / 2UL) * 65536UL) + (((fused_0n__k_851 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL)))], 0, 128UL);
      memset(&__outs_0[((((fused_0n__k_851 / 2UL) * 65536UL) + (((fused_0n__k_851 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL))) + 1920UL)], 0, 128UL);
    }
    memset(&__outs_0[((((fused_0n__k_851 / 2UL) * 65536UL) + ((fused_0n__k_851 % 2UL) * 32768UL)) + 30720UL)], 0, 2048UL);
  }
  for (uint64_t fused_0k__n_852 = 0UL; fused_0k__n_852 < 8UL; fused_0k__n_852 += 1UL) {
    int32_t* __origouts_820_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_852 * 32768UL) + (c * 4096UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_30 = &__origouts_820_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_62, A_list, B_list, &__origouts_820_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter2805 = 0UL; _fuseiter2805 < 14UL; _fuseiter2805 += 1UL) {
      for (uint64_t _fuseiter2806 = 0UL; _fuseiter2806 < 14UL; _fuseiter2806 += 1UL) {
        for (uint64_t _fuseiter2807 = 0UL; _fuseiter2807 < 32UL; _fuseiter2807 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_820_shr[((_fuseiter2805 * 448UL) + ((_fuseiter2806 * 32UL) + _fuseiter2807))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_852 * 32UL) + _fuseiter2807)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_852 * 32UL) + _fuseiter2807)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter2807 + (fused_0k__n_852 * 32UL)) / 128UL) * 32768UL) + (((_fuseiter2805 + 1UL) * 2048UL) + (((_fuseiter2806 + 1UL) * 128UL) + ((_fuseiter2807 + (fused_0k__n_852 * 32UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_820_shr);
  }
  return true;
}

static bool res4c_conv_1_cast_mul_add_relu_cast_reorder__128(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_64 = (void**)&__uninitialized_data[23657536UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_853 = 0UL; fused_0k_o__n_853 < 8UL; fused_0k_o__n_853 += 1UL) {
    int32_t* __origouts_830_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_830_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 0, ((uint64_t)(__cached_3 * 32) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 32768UL) + ((r * 2048UL) + (s * 128UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_853 * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_31 = &__origouts_830_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_64[0UL], A_list, B_list, &__origouts_830_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 1, 128, 4096, 18, 7, 7, __stream);
    for (uint64_t _fuseiter2840 = 0UL; _fuseiter2840 < 14UL; _fuseiter2840 += 1UL) {
      for (uint64_t _fuseiter2841 = 0UL; _fuseiter2841 < 14UL; _fuseiter2841 += 1UL) {
        for (uint64_t _fuseiter2842 = 0UL; _fuseiter2842 < 32UL; _fuseiter2842 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_830_shr[((_fuseiter2840 * 448UL) + ((_fuseiter2841 * 32UL) + _fuseiter2842))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_853 * 32UL) + _fuseiter2842)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_853 * 32UL) + _fuseiter2842)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter2842 + (fused_0k_o__n_853 * 32UL)) / 256UL) * 50176UL) + ((_fuseiter2840 * 3584UL) + ((_fuseiter2841 * 256UL) + ((_fuseiter2842 + (fused_0k_o__n_853 * 32UL)) % 256UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_830_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4c_conv_2_cast_mul_add_cast_add_cast__132(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_66 = *(void**)(__module_data + 232);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_854 = 0UL; fused_0n__k_854 < 8UL; fused_0n__k_854 += 1UL) {
    int32_t* __origouts_840_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0n__k_854 / 8UL) * 50176UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0n__k_854 % 8UL) * 32768UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_32 = &__origouts_840_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_66, A_list, B_list, &__origouts_840_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter2875 = 0UL; _fuseiter2875 < 14UL; _fuseiter2875 += 1UL) {
      for (uint64_t _fuseiter2876 = 0UL; _fuseiter2876 < 14UL; _fuseiter2876 += 1UL) {
        for (uint64_t _fuseiter2877 = 0UL; _fuseiter2877 < 128UL; _fuseiter2877 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_840_shr[((_fuseiter2875 * 1792UL) + ((_fuseiter2876 * 128UL) + _fuseiter2877))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_854 / 8UL) * 1024UL) + ((fused_0n__k_854 % 8UL) * 128UL)) + _fuseiter2877)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_854 % 8UL) * 128UL) + _fuseiter2877)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_854 / 8UL) * 200704UL) + ((fused_0n__k_854 % 8UL) * 25088UL)) + ((_fuseiter2875 * 1792UL) + ((_fuseiter2876 * 128UL) + _fuseiter2877)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0n__k_854 / 8UL) * 200704UL) + ((fused_0n__k_854 % 8UL) * 25088UL)) + ((_fuseiter2875 * 1792UL) + ((_fuseiter2876 * 128UL) + _fuseiter2877)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_840_shr);
  }
  return true;
}

static bool res4d_conv_0_cast_mul_add_relu_cast__136(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_68 = *(void**)(__module_data + 240);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_855 = 0UL; fused_0n__k_855 < 4UL; fused_0n__k_855 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_855 / 4UL) * 65536UL) + ((fused_0n__k_855 % 4UL) * 16384UL))], 0, 1024UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_855 / 4UL) * 65536UL) + (((fused_0n__k_855 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_855 / 4UL) * 65536UL) + (((fused_0n__k_855 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_855 / 4UL) * 65536UL) + ((fused_0n__k_855 % 4UL) * 16384UL)) + 15360UL)], 0, 1024UL);
  }
  for (uint64_t fused_0k__n_856 = 0UL; fused_0k__n_856 < 4UL; fused_0k__n_856 += 1UL) {
    int32_t* __origouts_850_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_856 * 65536UL) + (c * 8192UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_33 = &__origouts_850_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_68, A_list, B_list, &__origouts_850_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter2911 = 0UL; _fuseiter2911 < 14UL; _fuseiter2911 += 1UL) {
      for (uint64_t _fuseiter2912 = 0UL; _fuseiter2912 < 14UL; _fuseiter2912 += 1UL) {
        for (uint64_t _fuseiter2913 = 0UL; _fuseiter2913 < 64UL; _fuseiter2913 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_850_shr[((_fuseiter2911 * 896UL) + ((_fuseiter2912 * 64UL) + _fuseiter2913))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_856 * 64UL) + _fuseiter2913)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_856 * 64UL) + _fuseiter2913)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16::store(__cached_6, &__outs_0[(((fused_0k__n_856 * 16384UL) + 1088UL) + ((_fuseiter2911 * 1024UL) + ((_fuseiter2912 * 64UL) + _fuseiter2913)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_850_shr);
  }
  return true;
}

static bool res4d_conv_1_cast_mul_add_relu_cast_reorder__140(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_70 = (void**)&__uninitialized_data[23657544UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 768UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_857 = 0UL; fused_0k_o__n_857 < 4UL; fused_0k_o__n_857 += 1UL) {
    int32_t* __origouts_860_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[448UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_860_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 0, ((uint64_t)(__cached_3 * 64) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_857 * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_34 = &__origouts_860_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_70[0UL], A_list, B_list, &__origouts_860_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter2941 = 0UL; _fuseiter2941 < 14UL; _fuseiter2941 += 1UL) {
      for (uint64_t _fuseiter2942 = 0UL; _fuseiter2942 < 14UL; _fuseiter2942 += 1UL) {
        for (uint64_t _fuseiter2943 = 0UL; _fuseiter2943 < 64UL; _fuseiter2943 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_860_shr[((_fuseiter2941 * 896UL) + ((_fuseiter2942 * 64UL) + _fuseiter2943))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_857 * 64UL) + _fuseiter2943)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_857 * 64UL) + _fuseiter2943)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter2943 + (fused_0k_o__n_857 * 64UL)) / 128UL) * 25088UL) + ((_fuseiter2941 * 1792UL) + ((_fuseiter2942 * 128UL) + ((_fuseiter2943 + (fused_0k_o__n_857 * 64UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_860_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4d_conv_2_cast_mul_add_cast_add_cast__144(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_72 = *(void**)(__module_data + 248);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_858 = 0UL; fused_0k__n_858 < 8UL; fused_0k__n_858 += 1UL) {
    int32_t* __origouts_870_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_858 * 32768UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_35 = &__origouts_870_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_72, A_list, B_list, &__origouts_870_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter2976 = 0UL; _fuseiter2976 < 14UL; _fuseiter2976 += 1UL) {
      for (uint64_t _fuseiter2977 = 0UL; _fuseiter2977 < 14UL; _fuseiter2977 += 1UL) {
        for (uint64_t _fuseiter2978 = 0UL; _fuseiter2978 < 128UL; _fuseiter2978 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_870_shr[((_fuseiter2976 * 1792UL) + ((_fuseiter2977 * 128UL) + _fuseiter2978))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_858 * 128UL) + _fuseiter2978)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_858 * 128UL) + _fuseiter2978)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((fused_0k__n_858 * 25088UL) + ((_fuseiter2976 * 1792UL) + ((_fuseiter2977 * 128UL) + _fuseiter2978)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((fused_0k__n_858 * 25088UL) + ((_fuseiter2976 * 1792UL) + ((_fuseiter2977 * 128UL) + _fuseiter2978)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_870_shr);
  }
  return true;
}

static bool res4e_conv_0_cast_mul_add_relu_cast_reorder__148(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_74 = *(void**)(__module_data + 256);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_859 = 0UL; fused_0n__k_859 < 2UL; fused_0n__k_859 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_859 / 2UL) * 65536UL) + ((fused_0n__k_859 % 2UL) * 32768UL))], 0, 2048UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_859 / 2UL) * 65536UL) + (((fused_0n__k_859 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL)))], 0, 128UL);
      memset(&__outs_0[((((fused_0n__k_859 / 2UL) * 65536UL) + (((fused_0n__k_859 % 2UL) * 32768UL) + ((p1 + 1UL) * 2048UL))) + 1920UL)], 0, 128UL);
    }
    memset(&__outs_0[((((fused_0n__k_859 / 2UL) * 65536UL) + ((fused_0n__k_859 % 2UL) * 32768UL)) + 30720UL)], 0, 2048UL);
  }
  for (uint64_t p_o = 0UL; p_o < 2UL; p_o += 1UL) {
    int32_t* __origouts_880_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 8UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[((c * 25088UL) + (p_o * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(c * 32768UL)];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_36 = &__origouts_880_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_74, A_list, B_list, &__origouts_880_shr[0UL], 1, 1, 1, 8, 8, 7, __stream);
    for (uint64_t _fuseiter3012 = 0UL; _fuseiter3012 < 7UL; _fuseiter3012 += 1UL) {
      for (uint64_t _fuseiter3013 = 0UL; _fuseiter3013 < 14UL; _fuseiter3013 += 1UL) {
        for (uint64_t _fuseiter3014 = 0UL; _fuseiter3014 < 256UL; _fuseiter3014 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_880_shr[((_fuseiter3012 * 3584UL) + ((_fuseiter3013 * 256UL) + _fuseiter3014))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[_fuseiter3014]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[_fuseiter3014]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((_fuseiter3014 / 128UL) * 32768UL) + ((((_fuseiter3012 + (p_o * 7UL)) + 1UL) * 2048UL) + (((_fuseiter3013 + 1UL) * 128UL) + (_fuseiter3014 % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_880_shr);
  }
  return true;
}

static bool res4e_conv_1_cast_mul_add_relu_cast_reorder__152(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_64 = (void**)&__uninitialized_data[23657536UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0n__k_o_861 = 0UL; fused_0n__k_o_861 < 8UL; fused_0n__k_o_861 += 1UL) {
    int32_t* __origouts_890_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[320UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_890_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 0, ((uint64_t)(__cached_3 * 32) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0n__k_o_861 / 8UL) * 65536UL) + ((c_o * 32768UL) + ((r * 2048UL) + (s * 128UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0n__k_o_861 % 8UL) * 73728UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_37 = &__origouts_890_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_64[0UL], A_list, B_list, &__origouts_890_shr[(uint64_t)(((__cached_2 / 14) * 448) + ((__cached_2 % 14) * 32))], 1, 128, 4096, 18, 7, 7, __stream);
    for (uint64_t _fuseiter3047 = 0UL; _fuseiter3047 < 14UL; _fuseiter3047 += 1UL) {
      for (uint64_t _fuseiter3048 = 0UL; _fuseiter3048 < 14UL; _fuseiter3048 += 1UL) {
        for (uint64_t _fuseiter3049 = 0UL; _fuseiter3049 < 32UL; _fuseiter3049 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_890_shr[((_fuseiter3047 * 448UL) + ((_fuseiter3048 * 32UL) + _fuseiter3049))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((((fused_0n__k_o_861 / 8UL) * 256UL) + ((fused_0n__k_o_861 % 8UL) * 32UL)) + _fuseiter3049)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0n__k_o_861 % 8UL) * 32UL) + _fuseiter3049)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0n__k_o_861 / 8UL) * 50176UL) + ((((_fuseiter3049 + ((fused_0n__k_o_861 % 8UL) * 32UL)) / 256UL) * 50176UL) + ((_fuseiter3047 * 3584UL) + ((_fuseiter3048 * 256UL) + ((_fuseiter3049 + ((fused_0n__k_o_861 % 8UL) * 32UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_890_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4e_conv_2_cast_mul_add_cast_add_cast_reorder__156(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __outs_1, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_52 = *(void**)(__module_data + 200);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_862 = 0UL; fused_0n__k_862 < 8UL; fused_0n__k_862 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 14UL; p_o += 1UL) {
      int32_t* __origouts_900_shr = (int32_t*)sc_aligned_malloc(__stream, 7168UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_862 / 8UL) * 50176UL) + (p_o * 3584UL))];
      A_list[0UL] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0n__k_862 % 8UL) * 32768UL)];
      B_list[0UL] = __cached_1;
      void* _arg_cache_38 = &__origouts_900_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_52, A_list, B_list, &__origouts_900_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
      for (uint64_t _fuseiter3083 = 0UL; _fuseiter3083 < 14UL; _fuseiter3083 += 1UL) {
        for (uint64_t _fuseiter3084 = 0UL; _fuseiter3084 < 128UL; _fuseiter3084 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_900_shr[((_fuseiter3083 * 128UL) + _fuseiter3084)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((((fused_0n__k_862 / 8UL) * 1024UL) + ((fused_0n__k_862 % 8UL) * 128UL)) + _fuseiter3084)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_862 % 8UL) * 128UL) + _fuseiter3084)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_862 / 8UL) * 200704UL) + (((fused_0n__k_862 % 8UL) * 25088UL) + (p_o * 1792UL))) + ((_fuseiter3083 * 128UL) + _fuseiter3084))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0n__k_862 / 8UL) * 200704UL) + (((fused_0n__k_862 % 8UL) * 25088UL) + (p_o * 1792UL))) + ((_fuseiter3083 * 128UL) + _fuseiter3084))]);
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_1[(((fused_0n__k_862 / 8UL) * 200704UL) + ((((_fuseiter3084 + ((fused_0n__k_862 % 8UL) * 128UL)) / 256UL) * 50176UL) + ((p_o * 3584UL) + ((_fuseiter3083 * 256UL) + ((_fuseiter3084 + ((fused_0n__k_862 % 8UL) * 128UL)) % 256UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_900_shr);
    }
  }
  return true;
}

static bool reorder__157(uint8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 = 0UL; fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 < 28UL; fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 += 1UL) {
    for (uint64_t _fuseiter_3124 = 0UL; _fuseiter_3124 < 14UL; _fuseiter_3124 += 1UL) {
      for (uint64_t _fuseiter_3125 = 0UL; _fuseiter_3125 < 512UL; _fuseiter_3125 += 16UL) {
        vec_u8x16 __cached_0;
        __cached_0 = vec_u8x16::load(&__ins_0[(((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 / 28UL) * 200704UL) + ((((_fuseiter_3125 + (((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 / 14UL) % 2UL) * 512UL)) / 128UL) * 25088UL) + (((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 % 14UL) * 1792UL) + ((_fuseiter_3124 * 128UL) + ((_fuseiter_3125 + (((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 / 14UL) % 2UL) * 512UL)) % 128UL)))))]);
        vec_u8x16 __cached_1;
        __cached_1 = __cached_0;
        vec_u8x16::store(__cached_1, &__outs_0[(((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 / 28UL) * 200704UL) + ((((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 / 14UL) % 2UL) * 100352UL) + (((fused_0fused_0_fuseiter_3121___fuseiter_3122_863___fuseiter_3123_864 % 14UL) * 7168UL) + ((_fuseiter_3124 * 512UL) + _fuseiter_3125))))]);
      }
    }
  }
  return true;
}

static bool res4f_conv_0_cast_mul_add_relu_cast_reorder__161(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_76 = *(void**)(__module_data + 264);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0n__k_865 = 0UL; fused_0n__k_865 < 4UL; fused_0n__k_865 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_865 / 4UL) * 65536UL) + ((fused_0n__k_865 % 4UL) * 16384UL))], 0, 1024UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_865 / 4UL) * 65536UL) + (((fused_0n__k_865 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL)))], 0, 64UL);
      memset(&__outs_0[((((fused_0n__k_865 / 4UL) * 65536UL) + (((fused_0n__k_865 % 4UL) * 16384UL) + ((p1 + 1UL) * 1024UL))) + 960UL)], 0, 64UL);
    }
    memset(&__outs_0[((((fused_0n__k_865 / 4UL) * 65536UL) + ((fused_0n__k_865 % 4UL) * 16384UL)) + 15360UL)], 0, 1024UL);
  }
  for (uint64_t fused_0k__n_866 = 0UL; fused_0k__n_866 < 8UL; fused_0k__n_866 += 1UL) {
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_910_shr = (int32_t*)sc_aligned_malloc(__stream, 3584UL);
      void** A_list = (void**)&__rescheduled_0[0UL];
      void** B_list = (void**)&__rescheduled_0[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[((c * 100352UL) + (p_o * 14336UL))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[((fused_0k__n_866 * 32768UL) + (c * 16384UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_39 = &__origouts_910_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_76, A_list, B_list, &__origouts_910_shr[0UL], 1, 1, 1, 2, 8, 7, __stream);
      for (uint64_t _fuseiter3128 = 0UL; _fuseiter3128 < 2UL; _fuseiter3128 += 1UL) {
        for (uint64_t _fuseiter3129 = 0UL; _fuseiter3129 < 14UL; _fuseiter3129 += 1UL) {
          for (uint64_t _fuseiter3130 = 0UL; _fuseiter3130 < 32UL; _fuseiter3130 += 16UL) {
            vec_s32x16 __cached_2;
            __cached_2 = vec_s32x16::load(&__origouts_910_shr[((_fuseiter3128 * 448UL) + ((_fuseiter3129 * 32UL) + _fuseiter3130))]);
            vec_f32x16 __cached_3;
            __cached_3 = (vec_f32x16)(__cached_2);
            vec_f32x16 __cached_4;
            __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_866 * 32UL) + _fuseiter3130)]);
            __cached_3 = (__cached_3 * __cached_4);
            vec_f32x16 __cached_5;
            __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_866 * 32UL) + _fuseiter3130)]);
            __cached_3 = (__cached_3 + __cached_5);
            __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
            vec_s8x16 __cached_6;
            __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
            vec_s8x16 __cached_7;
            __cached_7 = __cached_6;
            vec_s8x16::store(__cached_7, &__outs_0[((((_fuseiter3130 + (fused_0k__n_866 * 32UL)) / 64UL) * 16384UL) + ((((_fuseiter3128 + (p_o * 2UL)) + 1UL) * 1024UL) + (((_fuseiter3129 + 1UL) * 64UL) + ((_fuseiter3130 + (fused_0k__n_866 * 32UL)) % 64UL))))]);
          }
        }
      }
      sc_aligned_free(__stream, __origouts_910_shr);
    }
  }
  return true;
}

static bool res4f_conv_1_cast_mul_add_relu_cast_reorder__165(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_70 = (void**)&__uninitialized_data[23657544UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 768UL);
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 196;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_867 = 0UL; fused_0k_o__n_867 < 4UL; fused_0k_o__n_867 += 1UL) {
    int32_t* __origouts_920_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_0[128UL];
    void** B_list = (void**)&__rescheduled_0[448UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_920_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 0, ((uint64_t)(__cached_3 * 64) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 4UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[((c_o * 16384UL) + ((r * 1024UL) + (s * 64UL)))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[((fused_0k_o__n_867 * 147456UL) + ((c_o * 36864UL) + ((r * 12288UL) + (s * 4096UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_40 = &__origouts_920_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_70[0UL], A_list, B_list, &__origouts_920_shr[(uint64_t)(((__cached_2 / 14) * 896) + ((__cached_2 % 14) * 64))], 1, 64, 4096, 36, 7, 7, __stream);
    for (uint64_t _fuseiter3163 = 0UL; _fuseiter3163 < 14UL; _fuseiter3163 += 1UL) {
      for (uint64_t _fuseiter3164 = 0UL; _fuseiter3164 < 14UL; _fuseiter3164 += 1UL) {
        for (uint64_t _fuseiter3165 = 0UL; _fuseiter3165 < 64UL; _fuseiter3165 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_920_shr[((_fuseiter3163 * 896UL) + ((_fuseiter3164 * 64UL) + _fuseiter3165))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_867 * 64UL) + _fuseiter3165)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_867 * 64UL) + _fuseiter3165)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[((((_fuseiter3165 + (fused_0k_o__n_867 * 64UL)) / 128UL) * 25088UL) + ((_fuseiter3163 * 1792UL) + ((_fuseiter3164 * 128UL) + ((_fuseiter3165 + (fused_0k_o__n_867 * 64UL)) % 128UL))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_920_shr);
  }
  sc_aligned_free(__stream, __rescheduled_0);
  return true;
}

static bool res4f_conv_2_cast_mul_add_cast_add_cast__170(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_78 = *(void**)(__module_data + 272);
  alignas(64) int8_t __rescheduled_0[128UL];
  for (uint64_t fused_0k__n_868 = 0UL; fused_0k__n_868 < 4UL; fused_0k__n_868 += 1UL) {
    int32_t* __origouts_930_shr = (int32_t*)sc_aligned_malloc(__stream, 200704UL);
    void** A_list = (void**)&__rescheduled_0[0UL];
    void** B_list = (void**)&__rescheduled_0[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(c * 25088UL)];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[((fused_0k__n_868 * 65536UL) + (c * 32768UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_41 = &__origouts_930_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_78, A_list, B_list, &__origouts_930_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter3198 = 0UL; _fuseiter3198 < 14UL; _fuseiter3198 += 1UL) {
      for (uint64_t _fuseiter3199 = 0UL; _fuseiter3199 < 14UL; _fuseiter3199 += 1UL) {
        for (uint64_t _fuseiter3200 = 0UL; _fuseiter3200 < 256UL; _fuseiter3200 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_930_shr[((_fuseiter3198 * 3584UL) + ((_fuseiter3199 * 256UL) + _fuseiter3200))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[((fused_0k__n_868 * 256UL) + _fuseiter3200)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[((fused_0k__n_868 * 256UL) + _fuseiter3200)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((fused_0k__n_868 * 50176UL) + ((_fuseiter3198 * 3584UL) + ((_fuseiter3199 * 256UL) + _fuseiter3200)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((fused_0k__n_868 * 50176UL) + ((_fuseiter3198 * 3584UL) + ((_fuseiter3199 * 256UL) + _fuseiter3200)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_930_shr);
  }
  return true;
}

static bool reorder__531(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_3232 = 0UL; _fuseiter_3232 < 128UL; _fuseiter_3232 += 1UL) {
    for (uint64_t _fuseiter_3233 = 0UL; _fuseiter_3233 < 4UL; _fuseiter_3233 += 1UL) {
      for (uint64_t _fuseiter_3236 = 0UL; _fuseiter_3236 < 64UL; _fuseiter_3236 += 1UL) {
        for (uint64_t _fuseiter_3237 = 0UL; _fuseiter_3237 < 16UL; _fuseiter_3237 += 1UL) {
          for (uint64_t _fuseiter_3238 = 0UL; _fuseiter_3238 < 4UL; _fuseiter_3238 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3237 + (_fuseiter_3232 * 16UL)) * 1024UL) + ((_fuseiter_3238 + (_fuseiter_3236 * 4UL)) + (_fuseiter_3233 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_3232 * 16384UL) + ((_fuseiter_3233 * 4096UL) + ((_fuseiter_3236 * 64UL) + ((_fuseiter_3237 * 4UL) + _fuseiter_3238))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__654(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 < 128UL; fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 % 128UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_869____itr_2_870____itr_3_871 % 128UL) * 16UL))]);
  }
  return true;
}

static bool mul__653(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 < 128UL; fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 % 128UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 / 128UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_872____itr_2_873____itr_3_874 % 128UL) * 16UL))]);
  }
  return true;
}

static bool mul__240(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_875____itr_2_876 = 0UL; fused_0fused_0__itr_0____itr_1_875____itr_2_876 < 786432UL; fused_0fused_0__itr_0____itr_1_875____itr_2_876 += 1UL) {
    for (uint64_t _fuseiter_3255 = 0UL; _fuseiter_3255 < 3UL; _fuseiter_3255 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_875____itr_2_876 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_875____itr_2_876 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_875____itr_2_876 % 3UL) * 3UL))) + _fuseiter_3255)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_875____itr_2_876 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_875____itr_2_876 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_875____itr_2_876 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_875____itr_2_876 % 3UL) * 3UL))) + _fuseiter_3255)] = __cached_2;
    }
  }
  return true;
}

static bool cast__241(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_877____itr_2_878 = 0UL; fused_0fused_0__itr_0____itr_1_877____itr_2_878 < 786432UL; fused_0fused_0__itr_0____itr_1_877____itr_2_878 += 1UL) {
    for (uint64_t _fuseiter3260 = 0UL; _fuseiter3260 < 3UL; _fuseiter3260 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_877____itr_2_878 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_877____itr_2_878 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_877____itr_2_878 % 3UL) * 3UL))) + _fuseiter3260)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_877____itr_2_878 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_877____itr_2_878 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_877____itr_2_878 % 3UL) * 3UL))) + _fuseiter3260)] = __cached_1;
    }
  }
  return true;
}

static bool reorder__537(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 = 0UL; fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 < 24UL; fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 += 1UL) {
    for (uint64_t _fuseiter_3264 = 0UL; _fuseiter_3264 < 3UL; _fuseiter_3264 += 1UL) {
      for (uint64_t _fuseiter_3265 = 0UL; _fuseiter_3265 < 64UL; _fuseiter_3265 += 1UL) {
        for (uint64_t _fuseiter_3266 = 0UL; _fuseiter_3266 < 128UL; _fuseiter_3266 += 1UL) {
          for (uint64_t _fuseiter_3267 = 0UL; _fuseiter_3267 < 4UL; _fuseiter_3267 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3266 + ((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 / 6UL) * 128UL)) * 4608UL) + ((((_fuseiter_3267 + (_fuseiter_3265 * 4UL)) + (((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 / 3UL) % 2UL) * 256UL)) * 9UL) + (((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 % 3UL) * 3UL) + _fuseiter_3264)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 / 6UL) * 589824UL) + ((((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 / 3UL) % 2UL) * 294912UL) + (((fused_0fused_0_fuseiter_3261___fuseiter_3262_879___fuseiter_3263_880 % 3UL) * 98304UL) + ((_fuseiter_3264 * 32768UL) + ((_fuseiter_3265 * 512UL) + ((_fuseiter_3266 * 4UL) + _fuseiter_3267))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool res5a_conv_0_cast_mul_add_relu_cast_reorder__681(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 280);
  for (uint64_t fused_0n__k_881 = 0UL; fused_0n__k_881 < 18UL; fused_0n__k_881 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_881 / 2UL) * 131072UL) + ((fused_0n__k_881 % 2UL) * 65536UL))], 0, 4096UL);
    for (uint64_t p1 = 0UL; p1 < 14UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_881 / 2UL) * 131072UL) + (((fused_0n__k_881 % 2UL) * 65536UL) + ((p1 + 1UL) * 4096UL)))], 0, 256UL);
      memset(&__outs_0[((((fused_0n__k_881 / 2UL) * 131072UL) + (((fused_0n__k_881 % 2UL) * 65536UL) + ((p1 + 1UL) * 4096UL))) + 3840UL)], 0, 256UL);
    }
    memset(&__outs_0[((((fused_0n__k_881 / 2UL) * 131072UL) + ((fused_0n__k_881 % 2UL) * 65536UL)) + 61440UL)], 0, 4096UL);
  }
  for (uint64_t fused_0n__k_882 = 0UL; fused_0n__k_882 < 72UL; fused_0n__k_882 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_940_shr = (int32_t*)sc_aligned_malloc(__stream, 50176UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_882 / 8UL) * 200704UL) + (c * 50176UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_882 % 8UL) * 65536UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_42 = &__origouts_940_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_80, A_list, B_list, &__origouts_940_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter3270 = 0UL; _fuseiter3270 < 14UL; _fuseiter3270 += 1UL) {
      for (uint64_t _fuseiter3271 = 0UL; _fuseiter3271 < 14UL; _fuseiter3271 += 1UL) {
        for (uint64_t _fuseiter3272 = 0UL; _fuseiter3272 < 64UL; _fuseiter3272 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_940_shr[((_fuseiter3270 * 896UL) + ((_fuseiter3271 * 64UL) + _fuseiter3272))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_882 % 8UL) * 64UL) + _fuseiter3272)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_882 % 8UL) * 64UL) + _fuseiter3272)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_882 / 8UL) * 131072UL) + ((((_fuseiter3272 + ((fused_0n__k_882 % 8UL) * 64UL)) / 256UL) * 65536UL) + (((_fuseiter3270 + 1UL) * 4096UL) + (((_fuseiter3271 + 1UL) * 256UL) + ((_fuseiter3272 + ((fused_0n__k_882 % 8UL) * 64UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_940_shr);
  }
  return true;
}

static bool mul__658(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 += 1UL) {
    for (uint64_t _fuseiter_3307 = 0UL; _fuseiter_3307 < 128UL; _fuseiter_3307 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 % 4UL) * 128UL)) + _fuseiter_3307)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_883____itr_2_884____itr_3_885 % 4UL) * 128UL)) + _fuseiter_3307)]);
    }
  }
  return true;
}

static bool mul__657(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 += 1UL) {
    for (uint64_t _fuseiter_3313 = 0UL; _fuseiter_3313 < 128UL; _fuseiter_3313 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 % 4UL) * 128UL)) + _fuseiter_3313)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_886____itr_2_887____itr_3_888 % 4UL) * 128UL)) + _fuseiter_3313)]);
    }
  }
  return true;
}

static bool res5a_conv_1_cast_mul_add_relu_cast_reorder__680(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657560UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_889 = 0UL; fused_0k_o__n_889 < 36UL; fused_0k_o__n_889 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
    int32_t* __origouts_950_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[192UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_950_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0k_o__n_889 % 9UL) * 131072UL) + ((c_o * 65536UL) + ((r * 4096UL) + (s * 256UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0k_o__n_889 / 9UL) * 589824UL) + ((c_o * 294912UL) + ((r * 98304UL) + (s * 32768UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_43 = &__origouts_950_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_84[0UL], A_list, B_list, &__origouts_950_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 1, 256, 32768, 18, 7, 7, __stream);
    for (uint64_t _fuseiter3317 = 0UL; _fuseiter3317 < 7UL; _fuseiter3317 += 1UL) {
      for (uint64_t _fuseiter3318 = 0UL; _fuseiter3318 < 7UL; _fuseiter3318 += 1UL) {
        for (uint64_t _fuseiter3319 = 0UL; _fuseiter3319 < 128UL; _fuseiter3319 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_950_shr[((_fuseiter3317 * 896UL) + ((_fuseiter3318 * 128UL) + _fuseiter3319))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0k_o__n_889 / 9UL) * 128UL) + _fuseiter3319)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0k_o__n_889 / 9UL) * 128UL) + _fuseiter3319)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_889 % 9UL) * 25088UL) + ((((_fuseiter3319 + ((fused_0k_o__n_889 / 9UL) * 128UL)) / 256UL) * 12544UL) + ((_fuseiter3317 * 1792UL) + ((_fuseiter3318 * 256UL) + ((_fuseiter3319 + ((fused_0k_o__n_889 / 9UL) * 128UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_950_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool res5a_conv_b_cast_mul_add_cast_reorder__682(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_86 = *(void**)(__module_data + 288);
  uint8_t* input_tmp = (uint8_t*)sc_aligned_malloc(__stream, 451584UL);
  for (uint64_t fused_0n__c_o_890 = 0UL; fused_0n__c_o_890 < 36UL; fused_0n__c_o_890 += 1UL) {
    for (uint64_t p = 0UL; p < 7UL; p += 1UL) {
      for (uint64_t q = 0UL; q < 7UL; q += 1UL) {
        for (uint64_t c_i = 0UL; c_i < 256UL; c_i += 64UL) {
          vec_u8x64 __cached_0;
          __cached_0 = vec_u8x64::load(&__ins_0[(((fused_0n__c_o_890 / 4UL) * 200704UL) + (((fused_0n__c_o_890 % 4UL) * 50176UL) + ((p * 7168UL) + ((q * 512UL) + c_i))))]);
          vec_u8x64 __cached_1;
          __cached_1 = __cached_0;
          vec_u8x64::store(__cached_1, &input_tmp[(((fused_0n__c_o_890 / 4UL) * 50176UL) + (((fused_0n__c_o_890 % 4UL) * 12544UL) + ((p * 1792UL) + ((q * 256UL) + c_i))))]);
        }
      }
    }
  }
  for (uint64_t fused_0k__n_891 = 0UL; fused_0k__n_891 < 1152UL; fused_0k__n_891 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_960_shr = (int32_t*)sc_aligned_malloc(__stream, 3136UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_2;
      __cached_2 = &input_tmp[(((fused_0k__n_891 % 9UL) * 50176UL) + (c * 12544UL))];
      A_list[c] = __cached_2;
      void* __cached_3;
      __cached_3 = &__ins_1[(((fused_0k__n_891 / 9UL) * 16384UL) + (c * 4096UL))];
      B_list[c] = __cached_3;
    }
    void* _arg_cache_44 = &__origouts_960_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_86, A_list, B_list, &__origouts_960_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter3352 = 0UL; _fuseiter3352 < 7UL; _fuseiter3352 += 1UL) {
      for (uint64_t _fuseiter3353 = 0UL; _fuseiter3353 < 7UL; _fuseiter3353 += 1UL) {
        vec_s32x16 __cached_4;
        __cached_4 = vec_s32x16::load(&__origouts_960_shr[((_fuseiter3352 * 112UL) + (_fuseiter3353 * 16UL))]);
        vec_f32x16 __cached_5;
        __cached_5 = (vec_f32x16)(__cached_4);
        vec_f32x16 __cached_6;
        __cached_6 = vec_f32x16::load(&__ins_2[((fused_0k__n_891 / 9UL) * 16UL)]);
        __cached_5 = (__cached_5 * __cached_6);
        vec_f32x16 __cached_7;
        __cached_7 = vec_f32x16::load(&__ins_3[((fused_0k__n_891 / 9UL) * 16UL)]);
        __cached_5 = (__cached_5 + __cached_7);
        vec_s8x16 __cached_8;
        __cached_8 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_5));
        vec_s8x16 __cached_9;
        __cached_9 = __cached_8;
        vec_s8x16::store(__cached_9, &__outs_0[(((fused_0k__n_891 % 9UL) * 100352UL) + (((((fused_0k__n_891 / 9UL) * 16UL) / 64UL) * 3136UL) + ((_fuseiter3352 * 448UL) + ((_fuseiter3353 * 64UL) + (((fused_0k__n_891 / 9UL) * 16UL) % 64UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_960_shr);
  }
  sc_aligned_free(__stream, input_tmp);
  return true;
}

static bool mul__660(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 += 1UL) {
    for (uint64_t _fuseiter_3383 = 0UL; _fuseiter_3383 < 64UL; _fuseiter_3383 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 % 32UL) * 64UL)) + _fuseiter_3383)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_892____itr_2_893____itr_3_894 % 32UL) * 64UL)) + _fuseiter_3383)]);
    }
  }
  return true;
}

static bool mul__659(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 += 1UL) {
    for (uint64_t _fuseiter_3389 = 0UL; _fuseiter_3389 < 64UL; _fuseiter_3389 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 % 32UL) * 64UL)) + _fuseiter_3389)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_895____itr_2_896____itr_3_897 % 32UL) * 64UL)) + _fuseiter_3389)]);
    }
  }
  return true;
}

static bool reorder__540(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_3391 = 0UL; _fuseiter_3391 < 32UL; _fuseiter_3391 += 1UL) {
    for (uint64_t _fuseiter_3392 = 0UL; _fuseiter_3392 < 2UL; _fuseiter_3392 += 1UL) {
      for (uint64_t _fuseiter_3395 = 0UL; _fuseiter_3395 < 64UL; _fuseiter_3395 += 1UL) {
        for (uint64_t _fuseiter_3396 = 0UL; _fuseiter_3396 < 64UL; _fuseiter_3396 += 1UL) {
          for (uint64_t _fuseiter_3397 = 0UL; _fuseiter_3397 < 4UL; _fuseiter_3397 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3396 + (_fuseiter_3391 * 64UL)) * 512UL) + ((_fuseiter_3397 + (_fuseiter_3395 * 4UL)) + (_fuseiter_3392 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_3391 * 32768UL) + ((_fuseiter_3392 * 16384UL) + ((_fuseiter_3395 * 256UL) + ((_fuseiter_3396 * 4UL) + _fuseiter_3397))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__662(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 += 1UL) {
    for (uint64_t _fuseiter_3402 = 0UL; _fuseiter_3402 < 128UL; _fuseiter_3402 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 % 4UL) * 128UL)) + _fuseiter_3402)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_898____itr_2_899____itr_3_900 % 4UL) * 128UL)) + _fuseiter_3402)]);
    }
  }
  return true;
}

static bool mul__661(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 += 1UL) {
    for (uint64_t _fuseiter_3408 = 0UL; _fuseiter_3408 < 128UL; _fuseiter_3408 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 % 4UL) * 128UL)) + _fuseiter_3408)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_901____itr_2_902____itr_3_903 % 4UL) * 128UL)) + _fuseiter_3408)]);
    }
  }
  return true;
}

static bool reorder__543(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0_fuseiter_3410___fuseiter_3411_904 = 0UL; fused_0_fuseiter_3410___fuseiter_3411_904 < 128UL; fused_0_fuseiter_3410___fuseiter_3411_904 += 1UL) {
    for (uint64_t _fuseiter_3414 = 0UL; _fuseiter_3414 < 16UL; _fuseiter_3414 += 1UL) {
      for (uint64_t _fuseiter_3415 = 0UL; _fuseiter_3415 < 128UL; _fuseiter_3415 += 1UL) {
        for (uint64_t _fuseiter_3416 = 0UL; _fuseiter_3416 < 4UL; _fuseiter_3416 += 1UL) {
          int8_t __cached_0;
          __cached_0 = __ins_0[(((_fuseiter_3415 + ((fused_0_fuseiter_3410___fuseiter_3411_904 / 32UL) * 128UL)) * 2048UL) + ((_fuseiter_3416 + (_fuseiter_3414 * 4UL)) + ((fused_0_fuseiter_3410___fuseiter_3411_904 % 32UL) * 64UL)))];
          int8_t __cached_1;
          __cached_1 = __cached_0;
          __outs_0[(((fused_0_fuseiter_3410___fuseiter_3411_904 / 32UL) * 262144UL) + (((fused_0_fuseiter_3410___fuseiter_3411_904 % 32UL) * 8192UL) + ((_fuseiter_3414 * 512UL) + ((_fuseiter_3415 * 4UL) + _fuseiter_3416))))] = __cached_1;
        }
      }
    }
  }
  return true;
}

static bool mul__249(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_905____itr_2_906 = 0UL; fused_0fused_0__itr_0____itr_1_905____itr_2_906 < 786432UL; fused_0fused_0__itr_0____itr_1_905____itr_2_906 += 1UL) {
    for (uint64_t _fuseiter_3421 = 0UL; _fuseiter_3421 < 3UL; _fuseiter_3421 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_905____itr_2_906 % 3UL) * 3UL))) + _fuseiter_3421)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_905____itr_2_906 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_905____itr_2_906 % 3UL) * 3UL))) + _fuseiter_3421)] = __cached_2;
    }
  }
  return true;
}

static bool cast__250(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_907____itr_2_908 = 0UL; fused_0fused_0__itr_0____itr_1_907____itr_2_908 < 786432UL; fused_0fused_0__itr_0____itr_1_907____itr_2_908 += 1UL) {
    for (uint64_t _fuseiter3426 = 0UL; _fuseiter3426 < 3UL; _fuseiter3426 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_907____itr_2_908 % 3UL) * 3UL))) + _fuseiter3426)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_907____itr_2_908 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_907____itr_2_908 % 3UL) * 3UL))) + _fuseiter3426)] = __cached_1;
    }
  }
  return true;
}

static bool mul__258(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_909____itr_2_910 = 0UL; fused_0fused_0__itr_0____itr_1_909____itr_2_910 < 786432UL; fused_0fused_0__itr_0____itr_1_909____itr_2_910 += 1UL) {
    for (uint64_t _fuseiter_3431 = 0UL; _fuseiter_3431 < 3UL; _fuseiter_3431 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_909____itr_2_910 % 3UL) * 3UL))) + _fuseiter_3431)];
      float __cached_1;
      __cached_1 = __ins_1[(fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 1536UL)];
      float __cached_2;
      __cached_2 = (__cached_0 * __cached_1);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_909____itr_2_910 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_909____itr_2_910 % 3UL) * 3UL))) + _fuseiter_3431)] = __cached_2;
    }
  }
  return true;
}

static bool cast__259(int8_t* __restrict__ __outs_0, float* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0__itr_0____itr_1_911____itr_2_912 = 0UL; fused_0fused_0__itr_0____itr_1_911____itr_2_912 < 786432UL; fused_0fused_0__itr_0____itr_1_911____itr_2_912 += 1UL) {
    for (uint64_t _fuseiter3436 = 0UL; _fuseiter3436 < 3UL; _fuseiter3436 += 1UL) {
      float __cached_0;
      __cached_0 = __ins_0[((((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_911____itr_2_912 % 3UL) * 3UL))) + _fuseiter3436)];
      int8_t __cached_1;
      __cached_1 = (int8_t)sc_max(sc_min(sc_round_and_cast<int32_t>(__cached_0), 127), -128);
      __outs_0[((((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 1536UL) * 4608UL) + ((((fused_0fused_0__itr_0____itr_1_911____itr_2_912 / 3UL) % 512UL) * 9UL) + ((fused_0fused_0__itr_0____itr_1_911____itr_2_912 % 3UL) * 3UL))) + _fuseiter3436)] = __cached_1;
    }
  }
  return true;
}

static bool mul__664(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 % 32UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_913____itr_2_914____itr_3_915 % 32UL) * 16UL))]);
  }
  return true;
}

static bool mul__663(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__ins_0[(((fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 % 32UL) * 16UL))]);
    float __cached_1;
    __cached_1 = __ins_1[0];
    vec_f32x16 __cached_2;
    __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
    vec_f32x16::store(__cached_2, &__outs_0[(((fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 / 32UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_916____itr_2_917____itr_3_918 % 32UL) * 16UL))]);
  }
  return true;
}

static bool reorder__546(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_3449 = 0UL; _fuseiter_3449 < 32UL; _fuseiter_3449 += 1UL) {
    for (uint64_t _fuseiter_3451 = 0UL; _fuseiter_3451 < 3UL; _fuseiter_3451 += 1UL) {
      for (uint64_t _fuseiter_3452 = 0UL; _fuseiter_3452 < 3UL; _fuseiter_3452 += 1UL) {
        for (uint64_t _fuseiter_3453 = 0UL; _fuseiter_3453 < 128UL; _fuseiter_3453 += 1UL) {
          for (uint64_t _fuseiter_3454 = 0UL; _fuseiter_3454 < 16UL; _fuseiter_3454 += 1UL) {
            for (uint64_t _fuseiter_3455 = 0UL; _fuseiter_3455 < 4UL; _fuseiter_3455 += 1UL) {
              int8_t __cached_0;
              __cached_0 = __ins_0[(((_fuseiter_3454 + (_fuseiter_3449 * 16UL)) * 4608UL) + (((_fuseiter_3455 + (_fuseiter_3453 * 4UL)) * 9UL) + ((_fuseiter_3451 * 3UL) + _fuseiter_3452)))];
              int8_t __cached_1;
              __cached_1 = __cached_0;
              __outs_0[((_fuseiter_3449 * 73728UL) + ((_fuseiter_3451 * 24576UL) + ((_fuseiter_3452 * 8192UL) + ((_fuseiter_3453 * 64UL) + ((_fuseiter_3454 * 4UL) + _fuseiter_3455)))))] = __cached_1;
            }
          }
        }
      }
    }
  }
  return true;
}

static bool res5a_conv_2_cast_mul_add_cast_add_cast__679(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_88 = *(void**)(__module_data + 296);
  for (uint64_t fused_0k__n_919 = 0UL; fused_0k__n_919 < 288UL; fused_0k__n_919 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_970_shr = (int32_t*)sc_aligned_malloc(__stream, 12544UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0k__n_919 % 9UL) * 25088UL) + (c * 12544UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0k__n_919 / 9UL) * 32768UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_45 = &__origouts_970_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_88, A_list, B_list, &__origouts_970_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
    for (uint64_t _fuseiter3458 = 0UL; _fuseiter3458 < 7UL; _fuseiter3458 += 1UL) {
      for (uint64_t _fuseiter3459 = 0UL; _fuseiter3459 < 7UL; _fuseiter3459 += 1UL) {
        for (uint64_t _fuseiter3460 = 0UL; _fuseiter3460 < 64UL; _fuseiter3460 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_970_shr[((_fuseiter3458 * 448UL) + ((_fuseiter3459 * 64UL) + _fuseiter3460))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0k__n_919 / 9UL) * 64UL) + _fuseiter3460)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0k__n_919 / 9UL) * 64UL) + _fuseiter3460)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = vec_s8x16::load(&__ins_4[((((fused_0k__n_919 % 9UL) * 100352UL) + ((fused_0k__n_919 / 9UL) * 3136UL)) + ((_fuseiter3458 * 448UL) + ((_fuseiter3459 * 64UL) + _fuseiter3460)))]);
          __cached_6 = (__cached_6 + __cached_7);
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16::store(__cached_8, &__outs_0[((((fused_0k__n_919 % 9UL) * 100352UL) + ((fused_0k__n_919 / 9UL) * 3136UL)) + ((_fuseiter3458 * 448UL) + ((_fuseiter3459 * 64UL) + _fuseiter3460)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_970_shr);
  }
  return true;
}

static bool res5b_conv_0_cast_mul_add_relu_cast_reorder__678(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_90 = *(void**)(__module_data + 304);
  for (uint64_t fused_0n__k_920 = 0UL; fused_0n__k_920 < 9UL; fused_0n__k_920 += 1UL) {
    memset(&__outs_0[(fused_0n__k_920 * 41472UL)], 0, 4608UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[((fused_0n__k_920 * 41472UL) + ((p1 + 1UL) * 4608UL))], 0, 512UL);
      memset(&__outs_0[(((fused_0n__k_920 * 41472UL) + ((p1 + 1UL) * 4608UL)) + 4096UL)], 0, 512UL);
    }
    memset(&__outs_0[((fused_0n__k_920 * 41472UL) + 36864UL)], 0, 4608UL);
  }
  for (uint64_t fused_0n__k_921 = 0UL; fused_0n__k_921 < 36UL; fused_0n__k_921 += 1UL) {
    int8_t* __rescheduled_2 = (int8_t*)sc_aligned_malloc(__stream, 512UL);
    int32_t* __origouts_980_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[256UL];
    for (uint64_t c = 0UL; c < 32UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_921 / 4UL) * 100352UL) + (c * 3136UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_921 % 4UL) * 262144UL) + (c * 8192UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_46 = &__origouts_980_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_90, A_list, B_list, &__origouts_980_shr[0UL], 1, 1, 1, 32, 8, 7, __stream);
    for (uint64_t _fuseiter3494 = 0UL; _fuseiter3494 < 7UL; _fuseiter3494 += 1UL) {
      for (uint64_t _fuseiter3495 = 0UL; _fuseiter3495 < 7UL; _fuseiter3495 += 1UL) {
        for (uint64_t _fuseiter3496 = 0UL; _fuseiter3496 < 128UL; _fuseiter3496 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_980_shr[((_fuseiter3494 * 896UL) + ((_fuseiter3495 * 128UL) + _fuseiter3496))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_921 % 4UL) * 128UL) + _fuseiter3496)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_921 % 4UL) * 128UL) + _fuseiter3496)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_921 / 4UL) * 41472UL) + ((((_fuseiter3496 + ((fused_0n__k_921 % 4UL) * 128UL)) / 512UL) * 41472UL) + (((_fuseiter3494 + 1UL) * 4608UL) + (((_fuseiter3495 + 1UL) * 512UL) + ((_fuseiter3496 + ((fused_0n__k_921 % 4UL) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_980_shr);
    sc_aligned_free(__stream, __rescheduled_2);
  }
  return true;
}

static bool res5b_conv_1_cast_mul_add_relu_cast_reorder__677(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_94 = (void**)&__uninitialized_data[23657576UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_922 = 0UL; fused_0k_o__n_922 < 288UL; fused_0k_o__n_922 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 256UL);
    int32_t* __origouts_990_shr = (int32_t*)sc_aligned_malloc(__stream, 3136UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[128UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_990_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))], 0, ((uint64_t)(__cached_3 * 16) * 4UL));
    for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
      for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
        void* __cached_4;
        __cached_4 = &__ins_0[(((fused_0k_o__n_922 % 9UL) * 41472UL) + ((r * 4608UL) + (s * 512UL)))];
        A_list[((r * 3UL) + s)] = __cached_4;
        void* __cached_5;
        __cached_5 = &__ins_1[(((fused_0k_o__n_922 / 9UL) * 73728UL) + ((r * 24576UL) + (s * 8192UL)))];
        B_list[((r * 3UL) + s)] = __cached_5;
      }
    }
    void* _arg_cache_47 = &__origouts_990_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_94[0UL], A_list, B_list, &__origouts_990_shr[(uint64_t)(((__cached_2 / 7) * 112) + ((__cached_2 % 7) * 16))], 1, 512, 8192, 9, 7, 7, __stream);
    for (uint64_t _fuseiter3529 = 0UL; _fuseiter3529 < 7UL; _fuseiter3529 += 1UL) {
      for (uint64_t _fuseiter3530 = 0UL; _fuseiter3530 < 7UL; _fuseiter3530 += 1UL) {
        vec_s32x16 __cached_6;
        __cached_6 = vec_s32x16::load(&__origouts_990_shr[((_fuseiter3529 * 112UL) + (_fuseiter3530 * 16UL))]);
        vec_f32x16 __cached_7;
        __cached_7 = (vec_f32x16)(__cached_6);
        vec_f32x16 __cached_8;
        __cached_8 = vec_f32x16::load(&__ins_2[((fused_0k_o__n_922 / 9UL) * 16UL)]);
        __cached_7 = (__cached_7 * __cached_8);
        vec_f32x16 __cached_9;
        __cached_9 = vec_f32x16::load(&__ins_3[((fused_0k_o__n_922 / 9UL) * 16UL)]);
        __cached_7 = (__cached_7 + __cached_9);
        __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
        vec_s8x16 __cached_10;
        __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
        vec_s8x16 __cached_11;
        __cached_11 = __cached_10;
        vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_922 % 9UL) * 25088UL) + (((((fused_0k_o__n_922 / 9UL) * 16UL) / 256UL) * 12544UL) + ((_fuseiter3529 * 1792UL) + ((_fuseiter3530 * 256UL) + (((fused_0k_o__n_922 / 9UL) * 16UL) % 256UL)))))]);
      }
    }
    sc_aligned_free(__stream, __origouts_990_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool mul__666(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 += 1UL) {
    for (uint64_t _fuseiter_3566 = 0UL; _fuseiter_3566 < 64UL; _fuseiter_3566 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 % 32UL) * 64UL)) + _fuseiter_3566)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_923____itr_2_924____itr_3_925 % 32UL) * 64UL)) + _fuseiter_3566)]);
    }
  }
  return true;
}

static bool mul__665(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 < 32UL; fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 += 1UL) {
    for (uint64_t _fuseiter_3572 = 0UL; _fuseiter_3572 < 64UL; _fuseiter_3572 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 % 32UL) * 64UL)) + _fuseiter_3572)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 / 32UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_926____itr_2_927____itr_3_928 % 32UL) * 64UL)) + _fuseiter_3572)]);
    }
  }
  return true;
}

static bool reorder__549(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_3574 = 0UL; _fuseiter_3574 < 32UL; _fuseiter_3574 += 1UL) {
    for (uint64_t _fuseiter_3575 = 0UL; _fuseiter_3575 < 2UL; _fuseiter_3575 += 1UL) {
      for (uint64_t _fuseiter_3578 = 0UL; _fuseiter_3578 < 64UL; _fuseiter_3578 += 1UL) {
        for (uint64_t _fuseiter_3579 = 0UL; _fuseiter_3579 < 64UL; _fuseiter_3579 += 1UL) {
          for (uint64_t _fuseiter_3580 = 0UL; _fuseiter_3580 < 4UL; _fuseiter_3580 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3579 + (_fuseiter_3574 * 64UL)) * 512UL) + ((_fuseiter_3580 + (_fuseiter_3578 * 4UL)) + (_fuseiter_3575 * 256UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_3574 * 32768UL) + ((_fuseiter_3575 * 16384UL) + ((_fuseiter_3578 * 256UL) + ((_fuseiter_3579 * 4UL) + _fuseiter_3580))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__668(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 += 1UL) {
    for (uint64_t _fuseiter_3585 = 0UL; _fuseiter_3585 < 32UL; _fuseiter_3585 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 % 16UL) * 32UL)) + _fuseiter_3585)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_929____itr_2_930____itr_3_931 % 16UL) * 32UL)) + _fuseiter_3585)]);
    }
  }
  return true;
}

static bool mul__667(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 < 16UL; fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 += 1UL) {
    for (uint64_t _fuseiter_3591 = 0UL; _fuseiter_3591 < 32UL; _fuseiter_3591 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 % 16UL) * 32UL)) + _fuseiter_3591)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 / 16UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_932____itr_2_933____itr_3_934 % 16UL) * 32UL)) + _fuseiter_3591)]);
    }
  }
  return true;
}

static bool reorder__552(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t _fuseiter_3593 = 0UL; _fuseiter_3593 < 16UL; _fuseiter_3593 += 1UL) {
    for (uint64_t _fuseiter_3594 = 0UL; _fuseiter_3594 < 4UL; _fuseiter_3594 += 1UL) {
      for (uint64_t _fuseiter_3597 = 0UL; _fuseiter_3597 < 128UL; _fuseiter_3597 += 1UL) {
        for (uint64_t _fuseiter_3598 = 0UL; _fuseiter_3598 < 32UL; _fuseiter_3598 += 1UL) {
          for (uint64_t _fuseiter_3599 = 0UL; _fuseiter_3599 < 4UL; _fuseiter_3599 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3598 + (_fuseiter_3593 * 32UL)) * 2048UL) + ((_fuseiter_3599 + (_fuseiter_3597 * 4UL)) + (_fuseiter_3594 * 512UL)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[((_fuseiter_3593 * 65536UL) + ((_fuseiter_3594 * 16384UL) + ((_fuseiter_3597 * 128UL) + ((_fuseiter_3598 * 4UL) + _fuseiter_3599))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool mul__670(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 += 1UL) {
    for (uint64_t _fuseiter_3604 = 0UL; _fuseiter_3604 < 128UL; _fuseiter_3604 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 % 4UL) * 128UL)) + _fuseiter_3604)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_935____itr_2_936____itr_3_937 % 4UL) * 128UL)) + _fuseiter_3604)]);
    }
  }
  return true;
}

static bool mul__669(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 += 1UL) {
    for (uint64_t _fuseiter_3610 = 0UL; _fuseiter_3610 < 128UL; _fuseiter_3610 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 % 4UL) * 128UL)) + _fuseiter_3610)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 / 4UL) * 512UL) + ((fused_0fused_0fused_0__itr_0____itr_1_938____itr_2_939____itr_3_940 % 4UL) * 128UL)) + _fuseiter_3610)]);
    }
  }
  return true;
}

static bool reorder__555(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 = 0UL; fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 < 24UL; fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 += 1UL) {
    for (uint64_t _fuseiter_3615 = 0UL; _fuseiter_3615 < 3UL; _fuseiter_3615 += 1UL) {
      for (uint64_t _fuseiter_3616 = 0UL; _fuseiter_3616 < 64UL; _fuseiter_3616 += 1UL) {
        for (uint64_t _fuseiter_3617 = 0UL; _fuseiter_3617 < 128UL; _fuseiter_3617 += 1UL) {
          for (uint64_t _fuseiter_3618 = 0UL; _fuseiter_3618 < 4UL; _fuseiter_3618 += 1UL) {
            int8_t __cached_0;
            __cached_0 = __ins_0[(((_fuseiter_3617 + ((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 / 6UL) * 128UL)) * 4608UL) + ((((_fuseiter_3618 + (_fuseiter_3616 * 4UL)) + (((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 / 3UL) % 2UL) * 256UL)) * 9UL) + (((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 % 3UL) * 3UL) + _fuseiter_3615)))];
            int8_t __cached_1;
            __cached_1 = __cached_0;
            __outs_0[(((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 / 6UL) * 589824UL) + ((((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 / 3UL) % 2UL) * 294912UL) + (((fused_0fused_0_fuseiter_3612___fuseiter_3613_941___fuseiter_3614_942 % 3UL) * 98304UL) + ((_fuseiter_3615 * 32768UL) + ((_fuseiter_3616 * 512UL) + ((_fuseiter_3617 * 4UL) + _fuseiter_3618))))))] = __cached_1;
          }
        }
      }
    }
  }
  return true;
}

static bool res5b_conv_2_cast_mul_add_cast_add_cast_reorder__676(uint8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_96 = *(void**)(__module_data + 312);
  for (uint64_t fused_0n__k_943 = 0UL; fused_0n__k_943 < 288UL; fused_0n__k_943 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    for (uint64_t p_o = 0UL; p_o < 7UL; p_o += 1UL) {
      int32_t* __origouts_1000_shr = (int32_t*)sc_aligned_malloc(__stream, 1792UL);
      void** A_list = (void**)&__rescheduled_1[0UL];
      void** B_list = (void**)&__rescheduled_1[64UL];
      for (uint64_t c = 0UL; c < 2UL; c += 1UL) {
        void* __cached_0;
        __cached_0 = &__ins_0[(((fused_0n__k_943 / 32UL) * 25088UL) + ((c * 12544UL) + (p_o * 1792UL)))];
        A_list[c] = __cached_0;
        void* __cached_1;
        __cached_1 = &__ins_1[(((fused_0n__k_943 % 32UL) * 32768UL) + (c * 16384UL))];
        B_list[c] = __cached_1;
      }
      void* _arg_cache_48 = &__origouts_1000_shr[0UL];
      dnnl_brgemm_list_call(__sc_kernel_cache_96, A_list, B_list, &__origouts_1000_shr[0UL], 1, 1, 1, 2, 7, 7, __stream);
      for (uint64_t _fuseiter3622 = 0UL; _fuseiter3622 < 7UL; _fuseiter3622 += 1UL) {
        for (uint64_t _fuseiter3623 = 0UL; _fuseiter3623 < 64UL; _fuseiter3623 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1000_shr[((_fuseiter3622 * 64UL) + _fuseiter3623)]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_943 % 32UL) * 64UL) + _fuseiter3623)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_943 % 32UL) * 64UL) + _fuseiter3623)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0n__k_943 / 32UL) * 100352UL) + (((fused_0n__k_943 % 32UL) * 3136UL) + (p_o * 448UL))) + ((_fuseiter3622 * 64UL) + _fuseiter3623))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_u8x16 __cached_9;
          __cached_9 = __cached_8;
          vec_u8x16::store(__cached_9, &__outs_0[(((fused_0n__k_943 / 32UL) * 100352UL) + ((((_fuseiter3623 + ((fused_0n__k_943 % 32UL) * 64UL)) / 512UL) * 25088UL) + ((p_o * 3584UL) + ((_fuseiter3622 * 512UL) + ((_fuseiter3623 + ((fused_0n__k_943 % 32UL) * 64UL)) % 512UL)))))]);
        }
      }
      sc_aligned_free(__stream, __origouts_1000_shr);
    }
  }
  return true;
}

static bool res5c_conv_0_cast_mul_add_relu_cast_reorder__675(int8_t* __restrict__ __outs_0, uint8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void*& __sc_kernel_cache_98 = *(void**)(__module_data + 320);
  for (uint64_t fused_0n__k_944 = 0UL; fused_0n__k_944 < 18UL; fused_0n__k_944 += 1UL) {
    memset(&__outs_0[(((fused_0n__k_944 / 2UL) * 41472UL) + ((fused_0n__k_944 % 2UL) * 20736UL))], 0, 2304UL);
    for (uint64_t p1 = 0UL; p1 < 7UL; p1 += 1UL) {
      memset(&__outs_0[(((fused_0n__k_944 / 2UL) * 41472UL) + (((fused_0n__k_944 % 2UL) * 20736UL) + ((p1 + 1UL) * 2304UL)))], 0, 256UL);
      memset(&__outs_0[((((fused_0n__k_944 / 2UL) * 41472UL) + (((fused_0n__k_944 % 2UL) * 20736UL) + ((p1 + 1UL) * 2304UL))) + 2048UL)], 0, 256UL);
    }
    memset(&__outs_0[((((fused_0n__k_944 / 2UL) * 41472UL) + ((fused_0n__k_944 % 2UL) * 20736UL)) + 18432UL)], 0, 2304UL);
  }
  for (uint64_t fused_0n__k_945 = 0UL; fused_0n__k_945 < 144UL; fused_0n__k_945 += 1UL) {
    alignas(64) int8_t __rescheduled_2[128UL];
    int32_t* __origouts_1010_shr = (int32_t*)sc_aligned_malloc(__stream, 6272UL);
    void** A_list = (void**)&__rescheduled_2[0UL];
    void** B_list = (void**)&__rescheduled_2[64UL];
    for (uint64_t c = 0UL; c < 4UL; c += 1UL) {
      void* __cached_0;
      __cached_0 = &__ins_0[(((fused_0n__k_945 / 16UL) * 100352UL) + (c * 25088UL))];
      A_list[c] = __cached_0;
      void* __cached_1;
      __cached_1 = &__ins_1[(((fused_0n__k_945 % 16UL) * 65536UL) + (c * 16384UL))];
      B_list[c] = __cached_1;
    }
    void* _arg_cache_49 = &__origouts_1010_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_98, A_list, B_list, &__origouts_1010_shr[0UL], 1, 1, 1, 4, 8, 7, __stream);
    for (uint64_t _fuseiter3662 = 0UL; _fuseiter3662 < 7UL; _fuseiter3662 += 1UL) {
      for (uint64_t _fuseiter3663 = 0UL; _fuseiter3663 < 7UL; _fuseiter3663 += 1UL) {
        for (uint64_t _fuseiter3664 = 0UL; _fuseiter3664 < 32UL; _fuseiter3664 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1010_shr[((_fuseiter3662 * 224UL) + ((_fuseiter3663 * 32UL) + _fuseiter3664))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0n__k_945 % 16UL) * 32UL) + _fuseiter3664)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0n__k_945 % 16UL) * 32UL) + _fuseiter3664)]);
          __cached_3 = (__cached_3 + __cached_5);
          __cached_3 = sc_max(__cached_3, vec_f32x16(0.f));
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_s8x16 __cached_7;
          __cached_7 = __cached_6;
          vec_s8x16::store(__cached_7, &__outs_0[(((fused_0n__k_945 / 16UL) * 41472UL) + ((((_fuseiter3664 + ((fused_0n__k_945 % 16UL) * 32UL)) / 256UL) * 20736UL) + (((_fuseiter3662 + 1UL) * 2304UL) + (((_fuseiter3663 + 1UL) * 256UL) + ((_fuseiter3664 + ((fused_0n__k_945 % 16UL) * 32UL)) % 256UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1010_shr);
  }
  return true;
}

static bool res5c_conv_1_cast_mul_add_relu_cast_reorder__674(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3) noexcept{
  void** __sc_kernel_cache_arr_100 = (void**)&__uninitialized_data[23657584UL];
  alignas(64) int8_t __rescheduled_0[128UL];
  int32_t* conv_os_blk_size = (int32_t*)&__rescheduled_0[0UL];
  int32_t* conv_os_acc_size = (int32_t*)&__rescheduled_0[64UL];
  int32_t __cached_0;
  __cached_0 = 49;
  conv_os_blk_size[0] = __cached_0;
  int32_t __cached_1;
  __cached_1 = 0;
  conv_os_acc_size[0] = __cached_1;
  for (uint64_t fused_0k_o__n_946 = 0UL; fused_0k_o__n_946 < 36UL; fused_0k_o__n_946 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 384UL);
    int32_t* __origouts_1020_shr = (int32_t*)sc_aligned_malloc(__stream, 25088UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[192UL];
    int32_t __cached_2;
    __cached_2 = conv_os_acc_size[0UL];
    int32_t __cached_3;
    __cached_3 = conv_os_blk_size[0UL];
    memset(&__origouts_1020_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 0, ((uint64_t)(__cached_3 * 128) * 4UL));
    for (uint64_t c_o = 0UL; c_o < 2UL; c_o += 1UL) {
      for (uint64_t r = 0UL; r < 3UL; r += 1UL) {
        for (uint64_t s = 0UL; s < 3UL; s += 1UL) {
          void* __cached_4;
          __cached_4 = &__ins_0[(((fused_0k_o__n_946 % 9UL) * 41472UL) + ((c_o * 20736UL) + ((r * 2304UL) + (s * 256UL))))];
          A_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_4;
          void* __cached_5;
          __cached_5 = &__ins_1[(((fused_0k_o__n_946 / 9UL) * 589824UL) + ((c_o * 294912UL) + ((r * 98304UL) + (s * 32768UL))))];
          B_list[(((c_o * 9UL) + (r * 3UL)) + s)] = __cached_5;
        }
      }
    }
    void* _arg_cache_50 = &__origouts_1020_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))];
    dnnl_brgemm_list_call(__sc_kernel_cache_arr_100[0UL], A_list, B_list, &__origouts_1020_shr[(uint64_t)(((__cached_2 / 7) * 896) + ((__cached_2 % 7) * 128))], 1, 256, 32768, 18, 7, 7, __stream);
    for (uint64_t _fuseiter3697 = 0UL; _fuseiter3697 < 7UL; _fuseiter3697 += 1UL) {
      for (uint64_t _fuseiter3698 = 0UL; _fuseiter3698 < 7UL; _fuseiter3698 += 1UL) {
        for (uint64_t _fuseiter3699 = 0UL; _fuseiter3699 < 128UL; _fuseiter3699 += 16UL) {
          vec_s32x16 __cached_6;
          __cached_6 = vec_s32x16::load(&__origouts_1020_shr[((_fuseiter3697 * 896UL) + ((_fuseiter3698 * 128UL) + _fuseiter3699))]);
          vec_f32x16 __cached_7;
          __cached_7 = (vec_f32x16)(__cached_6);
          vec_f32x16 __cached_8;
          __cached_8 = vec_f32x16::load(&__ins_2[(((fused_0k_o__n_946 / 9UL) * 128UL) + _fuseiter3699)]);
          __cached_7 = (__cached_7 * __cached_8);
          vec_f32x16 __cached_9;
          __cached_9 = vec_f32x16::load(&__ins_3[(((fused_0k_o__n_946 / 9UL) * 128UL) + _fuseiter3699)]);
          __cached_7 = (__cached_7 + __cached_9);
          __cached_7 = sc_max(__cached_7, vec_f32x16(0.f));
          vec_s8x16 __cached_10;
          __cached_10 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_7));
          vec_s8x16 __cached_11;
          __cached_11 = __cached_10;
          vec_s8x16::store(__cached_11, &__outs_0[(((fused_0k_o__n_946 % 9UL) * 25088UL) + ((((_fuseiter3699 + ((fused_0k_o__n_946 / 9UL) * 128UL)) / 512UL) * 25088UL) + ((_fuseiter3697 * 3584UL) + ((_fuseiter3698 * 512UL) + ((_fuseiter3699 + ((fused_0k_o__n_946 / 9UL) * 128UL)) % 512UL)))))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1020_shr);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

static bool mul__672(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 += 1UL) {
    for (uint64_t _fuseiter_3734 = 0UL; _fuseiter_3734 < 512UL; _fuseiter_3734 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 % 4UL) * 512UL)) + _fuseiter_3734)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_947____itr_2_948____itr_3_949 % 4UL) * 512UL)) + _fuseiter_3734)]);
    }
  }
  return true;
}

static bool mul__671(float* __restrict__ __outs_0, float* __restrict__ __ins_0, float* __restrict__ __ins_1) noexcept{
  for (uint64_t fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 = 0UL; fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 < 4UL; fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 += 1UL) {
    for (uint64_t _fuseiter_3740 = 0UL; _fuseiter_3740 < 512UL; _fuseiter_3740 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__ins_0[((((fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 % 4UL) * 512UL)) + _fuseiter_3740)]);
      float __cached_1;
      __cached_1 = __ins_1[0];
      vec_f32x16 __cached_2;
      __cached_2 = (__cached_0 * vec_f32x16(__cached_1));
      vec_f32x16::store(__cached_2, &__outs_0[((((fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 / 4UL) * 2048UL) + ((fused_0fused_0fused_0__itr_0____itr_1_950____itr_2_951____itr_3_952 % 4UL) * 512UL)) + _fuseiter_3740)]);
    }
  }
  return true;
}

static bool reorder__558(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 = 0UL; fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 < 512UL; fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 += 1UL) {
    for (uint64_t _fuseiter_3747 = 0UL; _fuseiter_3747 < 512UL; _fuseiter_3747 += 1UL) {
      for (uint64_t _fuseiter_3748 = 0UL; _fuseiter_3748 < 4UL; _fuseiter_3748 += 1UL) {
        int8_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_3747 + ((fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 / 128UL) * 512UL)) * 512UL) + (_fuseiter_3748 + ((fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 % 128UL) * 4UL)))];
        int8_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 / 128UL) * 262144UL) + (((fused_0fused_0fused_0fused_0_fuseiter_3742___fuseiter_3743_953___fuseiter_3744_954___fuseiter_3745_955___fuseiter_3746_956 % 128UL) * 2048UL) + ((_fuseiter_3747 * 4UL) + _fuseiter_3748)))] = __cached_1;
      }
    }
  }
  return true;
}

static bool res5c_conv_2_cast_mul_add_cast_add_cast_cast__673(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, uint8_t* __restrict__ __ins_4) noexcept{
  void*& __sc_kernel_cache_102 = *(void**)(__module_data + 328);
  for (uint64_t fused_0k__n_957 = 0UL; fused_0k__n_957 < 36UL; fused_0k__n_957 += 1UL) {
    alignas(64) int8_t __rescheduled_1[128UL];
    int32_t* __origouts_1030_shr = (int32_t*)sc_aligned_malloc(__stream, 100352UL);
    void** A_list = (void**)&__rescheduled_1[0UL];
    void** B_list = (void**)&__rescheduled_1[64UL];
    void* __cached_0;
    __cached_0 = &__ins_0[((fused_0k__n_957 % 9UL) * 25088UL)];
    A_list[0UL] = __cached_0;
    void* __cached_1;
    __cached_1 = &__ins_1[((fused_0k__n_957 / 9UL) * 262144UL)];
    B_list[0UL] = __cached_1;
    void* _arg_cache_51 = &__origouts_1030_shr[0UL];
    dnnl_brgemm_list_call(__sc_kernel_cache_102, A_list, B_list, &__origouts_1030_shr[0UL], 1, 1, 1, 1, 7, 7, __stream);
    for (uint64_t _fuseiter3751 = 0UL; _fuseiter3751 < 7UL; _fuseiter3751 += 1UL) {
      for (uint64_t _fuseiter3752 = 0UL; _fuseiter3752 < 7UL; _fuseiter3752 += 1UL) {
        for (uint64_t _fuseiter3753 = 0UL; _fuseiter3753 < 512UL; _fuseiter3753 += 16UL) {
          vec_s32x16 __cached_2;
          __cached_2 = vec_s32x16::load(&__origouts_1030_shr[((_fuseiter3751 * 3584UL) + ((_fuseiter3752 * 512UL) + _fuseiter3753))]);
          vec_f32x16 __cached_3;
          __cached_3 = (vec_f32x16)(__cached_2);
          vec_f32x16 __cached_4;
          __cached_4 = vec_f32x16::load(&__ins_2[(((fused_0k__n_957 / 9UL) * 512UL) + _fuseiter3753)]);
          __cached_3 = (__cached_3 * __cached_4);
          vec_f32x16 __cached_5;
          __cached_5 = vec_f32x16::load(&__ins_3[(((fused_0k__n_957 / 9UL) * 512UL) + _fuseiter3753)]);
          __cached_3 = (__cached_3 + __cached_5);
          vec_s8x16 __cached_6;
          __cached_6 = sc_saturated_cast<vec_s8x16>(sc_round_and_cast<vec_s32x16>(__cached_3));
          vec_u8x16 __cached_7;
          __cached_7 = vec_u8x16::load(&__ins_4[((((fused_0k__n_957 % 9UL) * 100352UL) + ((fused_0k__n_957 / 9UL) * 25088UL)) + ((_fuseiter3751 * 3584UL) + ((_fuseiter3752 * 512UL) + _fuseiter3753)))]);
          __cached_6 = (__cached_6 + (vec_s8x16)(__cached_7));
          vec_u8x16 __cached_8;
          __cached_8 = (vec_u8x16)(sc_max(__cached_6, vec_s8x16(0)));
          vec_s8x16 __cached_9;
          __cached_9 = (vec_s8x16)(__cached_8);
          vec_s8x16::store(__cached_9, &__outs_0[((((fused_0k__n_957 % 9UL) * 100352UL) + ((fused_0k__n_957 / 9UL) * 25088UL)) + ((_fuseiter3751 * 3584UL) + ((_fuseiter3752 * 512UL) + _fuseiter3753)))]);
        }
      }
    }
    sc_aligned_free(__stream, __origouts_1030_shr);
  }
  return true;
}

static bool reorder__105(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  for (uint64_t fused_0n__h_958 = 0UL; fused_0n__h_958 < 63UL; fused_0n__h_958 += 1UL) {
    for (uint64_t w = 0UL; w < 7UL; w += 1UL) {
      for (uint64_t c = 0UL; c < 2048UL; c += 64UL) {
        vec_s8x64 zmm0;
        vec_s8x64 __cached_0;
        __cached_0 = vec_s8x64::load(&__ins_0[(((fused_0n__h_958 / 7UL) * 100352UL) + (((c / 512UL) * 25088UL) + (((fused_0n__h_958 % 7UL) * 3584UL) + ((w * 512UL) + (c % 512UL)))))]);
        zmm0 = __cached_0;
        vec_s8x64 __cached_1;
        __cached_1 = zmm0;
        vec_s8x64::store(__cached_1, &__outs_0[(((fused_0n__h_958 / 7UL) * 100352UL) + (((fused_0n__h_958 % 7UL) * 14336UL) + ((w * 2048UL) + c)))]);
      }
    }
  }
  return true;
}

static void __init_const_globals(int8_t* __restrict__ backbone_output, int8_t* __restrict__ backbone_input, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
  float* folded_const_46 = (float*)&__module_data[91392UL];
  float* folded_const_41 = (float*)&__module_data[85120UL];
  float* folded_const_31 = (float*)&__module_data[72576UL];
  float* folded_const_26 = (float*)&__module_data[66304UL];
  float* folded_const_21 = (float*)&__module_data[60032UL];
  float* folded_const_16 = (float*)&__module_data[53760UL];
  float* folded_const_103 = (float*)&__module_data[115392UL];
  float* folded_const_80 = (float*)&__module_data[114752UL];
  float* folded_const_78 = (float*)&__module_data[113664UL];
  float* folded_const_75 = (float*)&__module_data[113088UL];
  float* folded_const_73 = (float*)&__module_data[112000UL];
  float* folded_const_72 = (float*)&__module_data[111744UL];
  float* folded_const_70 = (float*)&__module_data[111424UL];
  float* folded_const_68 = (float*)&__module_data[110336UL];
  float* folded_const_67 = (float*)&__module_data[108288UL];
  float* folded_const_66 = (float*)&__module_data[107776UL];
  float* folded_const_64 = (float*)&__module_data[107200UL];
  float* folded_const_62 = (float*)&__module_data[105088UL];
  float* folded_const_61 = (float*)&__module_data[104576UL];
  float* folded_const_57 = (float*)&__module_data[101888UL];
  float* folded_const_54 = (float*)&__module_data[100800UL];
  float* folded_const_52 = (float*)&__module_data[98688UL];
  float* folded_const_51 = (float*)&__module_data[98176UL];
  float* folded_const_49 = (float*)&__module_data[97600UL];
  float* folded_const_47 = (float*)&__module_data[95488UL];
  float* folded_const_43 = (float*)&__module_data[89280UL];
  float* folded_const_40 = (float*)&__module_data[84096UL];
  float* folded_const_38 = (float*)&__module_data[83008UL];
  float* folded_const_35 = (float*)&__module_data[77824UL];
  float* folded_const_33 = (float*)&__module_data[76736UL];
  float* folded_const_30 = (float*)&__module_data[71552UL];
  float* folded_const_28 = (float*)&__module_data[70464UL];
  float* folded_const_23 = (float*)&__module_data[64192UL];
  float* folded_const_20 = (float*)&__module_data[59008UL];
  float* folded_const_18 = (float*)&__module_data[57920UL];
  float* folded_const_15 = (float*)&__module_data[45568UL];
  float* folded_const_14 = (float*)&__module_data[43520UL];
  float* folded_const_12 = (float*)&__module_data[41408UL];
  float* folded_const_10 = (float*)&__module_data[33152UL];
  float* folded_const_9 = (float*)&__module_data[31104UL];
  float* folded_const_7 = (float*)&__module_data[28992UL];
  float* folded_const_5 = (float*)&__module_data[20736UL];
  float* folded_const_4 = (float*)&__module_data[18688UL];
  float* folded_const_2 = (float*)&__module_data[16576UL];
  float* folded_const_0 = (float*)&__module_data[8320UL];
  float* folded_const_154 = (float*)&__module_data[221120UL];
  float* folded_const_155 = (float*)&__module_data[221376UL];
  float* folded_const_152 = (float*)&__module_data[219840UL];
  float* folded_const_149 = (float*)&__module_data[218304UL];
  float* folded_const_146 = (float*)&__module_data[216768UL];
  float* folded_const_151 = (float*)&__module_data[219584UL];
  float* folded_const_148 = (float*)&__module_data[218048UL];
  float* folded_const_144 = (float*)&__module_data[214208UL];
  float* folded_const_153 = (float*)&__module_data[220864UL];
  float* folded_const_150 = (float*)&__module_data[219328UL];
  float* folded_const_147 = (float*)&__module_data[217792UL];
  float* folded_const_142 = (float*)&__module_data[211648UL];
  float* folded_const_139 = (float*)&__module_data[208576UL];
  float* folded_const_136 = (float*)&__module_data[205504UL];
  float* folded_const_133 = (float*)&__module_data[202432UL];
  float* folded_const_141 = (float*)&__module_data[211136UL];
  float* folded_const_138 = (float*)&__module_data[208064UL];
  float* folded_const_135 = (float*)&__module_data[204992UL];
  float* folded_const_145 = (float*)&__module_data[214720UL];
  float* folded_const_131 = (float*)&__module_data[197312UL];
  float* folded_const_143 = (float*)&__module_data[213696UL];
  float* folded_const_140 = (float*)&__module_data[210624UL];
  float* folded_const_137 = (float*)&__module_data[207552UL];
  float* folded_const_134 = (float*)&__module_data[204480UL];
  float* folded_const_129 = (float*)&__module_data[192192UL];
  float* folded_const_126 = (float*)&__module_data[186048UL];
  float* folded_const_123 = (float*)&__module_data[179904UL];
  float* folded_const_120 = (float*)&__module_data[173760UL];
  float* folded_const_117 = (float*)&__module_data[167616UL];
  float* folded_const_114 = (float*)&__module_data[161472UL];
  float* folded_const_128 = (float*)&__module_data[191168UL];
  float* folded_const_125 = (float*)&__module_data[185024UL];
  float* folded_const_122 = (float*)&__module_data[178880UL];
  float* folded_const_119 = (float*)&__module_data[172736UL];
  float* folded_const_116 = (float*)&__module_data[166592UL];
  float* folded_const_132 = (float*)&__module_data[198336UL];
  float* folded_const_112 = (float*)&__module_data[151232UL];
  float* folded_const_130 = (float*)&__module_data[196288UL];
  float* folded_const_127 = (float*)&__module_data[190144UL];
  float* folded_const_124 = (float*)&__module_data[184000UL];
  float* folded_const_121 = (float*)&__module_data[177856UL];
  float* folded_const_118 = (float*)&__module_data[171712UL];
  float* folded_const_115 = (float*)&__module_data[165568UL];
  int8_t* folded_const_156 = (int8_t*)&__uninitialized_data[0UL];
  float* folded_const_157 = (float*)&__uninitialized_data[589824UL];
  float* folded_const_86 = (float*)&__module_data[115028UL];
  float* folded_const_158 = (float*)&__uninitialized_data[593920UL];
  float* folded_const_159 = (float*)&__uninitialized_data[598016UL];
  float* folded_const_87 = (float*)&__module_data[115032UL];
  float* folded_const_160 = (float*)&__uninitialized_data[602112UL];
  float* folded_const_161 = (float*)&__uninitialized_data[606208UL];
  float* folded_const_88 = (float*)&__module_data[115036UL];
  float* folded_const_162 = (float*)&__uninitialized_data[610304UL];
  float* folded_const_163 = (float*)&__uninitialized_data[614400UL];
  float* folded_const_89 = (float*)&__module_data[115040UL];
  float* folded_const_164 = (float*)&__uninitialized_data[618496UL];
  float* folded_const_165 = (float*)&__uninitialized_data[622592UL];
  float* folded_const_90 = (float*)&__module_data[115044UL];
  float* folded_const_166 = (float*)&__uninitialized_data[626688UL];
  float* folded_const_36 = (float*)&__module_data[78848UL];
  float* folded_const_167 = (float*)&__uninitialized_data[630784UL];
  float* folded_const_91 = (float*)&__module_data[115048UL];
  float* folded_const_168 = (float*)&__uninitialized_data[634880UL];
  float* folded_const_169 = (float*)&__uninitialized_data[638976UL];
  float* folded_const_92 = (float*)&__module_data[115052UL];
  float* folded_const_170 = (float*)&__uninitialized_data[643072UL];
  float* folded_const_171 = (float*)&__uninitialized_data[647168UL];
  float* folded_const_93 = (float*)&__module_data[115056UL];
  float* folded_const_172 = (float*)&__uninitialized_data[649216UL];
  float* folded_const_173 = (float*)&__uninitialized_data[651264UL];
  float* folded_const_94 = (float*)&__module_data[115060UL];
  float* folded_const_174 = (float*)&__uninitialized_data[653312UL];
  float* folded_const_175 = (float*)&__uninitialized_data[655360UL];
  float* folded_const_95 = (float*)&__module_data[115064UL];
  float* folded_const_176 = (float*)&__uninitialized_data[657408UL];
  float* folded_const_177 = (float*)&__uninitialized_data[659456UL];
  float* folded_const_96 = (float*)&__module_data[115068UL];
  float* folded_const_178 = (float*)&__uninitialized_data[661504UL];
  float* folded_const_179 = (float*)&__uninitialized_data[663552UL];
  float* folded_const_97 = (float*)&__module_data[115072UL];
  float* folded_const_180 = (float*)&__uninitialized_data[665600UL];
  float* folded_const_181 = (float*)&__uninitialized_data[667648UL];
  float* folded_const_17 = (float*)&__module_data[57856UL];
  float* folded_const_182 = (float*)&__uninitialized_data[668672UL];
  float* folded_const_183 = (float*)&__uninitialized_data[669696UL];
  float* folded_const_19 = (float*)&__module_data[58944UL];
  float* folded_const_184 = (float*)&__uninitialized_data[670720UL];
  float* folded_const_185 = (float*)&__uninitialized_data[671744UL];
  float* folded_const_22 = (float*)&__module_data[64128UL];
  float* folded_const_186 = (float*)&__uninitialized_data[672768UL];
  float* folded_const_187 = (float*)&__uninitialized_data[673792UL];
  float* folded_const_24 = (float*)&__module_data[65216UL];
  float* folded_const_188 = (float*)&__uninitialized_data[674816UL];
  float* folded_const_25 = (float*)&__module_data[65280UL];
  float* folded_const_189 = (float*)&__uninitialized_data[675840UL];
  float* folded_const_27 = (float*)&__module_data[70400UL];
  float* folded_const_190 = (float*)&__uninitialized_data[676864UL];
  float* folded_const_191 = (float*)&__uninitialized_data[677888UL];
  float* folded_const_29 = (float*)&__module_data[71488UL];
  float* folded_const_192 = (float*)&__uninitialized_data[678912UL];
  float* folded_const_193 = (float*)&__uninitialized_data[679936UL];
  float* folded_const_32 = (float*)&__module_data[76672UL];
  float* folded_const_194 = (float*)&__uninitialized_data[680960UL];
  float* folded_const_195 = (float*)&__uninitialized_data[681984UL];
  float* folded_const_34 = (float*)&__module_data[77760UL];
  float* folded_const_196 = (float*)&__uninitialized_data[683008UL];
  float* folded_const_197 = (float*)&__uninitialized_data[684032UL];
  float* folded_const_37 = (float*)&__module_data[82944UL];
  float* folded_const_198 = (float*)&__uninitialized_data[685056UL];
  float* folded_const_199 = (float*)&__uninitialized_data[686080UL];
  float* folded_const_39 = (float*)&__module_data[84032UL];
  float* folded_const_200 = (float*)&__uninitialized_data[687104UL];
  float* folded_const_201 = (float*)&__uninitialized_data[688128UL];
  float* folded_const_42 = (float*)&__module_data[89216UL];
  float* folded_const_202 = (float*)&__uninitialized_data[689152UL];
  float* folded_const_203 = (float*)&__uninitialized_data[690176UL];
  float* folded_const_44 = (float*)&__module_data[90304UL];
  float* folded_const_204 = (float*)&__uninitialized_data[691200UL];
  float* folded_const_45 = (float*)&__module_data[90368UL];
  float* folded_const_205 = (float*)&__uninitialized_data[692224UL];
  float* folded_const_98 = (float*)&__module_data[115076UL];
  float* folded_const_206 = (float*)&__uninitialized_data[693248UL];
  float* folded_const_207 = (float*)&__uninitialized_data[694272UL];
  float* folded_const_99 = (float*)&__module_data[115080UL];
  float* folded_const_208 = (float*)&__uninitialized_data[695296UL];
  float* folded_const_209 = (float*)&__uninitialized_data[696320UL];
  float* folded_const_100 = (float*)&__module_data[115084UL];
  float* folded_const_210 = (float*)&__uninitialized_data[697344UL];
  float* folded_const_211 = (float*)&__uninitialized_data[698368UL];
  float* folded_const_101 = (float*)&__module_data[115088UL];
  float* folded_const_212 = (float*)&__uninitialized_data[699392UL];
  int8_t* folded_const_213 = (int8_t*)&__uninitialized_data[700416UL];
  int8_t* folded_const_214 = (int8_t*)&__uninitialized_data[962560UL];
  int8_t* folded_const_215 = (int8_t*)&__uninitialized_data[1224704UL];
  int8_t* folded_const_216 = (int8_t*)&__uninitialized_data[1486848UL];
  int8_t* folded_const_217 = (int8_t*)&__uninitialized_data[1748992UL];
  int8_t* folded_const_218 = (int8_t*)&__uninitialized_data[2011136UL];
  int8_t* folded_const_219 = (int8_t*)&__uninitialized_data[2273280UL];
  int8_t* folded_const_220 = (int8_t*)&__uninitialized_data[2535424UL];
  int8_t* folded_const_221 = (int8_t*)&__uninitialized_data[2797568UL];
  int8_t* folded_const_222 = (int8_t*)&__uninitialized_data[3059712UL];
  int8_t* folded_const_223 = (int8_t*)&__uninitialized_data[3321856UL];
  int8_t* folded_const_224 = (int8_t*)&__uninitialized_data[3584000UL];
  float* folded_const_225 = (float*)&__uninitialized_data[3588096UL];
  float* folded_const_48 = (float*)&__module_data[97536UL];
  float* folded_const_226 = (float*)&__uninitialized_data[3588608UL];
  float* folded_const_227 = (float*)&__uninitialized_data[3589120UL];
  float* folded_const_50 = (float*)&__module_data[98112UL];
  float* folded_const_228 = (float*)&__uninitialized_data[3589632UL];
  float* folded_const_229 = (float*)&__uninitialized_data[3590144UL];
  float* folded_const_53 = (float*)&__module_data[100736UL];
  float* folded_const_230 = (float*)&__uninitialized_data[3590656UL];
  float* folded_const_231 = (float*)&__uninitialized_data[3591168UL];
  float* folded_const_55 = (float*)&__module_data[101312UL];
  float* folded_const_232 = (float*)&__uninitialized_data[3591680UL];
  float* folded_const_56 = (float*)&__module_data[101376UL];
  float* folded_const_233 = (float*)&__uninitialized_data[3592192UL];
  float* folded_const_58 = (float*)&__module_data[103936UL];
  float* folded_const_234 = (float*)&__uninitialized_data[3592704UL];
  float* folded_const_59 = (float*)&__module_data[104000UL];
  float* folded_const_235 = (float*)&__uninitialized_data[3593216UL];
  float* folded_const_60 = (float*)&__module_data[104512UL];
  float* folded_const_236 = (float*)&__uninitialized_data[3593728UL];
  float* folded_const_237 = (float*)&__uninitialized_data[3594240UL];
  float* folded_const_63 = (float*)&__module_data[107136UL];
  float* folded_const_238 = (float*)&__uninitialized_data[3594752UL];
  float* folded_const_239 = (float*)&__uninitialized_data[3595264UL];
  float* folded_const_65 = (float*)&__module_data[107712UL];
  float* folded_const_240 = (float*)&__uninitialized_data[3595776UL];
  float* folded_const_241 = (float*)&__uninitialized_data[3596288UL];
  float* folded_const_69 = (float*)&__module_data[111360UL];
  float* folded_const_242 = (float*)&__uninitialized_data[3596544UL];
  float* folded_const_243 = (float*)&__uninitialized_data[3596800UL];
  float* folded_const_71 = (float*)&__module_data[111680UL];
  float* folded_const_244 = (float*)&__uninitialized_data[3597056UL];
  float* folded_const_245 = (float*)&__uninitialized_data[3597312UL];
  float* folded_const_74 = (float*)&__module_data[113024UL];
  float* folded_const_246 = (float*)&__uninitialized_data[3597568UL];
  float* folded_const_247 = (float*)&__uninitialized_data[3597824UL];
  float* folded_const_76 = (float*)&__module_data[113344UL];
  float* folded_const_248 = (float*)&__uninitialized_data[3598080UL];
  float* folded_const_77 = (float*)&__module_data[113408UL];
  float* folded_const_249 = (float*)&__uninitialized_data[3598336UL];
  float* folded_const_79 = (float*)&__module_data[114688UL];
  float* folded_const_250 = (float*)&__uninitialized_data[3598592UL];
  float* folded_const_251 = (float*)&__uninitialized_data[3598848UL];
  float* folded_const_81 = (float*)&__module_data[115008UL];
  float* folded_const_252 = (float*)&__uninitialized_data[3599104UL];
  float* folded_const_102 = (float*)&__module_data[115136UL];
  int8_t* folded_const_253 = (int8_t*)&__uninitialized_data[3599360UL];
  int8_t* folded_const_254 = (int8_t*)&__uninitialized_data[4189184UL];
  int8_t* folded_const_255 = (int8_t*)&__uninitialized_data[4779008UL];
  int8_t* folded_const_256 = (int8_t*)&__uninitialized_data[5368832UL];
  int8_t* folded_const_257 = (int8_t*)&__uninitialized_data[5958656UL];
  int8_t* folded_const_258 = (int8_t*)&__uninitialized_data[6548480UL];
  int8_t* folded_const_259 = (int8_t*)&__uninitialized_data[7072768UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[7220224UL];
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[7367680UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[7515136UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[7662592UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[7793664UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[7924736UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[7990272UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[8055808UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[8121344UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[8186880UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[8252416UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[8317952UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[8383488UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[8420352UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[8457216UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[8494080UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[8526848UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[8543232UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[8559616UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[8576000UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[8592384UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[8608768UL];
  float* folded_const_282 = (float*)&__uninitialized_data[8625152UL];
  float* folded_const_13 = (float*)&__module_data[43456UL];
  float* folded_const_283 = (float*)&__uninitialized_data[8627200UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[8629248UL];
  float* folded_const_110 = (float*)&__module_data[140992UL];
  float* folded_const_107 = (float*)&__module_data[128704UL];
  float* folded_const_104 = (float*)&__module_data[116416UL];
  float* folded_const_109 = (float*)&__module_data[138944UL];
  float* folded_const_106 = (float*)&__module_data[126656UL];
  float* folded_const_113 = (float*)&__module_data[153280UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[9153536UL];
  float* folded_const_286 = (float*)&__uninitialized_data[11250688UL];
  float* folded_const_85 = (float*)&__module_data[115024UL];
  float* folded_const_287 = (float*)&__uninitialized_data[11258880UL];
  float* folded_const_111 = (float*)&__module_data[149184UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[11267072UL];
  float* folded_const_289 = (float*)&__uninitialized_data[13626368UL];
  float* folded_const_11 = (float*)&__module_data[41344UL];
  float* folded_const_290 = (float*)&__uninitialized_data[13628416UL];
  float* folded_const_291 = (float*)&__uninitialized_data[13630464UL];
  float* folded_const_84 = (float*)&__module_data[115020UL];
  float* folded_const_292 = (float*)&__uninitialized_data[13638656UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[13646848UL];
  float* folded_const_294 = (float*)&__uninitialized_data[14695424UL];
  float* folded_const_8 = (float*)&__module_data[31040UL];
  float* folded_const_295 = (float*)&__uninitialized_data[14697472UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[14699520UL];
  float* folded_const_108 = (float*)&__module_data[136896UL];
  float* folded_const_105 = (float*)&__module_data[124608UL];
  float* folded_const_297 = (float*)&__uninitialized_data[15748096UL];
  float* folded_const_6 = (float*)&__module_data[28928UL];
  float* folded_const_298 = (float*)&__uninitialized_data[15750144UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[15752192UL];
  float* folded_const_300 = (float*)&__uninitialized_data[18111488UL];
  float* folded_const_83 = (float*)&__module_data[115016UL];
  float* folded_const_301 = (float*)&__uninitialized_data[18119680UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[18127872UL];
  float* folded_const_303 = (float*)&__uninitialized_data[19176448UL];
  float* folded_const_3 = (float*)&__module_data[18624UL];
  float* folded_const_304 = (float*)&__uninitialized_data[19178496UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[19180544UL];
  float* folded_const_306 = (float*)&__uninitialized_data[20229120UL];
  float* folded_const_1 = (float*)&__module_data[16512UL];
  float* folded_const_307 = (float*)&__uninitialized_data[20231168UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[20233216UL];
  float* folded_const_309 = (float*)&__uninitialized_data[22592512UL];
  float* folded_const_82 = (float*)&__module_data[115012UL];
  float* folded_const_310 = (float*)&__uninitialized_data[22600704UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[22608896UL];
  bool& is_init = *(bool*)(__module_data + 0);
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 18546688UL);
  // [f32 [1, 32, 1, 1, 32] @ ABCD32b]
  float* buffer_261 = (float*)&__rescheduled_0[0UL];
  reorder__481(buffer_261, folded_const_46);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_262 = (float*)&__rescheduled_0[4096UL];
  reorder__488(buffer_262, folded_const_41);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_263 = (float*)&__rescheduled_0[8192UL];
  reorder__504(buffer_263, folded_const_31);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_264 = (float*)&__rescheduled_0[12288UL];
  reorder__513(buffer_264, folded_const_26);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_265 = (float*)&__rescheduled_0[16384UL];
  reorder__520(buffer_265, folded_const_21);
  // [f32 [1, 4, 1, 1, 256] @ ABCD256b]
  float* buffer_266 = (float*)&__rescheduled_0[20480UL];
  reorder__529(buffer_266, folded_const_16);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_267 = (float*)&__rescheduled_0[24576UL];
  reorder__420(buffer_267, folded_const_103);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_268 = (float*)&__rescheduled_0[25600UL];
  reorder__424(buffer_268, folded_const_80);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_269 = (float*)&__rescheduled_0[25856UL];
  reorder__427(buffer_269, folded_const_78);
  // [f32 [1, 4, 1, 1, 16] @ ABCD16b]
  float* buffer_270 = (float*)&__rescheduled_0[26880UL];
  reorder__431(buffer_270, folded_const_75);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_271 = (float*)&__rescheduled_0[27136UL];
  reorder__434(buffer_271, folded_const_73);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_272 = (float*)&__rescheduled_0[28160UL];
  reorder__437(buffer_272, folded_const_72);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_273 = (float*)&__rescheduled_0[28416UL];
  reorder__440(buffer_273, folded_const_70);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_274 = (float*)&__rescheduled_0[28672UL];
  reorder__443(buffer_274, folded_const_68);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_275 = (float*)&__rescheduled_0[29696UL];
  reorder__446(buffer_275, folded_const_67);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_276 = (float*)&__rescheduled_0[31744UL];
  reorder__449(buffer_276, folded_const_66);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_277 = (float*)&__rescheduled_0[32256UL];
  reorder__452(buffer_277, folded_const_64);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_278 = (float*)&__rescheduled_0[32768UL];
  reorder__455(buffer_278, folded_const_62);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_279 = (float*)&__rescheduled_0[34816UL];
  reorder__458(buffer_279, folded_const_61);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_280 = (float*)&__rescheduled_0[35328UL];
  reorder__462(buffer_280, folded_const_57);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_281 = (float*)&__rescheduled_0[37376UL];
  reorder__466(buffer_281, folded_const_54);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_282 = (float*)&__rescheduled_0[37888UL];
  reorder__469(buffer_282, folded_const_52);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_283 = (float*)&__rescheduled_0[39936UL];
  reorder__472(buffer_283, folded_const_51);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_284 = (float*)&__rescheduled_0[40448UL];
  reorder__475(buffer_284, folded_const_49);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_285 = (float*)&__rescheduled_0[40960UL];
  reorder__478(buffer_285, folded_const_47);
  // [f32 [1, 2, 1, 1, 128] @ ABCD128b]
  float* buffer_286 = (float*)&__rescheduled_0[43008UL];
  reorder__485(buffer_286, folded_const_43);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_287 = (float*)&__rescheduled_0[44032UL];
  reorder__491(buffer_287, folded_const_40);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_288 = (float*)&__rescheduled_0[45056UL];
  reorder__494(buffer_288, folded_const_38);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_289 = (float*)&__rescheduled_0[46080UL];
  reorder__498(buffer_289, folded_const_35);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_290 = (float*)&__rescheduled_0[47104UL];
  reorder__501(buffer_290, folded_const_33);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_291 = (float*)&__rescheduled_0[48128UL];
  reorder__507(buffer_291, folded_const_30);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_292 = (float*)&__rescheduled_0[49152UL];
  reorder__510(buffer_292, folded_const_28);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_293 = (float*)&__rescheduled_0[50176UL];
  reorder__517(buffer_293, folded_const_23);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_294 = (float*)&__rescheduled_0[51200UL];
  reorder__523(buffer_294, folded_const_20);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_295 = (float*)&__rescheduled_0[52224UL];
  reorder__526(buffer_295, folded_const_18);
  // [f32 [1, 128, 1, 1, 16] @ ABCD16b]
  float* buffer_296 = (float*)&__rescheduled_0[53248UL];
  reorder__532(buffer_296, folded_const_15);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_297 = (float*)&__rescheduled_0[61440UL];
  reorder__535(buffer_297, folded_const_14);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_298 = (float*)&__rescheduled_0[63488UL];
  reorder__538(buffer_298, folded_const_12);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_299 = (float*)&__rescheduled_0[65536UL];
  reorder__541(buffer_299, folded_const_10);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_300 = (float*)&__rescheduled_0[73728UL];
  reorder__544(buffer_300, folded_const_9);
  // [f32 [1, 32, 1, 1, 16] @ ABCD16b]
  float* buffer_301 = (float*)&__rescheduled_0[75776UL];
  reorder__547(buffer_301, folded_const_7);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_302 = (float*)&__rescheduled_0[77824UL];
  reorder__550(buffer_302, folded_const_5);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_303 = (float*)&__rescheduled_0[86016UL];
  reorder__553(buffer_303, folded_const_4);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_304 = (float*)&__rescheduled_0[88064UL];
  reorder__556(buffer_304, folded_const_2);
  // [f32 [1, 4, 1, 1, 512] @ ABCD512b]
  float* buffer_305 = (float*)&__rescheduled_0[90112UL];
  reorder__559(buffer_305, folded_const_0);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_306 = (float*)&__rescheduled_0[98304UL];
  reorder__425(buffer_306, &res2a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 16] @ ABCD16b]
  float* buffer_307 = (float*)&__rescheduled_0[98560UL];
  reorder__432(buffer_307, &res2b_bias_1[0]);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_308 = (float*)&__rescheduled_0[98816UL];
  reorder__438(buffer_308, &res2c_bias_0[0]);
  // [f32 [1, 2, 1, 1, 32] @ ABCD32b]
  float* buffer_309 = (float*)&__rescheduled_0[99072UL];
  reorder__441(buffer_309, &res2c_bias_1[0]);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_310 = (float*)&__rescheduled_0[99328UL];
  reorder__450(buffer_310, &res3a_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_311 = (float*)&__rescheduled_0[99840UL];
  reorder__453(buffer_311, &res3a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_312 = (float*)&__rescheduled_0[100352UL];
  reorder__459(buffer_312, &res3b_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_313 = (float*)&__rescheduled_0[100864UL];
  reorder__467(buffer_313, &res3c_bias_1[0]);
  // [f32 [1, 2, 1, 1, 64] @ ABCD64b]
  float* buffer_314 = (float*)&__rescheduled_0[101376UL];
  reorder__473(buffer_314, &res3d_bias_0[0]);
  // [f32 [1, 4, 1, 1, 32] @ ABCD32b]
  float* buffer_315 = (float*)&__rescheduled_0[101888UL];
  reorder__476(buffer_315, &res3d_bias_1[0]);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_316 = (float*)&__rescheduled_0[102400UL];
  reorder__421(buffer_316, &res2a_bias_b[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_317 = (float*)&__rescheduled_0[103424UL];
  reorder__428(buffer_317, &res2a_bias_2[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_318 = (float*)&__rescheduled_0[104448UL];
  reorder__435(buffer_318, &res2b_bias_2[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_319 = (float*)&__rescheduled_0[105472UL];
  reorder__444(buffer_319, &res2c_bias_2[0]);
  // [f32 [1, 2, 1, 1, 128] @ ABCD128b]
  float* buffer_320 = (float*)&__rescheduled_0[106496UL];
  reorder__486(buffer_320, &res4a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_321 = (float*)&__rescheduled_0[107520UL];
  reorder__492(buffer_321, &res4b_bias_0[0]);
  // [f32 [1, 16, 1, 1, 16] @ ABCD16b]
  float* buffer_322 = (float*)&__rescheduled_0[108544UL];
  reorder__495(buffer_322, &res4b_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_323 = (float*)&__rescheduled_0[109568UL];
  reorder__499(buffer_323, &res4c_bias_0[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_324 = (float*)&__rescheduled_0[110592UL];
  reorder__502(buffer_324, &res4c_bias_1[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_325 = (float*)&__rescheduled_0[111616UL];
  reorder__508(buffer_325, &res4d_bias_0[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_326 = (float*)&__rescheduled_0[112640UL];
  reorder__511(buffer_326, &res4d_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_327 = (float*)&__rescheduled_0[113664UL];
  reorder__518(buffer_327, &res4e_bias_1[0]);
  // [f32 [1, 8, 1, 1, 32] @ ABCD32b]
  float* buffer_328 = (float*)&__rescheduled_0[114688UL];
  reorder__524(buffer_328, &res4f_bias_0[0]);
  // [f32 [1, 4, 1, 1, 64] @ ABCD64b]
  float* buffer_329 = (float*)&__rescheduled_0[115712UL];
  reorder__527(buffer_329, &res4f_bias_1[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_330 = (float*)&__rescheduled_0[116736UL];
  reorder__447(buffer_330, &res3a_bias_b[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_331 = (float*)&__rescheduled_0[118784UL];
  reorder__456(buffer_331, &res3a_bias_2[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_332 = (float*)&__rescheduled_0[120832UL];
  reorder__463(buffer_332, &res3b_bias_2[0]);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_333 = (float*)&__rescheduled_0[122880UL];
  reorder__470(buffer_333, &res3c_bias_2[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_334 = (float*)&__rescheduled_0[124928UL];
  reorder__479(buffer_334, &res3d_bias_2[0]);
  // [f32 [1, 8, 1, 1, 64] @ ABCD64b]
  float* buffer_335 = (float*)&__rescheduled_0[126976UL];
  reorder__536(buffer_335, &res5a_bias_0[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_336 = (float*)&__rescheduled_0[129024UL];
  reorder__539(buffer_336, &res5a_bias_1[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_337 = (float*)&__rescheduled_0[131072UL];
  reorder__545(buffer_337, &res5b_bias_0[0]);
  // [f32 [1, 32, 1, 1, 16] @ ABCD16b]
  float* buffer_338 = (float*)&__rescheduled_0[133120UL];
  reorder__548(buffer_338, &res5b_bias_1[0]);
  // [f32 [1, 16, 1, 1, 32] @ ABCD32b]
  float* buffer_339 = (float*)&__rescheduled_0[135168UL];
  reorder__554(buffer_339, &res5c_bias_0[0]);
  // [f32 [1, 4, 1, 1, 128] @ ABCD128b]
  float* buffer_340 = (float*)&__rescheduled_0[137216UL];
  reorder__557(buffer_340, &res5c_bias_1[0]);
  // [f32 [1, 32, 1, 1, 32] @ ABCD32b]
  float* buffer_341 = (float*)&__rescheduled_0[139264UL];
  reorder__482(buffer_341, &res4a_bias_b[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_342 = (float*)&__rescheduled_0[143360UL];
  reorder__489(buffer_342, &res4a_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_343 = (float*)&__rescheduled_0[147456UL];
  reorder__505(buffer_343, &res4c_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_344 = (float*)&__rescheduled_0[151552UL];
  reorder__514(buffer_344, &res4d_bias_2[0]);
  // [f32 [1, 8, 1, 1, 128] @ ABCD128b]
  float* buffer_345 = (float*)&__rescheduled_0[155648UL];
  reorder__521(buffer_345, &res4e_bias_2[0]);
  // [f32 [1, 4, 1, 1, 256] @ ABCD256b]
  float* buffer_346 = (float*)&__rescheduled_0[159744UL];
  reorder__530(buffer_346, &res4f_bias_2[0]);
  // [f32 [1, 128, 1, 1, 16] @ ABCD16b]
  float* buffer_347 = (float*)&__rescheduled_0[163840UL];
  reorder__533(buffer_347, &res5a_bias_b[0]);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_348 = (float*)&__rescheduled_0[172032UL];
  reorder__542(buffer_348, &res5a_bias_2[0]);
  // [f32 [1, 32, 1, 1, 64] @ ABCD64b]
  float* buffer_349 = (float*)&__rescheduled_0[180224UL];
  reorder__551(buffer_349, &res5b_bias_2[0]);
  // [f32 [1, 4, 1, 1, 512] @ ABCD512b]
  float* buffer_350 = (float*)&__rescheduled_0[188416UL];
  reorder__560(buffer_350, &res5c_bias_2[0]);
  // [f32 [64, 64, 1, 1] @ ABCD]
  float* buffer_351 = (float*)&__rescheduled_0[196608UL];
  mul__111(buffer_351, res2a_weight_0, folded_const_154);
  // [s8 [64, 64, 1, 1] @ ABCD]
  int8_t* buffer_352 = (int8_t*)&__rescheduled_0[2555904UL];
  cast__112(buffer_352, buffer_351);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_353 = (float*)&__rescheduled_0[196608UL];
  mul__108(buffer_353, res2a_weight_b, folded_const_155);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_354 = (int8_t*)&__rescheduled_0[2560000UL];
  cast__109(buffer_354, buffer_353);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_355 = (float*)&__rescheduled_0[196608UL];
  mul__117(buffer_355, res2a_weight_2, folded_const_152);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_356 = (int8_t*)&__rescheduled_0[2576384UL];
  cast__118(buffer_356, buffer_355);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_357 = (float*)&__rescheduled_0[196608UL];
  mul__126(buffer_357, res2b_weight_2, folded_const_149);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_358 = (int8_t*)&__rescheduled_0[2592768UL];
  cast__127(buffer_358, buffer_357);
  // [f32 [256, 64, 1, 1] @ ABCD]
  float* buffer_359 = (float*)&__rescheduled_0[196608UL];
  mul__135(buffer_359, res2c_weight_2, folded_const_146);
  // [s8 [256, 64, 1, 1] @ ABCD]
  int8_t* buffer_360 = (int8_t*)&__rescheduled_0[2609152UL];
  cast__136(buffer_360, buffer_359);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_361 = (float*)&__rescheduled_0[196608UL];
  mul__120(buffer_361, res2b_weight_0, folded_const_151);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_362 = (int8_t*)&__rescheduled_0[2625536UL];
  cast__121(buffer_362, buffer_361);
  // [f32 [64, 256, 1, 1] @ ABCD]
  float* buffer_363 = (float*)&__rescheduled_0[196608UL];
  mul__129(buffer_363, res2c_weight_0, folded_const_148);
  // [s8 [64, 256, 1, 1] @ ABCD]
  int8_t* buffer_364 = (int8_t*)&__rescheduled_0[2641920UL];
  cast__130(buffer_364, buffer_363);
  // [f32 [128, 256, 1, 1] @ ABCD]
  float* buffer_365 = (float*)&__rescheduled_0[196608UL];
  mul__141(buffer_365, res3a_weight_0, folded_const_144);
  // [s8 [128, 256, 1, 1] @ ABCD]
  int8_t* buffer_366 = (int8_t*)&__rescheduled_0[2658304UL];
  cast__142(buffer_366, buffer_365);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_367 = (float*)&__rescheduled_0[196608UL];
  mul__114(buffer_367, res2a_weight_1, folded_const_153);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_368 = (int8_t*)&__rescheduled_0[2691072UL];
  cast__115(buffer_368, buffer_367);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_369 = (float*)&__rescheduled_0[196608UL];
  mul__123(buffer_369, res2b_weight_1, folded_const_150);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_370 = (int8_t*)&__rescheduled_0[2727936UL];
  cast__124(buffer_370, buffer_369);
  // [f32 [64, 64, 3, 3] @ ABCD]
  float* buffer_371 = (float*)&__rescheduled_0[196608UL];
  mul__132(buffer_371, res2c_weight_1, folded_const_147);
  // [s8 [64, 64, 3, 3] @ ABCD]
  int8_t* buffer_372 = (int8_t*)&__rescheduled_0[2764800UL];
  cast__133(buffer_372, buffer_371);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_373 = (float*)&__rescheduled_0[196608UL];
  mul__147(buffer_373, res3a_weight_2, folded_const_142);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_374 = (int8_t*)&__rescheduled_0[2801664UL];
  cast__148(buffer_374, buffer_373);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_375 = (float*)&__rescheduled_0[196608UL];
  mul__156(buffer_375, res3b_weight_2, folded_const_139);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_376 = (int8_t*)&__rescheduled_0[2867200UL];
  cast__157(buffer_376, buffer_375);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_377 = (float*)&__rescheduled_0[196608UL];
  mul__165(buffer_377, res3c_weight_2, folded_const_136);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_378 = (int8_t*)&__rescheduled_0[2932736UL];
  cast__166(buffer_378, buffer_377);
  // [f32 [512, 128, 1, 1] @ ABCD]
  float* buffer_379 = (float*)&__rescheduled_0[196608UL];
  mul__174(buffer_379, res3d_weight_2, folded_const_133);
  // [s8 [512, 128, 1, 1] @ ABCD]
  int8_t* buffer_380 = (int8_t*)&__rescheduled_0[2998272UL];
  cast__175(buffer_380, buffer_379);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_381 = (float*)&__rescheduled_0[196608UL];
  mul__150(buffer_381, res3b_weight_0, folded_const_141);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_382 = (int8_t*)&__rescheduled_0[3063808UL];
  cast__151(buffer_382, buffer_381);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_383 = (float*)&__rescheduled_0[196608UL];
  mul__159(buffer_383, res3c_weight_0, folded_const_138);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_384 = (int8_t*)&__rescheduled_0[3129344UL];
  cast__160(buffer_384, buffer_383);
  // [f32 [128, 512, 1, 1] @ ABCD]
  float* buffer_385 = (float*)&__rescheduled_0[196608UL];
  mul__168(buffer_385, res3d_weight_0, folded_const_135);
  // [s8 [128, 512, 1, 1] @ ABCD]
  int8_t* buffer_386 = (int8_t*)&__rescheduled_0[3194880UL];
  cast__169(buffer_386, buffer_385);
  // [f32 [512, 256, 1, 1] @ ABCD]
  float* buffer_387 = (float*)&__rescheduled_0[196608UL];
  mul__138(buffer_387, res3a_weight_b, folded_const_145);
  // [s8 [512, 256, 1, 1] @ ABCD]
  int8_t* buffer_388 = (int8_t*)&__rescheduled_0[3260416UL];
  cast__139(buffer_388, buffer_387);
  // [f32 [256, 512, 1, 1] @ ABCD]
  float* buffer_389 = (float*)&__rescheduled_0[196608UL];
  mul__180(buffer_389, res4a_weight_0, folded_const_131);
  // [s8 [256, 512, 1, 1] @ ABCD]
  int8_t* buffer_390 = (int8_t*)&__rescheduled_0[3391488UL];
  cast__181(buffer_390, buffer_389);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_391 = (float*)&__rescheduled_0[196608UL];
  mul__144(buffer_391, res3a_weight_1, folded_const_143);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_392 = (int8_t*)&__rescheduled_0[3522560UL];
  cast__145(buffer_392, buffer_391);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_393 = (float*)&__rescheduled_0[196608UL];
  mul__153(buffer_393, res3b_weight_1, folded_const_140);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_394 = (int8_t*)&__rescheduled_0[3670016UL];
  cast__154(buffer_394, buffer_393);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_395 = (float*)&__rescheduled_0[196608UL];
  mul__162(buffer_395, res3c_weight_1, folded_const_137);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_396 = (int8_t*)&__rescheduled_0[3817472UL];
  cast__163(buffer_396, buffer_395);
  // [f32 [128, 128, 3, 3] @ ABCD]
  float* buffer_397 = (float*)&__rescheduled_0[196608UL];
  mul__171(buffer_397, res3d_weight_1, folded_const_134);
  // [s8 [128, 128, 3, 3] @ ABCD]
  int8_t* buffer_398 = (int8_t*)&__rescheduled_0[3964928UL];
  cast__172(buffer_398, buffer_397);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_399 = (float*)&__rescheduled_0[196608UL];
  mul__186(buffer_399, res4a_weight_2, folded_const_129);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_400 = (int8_t*)&__rescheduled_0[4112384UL];
  cast__187(buffer_400, buffer_399);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_401 = (float*)&__rescheduled_0[196608UL];
  mul__195(buffer_401, res4b_weight_2, folded_const_126);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_402 = (int8_t*)&__rescheduled_0[4374528UL];
  cast__196(buffer_402, buffer_401);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_403 = (float*)&__rescheduled_0[196608UL];
  mul__204(buffer_403, res4c_weight_2, folded_const_123);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_404 = (int8_t*)&__rescheduled_0[4636672UL];
  cast__205(buffer_404, buffer_403);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_405 = (float*)&__rescheduled_0[196608UL];
  mul__213(buffer_405, res4d_weight_2, folded_const_120);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_406 = (int8_t*)&__rescheduled_0[4898816UL];
  cast__214(buffer_406, buffer_405);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_407 = (float*)&__rescheduled_0[196608UL];
  mul__222(buffer_407, res4e_weight_2, folded_const_117);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_408 = (int8_t*)&__rescheduled_0[5160960UL];
  cast__223(buffer_408, buffer_407);
  // [f32 [1024, 256, 1, 1] @ ABCD]
  float* buffer_409 = (float*)&__rescheduled_0[196608UL];
  mul__231(buffer_409, res4f_weight_2, folded_const_114);
  // [s8 [1024, 256, 1, 1] @ ABCD]
  int8_t* buffer_410 = (int8_t*)&__rescheduled_0[5423104UL];
  cast__232(buffer_410, buffer_409);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_411 = (float*)&__rescheduled_0[196608UL];
  mul__189(buffer_411, res4b_weight_0, folded_const_128);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_412 = (int8_t*)&__rescheduled_0[5685248UL];
  cast__190(buffer_412, buffer_411);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_413 = (float*)&__rescheduled_0[196608UL];
  mul__198(buffer_413, res4c_weight_0, folded_const_125);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_414 = (int8_t*)&__rescheduled_0[5947392UL];
  cast__199(buffer_414, buffer_413);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_415 = (float*)&__rescheduled_0[196608UL];
  mul__207(buffer_415, res4d_weight_0, folded_const_122);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_416 = (int8_t*)&__rescheduled_0[6209536UL];
  cast__208(buffer_416, buffer_415);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_417 = (float*)&__rescheduled_0[196608UL];
  mul__216(buffer_417, res4e_weight_0, folded_const_119);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_418 = (int8_t*)&__rescheduled_0[6471680UL];
  cast__217(buffer_418, buffer_417);
  // [f32 [256, 1024, 1, 1] @ ABCD]
  float* buffer_419 = (float*)&__rescheduled_0[196608UL];
  mul__225(buffer_419, res4f_weight_0, folded_const_116);
  // [s8 [256, 1024, 1, 1] @ ABCD]
  int8_t* buffer_420 = (int8_t*)&__rescheduled_0[6733824UL];
  cast__226(buffer_420, buffer_419);
  // [f32 [1024, 512, 1, 1] @ ABCD]
  float* buffer_421 = (float*)&__rescheduled_0[196608UL];
  mul__177(buffer_421, res4a_weight_b, folded_const_132);
  // [s8 [1024, 512, 1, 1] @ ABCD]
  int8_t* buffer_422 = (int8_t*)&__rescheduled_0[6995968UL];
  cast__178(buffer_422, buffer_421);
  // [f32 [512, 1024, 1, 1] @ ABCD]
  float* buffer_423 = (float*)&__rescheduled_0[196608UL];
  mul__237(buffer_423, res5a_weight_0, folded_const_112);
  // [s8 [512, 1024, 1, 1] @ ABCD]
  int8_t* buffer_424 = (int8_t*)&__rescheduled_0[7520256UL];
  cast__238(buffer_424, buffer_423);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_425 = (float*)&__rescheduled_0[196608UL];
  mul__183(buffer_425, res4a_weight_1, folded_const_130);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_426 = (int8_t*)&__rescheduled_0[8044544UL];
  cast__184(buffer_426, buffer_425);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_427 = (float*)&__rescheduled_0[196608UL];
  mul__192(buffer_427, res4b_weight_1, folded_const_127);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_428 = (int8_t*)&__rescheduled_0[8634368UL];
  cast__193(buffer_428, buffer_427);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_429 = (float*)&__rescheduled_0[196608UL];
  mul__201(buffer_429, res4c_weight_1, folded_const_124);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_430 = (int8_t*)&__rescheduled_0[9224192UL];
  cast__202(buffer_430, buffer_429);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_431 = (float*)&__rescheduled_0[196608UL];
  mul__210(buffer_431, res4d_weight_1, folded_const_121);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_432 = (int8_t*)&__rescheduled_0[9814016UL];
  cast__211(buffer_432, buffer_431);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_433 = (float*)&__rescheduled_0[196608UL];
  mul__219(buffer_433, res4e_weight_1, folded_const_118);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_434 = (int8_t*)&__rescheduled_0[10403840UL];
  cast__220(buffer_434, buffer_433);
  // [f32 [256, 256, 3, 3] @ ABCD]
  float* buffer_435 = (float*)&__rescheduled_0[196608UL];
  mul__228(buffer_435, res4f_weight_1, folded_const_115);
  // [s8 [256, 256, 3, 3] @ ABCD]
  int8_t* buffer_436 = (int8_t*)&__rescheduled_0[10993664UL];
  cast__229(buffer_436, buffer_435);
  reorder__525(folded_const_156, buffer_436);
  mul__652(folded_const_157, buffer_346, folded_const_86);
  mul__651(folded_const_158, buffer_266, folded_const_86);
  mul__646(folded_const_159, buffer_345, folded_const_87);
  mul__645(folded_const_160, buffer_265, folded_const_87);
  mul__640(folded_const_161, buffer_344, folded_const_88);
  mul__639(folded_const_162, buffer_264, folded_const_88);
  mul__634(folded_const_163, buffer_343, folded_const_89);
  mul__633(folded_const_164, buffer_263, folded_const_89);
  mul__628(folded_const_165, &res4b_bias_2[0], folded_const_90);
  mul__627(folded_const_166, &folded_const_36[0UL], folded_const_90);
  mul__622(folded_const_167, buffer_342, folded_const_91);
  mul__621(folded_const_168, buffer_262, folded_const_91);
  mul__616(folded_const_169, buffer_341, folded_const_92);
  mul__615(folded_const_170, buffer_261, folded_const_92);
  mul__614(folded_const_171, buffer_334, folded_const_93);
  mul__613(folded_const_172, buffer_285, folded_const_93);
  mul__608(folded_const_173, buffer_333, folded_const_94);
  mul__607(folded_const_174, buffer_282, folded_const_94);
  mul__602(folded_const_175, buffer_332, folded_const_95);
  mul__601(folded_const_176, buffer_280, folded_const_95);
  mul__596(folded_const_177, buffer_331, folded_const_96);
  mul__595(folded_const_178, buffer_278, folded_const_96);
  mul__590(folded_const_179, buffer_330, folded_const_97);
  mul__589(folded_const_180, buffer_275, folded_const_97);
  mul__650(folded_const_181, buffer_329, folded_const_17);
  mul__649(folded_const_182, buffer_295, folded_const_17);
  mul__648(folded_const_183, buffer_328, folded_const_19);
  mul__647(folded_const_184, buffer_294, folded_const_19);
  mul__644(folded_const_185, buffer_327, folded_const_22);
  mul__643(folded_const_186, buffer_293, folded_const_22);
  mul__642(folded_const_187, &res4e_bias_0[0], folded_const_24);
  mul__641(folded_const_188, &folded_const_25[0UL], folded_const_24);
  mul__638(folded_const_189, buffer_326, folded_const_27);
  mul__637(folded_const_190, buffer_292, folded_const_27);
  mul__636(folded_const_191, buffer_325, folded_const_29);
  mul__635(folded_const_192, buffer_291, folded_const_29);
  mul__632(folded_const_193, buffer_324, folded_const_32);
  mul__631(folded_const_194, buffer_290, folded_const_32);
  mul__630(folded_const_195, buffer_323, folded_const_34);
  mul__629(folded_const_196, buffer_289, folded_const_34);
  mul__626(folded_const_197, buffer_322, folded_const_37);
  mul__625(folded_const_198, buffer_288, folded_const_37);
  mul__624(folded_const_199, buffer_321, folded_const_39);
  mul__623(folded_const_200, buffer_287, folded_const_39);
  mul__620(folded_const_201, buffer_320, folded_const_42);
  mul__619(folded_const_202, buffer_286, folded_const_42);
  mul__618(folded_const_203, &res4a_bias_0[0], folded_const_44);
  mul__617(folded_const_204, &folded_const_45[0UL], folded_const_44);
  mul__588(folded_const_205, buffer_319, folded_const_98);
  mul__587(folded_const_206, buffer_274, folded_const_98);
  mul__582(folded_const_207, buffer_318, folded_const_99);
  mul__581(folded_const_208, buffer_271, folded_const_99);
  mul__576(folded_const_209, buffer_317, folded_const_100);
  mul__575(folded_const_210, buffer_269, folded_const_100);
  mul__570(folded_const_211, buffer_316, folded_const_101);
  mul__569(folded_const_212, buffer_267, folded_const_101);
  reorder__522(folded_const_213, buffer_420);
  reorder__515(folded_const_214, buffer_418);
  reorder__506(folded_const_215, buffer_416);
  reorder__497(folded_const_216, buffer_414);
  reorder__490(folded_const_217, buffer_412);
  reorder__528(folded_const_218, buffer_410);
  reorder__519(folded_const_219, buffer_408);
  reorder__512(folded_const_220, buffer_406);
  reorder__503(folded_const_221, buffer_404);
  reorder__496(folded_const_222, buffer_402);
  reorder__487(folded_const_223, buffer_400);
  reorder__422(folded_const_224, buffer_352);
  mul__612(folded_const_225, buffer_315, folded_const_48);
  mul__611(folded_const_226, buffer_284, folded_const_48);
  mul__610(folded_const_227, buffer_314, folded_const_50);
  mul__609(folded_const_228, buffer_283, folded_const_50);
  mul__606(folded_const_229, buffer_313, folded_const_53);
  mul__605(folded_const_230, buffer_281, folded_const_53);
  mul__604(folded_const_231, &res3c_bias_0[0], folded_const_55);
  mul__603(folded_const_232, &folded_const_56[0UL], folded_const_55);
  mul__600(folded_const_233, &res3b_bias_1[0], folded_const_58);
  mul__599(folded_const_234, &folded_const_59[0UL], folded_const_58);
  mul__598(folded_const_235, buffer_312, folded_const_60);
  mul__597(folded_const_236, buffer_279, folded_const_60);
  mul__594(folded_const_237, buffer_311, folded_const_63);
  mul__593(folded_const_238, buffer_277, folded_const_63);
  mul__592(folded_const_239, buffer_310, folded_const_65);
  mul__591(folded_const_240, buffer_276, folded_const_65);
  mul__586(folded_const_241, buffer_309, folded_const_69);
  mul__585(folded_const_242, buffer_273, folded_const_69);
  mul__584(folded_const_243, buffer_308, folded_const_71);
  mul__583(folded_const_244, buffer_272, folded_const_71);
  mul__580(folded_const_245, buffer_307, folded_const_74);
  mul__579(folded_const_246, buffer_270, folded_const_74);
  mul__578(folded_const_247, &res2b_bias_0[0], folded_const_76);
  mul__577(folded_const_248, &folded_const_77[0UL], folded_const_76);
  mul__574(folded_const_249, buffer_306, folded_const_79);
  mul__573(folded_const_250, buffer_268, folded_const_79);
  mul__572(folded_const_251, &res2a_bias_0[0], folded_const_81);
  mul__571(folded_const_252, &folded_const_102[0UL], folded_const_81);
  reorder__516(folded_const_253, buffer_434);
  reorder__509(folded_const_254, buffer_432);
  reorder__500(folded_const_255, buffer_430);
  reorder__493(folded_const_256, buffer_428);
  reorder__484(folded_const_257, buffer_426);
  reorder__480(folded_const_258, buffer_422);
  reorder__474(folded_const_259, buffer_398);
  reorder__465(folded_const_260, buffer_396);
  reorder__460(folded_const_261, buffer_394);
  reorder__451(folded_const_262, buffer_392);
  reorder__483(folded_const_263, buffer_390);
  reorder__445(folded_const_264, buffer_388);
  reorder__471(folded_const_265, buffer_386);
  reorder__464(folded_const_266, buffer_384);
  reorder__457(folded_const_267, buffer_382);
  reorder__477(folded_const_268, buffer_380);
  reorder__468(folded_const_269, buffer_378);
  reorder__461(folded_const_270, buffer_376);
  reorder__454(folded_const_271, buffer_374);
  reorder__439(folded_const_272, buffer_372);
  reorder__430(folded_const_273, buffer_370);
  reorder__423(folded_const_274, buffer_368);
  reorder__448(folded_const_275, buffer_366);
  reorder__436(folded_const_276, buffer_364);
  reorder__429(folded_const_277, buffer_362);
  reorder__442(folded_const_278, buffer_360);
  reorder__433(folded_const_279, buffer_358);
  reorder__426(folded_const_280, buffer_356);
  reorder__419(folded_const_281, buffer_354);
  mul__656(folded_const_282, buffer_335, folded_const_13);
  mul__655(folded_const_283, buffer_297, folded_const_13);
  reorder__534(folded_const_284, buffer_424);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_566 = (float*)&__rescheduled_0[196608UL];
  mul__243(buffer_566, res5a_weight_2, folded_const_110);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_567 = (int8_t*)&__rescheduled_0[9633792UL];
  cast__244(buffer_567, buffer_566);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_568 = (float*)&__rescheduled_0[196608UL];
  mul__252(buffer_568, res5b_weight_2, folded_const_107);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_569 = (int8_t*)&__rescheduled_0[11993088UL];
  cast__253(buffer_569, buffer_568);
  // [f32 [2048, 512, 1, 1] @ ABCD]
  float* buffer_570 = (float*)&__rescheduled_0[196608UL];
  mul__261(buffer_570, res5c_weight_2, folded_const_104);
  // [s8 [2048, 512, 1, 1] @ ABCD]
  int8_t* buffer_571 = (int8_t*)&__rescheduled_0[13041664UL];
  cast__262(buffer_571, buffer_570);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_572 = (float*)&__rescheduled_0[196608UL];
  mul__246(buffer_572, res5b_weight_0, folded_const_109);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_573 = (int8_t*)&__rescheduled_0[14090240UL];
  cast__247(buffer_573, buffer_572);
  // [f32 [512, 2048, 1, 1] @ ABCD]
  float* buffer_574 = (float*)&__rescheduled_0[196608UL];
  mul__255(buffer_574, res5c_weight_0, folded_const_106);
  // [s8 [512, 2048, 1, 1] @ ABCD]
  int8_t* buffer_575 = (int8_t*)&__rescheduled_0[15138816UL];
  cast__256(buffer_575, buffer_574);
  // [f32 [2048, 1024, 1, 1] @ ABCD]
  float* buffer_576 = (float*)&__rescheduled_0[196608UL];
  mul__234(buffer_576, res5a_weight_b, folded_const_113);
  // [s8 [2048, 1024, 1, 1] @ ABCD]
  int8_t* buffer_577 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__235(buffer_577, buffer_576);
  reorder__531(folded_const_285, buffer_577);
  mul__654(folded_const_286, buffer_347, folded_const_85);
  mul__653(folded_const_287, buffer_296, folded_const_85);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_582 = (float*)&__rescheduled_0[196608UL];
  mul__240(buffer_582, res5a_weight_1, folded_const_111);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_583 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__241(buffer_583, buffer_582);
  reorder__537(folded_const_288, buffer_583);
  mul__658(folded_const_289, buffer_336, folded_const_11);
  mul__657(folded_const_290, buffer_298, folded_const_11);
  mul__660(folded_const_291, buffer_348, folded_const_84);
  mul__659(folded_const_292, buffer_299, folded_const_84);
  reorder__540(folded_const_293, buffer_567);
  mul__662(folded_const_294, buffer_337, folded_const_8);
  mul__661(folded_const_295, buffer_300, folded_const_8);
  reorder__543(folded_const_296, buffer_573);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_596 = (float*)&__rescheduled_0[196608UL];
  mul__249(buffer_596, res5b_weight_1, folded_const_108);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_597 = (int8_t*)&__rescheduled_0[16187392UL];
  cast__250(buffer_597, buffer_596);
  // [f32 [512, 512, 3, 3] @ ABCD]
  float* buffer_598 = (float*)&__rescheduled_0[196608UL];
  mul__258(buffer_598, res5c_weight_1, folded_const_105);
  // [s8 [512, 512, 3, 3] @ ABCD]
  int8_t* buffer_599 = (int8_t*)&__rescheduled_0[9633792UL];
  cast__259(buffer_599, buffer_598);
  mul__664(folded_const_297, buffer_338, folded_const_6);
  mul__663(folded_const_298, buffer_301, folded_const_6);
  reorder__546(folded_const_299, buffer_597);
  mul__666(folded_const_300, buffer_349, folded_const_83);
  mul__665(folded_const_301, buffer_302, folded_const_83);
  reorder__549(folded_const_302, buffer_569);
  mul__668(folded_const_303, buffer_339, folded_const_3);
  mul__667(folded_const_304, buffer_303, folded_const_3);
  reorder__552(folded_const_305, buffer_575);
  mul__670(folded_const_306, buffer_340, folded_const_1);
  mul__669(folded_const_307, buffer_304, folded_const_1);
  reorder__555(folded_const_308, buffer_599);
  mul__672(folded_const_309, buffer_350, folded_const_82);
  mul__671(folded_const_310, buffer_305, folded_const_82);
  reorder__558(folded_const_311, buffer_571);
  is_init = true;
  sc_aligned_free(__stream, __rescheduled_0);
}

extern "C" void sc_init_rn50_backbone_bs9() {
  bool& is_init = *(bool*)(__module_data + 0);
  void*& __sc_kernel_cache = *(void**)(__module_data + 8);
  uint8_t* __brgemm_attrs = (uint8_t*)&__module_data[384UL];
  void*& __sc_kernel_cache_2 = *(void**)(__module_data + 16);
  uint8_t* __brgemm_attrs_1 = (uint8_t*)&__module_data[512UL];
  void*& __sc_kernel_cache_4 = *(void**)(__module_data + 24);
  uint8_t* __brgemm_attrs_3 = (uint8_t*)&__module_data[640UL];
  void*& __sc_kernel_cache_6 = *(void**)(__module_data + 32);
  uint8_t* __brgemm_attrs_5 = (uint8_t*)&__module_data[768UL];
  void*& __sc_kernel_cache_8 = *(void**)(__module_data + 40);
  uint8_t* __brgemm_attrs_7 = (uint8_t*)&__module_data[896UL];
  void*& __sc_kernel_cache_10 = *(void**)(__module_data + 48);
  uint8_t* __brgemm_attrs_9 = (uint8_t*)&__module_data[1024UL];
  void*& __sc_kernel_cache_12 = *(void**)(__module_data + 56);
  uint8_t* __brgemm_attrs_11 = (uint8_t*)&__module_data[1152UL];
  void*& __sc_kernel_cache_14 = *(void**)(__module_data + 64);
  uint8_t* __brgemm_attrs_13 = (uint8_t*)&__module_data[1280UL];
  void*& __sc_kernel_cache_16 = *(void**)(__module_data + 72);
  uint8_t* __brgemm_attrs_15 = (uint8_t*)&__module_data[1408UL];
  void*& __sc_kernel_cache_18 = *(void**)(__module_data + 80);
  uint8_t* __brgemm_attrs_17 = (uint8_t*)&__module_data[1536UL];
  void*& __sc_kernel_cache_20 = *(void**)(__module_data + 88);
  uint8_t* __brgemm_attrs_19 = (uint8_t*)&__module_data[1664UL];
  void*& __sc_kernel_cache_22 = *(void**)(__module_data + 96);
  uint8_t* __brgemm_attrs_21 = (uint8_t*)&__module_data[1792UL];
  void*& __sc_kernel_cache_24 = *(void**)(__module_data + 104);
  uint8_t* __brgemm_attrs_23 = (uint8_t*)&__module_data[1920UL];
  void*& __sc_kernel_cache_26 = *(void**)(__module_data + 112);
  uint8_t* __brgemm_attrs_25 = (uint8_t*)&__module_data[2048UL];
  void*& __sc_kernel_cache_28 = *(void**)(__module_data + 120);
  uint8_t* __brgemm_attrs_27 = (uint8_t*)&__module_data[2176UL];
  void*& __sc_kernel_cache_31 = *(void**)(__module_data + 128);
  uint8_t* __brgemm_attrs_30 = (uint8_t*)&__module_data[3328UL];
  void*& __sc_kernel_cache_33 = *(void**)(__module_data + 136);
  uint8_t* __brgemm_attrs_32 = (uint8_t*)&__module_data[3456UL];
  void*& __sc_kernel_cache_35 = *(void**)(__module_data + 144);
  uint8_t* __brgemm_attrs_34 = (uint8_t*)&__module_data[3584UL];
  void*& __sc_kernel_cache_37 = *(void**)(__module_data + 152);
  uint8_t* __brgemm_attrs_36 = (uint8_t*)&__module_data[3712UL];
  void*& __sc_kernel_cache_39 = *(void**)(__module_data + 160);
  uint8_t* __brgemm_attrs_38 = (uint8_t*)&__module_data[3840UL];
  void*& __sc_kernel_cache_40 = *(void**)(__module_data + 168);
  void*& __sc_kernel_cache_42 = *(void**)(__module_data + 176);
  uint8_t* __brgemm_attrs_41 = (uint8_t*)&__module_data[3968UL];
  void*& __sc_kernel_cache_44 = *(void**)(__module_data + 184);
  uint8_t* __brgemm_attrs_43 = (uint8_t*)&__module_data[4096UL];
  void*& __sc_kernel_cache_50 = *(void**)(__module_data + 192);
  uint8_t* __brgemm_attrs_49 = (uint8_t*)&__module_data[4800UL];
  void*& __sc_kernel_cache_52 = *(void**)(__module_data + 200);
  uint8_t* __brgemm_attrs_51 = (uint8_t*)&__module_data[4928UL];
  void*& __sc_kernel_cache_54 = *(void**)(__module_data + 208);
  uint8_t* __brgemm_attrs_53 = (uint8_t*)&__module_data[5056UL];
  void*& __sc_kernel_cache_60 = *(void**)(__module_data + 216);
  uint8_t* __brgemm_attrs_59 = (uint8_t*)&__module_data[5568UL];
  void*& __sc_kernel_cache_62 = *(void**)(__module_data + 224);
  uint8_t* __brgemm_attrs_61 = (uint8_t*)&__module_data[5696UL];
  void*& __sc_kernel_cache_66 = *(void**)(__module_data + 232);
  uint8_t* __brgemm_attrs_65 = (uint8_t*)&__module_data[5952UL];
  void*& __sc_kernel_cache_68 = *(void**)(__module_data + 240);
  uint8_t* __brgemm_attrs_67 = (uint8_t*)&__module_data[6080UL];
  void*& __sc_kernel_cache_72 = *(void**)(__module_data + 248);
  uint8_t* __brgemm_attrs_71 = (uint8_t*)&__module_data[6336UL];
  void*& __sc_kernel_cache_74 = *(void**)(__module_data + 256);
  uint8_t* __brgemm_attrs_73 = (uint8_t*)&__module_data[6464UL];
  void*& __sc_kernel_cache_76 = *(void**)(__module_data + 264);
  uint8_t* __brgemm_attrs_75 = (uint8_t*)&__module_data[6592UL];
  void*& __sc_kernel_cache_78 = *(void**)(__module_data + 272);
  uint8_t* __brgemm_attrs_77 = (uint8_t*)&__module_data[6720UL];
  void*& __sc_kernel_cache_80 = *(void**)(__module_data + 280);
  uint8_t* __brgemm_attrs_79 = (uint8_t*)&__module_data[6848UL];
  void*& __sc_kernel_cache_86 = *(void**)(__module_data + 288);
  uint8_t* __brgemm_attrs_85 = (uint8_t*)&__module_data[7232UL];
  void*& __sc_kernel_cache_88 = *(void**)(__module_data + 296);
  uint8_t* __brgemm_attrs_87 = (uint8_t*)&__module_data[7360UL];
  void*& __sc_kernel_cache_90 = *(void**)(__module_data + 304);
  uint8_t* __brgemm_attrs_89 = (uint8_t*)&__module_data[7488UL];
  void*& __sc_kernel_cache_96 = *(void**)(__module_data + 312);
  uint8_t* __brgemm_attrs_95 = (uint8_t*)&__module_data[7808UL];
  void*& __sc_kernel_cache_98 = *(void**)(__module_data + 320);
  uint8_t* __brgemm_attrs_97 = (uint8_t*)&__module_data[7936UL];
  void*& __sc_kernel_cache_102 = *(void**)(__module_data + 328);
  uint8_t* __brgemm_attrs_101 = (uint8_t*)&__module_data[8192UL];
  void** __brgemm_bd_mask_arr = (void**)&__uninitialized_data[23657472UL];
  uint8_t* __brgemm_full_bd_mask = (uint8_t*)&__module_data[2432UL];
  void** __brgemm_bd_mask_arr_47 = (void**)&__uninitialized_data[23657504UL];
  uint8_t* __brgemm_full_bd_mask_46 = (uint8_t*)&__module_data[4352UL];
  void** __brgemm_bd_mask_arr_57 = (void**)&__uninitialized_data[23657520UL];
  uint8_t* __brgemm_full_bd_mask_56 = (uint8_t*)&__module_data[5312UL];
  void** __brgemm_bd_mask_arr_83 = (void**)&__uninitialized_data[23657552UL];
  uint8_t* __brgemm_full_bd_mask_82 = (uint8_t*)&__module_data[7104UL];
  void** __brgemm_bd_mask_arr_93 = (void**)&__uninitialized_data[23657568UL];
  uint8_t* __brgemm_full_bd_mask_92 = (uint8_t*)&__module_data[7736UL];
  void** __sc_kernel_cache_arr = (void**)&__uninitialized_data[23657488UL];
  uint8_t* __brgemm_attrs_29 = (uint8_t*)&__module_data[2304UL];
  void** __sc_kernel_cache_arr_48 = (void**)&__uninitialized_data[23657512UL];
  uint8_t* __brgemm_attrs_45 = (uint8_t*)&__module_data[4224UL];
  void** __sc_kernel_cache_arr_58 = (void**)&__uninitialized_data[23657528UL];
  uint8_t* __brgemm_attrs_55 = (uint8_t*)&__module_data[5184UL];
  void** __sc_kernel_cache_arr_64 = (void**)&__uninitialized_data[23657536UL];
  uint8_t* __brgemm_attrs_63 = (uint8_t*)&__module_data[5824UL];
  void** __sc_kernel_cache_arr_70 = (void**)&__uninitialized_data[23657544UL];
  uint8_t* __brgemm_attrs_69 = (uint8_t*)&__module_data[6208UL];
  void** __sc_kernel_cache_arr_84 = (void**)&__uninitialized_data[23657560UL];
  uint8_t* __brgemm_attrs_81 = (uint8_t*)&__module_data[6976UL];
  void** __sc_kernel_cache_arr_94 = (void**)&__uninitialized_data[23657576UL];
  uint8_t* __brgemm_attrs_91 = (uint8_t*)&__module_data[7616UL];
  void** __sc_kernel_cache_arr_100 = (void**)&__uninitialized_data[23657584UL];
  uint8_t* __brgemm_attrs_99 = (uint8_t*)&__module_data[8064UL];
  is_init = false;
  __sc_kernel_cache = dnnl_brgemm_list_func(112, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs, ((void*)0), ((void*)0));
  __sc_kernel_cache_2 = dnnl_brgemm_list_func(14, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_1, ((void*)0), ((void*)0));
  __sc_kernel_cache_4 = dnnl_brgemm_list_func(112, 16, 64, 64, 16, 16, 0.f, 7, 7, __brgemm_attrs_3, ((void*)0), ((void*)0));
  __sc_kernel_cache_6 = dnnl_brgemm_list_func(56, 64, 32, 32, 64, 64, 0.f, 7, 7, __brgemm_attrs_5, ((void*)0), ((void*)0));
  __sc_kernel_cache_8 = dnnl_brgemm_list_func(392, 64, 32, 32, 64, 64, 0.f, 8, 7, __brgemm_attrs_7, ((void*)0), ((void*)0));
  __sc_kernel_cache_10 = dnnl_brgemm_list_func(28, 16, 64, 64, 16, 16, 0.f, 7, 7, __brgemm_attrs_9, ((void*)0), ((void*)0));
  __sc_kernel_cache_12 = dnnl_brgemm_list_func(56, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_11, ((void*)0), ((void*)0));
  __sc_kernel_cache_14 = dnnl_brgemm_list_func(56, 32, 64, 64, 32, 32, 0.f, 8, 7, __brgemm_attrs_13, ((void*)0), ((void*)0));
  __sc_kernel_cache_16 = dnnl_brgemm_list_func(28, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_15, ((void*)0), ((void*)0));
  __sc_kernel_cache_18 = dnnl_brgemm_list_func(56, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_17, ((void*)0), ((void*)0));
  __sc_kernel_cache_20 = dnnl_brgemm_list_func(112, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_19, ((void*)0), ((void*)0));
  __sc_kernel_cache_22 = dnnl_brgemm_list_func(28, 32, 128, 256, 32, 32, 0.f, 7, 7, __brgemm_attrs_21, ((void*)0), ((void*)0));
  __sc_kernel_cache_24 = dnnl_brgemm_list_func(196, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_23, ((void*)0), ((void*)0));
  __sc_kernel_cache_26 = dnnl_brgemm_list_func(392, 32, 32, 32, 32, 32, 0.f, 7, 7, __brgemm_attrs_25, ((void*)0), ((void*)0));
  __sc_kernel_cache_28 = dnnl_brgemm_list_func(112, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_27, ((void*)0), ((void*)0));
  __sc_kernel_cache_31 = dnnl_brgemm_list_func(28, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_30, ((void*)0), ((void*)0));
  __sc_kernel_cache_33 = dnnl_brgemm_list_func(784, 128, 64, 64, 128, 128, 0.f, 8, 7, __brgemm_attrs_32, ((void*)0), ((void*)0));
  __sc_kernel_cache_35 = dnnl_brgemm_list_func(28, 32, 64, 64, 32, 32, 0.f, 7, 7, __brgemm_attrs_34, ((void*)0), ((void*)0));
  __sc_kernel_cache_37 = dnnl_brgemm_list_func(56, 64, 128, 128, 64, 64, 0.f, 7, 7, __brgemm_attrs_36, ((void*)0), ((void*)0));
  __sc_kernel_cache_39 = dnnl_brgemm_list_func(56, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_38, ((void*)0), ((void*)0));
  __sc_kernel_cache_40 = dnnl_brgemm_list_func(28, 32, 128, 128, 32, 32, 0.f, 7, 7, __brgemm_attrs_21, ((void*)0), ((void*)0));
  __sc_kernel_cache_42 = dnnl_brgemm_list_func(196, 128, 32, 32, 128, 128, 0.f, 7, 7, __brgemm_attrs_41, ((void*)0), ((void*)0));
  __sc_kernel_cache_44 = dnnl_brgemm_list_func(56, 256, 128, 128, 256, 256, 0.f, 8, 7, __brgemm_attrs_43, ((void*)0), ((void*)0));
  __sc_kernel_cache_50 = dnnl_brgemm_list_func(98, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_49, ((void*)0), ((void*)0));
  __sc_kernel_cache_52 = dnnl_brgemm_list_func(14, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_51, ((void*)0), ((void*)0));
  __sc_kernel_cache_54 = dnnl_brgemm_list_func(28, 64, 1024, 1024, 64, 64, 0.f, 8, 7, __brgemm_attrs_53, ((void*)0), ((void*)0));
  __sc_kernel_cache_60 = dnnl_brgemm_list_func(98, 1024, 64, 64, 1024, 1024, 0.f, 7, 7, __brgemm_attrs_59, ((void*)0), ((void*)0));
  __sc_kernel_cache_62 = dnnl_brgemm_list_func(196, 32, 128, 128, 32, 32, 0.f, 8, 7, __brgemm_attrs_61, ((void*)0), ((void*)0));
  __sc_kernel_cache_66 = dnnl_brgemm_list_func(196, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_65, ((void*)0), ((void*)0));
  __sc_kernel_cache_68 = dnnl_brgemm_list_func(196, 64, 128, 128, 64, 64, 0.f, 8, 7, __brgemm_attrs_67, ((void*)0), ((void*)0));
  __sc_kernel_cache_72 = dnnl_brgemm_list_func(196, 128, 128, 128, 128, 128, 0.f, 7, 7, __brgemm_attrs_71, ((void*)0), ((void*)0));
  __sc_kernel_cache_74 = dnnl_brgemm_list_func(98, 256, 128, 128, 256, 256, 0.f, 8, 7, __brgemm_attrs_73, ((void*)0), ((void*)0));
  __sc_kernel_cache_76 = dnnl_brgemm_list_func(28, 32, 512, 512, 32, 32, 0.f, 8, 7, __brgemm_attrs_75, ((void*)0), ((void*)0));
  __sc_kernel_cache_78 = dnnl_brgemm_list_func(196, 256, 128, 128, 256, 256, 0.f, 7, 7, __brgemm_attrs_77, ((void*)0), ((void*)0));
  __sc_kernel_cache_80 = dnnl_brgemm_list_func(196, 64, 256, 256, 64, 64, 0.f, 8, 7, __brgemm_attrs_79, ((void*)0), ((void*)0));
  __sc_kernel_cache_86 = dnnl_brgemm_list_func(49, 16, 256, 256, 16, 16, 0.f, 8, 7, __brgemm_attrs_85, ((void*)0), ((void*)0));
  __sc_kernel_cache_88 = dnnl_brgemm_list_func(49, 64, 256, 256, 64, 64, 0.f, 7, 7, __brgemm_attrs_87, ((void*)0), ((void*)0));
  __sc_kernel_cache_90 = dnnl_brgemm_list_func(49, 128, 64, 64, 128, 128, 0.f, 8, 7, __brgemm_attrs_89, ((void*)0), ((void*)0));
  __sc_kernel_cache_96 = dnnl_brgemm_list_func(7, 64, 256, 256, 64, 64, 0.f, 7, 7, __brgemm_attrs_95, ((void*)0), ((void*)0));
  __sc_kernel_cache_98 = dnnl_brgemm_list_func(49, 32, 512, 512, 32, 32, 0.f, 8, 7, __brgemm_attrs_97, ((void*)0), ((void*)0));
  __sc_kernel_cache_102 = dnnl_brgemm_list_func(49, 512, 512, 512, 512, 512, 0.f, 7, 7, __brgemm_attrs_101, ((void*)0), ((void*)0));
  __brgemm_bd_mask_arr[0] = &__brgemm_full_bd_mask[(0 * 419)];
  __brgemm_bd_mask_arr[1] = &__brgemm_full_bd_mask[(1 * 419)];
  __brgemm_bd_mask_arr_47[0] = &__brgemm_full_bd_mask_46[(0 * 404)];
  __brgemm_bd_mask_arr_57[0] = &__brgemm_full_bd_mask_56[(0 * 222)];
  __brgemm_bd_mask_arr_83[0] = &__brgemm_full_bd_mask_82[(0 * 103)];
  __brgemm_bd_mask_arr_93[0] = &__brgemm_full_bd_mask_92[(0 * 61)];
  __sc_kernel_cache_arr[0] = dnnl_brgemm_list_func(419, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_29, __brgemm_bd_mask_arr[0], ((void*)0));
  __sc_kernel_cache_arr[1] = dnnl_brgemm_list_func(419, 128, 64, 64, 128, 128, 0.f, 7, 7, __brgemm_attrs_29, __brgemm_bd_mask_arr[1], ((void*)0));
  __sc_kernel_cache_arr_48[0] = dnnl_brgemm_list_func(404, 128, 256, 512, 128, 128, 0.f, 7, 7, __brgemm_attrs_45, __brgemm_bd_mask_arr_47[0], ((void*)0));
  __sc_kernel_cache_arr_58[0] = dnnl_brgemm_list_func(222, 16, 256, 256, 16, 16, 0.f, 7, 7, __brgemm_attrs_55, __brgemm_bd_mask_arr_57[0], ((void*)0));
  __sc_kernel_cache_arr_64[0] = dnnl_brgemm_list_func(222, 32, 128, 128, 32, 32, 0.f, 7, 7, __brgemm_attrs_63, __brgemm_bd_mask_arr_57[0], ((void*)0));
  __sc_kernel_cache_arr_70[0] = dnnl_brgemm_list_func(222, 64, 64, 64, 64, 64, 0.f, 7, 7, __brgemm_attrs_69, __brgemm_bd_mask_arr_57[0], ((void*)0));
  __sc_kernel_cache_arr_84[0] = dnnl_brgemm_list_func(103, 128, 256, 512, 128, 128, 0.f, 7, 7, __brgemm_attrs_81, __brgemm_bd_mask_arr_83[0], ((void*)0));
  __sc_kernel_cache_arr_94[0] = dnnl_brgemm_list_func(61, 16, 512, 512, 16, 16, 0.f, 7, 7, __brgemm_attrs_91, __brgemm_bd_mask_arr_93[0], ((void*)0));
  __sc_kernel_cache_arr_100[0] = dnnl_brgemm_list_func(61, 128, 256, 256, 128, 128, 0.f, 7, 7, __brgemm_attrs_99, __brgemm_bd_mask_arr_93[0], ((void*)0));
}

