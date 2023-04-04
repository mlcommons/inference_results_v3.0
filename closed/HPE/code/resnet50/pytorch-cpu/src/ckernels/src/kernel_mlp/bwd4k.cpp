
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t mlp_training_backward_4k_data[64];
static constexpr int8_t* __module_data = mlp_training_backward_4k_data;
alignas(64) static int8_t __uninitialized_data[0UL];

static bool reorder__17(uint16_t* __outs_0, uint16_t* __ins_0);
static bool select_one_mul_reduce__41(uint16_t* __outs_0, uint16_t* __outs_1, uint16_t* __ins_0, uint16_t* __ins_1);
static bool reorder__18(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_select_one_cast_mul__36(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool reduce__6(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__19(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__26(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__27(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_cast__35(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1);
static bool reorder__28(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_select_one_cast_mul__38(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool matmul_core_cast_reorder__40(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1);
static bool reduce__11(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__21(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__20(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_cast__39(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1);
static bool reorder__22(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__23(uint16_t* __outs_0, uint16_t* __ins_0);
static bool reorder__24(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_cast__37(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1);
static bool reorder__25(uint16_t* __outs_0, uint16_t* __ins_0);
static void reorder__170_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void select_one_mul_reduce__410_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__180_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_select_one_cast_mul__360_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__190_closure_4_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__260_closure_5_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__270_closure_6_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast__350_closure_7_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__280_closure_8_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_select_one_cast_mul__380_closure_9_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_reorder__400_closure_10_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__210_closure_11_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__200_closure_12_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast__390_closure_13_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__220_closure_14_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__230_closure_15_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__240_closure_16_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast__370_closure_17_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__250_closure_18_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__170_closure_0(uint64_t fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void select_one_mul_reduce__410_closure_1(uint64_t __itr_0_0outer, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__180_closure_2(uint64_t fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_select_one_cast_mul__360_closure_3(uint64_t fused_0m_o__n_o_10, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void reorder__190_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__260_closure_5(uint64_t _fuseiter_65, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__270_closure_6(uint64_t fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__350_closure_7(uint64_t fused_0m_o__n_o_16, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__280_closure_8(uint64_t fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_select_one_cast_mul__380_closure_9(uint64_t fused_0m_o__n_o_19, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_reorder__400_closure_10(uint64_t fused_0m_o__n_o_20, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__210_closure_11(uint64_t fused_0_fuseiter_101___fuseiter_102_21, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__200_closure_12(uint64_t _fuseiter_106_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__390_closure_13(uint64_t fused_0m_o__n_o_22, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__220_closure_14(uint64_t fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__230_closure_15(uint64_t _fuseiter_118, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__240_closure_16(uint64_t fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__370_closure_17(uint64_t fused_0m_o__n_o_28, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__250_closure_18(uint64_t fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer, uint16_t* __ins_0, uint16_t* __outs_0);


extern "C" void mlp_training_backward_4k(uint16_t* out_grad_bias2, uint16_t* out_grad_weight2, uint16_t* out_grad_bias1, uint16_t* out_grad_weight1, uint16_t* out_grad_bias0, uint16_t* out_grad_weight0, uint16_t* out_grad_input0, uint16_t* gradient, uint16_t* in_relu_output, uint16_t* data_input2, uint16_t* weight_input2, uint16_t* data_input1, uint16_t* weight_input1, uint16_t* data_input0, uint16_t* weight_input0){
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 11796480UL);
  uint16_t* buffer_8 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__17(buffer_8, &weight_input2[0UL]);
  uint16_t* buffer_9 = (uint16_t*)&__rescheduled_0[6291456UL];
  select_one_mul_reduce__41(buffer_9, out_grad_bias2, in_relu_output, gradient);
  uint16_t* buffer_11 = (uint16_t*)&__rescheduled_0[8388608UL];
  reorder__18(buffer_11, &weight_input1[0UL]);
  uint16_t* buffer_12 = (uint16_t*)&__rescheduled_0[8650752UL];
  matmul_core_select_one_cast_mul__36(buffer_12, buffer_9, buffer_8, data_input2);
  reduce__6(out_grad_bias1, buffer_12);
  uint16_t* buffer_14 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__19(buffer_14, &weight_input0[0UL]);
  uint16_t* buffer_15 = (uint16_t*)&__rescheduled_0[4194304UL];
  reorder__26(buffer_15, &data_input2[0UL]);
  uint16_t* buffer_16 = (uint16_t*)&__rescheduled_0[10747904UL];
  reorder__27(buffer_16, buffer_9);
  uint16_t* buffer_17 = (uint16_t*)&__rescheduled_0[6291456UL];
  matmul_core_cast__35(buffer_17, buffer_15, buffer_16);
  reorder__28(out_grad_weight2, buffer_17);
  uint16_t* buffer_19 = (uint16_t*)&__rescheduled_0[4194304UL];
  matmul_core_select_one_cast_mul__38(buffer_19, buffer_12, buffer_11, data_input1);
  matmul_core_cast_reorder__40(out_grad_input0, buffer_19, buffer_14);
  reduce__11(out_grad_bias0, buffer_19);
  uint16_t* buffer_22 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__21(buffer_22, buffer_19);
  uint16_t* buffer_23 = (uint16_t*)&__rescheduled_0[4194304UL];
  reorder__20(buffer_23, &data_input0[0UL]);
  uint16_t* buffer_24 = (uint16_t*)&__rescheduled_0[4300800UL];
  matmul_core_cast__39(buffer_24, buffer_23, buffer_22);
  reorder__22(out_grad_weight0, buffer_24);
  uint16_t* buffer_26 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__23(buffer_26, &data_input1[0UL]);
  uint16_t* buffer_27 = (uint16_t*)&__rescheduled_0[4194304UL];
  reorder__24(buffer_27, buffer_12);
  uint16_t* buffer_28 = (uint16_t*)&__rescheduled_0[6291456UL];
  matmul_core_cast__37(buffer_28, buffer_26, buffer_27);
  reorder__25(out_grad_weight1, buffer_28);
  sc_aligned_free(__stream, __rescheduled_0);
}

static bool reorder__17(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs0[2UL];
  __tempargs0[0UL] = __ins_0;
  __tempargs0[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__170_closure_0_0wrapper, __stream, __module_data, 0UL, 64UL, 1UL, __tempargs0);
  return true;
}

static bool select_one_mul_reduce__41(uint16_t* __outs_0, uint16_t* __outs_1, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs1[3UL];
  __tempargs1[0UL] = __ins_0;
  __tempargs1[1UL] = __ins_1;
  __tempargs1[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&select_one_mul_reduce__410_closure_1_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs1);
  for (uint64_t _fuseiter_43 = 0UL; _fuseiter_43 < 128UL; _fuseiter_43 += 16UL) {
    vec_f32x16 reduce__0;
    reduce__0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_42 = 0UL; _fuseiter_42 < 4096UL; _fuseiter_42 += 1UL) {
      vec_f32x16 __cached_4;
      __cached_4 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__outs_0[((_fuseiter_42 * 128UL) + _fuseiter_43)]))) << vec_u32x16(16UL)));
      reduce__0 = (__cached_4 + reduce__0);
    }
    vec_u16x16 __cached_5;
    __cached_5 = tobf16(reduce__0);
    vec_u16x16::store(__cached_5, &__outs_1[_fuseiter_43]);
  }
  return true;
}

static bool reorder__18(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs2[2UL];
  __tempargs2[0UL] = __ins_0;
  __tempargs2[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__180_closure_2_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs2);
  return true;
}

static bool matmul_core_select_one_cast_mul__36(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs3[4UL];
  __tempargs3[0UL] = __ins_0;
  __tempargs3[1UL] = __ins_1;
  __tempargs3[2UL] = __ins_2;
  __tempargs3[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_select_one_cast_mul__360_closure_3_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs3);
  return true;
}

static bool reduce__6(uint16_t* __outs_0, uint16_t* __ins_0){
  for (uint64_t _fuseiter_59 = 0UL; _fuseiter_59 < 256UL; _fuseiter_59 += 16UL) {
    vec_f32x16 reduce__1;
    reduce__1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_58 = 0UL; _fuseiter_58 < 4096UL; _fuseiter_58 += 1UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((_fuseiter_58 * 256UL) + _fuseiter_59)]))) << vec_u32x16(16UL)));
      reduce__1 = (__cached_0 + reduce__1);
    }
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(reduce__1);
    vec_u16x16::store(__cached_1, &__outs_0[_fuseiter_59]);
  }
  return true;
}

static bool reorder__19(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs4[2UL];
  __tempargs4[0UL] = __ins_0;
  __tempargs4[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__190_closure_4_0wrapper, __stream, __module_data, 0UL, 16UL, 1UL, __tempargs4);
  return true;
}

static bool reorder__26(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs5[2UL];
  __tempargs5[0UL] = __ins_0;
  __tempargs5[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__260_closure_5_0wrapper, __stream, __module_data, 0UL, 256UL, 8UL, __tempargs5);
  return true;
}

static bool reorder__27(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs6[2UL];
  __tempargs6[0UL] = __ins_0;
  __tempargs6[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__270_closure_6_0wrapper, __stream, __module_data, 0UL, 1024UL, 1UL, __tempargs6);
  return true;
}

static bool matmul_core_cast__35(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs7[3UL];
  __tempargs7[0UL] = __ins_0;
  __tempargs7[1UL] = __ins_1;
  __tempargs7[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast__350_closure_7_0wrapper, __stream, __module_data, 0UL, 32UL, 1UL, __tempargs7);
  return true;
}

static bool reorder__28(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs8[2UL];
  __tempargs8[0UL] = __ins_0;
  __tempargs8[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__280_closure_8_0wrapper, __stream, __module_data, 0UL, 64UL, 1UL, __tempargs8);
  return true;
}

static bool matmul_core_select_one_cast_mul__38(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs9[4UL];
  __tempargs9[0UL] = __ins_0;
  __tempargs9[1UL] = __ins_1;
  __tempargs9[2UL] = __ins_2;
  __tempargs9[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_select_one_cast_mul__380_closure_9_0wrapper, __stream, __module_data, 0UL, 512UL, 1UL, __tempargs9);
  return true;
}

static bool matmul_core_cast_reorder__40(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs10[3UL];
  __tempargs10[0UL] = __ins_0;
  __tempargs10[1UL] = __ins_1;
  __tempargs10[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_reorder__400_closure_10_0wrapper, __stream, __module_data, 0UL, 128UL, 1UL, __tempargs10);
  return true;
}

static bool reduce__11(uint16_t* __outs_0, uint16_t* __ins_0){
  for (uint64_t _fuseiter_100 = 0UL; _fuseiter_100 < 512UL; _fuseiter_100 += 16UL) {
    vec_f32x16 reduce__18;
    reduce__18 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_99 = 0UL; _fuseiter_99 < 4096UL; _fuseiter_99 += 1UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((_fuseiter_99 * 512UL) + _fuseiter_100)]))) << vec_u32x16(16UL)));
      reduce__18 = (__cached_0 + reduce__18);
    }
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(reduce__18);
    vec_u16x16::store(__cached_1, &__outs_0[_fuseiter_100]);
  }
  return true;
}

static bool reorder__21(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs11[2UL];
  __tempargs11[0UL] = __ins_0;
  __tempargs11[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__210_closure_11_0wrapper, __stream, __module_data, 0UL, 1024UL, 1UL, __tempargs11);
  return true;
}

static bool reorder__20(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs12[2UL];
  __tempargs12[0UL] = __ins_0;
  __tempargs12[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__200_closure_12_0wrapper, __stream, __module_data, 0UL, 128UL, 1UL, __tempargs12);
  return true;
}

static bool matmul_core_cast__39(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs13[3UL];
  __tempargs13[0UL] = __ins_0;
  __tempargs13[1UL] = __ins_1;
  __tempargs13[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast__390_closure_13_0wrapper, __stream, __module_data, 0UL, 16UL, 1UL, __tempargs13);
  return true;
}

static bool reorder__22(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs14[2UL];
  __tempargs14[0UL] = __ins_0;
  __tempargs14[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__220_closure_14_0wrapper, __stream, __module_data, 0UL, 16UL, 1UL, __tempargs14);
  return true;
}

static bool reorder__23(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs15[2UL];
  __tempargs15[0UL] = __ins_0;
  __tempargs15[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__230_closure_15_0wrapper, __stream, __module_data, 0UL, 512UL, 8UL, __tempargs15);
  return true;
}

static bool reorder__24(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs16[2UL];
  __tempargs16[0UL] = __ins_0;
  __tempargs16[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__240_closure_16_0wrapper, __stream, __module_data, 0UL, 2048UL, 1UL, __tempargs16);
  return true;
}

static bool matmul_core_cast__37(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs17[3UL];
  __tempargs17[0UL] = __ins_0;
  __tempargs17[1UL] = __ins_1;
  __tempargs17[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast__370_closure_17_0wrapper, __stream, __module_data, 0UL, 128UL, 1UL, __tempargs17);
  return true;
}

static bool reorder__25(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs18[2UL];
  __tempargs18[0UL] = __ins_0;
  __tempargs18[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__250_closure_18_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs18);
  return true;
}

extern "C" void sc_init_mlp_training_backward_4k() {
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  void*& __sc_kernel_cache_3 = *(void**)(__module_data + 8);
  void*& __sc_kernel_cache_4 = *(void**)(__module_data + 16);
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  void*& __sc_kernel_cache_6 = *(void**)(__module_data + 32);
  __sc_kernel_cache = dnnl_brgemm_func(64, 64, 64, 128, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_3 = dnnl_brgemm_func(32, 32, 64, 4096, 32, 32, 64, 2048, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_4 = dnnl_brgemm_func(64, 64, 64, 256, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_5 = dnnl_brgemm_func(32, 16, 64, 512, 16, 16, 64, 1024, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_6 = dnnl_brgemm_func(13, 32, 64, 4096, 32, 32, 64, 2048, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
}

static void reorder__170_closure_0(uint64_t fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner = 0UL; fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner < 256UL; fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner += 1UL) {
    for (uint64_t _fuseiter_35 = 0UL; _fuseiter_35 < 2UL; _fuseiter_35 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 8192UL) * 16384UL) + ((((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 2048UL) % 4UL) * 4096UL) + ((((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 64UL) % 32UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) % 64UL) * 2UL) + _fuseiter_35))))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((((_fuseiter_35 + (((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 2048UL) % 4UL) * 64UL)) / 64UL) * 8192UL) + (((((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 8192UL) * 64UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 8192UL) * 64UL)) % 64UL) / 2UL) * 128UL) + (((((_fuseiter_35 + (((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 2048UL) % 4UL) * 64UL)) % 64UL) * 2UL) + ((((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_31___fuseiter_32_5___fuseiter_33_6___fuseiter_34_7_0inner) / 8192UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__170_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__170_closure_0(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void select_one_mul_reduce__410_closure_1(uint64_t __itr_0_0outer, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  for (uint64_t __itr_0_0inner = 0UL; __itr_0_0inner < 16UL; __itr_0_0inner += 1UL) {
    uint16_t _select_one_buf_0_shr[16];
    for (uint64_t _fuseiter37 = 0UL; _fuseiter37 < 128UL; _fuseiter37 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter37)]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_0 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_0)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_1[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter37)]))) << vec_u32x16(16UL)));
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_1 = (__cached_2 * __cached_1);
      __cached_3 = tobf16(_arg_cache_1);
      vec_u16x16::store(__cached_3, &__outs_0[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter37)]);
    }
  }
}

static void select_one_mul_reduce__410_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  select_one_mul_reduce__410_closure_1(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__180_closure_2(uint64_t fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner = 0UL; fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner < 4UL; fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner += 1UL) {
    for (uint64_t _fuseiter_47 = 0UL; _fuseiter_47 < 64UL; _fuseiter_47 += 1UL) {
      for (uint64_t _fuseiter_48 = 0UL; _fuseiter_48 < 2UL; _fuseiter_48 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 256UL) * 32768UL) + ((((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 32UL) % 8UL) * 4096UL) + (((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) % 32UL) * 128UL) + ((_fuseiter_47 * 2UL) + _fuseiter_48))))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((_fuseiter_48 + ((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 32UL) % 8UL) * 64UL)) / 64UL) * 16384UL) + ((((_fuseiter_47 + ((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 256UL) * 64UL)) / 64UL) * 4096UL) + (((((_fuseiter_47 + ((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 256UL) * 64UL)) % 64UL) / 2UL) * 128UL) + (((((_fuseiter_48 + ((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 32UL) % 8UL) * 64UL)) % 64UL) * 2UL) + (((_fuseiter_47 + ((((fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0outer * 4UL) + fused_0fused_0_fuseiter_44___fuseiter_45_8___fuseiter_46_9_0inner) / 256UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
      }
    }
  }
}

static void reorder__180_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__180_closure_2(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_select_one_cast_mul__360_closure_3(uint64_t fused_0m_o__n_o_10, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_140_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache, &__ins_0[((fused_0m_o__n_o_10 / 4UL) * 8192UL)], &__ins_1[((fused_0m_o__n_o_10 % 4UL) * 8192UL)], &__origouts_140_shr[0UL], 2, __stream);
  for (uint64_t _fuseiter49 = 0UL; _fuseiter49 < 64UL; _fuseiter49 += 1UL) {
    for (uint64_t _fuseiter50 = 0UL; _fuseiter50 < 64UL; _fuseiter50 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[((((fused_0m_o__n_o_10 / 4UL) * 16384UL) + ((fused_0m_o__n_o_10 % 4UL) * 64UL)) + ((_fuseiter49 * 256UL) + _fuseiter50))]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_2 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_2)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = vec_f32x16::load(&__origouts_140_shr[((_fuseiter49 * 64UL) + _fuseiter50)]);
      vec_f32x16 __cached_3;
      __cached_3 = __cached_2;
      vec_u16x16 __cached_4;
      vec_f32x16 _arg_cache_3 = (__cached_3 * __cached_1);
      __cached_4 = tobf16(_arg_cache_3);
      vec_u16x16::store(__cached_4, &__outs_0[((((fused_0m_o__n_o_10 / 4UL) * 16384UL) + ((fused_0m_o__n_o_10 % 4UL) * 64UL)) + ((_fuseiter49 * 256UL) + _fuseiter50))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_select_one_cast_mul__360_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_select_one_cast_mul__360_closure_3(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void reorder__190_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner = 0UL; fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner < 256UL; fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner += 1UL) {
    for (uint64_t _fuseiter_64 = 0UL; _fuseiter_64 < 2UL; _fuseiter_64 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) % 64UL) * 2UL) + _fuseiter_64)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_64 + (((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 64UL) % 8UL) * 2UL)) / 16UL) * 8192UL) + (((((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 512UL) * 64UL)) / 64UL) * 1024UL) + ((((((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 512UL) * 64UL)) % 64UL) / 2UL) * 32UL) + ((((_fuseiter_64 + (((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 64UL) % 8UL) * 2UL)) % 16UL) * 2UL) + ((((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_60___fuseiter_61_11___fuseiter_62_12___fuseiter_63_13_0inner) / 512UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__190_closure_4_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__190_closure_4(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__260_closure_5(uint64_t _fuseiter_65, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_66 = 0UL; _fuseiter_66 < 4096UL; _fuseiter_66 += 32UL) {
    vec_u16x32 row1_2;
    vec_u16x32 row2_3;
    vec_u16x32 row3_4;
    vec_u16x32 row4_5;
    vec_u16x32 row5_6;
    vec_u16x32 row6_7;
    vec_u16x32 row7_8;
    vec_u16x32 row8_9;
    vec_u16x32 row9_10;
    vec_u16x32 row10_11;
    vec_u16x32 row11_12;
    vec_u16x32 row12_13;
    vec_u16x32 row13_14;
    vec_u16x32 row14_15;
    vec_u16x32 row15_16;
    vec_u16x32 row16_17;
    vec_u16x8 __cached_0;
    __cached_0 = vec_u16x8::load(&__ins_0[((_fuseiter_66 * 256UL) + _fuseiter_65)]);
    row1_2 = vec_u16x32(__cached_0);
    vec_u16x8 __cached_1;
    __cached_1 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 8UL) * 256UL) + _fuseiter_65)]);
    row1_2 = sc_select(65280, vec_u16x32(__cached_1), row1_2);
    vec_u16x8 __cached_2;
    __cached_2 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 16UL) * 256UL) + _fuseiter_65)]);
    row1_2 = sc_select(16711680, vec_u16x32(__cached_2), row1_2);
    vec_u16x8 __cached_3;
    __cached_3 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 24UL) * 256UL) + _fuseiter_65)]);
    row1_2 = sc_select(-16777216, vec_u16x32(__cached_3), row1_2);
    vec_u16x8 __cached_4;
    __cached_4 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 1UL) * 256UL) + _fuseiter_65)]);
    row2_3 = vec_u16x32(__cached_4);
    vec_u16x8 __cached_5;
    __cached_5 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 9UL) * 256UL) + _fuseiter_65)]);
    row2_3 = sc_select(65280, vec_u16x32(__cached_5), row2_3);
    vec_u16x8 __cached_6;
    __cached_6 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 17UL) * 256UL) + _fuseiter_65)]);
    row2_3 = sc_select(16711680, vec_u16x32(__cached_6), row2_3);
    vec_u16x8 __cached_7;
    __cached_7 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 25UL) * 256UL) + _fuseiter_65)]);
    row2_3 = sc_select(-16777216, vec_u16x32(__cached_7), row2_3);
    vec_u16x8 __cached_8;
    __cached_8 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 2UL) * 256UL) + _fuseiter_65)]);
    row3_4 = vec_u16x32(__cached_8);
    vec_u16x8 __cached_9;
    __cached_9 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 10UL) * 256UL) + _fuseiter_65)]);
    row3_4 = sc_select(65280, vec_u16x32(__cached_9), row3_4);
    vec_u16x8 __cached_10;
    __cached_10 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 18UL) * 256UL) + _fuseiter_65)]);
    row3_4 = sc_select(16711680, vec_u16x32(__cached_10), row3_4);
    vec_u16x8 __cached_11;
    __cached_11 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 26UL) * 256UL) + _fuseiter_65)]);
    row3_4 = sc_select(-16777216, vec_u16x32(__cached_11), row3_4);
    vec_u16x8 __cached_12;
    __cached_12 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 3UL) * 256UL) + _fuseiter_65)]);
    row4_5 = vec_u16x32(__cached_12);
    vec_u16x8 __cached_13;
    __cached_13 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 11UL) * 256UL) + _fuseiter_65)]);
    row4_5 = sc_select(65280, vec_u16x32(__cached_13), row4_5);
    vec_u16x8 __cached_14;
    __cached_14 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 19UL) * 256UL) + _fuseiter_65)]);
    row4_5 = sc_select(16711680, vec_u16x32(__cached_14), row4_5);
    vec_u16x8 __cached_15;
    __cached_15 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 27UL) * 256UL) + _fuseiter_65)]);
    row4_5 = sc_select(-16777216, vec_u16x32(__cached_15), row4_5);
    vec_u16x8 __cached_16;
    __cached_16 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 4UL) * 256UL) + _fuseiter_65)]);
    row5_6 = vec_u16x32(__cached_16);
    vec_u16x8 __cached_17;
    __cached_17 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 12UL) * 256UL) + _fuseiter_65)]);
    row5_6 = sc_select(65280, vec_u16x32(__cached_17), row5_6);
    vec_u16x8 __cached_18;
    __cached_18 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 20UL) * 256UL) + _fuseiter_65)]);
    row5_6 = sc_select(16711680, vec_u16x32(__cached_18), row5_6);
    vec_u16x8 __cached_19;
    __cached_19 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 28UL) * 256UL) + _fuseiter_65)]);
    row5_6 = sc_select(-16777216, vec_u16x32(__cached_19), row5_6);
    vec_u16x8 __cached_20;
    __cached_20 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 5UL) * 256UL) + _fuseiter_65)]);
    row6_7 = vec_u16x32(__cached_20);
    vec_u16x8 __cached_21;
    __cached_21 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 13UL) * 256UL) + _fuseiter_65)]);
    row6_7 = sc_select(65280, vec_u16x32(__cached_21), row6_7);
    vec_u16x8 __cached_22;
    __cached_22 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 21UL) * 256UL) + _fuseiter_65)]);
    row6_7 = sc_select(16711680, vec_u16x32(__cached_22), row6_7);
    vec_u16x8 __cached_23;
    __cached_23 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 29UL) * 256UL) + _fuseiter_65)]);
    row6_7 = sc_select(-16777216, vec_u16x32(__cached_23), row6_7);
    vec_u16x8 __cached_24;
    __cached_24 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 6UL) * 256UL) + _fuseiter_65)]);
    row7_8 = vec_u16x32(__cached_24);
    vec_u16x8 __cached_25;
    __cached_25 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 14UL) * 256UL) + _fuseiter_65)]);
    row7_8 = sc_select(65280, vec_u16x32(__cached_25), row7_8);
    vec_u16x8 __cached_26;
    __cached_26 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 22UL) * 256UL) + _fuseiter_65)]);
    row7_8 = sc_select(16711680, vec_u16x32(__cached_26), row7_8);
    vec_u16x8 __cached_27;
    __cached_27 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 30UL) * 256UL) + _fuseiter_65)]);
    row7_8 = sc_select(-16777216, vec_u16x32(__cached_27), row7_8);
    vec_u16x8 __cached_28;
    __cached_28 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 7UL) * 256UL) + _fuseiter_65)]);
    row8_9 = vec_u16x32(__cached_28);
    vec_u16x8 __cached_29;
    __cached_29 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 15UL) * 256UL) + _fuseiter_65)]);
    row8_9 = sc_select(65280, vec_u16x32(__cached_29), row8_9);
    vec_u16x8 __cached_30;
    __cached_30 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 23UL) * 256UL) + _fuseiter_65)]);
    row8_9 = sc_select(16711680, vec_u16x32(__cached_30), row8_9);
    vec_u16x8 __cached_31;
    __cached_31 = vec_u16x8::load(&__ins_0[(((_fuseiter_66 + 31UL) * 256UL) + _fuseiter_65)]);
    row8_9 = sc_select(-16777216, vec_u16x32(__cached_31), row8_9);
    row9_10 = sc_unpack_low(row1_2, row2_3, 16);
    row10_11 = sc_unpack_high(row1_2, row2_3, 16);
    row11_12 = sc_unpack_low(row3_4, row4_5, 16);
    row12_13 = sc_unpack_high(row3_4, row4_5, 16);
    row13_14 = sc_unpack_low(row5_6, row6_7, 16);
    row14_15 = sc_unpack_high(row5_6, row6_7, 16);
    row15_16 = sc_unpack_low(row7_8, row8_9, 16);
    row16_17 = sc_unpack_high(row7_8, row8_9, 16);
    row1_2 = sc_unpack_low(row9_10, row11_12, 32);
    row2_3 = sc_unpack_high(row9_10, row11_12, 32);
    row3_4 = sc_unpack_low(row10_11, row12_13, 32);
    row4_5 = sc_unpack_high(row10_11, row12_13, 32);
    row5_6 = sc_unpack_low(row13_14, row15_16, 32);
    row6_7 = sc_unpack_high(row13_14, row15_16, 32);
    row7_8 = sc_unpack_low(row14_15, row16_17, 32);
    row8_9 = sc_unpack_high(row14_15, row16_17, 32);
    row9_10 = sc_unpack_low(row1_2, row5_6, 64);
    row10_11 = sc_unpack_high(row1_2, row5_6, 64);
    row11_12 = sc_unpack_low(row2_3, row6_7, 64);
    row12_13 = sc_unpack_high(row2_3, row6_7, 64);
    row13_14 = sc_unpack_low(row3_4, row7_8, 64);
    row14_15 = sc_unpack_high(row3_4, row7_8, 64);
    row15_16 = sc_unpack_low(row4_5, row8_9, 64);
    row16_17 = sc_unpack_high(row4_5, row8_9, 64);
    vec_u16x32 __cached_32;
    __cached_32 = row9_10;
    vec_u16x32::store(__cached_32, &__outs_0[((_fuseiter_65 * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_33;
    __cached_33 = row10_11;
    vec_u16x32::store(__cached_33, &__outs_0[(((_fuseiter_65 + 1UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_34;
    __cached_34 = row11_12;
    vec_u16x32::store(__cached_34, &__outs_0[(((_fuseiter_65 + 2UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_35;
    __cached_35 = row12_13;
    vec_u16x32::store(__cached_35, &__outs_0[(((_fuseiter_65 + 3UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_36;
    __cached_36 = row13_14;
    vec_u16x32::store(__cached_36, &__outs_0[(((_fuseiter_65 + 4UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_37;
    __cached_37 = row14_15;
    vec_u16x32::store(__cached_37, &__outs_0[(((_fuseiter_65 + 5UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_38;
    __cached_38 = row15_16;
    vec_u16x32::store(__cached_38, &__outs_0[(((_fuseiter_65 + 6UL) * 4096UL) + _fuseiter_66)]);
    vec_u16x32 __cached_39;
    __cached_39 = row16_17;
    vec_u16x32::store(__cached_39, &__outs_0[(((_fuseiter_65 + 7UL) * 4096UL) + _fuseiter_66)]);
  }
}

static void reorder__260_closure_5_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__260_closure_5(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__270_closure_6(uint64_t fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner = 0UL; fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner < 8UL; fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner += 1UL) {
    for (uint64_t _fuseiter_70 = 0UL; _fuseiter_70 < 32UL; _fuseiter_70 += 1UL) {
      for (uint64_t _fuseiter_71 = 0UL; _fuseiter_71 < 2UL; _fuseiter_71 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_71 + ((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) / 32UL) % 64UL) * 64UL)) * 128UL) + (_fuseiter_70 + ((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) / 2048UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) / 2048UL) * 131072UL) + ((((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) / 32UL) % 64UL) * 2048UL) + (((((fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0outer * 8UL) + fused_0fused_0_fuseiter_67___fuseiter_68_14___fuseiter_69_15_0inner) % 32UL) * 64UL) + ((_fuseiter_70 * 2UL) + _fuseiter_71))))] = __cached_1;
      }
    }
  }
}

static void reorder__270_closure_6_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__270_closure_6(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__350_closure_7(uint64_t fused_0m_o__n_o_16, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_3 = *(void**)(__module_data + 8);
  float* __origouts_150_shr = (float*)sc_thread_aligned_malloc(__stream, 4096UL);
  dnnl_brgemm_call(__sc_kernel_cache_3, &__ins_0[((fused_0m_o__n_o_16 / 4UL) * 131072UL)], &__ins_1[((fused_0m_o__n_o_16 % 4UL) * 131072UL)], &__origouts_150_shr[0UL], 64, __stream);
  for (uint64_t _fuseiter74 = 0UL; _fuseiter74 < 32UL; _fuseiter74 += 1UL) {
    for (uint64_t _fuseiter75 = 0UL; _fuseiter75 < 32UL; _fuseiter75 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_150_shr[((_fuseiter74 * 32UL) + _fuseiter75)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_16 / 4UL) * 4096UL) + ((fused_0m_o__n_o_16 % 4UL) * 1024UL)) + ((_fuseiter74 * 32UL) + _fuseiter75))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_150_shr);
}

static void matmul_core_cast__350_closure_7_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__350_closure_7(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__280_closure_8(uint64_t fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner = 0UL; fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner < 16UL; fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner += 1UL) {
    for (uint64_t _fuseiter_80 = 0UL; _fuseiter_80 < 32UL; _fuseiter_80 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 128UL) * 4096UL) + ((((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 32UL) % 4UL) * 1024UL) + (((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) % 32UL) * 32UL) + _fuseiter_80)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_80 + (((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 32UL) % 4UL) * 32UL)) / 64UL) * 16384UL) + (((((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 128UL) * 32UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 128UL) * 32UL)) % 64UL) / 2UL) * 128UL) + ((((_fuseiter_80 + (((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 32UL) % 4UL) * 32UL)) % 64UL) * 2UL) + ((((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0outer * 16UL) + fused_0fused_0_fuseiter_77___fuseiter_78_17___fuseiter_79_18_0inner) / 128UL) * 32UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__280_closure_8_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__280_closure_8(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_select_one_cast_mul__380_closure_9(uint64_t fused_0m_o__n_o_19, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_4 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_160_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_4, &__ins_0[((fused_0m_o__n_o_19 / 8UL) * 16384UL)], &__ins_1[((fused_0m_o__n_o_19 % 8UL) * 16384UL)], &__origouts_160_shr[0UL], 4, __stream);
  for (uint64_t _fuseiter81 = 0UL; _fuseiter81 < 64UL; _fuseiter81 += 1UL) {
    for (uint64_t _fuseiter82 = 0UL; _fuseiter82 < 64UL; _fuseiter82 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[((((fused_0m_o__n_o_19 / 8UL) * 32768UL) + ((fused_0m_o__n_o_19 % 8UL) * 64UL)) + ((_fuseiter81 * 512UL) + _fuseiter82))]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_4 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_4)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = vec_f32x16::load(&__origouts_160_shr[((_fuseiter81 * 64UL) + _fuseiter82)]);
      vec_f32x16 __cached_3;
      __cached_3 = __cached_2;
      vec_u16x16 __cached_4;
      vec_f32x16 _arg_cache_5 = (__cached_3 * __cached_1);
      __cached_4 = tobf16(_arg_cache_5);
      vec_u16x16::store(__cached_4, &__outs_0[((((fused_0m_o__n_o_19 / 8UL) * 32768UL) + ((fused_0m_o__n_o_19 % 8UL) * 64UL)) + ((_fuseiter81 * 512UL) + _fuseiter82))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_select_one_cast_mul__380_closure_9_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_select_one_cast_mul__380_closure_9(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_reorder__400_closure_10(uint64_t fused_0m_o__n_o_20, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_5 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 2112UL);
  float* __origouts_170_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_5, &__ins_0[(fused_0m_o__n_o_20 * 16384UL)], &__ins_1[0UL], &__origouts_170_shr[0UL], 8, __stream);
  uint16_t* _cast_buf_0_shr = (uint16_t*)&__rescheduled_1[2048UL];
  for (uint64_t _fuseiter92 = 0UL; _fuseiter92 < 32UL; _fuseiter92 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__origouts_170_shr[(_fuseiter92 * 16UL)]);
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(__cached_0);
    vec_u16x16::store(__cached_1, &_cast_buf_0_shr[0UL]);
    for (uint64_t _fuseiter_98 = 0UL; _fuseiter_98 < 16UL; _fuseiter_98 += 1UL) {
      if (((_fuseiter_98 < 13UL) && ((_fuseiter92 + (fused_0m_o__n_o_20 * 32UL)) < 4096UL))) {
        uint16_t __cached_2;
        __cached_2 = _cast_buf_0_shr[_fuseiter_98];
        uint16_t __cached_3;
        __cached_3 = __cached_2;
        __outs_0[(((_fuseiter92 + (fused_0m_o__n_o_20 * 32UL)) * 13UL) + _fuseiter_98)] = __cached_3;
      }
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_reorder__400_closure_10_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_reorder__400_closure_10(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__210_closure_11(uint64_t fused_0_fuseiter_101___fuseiter_102_21, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_103 = 0UL; _fuseiter_103 < 32UL; _fuseiter_103 += 1UL) {
    for (uint64_t _fuseiter_104 = 0UL; _fuseiter_104 < 32UL; _fuseiter_104 += 1UL) {
      for (uint64_t _fuseiter_105 = 0UL; _fuseiter_105 < 2UL; _fuseiter_105 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_105 + (_fuseiter_103 * 2UL)) + ((fused_0_fuseiter_101___fuseiter_102_21 % 64UL) * 64UL)) * 512UL) + (_fuseiter_104 + ((fused_0_fuseiter_101___fuseiter_102_21 / 64UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_101___fuseiter_102_21 / 64UL) * 131072UL) + (((fused_0_fuseiter_101___fuseiter_102_21 % 64UL) * 2048UL) + ((_fuseiter_103 * 64UL) + ((_fuseiter_104 * 2UL) + _fuseiter_105))))] = __cached_1;
      }
    }
  }
}

static void reorder__210_closure_11_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__210_closure_11(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__200_closure_12(uint64_t _fuseiter_106_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_106_0inner = 0UL; _fuseiter_106_0inner < 32UL; _fuseiter_106_0inner += 1UL) {
    for (uint64_t _fuseiter_107 = 0UL; _fuseiter_107 < 13UL; _fuseiter_107 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[((((_fuseiter_106_0outer * 32UL) + _fuseiter_106_0inner) * 13UL) + _fuseiter_107)];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((_fuseiter_107 * 4096UL) + ((_fuseiter_106_0outer * 32UL) + _fuseiter_106_0inner))] = __cached_1;
    }
  }
}

static void reorder__200_closure_12_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__200_closure_12(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__390_closure_13(uint64_t fused_0m_o__n_o_22, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_6 = *(void**)(__module_data + 32);
  float* __origouts_180_shr = (float*)sc_thread_aligned_malloc(__stream, 1664UL);
  dnnl_brgemm_call(__sc_kernel_cache_6, &__ins_0[((fused_0m_o__n_o_22 / 16UL) * 53248UL)], &__ins_1[((fused_0m_o__n_o_22 % 16UL) * 131072UL)], &__origouts_180_shr[0UL], 64, __stream);
  for (uint64_t _fuseiter110 = 0UL; _fuseiter110 < 13UL; _fuseiter110 += 1UL) {
    for (uint64_t _fuseiter111 = 0UL; _fuseiter111 < 32UL; _fuseiter111 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_180_shr[((_fuseiter110 * 32UL) + _fuseiter111)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_22 / 16UL) * 6656UL) + ((fused_0m_o__n_o_22 % 16UL) * 416UL)) + ((_fuseiter110 * 32UL) + _fuseiter111))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_180_shr);
}

static void matmul_core_cast__390_closure_13_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__390_closure_13(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__220_closure_14(uint64_t fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner = 0UL; fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner < 256UL; fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner += 1UL) {
    for (uint64_t _fuseiter_117 = 0UL; _fuseiter_117 < 2UL; _fuseiter_117 += 1UL) {
      if (((((_fuseiter_117 + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 2UL)) < 16UL) && ((_fuseiter_117 + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 2UL)) < 13UL)) && (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 512UL) * 64UL)) < 512UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_117 + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 2UL)) / 13UL) * 6656UL) + (((((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 512UL) * 64UL)) / 32UL) * 416UL) + ((((_fuseiter_117 + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 2UL)) % 13UL) * 32UL) + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 512UL) * 64UL)) % 32UL))))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) % 64UL) * 2UL) + _fuseiter_117)))] = __cached_1;
      } else {
        uint16_t __cached_2;
        __cached_2 = 0UL;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_113___fuseiter_114_23___fuseiter_115_24___fuseiter_116_25_0inner) % 64UL) * 2UL) + _fuseiter_117)))] = __cached_2;
      }
    }
  }
}

static void reorder__220_closure_14_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__220_closure_14(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__230_closure_15(uint64_t _fuseiter_118, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_119 = 0UL; _fuseiter_119 < 4096UL; _fuseiter_119 += 32UL) {
    vec_u16x32 row1_19;
    vec_u16x32 row2_20;
    vec_u16x32 row3_21;
    vec_u16x32 row4_22;
    vec_u16x32 row5_23;
    vec_u16x32 row6_24;
    vec_u16x32 row7_25;
    vec_u16x32 row8_26;
    vec_u16x32 row9_27;
    vec_u16x32 row10_28;
    vec_u16x32 row11_29;
    vec_u16x32 row12_30;
    vec_u16x32 row13_31;
    vec_u16x32 row14_32;
    vec_u16x32 row15_33;
    vec_u16x32 row16_34;
    vec_u16x8 __cached_0;
    __cached_0 = vec_u16x8::load(&__ins_0[((_fuseiter_119 * 512UL) + _fuseiter_118)]);
    row1_19 = vec_u16x32(__cached_0);
    vec_u16x8 __cached_1;
    __cached_1 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 8UL) * 512UL) + _fuseiter_118)]);
    row1_19 = sc_select(65280, vec_u16x32(__cached_1), row1_19);
    vec_u16x8 __cached_2;
    __cached_2 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 16UL) * 512UL) + _fuseiter_118)]);
    row1_19 = sc_select(16711680, vec_u16x32(__cached_2), row1_19);
    vec_u16x8 __cached_3;
    __cached_3 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 24UL) * 512UL) + _fuseiter_118)]);
    row1_19 = sc_select(-16777216, vec_u16x32(__cached_3), row1_19);
    vec_u16x8 __cached_4;
    __cached_4 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 1UL) * 512UL) + _fuseiter_118)]);
    row2_20 = vec_u16x32(__cached_4);
    vec_u16x8 __cached_5;
    __cached_5 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 9UL) * 512UL) + _fuseiter_118)]);
    row2_20 = sc_select(65280, vec_u16x32(__cached_5), row2_20);
    vec_u16x8 __cached_6;
    __cached_6 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 17UL) * 512UL) + _fuseiter_118)]);
    row2_20 = sc_select(16711680, vec_u16x32(__cached_6), row2_20);
    vec_u16x8 __cached_7;
    __cached_7 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 25UL) * 512UL) + _fuseiter_118)]);
    row2_20 = sc_select(-16777216, vec_u16x32(__cached_7), row2_20);
    vec_u16x8 __cached_8;
    __cached_8 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 2UL) * 512UL) + _fuseiter_118)]);
    row3_21 = vec_u16x32(__cached_8);
    vec_u16x8 __cached_9;
    __cached_9 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 10UL) * 512UL) + _fuseiter_118)]);
    row3_21 = sc_select(65280, vec_u16x32(__cached_9), row3_21);
    vec_u16x8 __cached_10;
    __cached_10 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 18UL) * 512UL) + _fuseiter_118)]);
    row3_21 = sc_select(16711680, vec_u16x32(__cached_10), row3_21);
    vec_u16x8 __cached_11;
    __cached_11 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 26UL) * 512UL) + _fuseiter_118)]);
    row3_21 = sc_select(-16777216, vec_u16x32(__cached_11), row3_21);
    vec_u16x8 __cached_12;
    __cached_12 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 3UL) * 512UL) + _fuseiter_118)]);
    row4_22 = vec_u16x32(__cached_12);
    vec_u16x8 __cached_13;
    __cached_13 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 11UL) * 512UL) + _fuseiter_118)]);
    row4_22 = sc_select(65280, vec_u16x32(__cached_13), row4_22);
    vec_u16x8 __cached_14;
    __cached_14 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 19UL) * 512UL) + _fuseiter_118)]);
    row4_22 = sc_select(16711680, vec_u16x32(__cached_14), row4_22);
    vec_u16x8 __cached_15;
    __cached_15 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 27UL) * 512UL) + _fuseiter_118)]);
    row4_22 = sc_select(-16777216, vec_u16x32(__cached_15), row4_22);
    vec_u16x8 __cached_16;
    __cached_16 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 4UL) * 512UL) + _fuseiter_118)]);
    row5_23 = vec_u16x32(__cached_16);
    vec_u16x8 __cached_17;
    __cached_17 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 12UL) * 512UL) + _fuseiter_118)]);
    row5_23 = sc_select(65280, vec_u16x32(__cached_17), row5_23);
    vec_u16x8 __cached_18;
    __cached_18 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 20UL) * 512UL) + _fuseiter_118)]);
    row5_23 = sc_select(16711680, vec_u16x32(__cached_18), row5_23);
    vec_u16x8 __cached_19;
    __cached_19 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 28UL) * 512UL) + _fuseiter_118)]);
    row5_23 = sc_select(-16777216, vec_u16x32(__cached_19), row5_23);
    vec_u16x8 __cached_20;
    __cached_20 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 5UL) * 512UL) + _fuseiter_118)]);
    row6_24 = vec_u16x32(__cached_20);
    vec_u16x8 __cached_21;
    __cached_21 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 13UL) * 512UL) + _fuseiter_118)]);
    row6_24 = sc_select(65280, vec_u16x32(__cached_21), row6_24);
    vec_u16x8 __cached_22;
    __cached_22 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 21UL) * 512UL) + _fuseiter_118)]);
    row6_24 = sc_select(16711680, vec_u16x32(__cached_22), row6_24);
    vec_u16x8 __cached_23;
    __cached_23 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 29UL) * 512UL) + _fuseiter_118)]);
    row6_24 = sc_select(-16777216, vec_u16x32(__cached_23), row6_24);
    vec_u16x8 __cached_24;
    __cached_24 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 6UL) * 512UL) + _fuseiter_118)]);
    row7_25 = vec_u16x32(__cached_24);
    vec_u16x8 __cached_25;
    __cached_25 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 14UL) * 512UL) + _fuseiter_118)]);
    row7_25 = sc_select(65280, vec_u16x32(__cached_25), row7_25);
    vec_u16x8 __cached_26;
    __cached_26 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 22UL) * 512UL) + _fuseiter_118)]);
    row7_25 = sc_select(16711680, vec_u16x32(__cached_26), row7_25);
    vec_u16x8 __cached_27;
    __cached_27 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 30UL) * 512UL) + _fuseiter_118)]);
    row7_25 = sc_select(-16777216, vec_u16x32(__cached_27), row7_25);
    vec_u16x8 __cached_28;
    __cached_28 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 7UL) * 512UL) + _fuseiter_118)]);
    row8_26 = vec_u16x32(__cached_28);
    vec_u16x8 __cached_29;
    __cached_29 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 15UL) * 512UL) + _fuseiter_118)]);
    row8_26 = sc_select(65280, vec_u16x32(__cached_29), row8_26);
    vec_u16x8 __cached_30;
    __cached_30 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 23UL) * 512UL) + _fuseiter_118)]);
    row8_26 = sc_select(16711680, vec_u16x32(__cached_30), row8_26);
    vec_u16x8 __cached_31;
    __cached_31 = vec_u16x8::load(&__ins_0[(((_fuseiter_119 + 31UL) * 512UL) + _fuseiter_118)]);
    row8_26 = sc_select(-16777216, vec_u16x32(__cached_31), row8_26);
    row9_27 = sc_unpack_low(row1_19, row2_20, 16);
    row10_28 = sc_unpack_high(row1_19, row2_20, 16);
    row11_29 = sc_unpack_low(row3_21, row4_22, 16);
    row12_30 = sc_unpack_high(row3_21, row4_22, 16);
    row13_31 = sc_unpack_low(row5_23, row6_24, 16);
    row14_32 = sc_unpack_high(row5_23, row6_24, 16);
    row15_33 = sc_unpack_low(row7_25, row8_26, 16);
    row16_34 = sc_unpack_high(row7_25, row8_26, 16);
    row1_19 = sc_unpack_low(row9_27, row11_29, 32);
    row2_20 = sc_unpack_high(row9_27, row11_29, 32);
    row3_21 = sc_unpack_low(row10_28, row12_30, 32);
    row4_22 = sc_unpack_high(row10_28, row12_30, 32);
    row5_23 = sc_unpack_low(row13_31, row15_33, 32);
    row6_24 = sc_unpack_high(row13_31, row15_33, 32);
    row7_25 = sc_unpack_low(row14_32, row16_34, 32);
    row8_26 = sc_unpack_high(row14_32, row16_34, 32);
    row9_27 = sc_unpack_low(row1_19, row5_23, 64);
    row10_28 = sc_unpack_high(row1_19, row5_23, 64);
    row11_29 = sc_unpack_low(row2_20, row6_24, 64);
    row12_30 = sc_unpack_high(row2_20, row6_24, 64);
    row13_31 = sc_unpack_low(row3_21, row7_25, 64);
    row14_32 = sc_unpack_high(row3_21, row7_25, 64);
    row15_33 = sc_unpack_low(row4_22, row8_26, 64);
    row16_34 = sc_unpack_high(row4_22, row8_26, 64);
    vec_u16x32 __cached_32;
    __cached_32 = row9_27;
    vec_u16x32::store(__cached_32, &__outs_0[((_fuseiter_118 * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_33;
    __cached_33 = row10_28;
    vec_u16x32::store(__cached_33, &__outs_0[(((_fuseiter_118 + 1UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_34;
    __cached_34 = row11_29;
    vec_u16x32::store(__cached_34, &__outs_0[(((_fuseiter_118 + 2UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_35;
    __cached_35 = row12_30;
    vec_u16x32::store(__cached_35, &__outs_0[(((_fuseiter_118 + 3UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_36;
    __cached_36 = row13_31;
    vec_u16x32::store(__cached_36, &__outs_0[(((_fuseiter_118 + 4UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_37;
    __cached_37 = row14_32;
    vec_u16x32::store(__cached_37, &__outs_0[(((_fuseiter_118 + 5UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_38;
    __cached_38 = row15_33;
    vec_u16x32::store(__cached_38, &__outs_0[(((_fuseiter_118 + 6UL) * 4096UL) + _fuseiter_119)]);
    vec_u16x32 __cached_39;
    __cached_39 = row16_34;
    vec_u16x32::store(__cached_39, &__outs_0[(((_fuseiter_118 + 7UL) * 4096UL) + _fuseiter_119)]);
  }
}

static void reorder__230_closure_15_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__230_closure_15(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__240_closure_16(uint64_t fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner = 0UL; fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner < 8UL; fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner += 1UL) {
    for (uint64_t _fuseiter_123 = 0UL; _fuseiter_123 < 32UL; _fuseiter_123 += 1UL) {
      for (uint64_t _fuseiter_124 = 0UL; _fuseiter_124 < 2UL; _fuseiter_124 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_124 + ((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) / 32UL) % 64UL) * 64UL)) * 256UL) + (_fuseiter_123 + ((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) / 2048UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) / 2048UL) * 131072UL) + ((((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) / 32UL) % 64UL) * 2048UL) + (((((fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0outer * 8UL) + fused_0fused_0_fuseiter_120___fuseiter_121_26___fuseiter_122_27_0inner) % 32UL) * 64UL) + ((_fuseiter_123 * 2UL) + _fuseiter_124))))] = __cached_1;
      }
    }
  }
}

static void reorder__240_closure_16_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__240_closure_16(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__370_closure_17(uint64_t fused_0m_o__n_o_28, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_3 = *(void**)(__module_data + 8);
  float* __origouts_190_shr = (float*)sc_thread_aligned_malloc(__stream, 4096UL);
  dnnl_brgemm_call(__sc_kernel_cache_3, &__ins_0[((fused_0m_o__n_o_28 / 8UL) * 131072UL)], &__ins_1[((fused_0m_o__n_o_28 % 8UL) * 131072UL)], &__origouts_190_shr[0UL], 64, __stream);
  for (uint64_t _fuseiter127 = 0UL; _fuseiter127 < 32UL; _fuseiter127 += 1UL) {
    for (uint64_t _fuseiter128 = 0UL; _fuseiter128 < 32UL; _fuseiter128 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_190_shr[((_fuseiter127 * 32UL) + _fuseiter128)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_28 / 8UL) * 8192UL) + ((fused_0m_o__n_o_28 % 8UL) * 1024UL)) + ((_fuseiter127 * 32UL) + _fuseiter128))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_190_shr);
}

static void matmul_core_cast__370_closure_17_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__370_closure_17(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__250_closure_18(uint64_t fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner = 0UL; fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner < 16UL; fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner += 1UL) {
    for (uint64_t _fuseiter_133 = 0UL; _fuseiter_133 < 32UL; _fuseiter_133 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 256UL) * 8192UL) + ((((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 32UL) % 8UL) * 1024UL) + (((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) % 32UL) * 32UL) + _fuseiter_133)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_133 + (((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 32UL) % 8UL) * 32UL)) / 64UL) * 32768UL) + (((((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 256UL) * 32UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 256UL) * 32UL)) % 64UL) / 2UL) * 128UL) + ((((_fuseiter_133 + (((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 32UL) % 8UL) * 32UL)) % 64UL) * 2UL) + ((((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0outer * 16UL) + fused_0fused_0_fuseiter_130___fuseiter_131_29___fuseiter_132_30_0inner) / 256UL) * 32UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__250_closure_18_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__250_closure_18(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

