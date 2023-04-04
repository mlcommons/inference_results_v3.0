
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t mlp_training_backward_128k_data[64];
static constexpr int8_t* __module_data = mlp_training_backward_128k_data;
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
static void reorder__170_closure_0(uint64_t fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void select_one_mul_reduce__410_closure_1(uint64_t __itr_0_0outer, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__180_closure_2(uint64_t fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_select_one_cast_mul__360_closure_3(uint64_t fused_0m_o__n_o_39, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void reorder__190_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__260_closure_5(uint64_t _fuseiter_199, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__270_closure_6(uint64_t fused_0_fuseiter_201___fuseiter_202_43, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__350_closure_7(uint64_t fused_0m_o__n_o_44, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__280_closure_8(uint64_t fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_select_one_cast_mul__380_closure_9(uint64_t fused_0m_o__n_o_47, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_reorder__400_closure_10(uint64_t fused_0m_o__n_o_48, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__210_closure_11(uint64_t fused_0_fuseiter_235___fuseiter_236_49, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__200_closure_12(uint64_t _fuseiter_240_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__390_closure_13(uint64_t fused_0m_o__n_o_50, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__220_closure_14(uint64_t fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__230_closure_15(uint64_t _fuseiter_252, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__240_closure_16(uint64_t fused_0_fuseiter_254___fuseiter_255_54, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast__370_closure_17(uint64_t fused_0m_o__n_o_55, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0);
static void reorder__250_closure_18(uint64_t fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer, uint16_t* __ins_0, uint16_t* __outs_0);


extern "C" void mlp_training_backward_128k(uint16_t* out_grad_bias2, uint16_t* out_grad_weight2, uint16_t* out_grad_bias1, uint16_t* out_grad_weight1, uint16_t* out_grad_bias0, uint16_t* out_grad_weight0, uint16_t* out_grad_input0, uint16_t* gradient, uint16_t* in_relu_output, uint16_t* data_input2, uint16_t* weight_input2, uint16_t* data_input1, uint16_t* weight_input1, uint16_t* data_input0, uint16_t* weight_input0){
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 369360896UL);
  uint16_t* buffer_8 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__17(buffer_8, &weight_input2[0UL]);
  uint16_t* buffer_9 = (uint16_t*)&__rescheduled_0[201326592UL];
  select_one_mul_reduce__41(buffer_9, out_grad_bias2, in_relu_output, gradient);
  uint16_t* buffer_11 = (uint16_t*)&__rescheduled_0[268435456UL];
  reorder__18(buffer_11, &weight_input1[0UL]);
  uint16_t* buffer_12 = (uint16_t*)&__rescheduled_0[268697600UL];
  matmul_core_select_one_cast_mul__36(buffer_12, buffer_9, buffer_8, data_input2);
  reduce__6(out_grad_bias1, buffer_12);
  uint16_t* buffer_14 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__19(buffer_14, &weight_input0[0UL]);
  uint16_t* buffer_15 = (uint16_t*)&__rescheduled_0[134217728UL];
  reorder__26(buffer_15, &data_input2[0UL]);
  uint16_t* buffer_16 = (uint16_t*)&__rescheduled_0[335806464UL];
  reorder__27(buffer_16, buffer_9);
  uint16_t* buffer_17 = (uint16_t*)&__rescheduled_0[201326592UL];
  matmul_core_cast__35(buffer_17, buffer_15, buffer_16);
  reorder__28(out_grad_weight2, buffer_17);
  uint16_t* buffer_19 = (uint16_t*)&__rescheduled_0[134217728UL];
  matmul_core_select_one_cast_mul__38(buffer_19, buffer_12, buffer_11, data_input1);
  matmul_core_cast_reorder__40(out_grad_input0, buffer_19, buffer_14);
  reduce__11(out_grad_bias0, buffer_19);
  uint16_t* buffer_22 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__21(buffer_22, buffer_19);
  uint16_t* buffer_23 = (uint16_t*)&__rescheduled_0[134217728UL];
  reorder__20(buffer_23, &data_input0[0UL]);
  uint16_t* buffer_24 = (uint16_t*)&__rescheduled_0[137625600UL];
  matmul_core_cast__39(buffer_24, buffer_23, buffer_22);
  reorder__22(out_grad_weight0, buffer_24);
  uint16_t* buffer_26 = (uint16_t*)&__rescheduled_0[0UL];
  reorder__23(buffer_26, &data_input1[0UL]);
  uint16_t* buffer_27 = (uint16_t*)&__rescheduled_0[134217728UL];
  reorder__24(buffer_27, buffer_12);
  uint16_t* buffer_28 = (uint16_t*)&__rescheduled_0[201326592UL];
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
  sc_parallel_call_cpu_with_env((void*)&select_one_mul_reduce__410_closure_1_0wrapper, __stream, __module_data, 0UL, 8192UL, 1UL, __tempargs1);
  for (uint64_t _fuseiter_177 = 0UL; _fuseiter_177 < 128UL; _fuseiter_177 += 16UL) {
    vec_f32x16 reduce__35;
    reduce__35 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_176 = 0UL; _fuseiter_176 < 131072UL; _fuseiter_176 += 1UL) {
      vec_f32x16 __cached_4;
      __cached_4 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__outs_0[((_fuseiter_176 * 128UL) + _fuseiter_177)]))) << vec_u32x16(16UL)));
      reduce__35 = (__cached_4 + reduce__35);
    }
    vec_u16x16 __cached_5;
    __cached_5 = tobf16(reduce__35);
    vec_u16x16::store(__cached_5, &__outs_1[_fuseiter_177]);
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
  sc_parallel_call_cpu_with_env((void*)&matmul_core_select_one_cast_mul__360_closure_3_0wrapper, __stream, __module_data, 0UL, 8192UL, 1UL, __tempargs3);
  return true;
}

static bool reduce__6(uint16_t* __outs_0, uint16_t* __ins_0){
  for (uint64_t _fuseiter_193 = 0UL; _fuseiter_193 < 256UL; _fuseiter_193 += 16UL) {
    vec_f32x16 reduce__36;
    reduce__36 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_192 = 0UL; _fuseiter_192 < 131072UL; _fuseiter_192 += 1UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((_fuseiter_192 * 256UL) + _fuseiter_193)]))) << vec_u32x16(16UL)));
      reduce__36 = (__cached_0 + reduce__36);
    }
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(reduce__36);
    vec_u16x16::store(__cached_1, &__outs_0[_fuseiter_193]);
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
  sc_parallel_call_cpu_with_env((void*)&reorder__270_closure_6_0wrapper, __stream, __module_data, 0UL, 8192UL, 1UL, __tempargs6);
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
  sc_parallel_call_cpu_with_env((void*)&matmul_core_select_one_cast_mul__380_closure_9_0wrapper, __stream, __module_data, 0UL, 16384UL, 1UL, __tempargs9);
  return true;
}

static bool matmul_core_cast_reorder__40(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1){
  generic_val __tempargs10[3UL];
  __tempargs10[0UL] = __ins_0;
  __tempargs10[1UL] = __ins_1;
  __tempargs10[2UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_reorder__400_closure_10_0wrapper, __stream, __module_data, 0UL, 2048UL, 1UL, __tempargs10);
  return true;
}

static bool reduce__11(uint16_t* __outs_0, uint16_t* __ins_0){
  for (uint64_t _fuseiter_234 = 0UL; _fuseiter_234 < 512UL; _fuseiter_234 += 16UL) {
    vec_f32x16 reduce__53;
    reduce__53 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16(0UL))) << vec_u32x16(16UL)));
    for (uint64_t _fuseiter_233 = 0UL; _fuseiter_233 < 131072UL; _fuseiter_233 += 1UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((_fuseiter_233 * 512UL) + _fuseiter_234)]))) << vec_u32x16(16UL)));
      reduce__53 = (__cached_0 + reduce__53);
    }
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(reduce__53);
    vec_u16x16::store(__cached_1, &__outs_0[_fuseiter_234]);
  }
  return true;
}

static bool reorder__21(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs11[2UL];
  __tempargs11[0UL] = __ins_0;
  __tempargs11[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__210_closure_11_0wrapper, __stream, __module_data, 0UL, 32768UL, 1UL, __tempargs11);
  return true;
}

static bool reorder__20(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs12[2UL];
  __tempargs12[0UL] = __ins_0;
  __tempargs12[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__200_closure_12_0wrapper, __stream, __module_data, 0UL, 4096UL, 1UL, __tempargs12);
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
  sc_parallel_call_cpu_with_env((void*)&reorder__240_closure_16_0wrapper, __stream, __module_data, 0UL, 16384UL, 1UL, __tempargs16);
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

extern "C" void sc_init_mlp_training_backward_128k() {
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 8);
  void*& __sc_kernel_cache_10 = *(void**)(__module_data + 16);
  void*& __sc_kernel_cache_11 = *(void**)(__module_data + 24);
  void*& __sc_kernel_cache_12 = *(void**)(__module_data + 32);
  __sc_kernel_cache = dnnl_brgemm_func(64, 64, 64, 128, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_9 = dnnl_brgemm_func(32, 32, 64, 131072, 32, 32, 64, 2048, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_10 = dnnl_brgemm_func(64, 64, 64, 256, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_11 = dnnl_brgemm_func(64, 16, 64, 512, 16, 16, 64, 1024, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_12 = dnnl_brgemm_func(13, 32, 64, 131072, 32, 32, 64, 2048, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
}

static void reorder__170_closure_0(uint64_t fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner = 0UL; fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner < 256UL; fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner += 1UL) {
    for (uint64_t _fuseiter_169 = 0UL; _fuseiter_169 < 2UL; _fuseiter_169 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 8192UL) * 16384UL) + ((((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 2048UL) % 4UL) * 4096UL) + ((((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 64UL) % 32UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) % 64UL) * 2UL) + _fuseiter_169))))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((((_fuseiter_169 + (((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 2048UL) % 4UL) * 64UL)) / 64UL) * 8192UL) + (((((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 8192UL) * 64UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 8192UL) * 64UL)) % 64UL) / 2UL) * 128UL) + (((((_fuseiter_169 + (((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 2048UL) % 4UL) * 64UL)) % 64UL) * 2UL) + ((((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_165___fuseiter_166_34___fuseiter_167_35___fuseiter_168_36_0inner) / 8192UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__170_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__170_closure_0(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void select_one_mul_reduce__410_closure_1(uint64_t __itr_0_0outer, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  for (uint64_t __itr_0_0inner = 0UL; __itr_0_0inner < 16UL; __itr_0_0inner += 1UL) {
    uint16_t _select_one_buf_0_shr[16];
    for (uint64_t _fuseiter171 = 0UL; _fuseiter171 < 128UL; _fuseiter171 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_0[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter171)]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_0 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_0)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_1[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter171)]))) << vec_u32x16(16UL)));
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_1 = (__cached_2 * __cached_1);
      __cached_3 = tobf16(_arg_cache_1);
      vec_u16x16::store(__cached_3, &__outs_0[((((__itr_0_0outer * 16UL) + __itr_0_0inner) * 128UL) + _fuseiter171)]);
    }
  }
}

static void select_one_mul_reduce__410_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  select_one_mul_reduce__410_closure_1(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__180_closure_2(uint64_t fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner = 0UL; fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner < 4UL; fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner += 1UL) {
    for (uint64_t _fuseiter_181 = 0UL; _fuseiter_181 < 64UL; _fuseiter_181 += 1UL) {
      for (uint64_t _fuseiter_182 = 0UL; _fuseiter_182 < 2UL; _fuseiter_182 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 256UL) * 32768UL) + ((((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 32UL) % 8UL) * 4096UL) + (((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) % 32UL) * 128UL) + ((_fuseiter_181 * 2UL) + _fuseiter_182))))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((_fuseiter_182 + ((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 32UL) % 8UL) * 64UL)) / 64UL) * 16384UL) + ((((_fuseiter_181 + ((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 256UL) * 64UL)) / 64UL) * 4096UL) + (((((_fuseiter_181 + ((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 256UL) * 64UL)) % 64UL) / 2UL) * 128UL) + (((((_fuseiter_182 + ((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 32UL) % 8UL) * 64UL)) % 64UL) * 2UL) + (((_fuseiter_181 + ((((fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0outer * 4UL) + fused_0fused_0_fuseiter_178___fuseiter_179_37___fuseiter_180_38_0inner) / 256UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
      }
    }
  }
}

static void reorder__180_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__180_closure_2(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_select_one_cast_mul__360_closure_3(uint64_t fused_0m_o__n_o_39, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_340_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache, &__ins_0[((fused_0m_o__n_o_39 / 4UL) * 8192UL)], &__ins_1[((fused_0m_o__n_o_39 % 4UL) * 8192UL)], &__origouts_340_shr[0UL], 2, __stream);
  for (uint64_t _fuseiter183 = 0UL; _fuseiter183 < 64UL; _fuseiter183 += 1UL) {
    for (uint64_t _fuseiter184 = 0UL; _fuseiter184 < 64UL; _fuseiter184 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[((((fused_0m_o__n_o_39 / 4UL) * 16384UL) + ((fused_0m_o__n_o_39 % 4UL) * 64UL)) + ((_fuseiter183 * 256UL) + _fuseiter184))]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_2 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_2)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = vec_f32x16::load(&__origouts_340_shr[((_fuseiter183 * 64UL) + _fuseiter184)]);
      vec_f32x16 __cached_3;
      __cached_3 = __cached_2;
      vec_u16x16 __cached_4;
      vec_f32x16 _arg_cache_3 = (__cached_3 * __cached_1);
      __cached_4 = tobf16(_arg_cache_3);
      vec_u16x16::store(__cached_4, &__outs_0[((((fused_0m_o__n_o_39 / 4UL) * 16384UL) + ((fused_0m_o__n_o_39 % 4UL) * 64UL)) + ((_fuseiter183 * 256UL) + _fuseiter184))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_select_one_cast_mul__360_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_select_one_cast_mul__360_closure_3(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void reorder__190_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner = 0UL; fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner < 256UL; fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner += 1UL) {
    for (uint64_t _fuseiter_198 = 0UL; _fuseiter_198 < 2UL; _fuseiter_198 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) % 64UL) * 2UL) + _fuseiter_198)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_198 + (((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 64UL) % 8UL) * 2UL)) / 16UL) * 8192UL) + (((((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 512UL) * 64UL)) / 64UL) * 1024UL) + ((((((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 512UL) * 64UL)) % 64UL) / 2UL) * 32UL) + ((((_fuseiter_198 + (((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 64UL) % 8UL) * 2UL)) % 16UL) * 2UL) + ((((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_194___fuseiter_195_40___fuseiter_196_41___fuseiter_197_42_0inner) / 512UL) * 64UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__190_closure_4_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__190_closure_4(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__260_closure_5(uint64_t _fuseiter_199, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_200 = 0UL; _fuseiter_200 < 131072UL; _fuseiter_200 += 32UL) {
    vec_u16x32 row1_37;
    vec_u16x32 row2_38;
    vec_u16x32 row3_39;
    vec_u16x32 row4_40;
    vec_u16x32 row5_41;
    vec_u16x32 row6_42;
    vec_u16x32 row7_43;
    vec_u16x32 row8_44;
    vec_u16x32 row9_45;
    vec_u16x32 row10_46;
    vec_u16x32 row11_47;
    vec_u16x32 row12_48;
    vec_u16x32 row13_49;
    vec_u16x32 row14_50;
    vec_u16x32 row15_51;
    vec_u16x32 row16_52;
    vec_u16x8 __cached_0;
    __cached_0 = vec_u16x8::load(&__ins_0[((_fuseiter_200 * 256UL) + _fuseiter_199)]);
    row1_37 = vec_u16x32(__cached_0);
    vec_u16x8 __cached_1;
    __cached_1 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 8UL) * 256UL) + _fuseiter_199)]);
    row1_37 = sc_select(65280, vec_u16x32(__cached_1), row1_37);
    vec_u16x8 __cached_2;
    __cached_2 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 16UL) * 256UL) + _fuseiter_199)]);
    row1_37 = sc_select(16711680, vec_u16x32(__cached_2), row1_37);
    vec_u16x8 __cached_3;
    __cached_3 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 24UL) * 256UL) + _fuseiter_199)]);
    row1_37 = sc_select(-16777216, vec_u16x32(__cached_3), row1_37);
    vec_u16x8 __cached_4;
    __cached_4 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 1UL) * 256UL) + _fuseiter_199)]);
    row2_38 = vec_u16x32(__cached_4);
    vec_u16x8 __cached_5;
    __cached_5 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 9UL) * 256UL) + _fuseiter_199)]);
    row2_38 = sc_select(65280, vec_u16x32(__cached_5), row2_38);
    vec_u16x8 __cached_6;
    __cached_6 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 17UL) * 256UL) + _fuseiter_199)]);
    row2_38 = sc_select(16711680, vec_u16x32(__cached_6), row2_38);
    vec_u16x8 __cached_7;
    __cached_7 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 25UL) * 256UL) + _fuseiter_199)]);
    row2_38 = sc_select(-16777216, vec_u16x32(__cached_7), row2_38);
    vec_u16x8 __cached_8;
    __cached_8 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 2UL) * 256UL) + _fuseiter_199)]);
    row3_39 = vec_u16x32(__cached_8);
    vec_u16x8 __cached_9;
    __cached_9 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 10UL) * 256UL) + _fuseiter_199)]);
    row3_39 = sc_select(65280, vec_u16x32(__cached_9), row3_39);
    vec_u16x8 __cached_10;
    __cached_10 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 18UL) * 256UL) + _fuseiter_199)]);
    row3_39 = sc_select(16711680, vec_u16x32(__cached_10), row3_39);
    vec_u16x8 __cached_11;
    __cached_11 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 26UL) * 256UL) + _fuseiter_199)]);
    row3_39 = sc_select(-16777216, vec_u16x32(__cached_11), row3_39);
    vec_u16x8 __cached_12;
    __cached_12 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 3UL) * 256UL) + _fuseiter_199)]);
    row4_40 = vec_u16x32(__cached_12);
    vec_u16x8 __cached_13;
    __cached_13 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 11UL) * 256UL) + _fuseiter_199)]);
    row4_40 = sc_select(65280, vec_u16x32(__cached_13), row4_40);
    vec_u16x8 __cached_14;
    __cached_14 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 19UL) * 256UL) + _fuseiter_199)]);
    row4_40 = sc_select(16711680, vec_u16x32(__cached_14), row4_40);
    vec_u16x8 __cached_15;
    __cached_15 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 27UL) * 256UL) + _fuseiter_199)]);
    row4_40 = sc_select(-16777216, vec_u16x32(__cached_15), row4_40);
    vec_u16x8 __cached_16;
    __cached_16 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 4UL) * 256UL) + _fuseiter_199)]);
    row5_41 = vec_u16x32(__cached_16);
    vec_u16x8 __cached_17;
    __cached_17 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 12UL) * 256UL) + _fuseiter_199)]);
    row5_41 = sc_select(65280, vec_u16x32(__cached_17), row5_41);
    vec_u16x8 __cached_18;
    __cached_18 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 20UL) * 256UL) + _fuseiter_199)]);
    row5_41 = sc_select(16711680, vec_u16x32(__cached_18), row5_41);
    vec_u16x8 __cached_19;
    __cached_19 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 28UL) * 256UL) + _fuseiter_199)]);
    row5_41 = sc_select(-16777216, vec_u16x32(__cached_19), row5_41);
    vec_u16x8 __cached_20;
    __cached_20 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 5UL) * 256UL) + _fuseiter_199)]);
    row6_42 = vec_u16x32(__cached_20);
    vec_u16x8 __cached_21;
    __cached_21 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 13UL) * 256UL) + _fuseiter_199)]);
    row6_42 = sc_select(65280, vec_u16x32(__cached_21), row6_42);
    vec_u16x8 __cached_22;
    __cached_22 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 21UL) * 256UL) + _fuseiter_199)]);
    row6_42 = sc_select(16711680, vec_u16x32(__cached_22), row6_42);
    vec_u16x8 __cached_23;
    __cached_23 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 29UL) * 256UL) + _fuseiter_199)]);
    row6_42 = sc_select(-16777216, vec_u16x32(__cached_23), row6_42);
    vec_u16x8 __cached_24;
    __cached_24 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 6UL) * 256UL) + _fuseiter_199)]);
    row7_43 = vec_u16x32(__cached_24);
    vec_u16x8 __cached_25;
    __cached_25 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 14UL) * 256UL) + _fuseiter_199)]);
    row7_43 = sc_select(65280, vec_u16x32(__cached_25), row7_43);
    vec_u16x8 __cached_26;
    __cached_26 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 22UL) * 256UL) + _fuseiter_199)]);
    row7_43 = sc_select(16711680, vec_u16x32(__cached_26), row7_43);
    vec_u16x8 __cached_27;
    __cached_27 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 30UL) * 256UL) + _fuseiter_199)]);
    row7_43 = sc_select(-16777216, vec_u16x32(__cached_27), row7_43);
    vec_u16x8 __cached_28;
    __cached_28 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 7UL) * 256UL) + _fuseiter_199)]);
    row8_44 = vec_u16x32(__cached_28);
    vec_u16x8 __cached_29;
    __cached_29 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 15UL) * 256UL) + _fuseiter_199)]);
    row8_44 = sc_select(65280, vec_u16x32(__cached_29), row8_44);
    vec_u16x8 __cached_30;
    __cached_30 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 23UL) * 256UL) + _fuseiter_199)]);
    row8_44 = sc_select(16711680, vec_u16x32(__cached_30), row8_44);
    vec_u16x8 __cached_31;
    __cached_31 = vec_u16x8::load(&__ins_0[(((_fuseiter_200 + 31UL) * 256UL) + _fuseiter_199)]);
    row8_44 = sc_select(-16777216, vec_u16x32(__cached_31), row8_44);
    row9_45 = sc_unpack_low(row1_37, row2_38, 16);
    row10_46 = sc_unpack_high(row1_37, row2_38, 16);
    row11_47 = sc_unpack_low(row3_39, row4_40, 16);
    row12_48 = sc_unpack_high(row3_39, row4_40, 16);
    row13_49 = sc_unpack_low(row5_41, row6_42, 16);
    row14_50 = sc_unpack_high(row5_41, row6_42, 16);
    row15_51 = sc_unpack_low(row7_43, row8_44, 16);
    row16_52 = sc_unpack_high(row7_43, row8_44, 16);
    row1_37 = sc_unpack_low(row9_45, row11_47, 32);
    row2_38 = sc_unpack_high(row9_45, row11_47, 32);
    row3_39 = sc_unpack_low(row10_46, row12_48, 32);
    row4_40 = sc_unpack_high(row10_46, row12_48, 32);
    row5_41 = sc_unpack_low(row13_49, row15_51, 32);
    row6_42 = sc_unpack_high(row13_49, row15_51, 32);
    row7_43 = sc_unpack_low(row14_50, row16_52, 32);
    row8_44 = sc_unpack_high(row14_50, row16_52, 32);
    row9_45 = sc_unpack_low(row1_37, row5_41, 64);
    row10_46 = sc_unpack_high(row1_37, row5_41, 64);
    row11_47 = sc_unpack_low(row2_38, row6_42, 64);
    row12_48 = sc_unpack_high(row2_38, row6_42, 64);
    row13_49 = sc_unpack_low(row3_39, row7_43, 64);
    row14_50 = sc_unpack_high(row3_39, row7_43, 64);
    row15_51 = sc_unpack_low(row4_40, row8_44, 64);
    row16_52 = sc_unpack_high(row4_40, row8_44, 64);
    vec_u16x32 __cached_32;
    __cached_32 = row9_45;
    vec_u16x32::store(__cached_32, &__outs_0[((_fuseiter_199 * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_33;
    __cached_33 = row10_46;
    vec_u16x32::store(__cached_33, &__outs_0[(((_fuseiter_199 + 1UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_34;
    __cached_34 = row11_47;
    vec_u16x32::store(__cached_34, &__outs_0[(((_fuseiter_199 + 2UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_35;
    __cached_35 = row12_48;
    vec_u16x32::store(__cached_35, &__outs_0[(((_fuseiter_199 + 3UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_36;
    __cached_36 = row13_49;
    vec_u16x32::store(__cached_36, &__outs_0[(((_fuseiter_199 + 4UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_37;
    __cached_37 = row14_50;
    vec_u16x32::store(__cached_37, &__outs_0[(((_fuseiter_199 + 5UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_38;
    __cached_38 = row15_51;
    vec_u16x32::store(__cached_38, &__outs_0[(((_fuseiter_199 + 6UL) * 131072UL) + _fuseiter_200)]);
    vec_u16x32 __cached_39;
    __cached_39 = row16_52;
    vec_u16x32::store(__cached_39, &__outs_0[(((_fuseiter_199 + 7UL) * 131072UL) + _fuseiter_200)]);
  }
}

static void reorder__260_closure_5_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__260_closure_5(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__270_closure_6(uint64_t fused_0_fuseiter_201___fuseiter_202_43, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_203 = 0UL; _fuseiter_203 < 32UL; _fuseiter_203 += 1UL) {
    for (uint64_t _fuseiter_204 = 0UL; _fuseiter_204 < 32UL; _fuseiter_204 += 1UL) {
      for (uint64_t _fuseiter_205 = 0UL; _fuseiter_205 < 2UL; _fuseiter_205 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_205 + (_fuseiter_203 * 2UL)) + ((fused_0_fuseiter_201___fuseiter_202_43 % 2048UL) * 64UL)) * 128UL) + (_fuseiter_204 + ((fused_0_fuseiter_201___fuseiter_202_43 / 2048UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_201___fuseiter_202_43 / 2048UL) * 4194304UL) + (((fused_0_fuseiter_201___fuseiter_202_43 % 2048UL) * 2048UL) + ((_fuseiter_203 * 64UL) + ((_fuseiter_204 * 2UL) + _fuseiter_205))))] = __cached_1;
      }
    }
  }
}

static void reorder__270_closure_6_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__270_closure_6(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__350_closure_7(uint64_t fused_0m_o__n_o_44, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 8);
  float* __origouts_350_shr = (float*)sc_thread_aligned_malloc(__stream, 4096UL);
  dnnl_brgemm_call(__sc_kernel_cache_9, &__ins_0[((fused_0m_o__n_o_44 / 4UL) * 4194304UL)], &__ins_1[((fused_0m_o__n_o_44 % 4UL) * 4194304UL)], &__origouts_350_shr[0UL], 2048, __stream);
  for (uint64_t _fuseiter208 = 0UL; _fuseiter208 < 32UL; _fuseiter208 += 1UL) {
    for (uint64_t _fuseiter209 = 0UL; _fuseiter209 < 32UL; _fuseiter209 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_350_shr[((_fuseiter208 * 32UL) + _fuseiter209)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_44 / 4UL) * 4096UL) + ((fused_0m_o__n_o_44 % 4UL) * 1024UL)) + ((_fuseiter208 * 32UL) + _fuseiter209))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_350_shr);
}

static void matmul_core_cast__350_closure_7_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__350_closure_7(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__280_closure_8(uint64_t fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner = 0UL; fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner < 16UL; fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner += 1UL) {
    for (uint64_t _fuseiter_214 = 0UL; _fuseiter_214 < 32UL; _fuseiter_214 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 128UL) * 4096UL) + ((((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 32UL) % 4UL) * 1024UL) + (((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) % 32UL) * 32UL) + _fuseiter_214)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_214 + (((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 32UL) % 4UL) * 32UL)) / 64UL) * 16384UL) + (((((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 128UL) * 32UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 128UL) * 32UL)) % 64UL) / 2UL) * 128UL) + ((((_fuseiter_214 + (((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 32UL) % 4UL) * 32UL)) % 64UL) * 2UL) + ((((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0outer * 16UL) + fused_0fused_0_fuseiter_211___fuseiter_212_45___fuseiter_213_46_0inner) / 128UL) * 32UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__280_closure_8_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__280_closure_8(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_select_one_cast_mul__380_closure_9(uint64_t fused_0m_o__n_o_47, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_10 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_360_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_10, &__ins_0[((fused_0m_o__n_o_47 / 8UL) * 16384UL)], &__ins_1[((fused_0m_o__n_o_47 % 8UL) * 16384UL)], &__origouts_360_shr[0UL], 4, __stream);
  for (uint64_t _fuseiter215 = 0UL; _fuseiter215 < 64UL; _fuseiter215 += 1UL) {
    for (uint64_t _fuseiter216 = 0UL; _fuseiter216 < 64UL; _fuseiter216 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[((((fused_0m_o__n_o_47 / 8UL) * 32768UL) + ((fused_0m_o__n_o_47 % 8UL) * 64UL)) + ((_fuseiter215 * 512UL) + _fuseiter216))]))) << vec_u32x16(16UL)));
      vec_f32x16 __cached_1;
      vec_u16x16 _arg_cache_4 = sc_select((__cached_0 > vec_f32x16(0.f)), vec_u16x16(16256UL), vec_u16x16(0UL));
      __cached_1 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(_arg_cache_4)) << vec_u32x16(16UL)));
      vec_f32x16 __cached_2;
      __cached_2 = vec_f32x16::load(&__origouts_360_shr[((_fuseiter215 * 64UL) + _fuseiter216)]);
      vec_f32x16 __cached_3;
      __cached_3 = __cached_2;
      vec_u16x16 __cached_4;
      vec_f32x16 _arg_cache_5 = (__cached_3 * __cached_1);
      __cached_4 = tobf16(_arg_cache_5);
      vec_u16x16::store(__cached_4, &__outs_0[((((fused_0m_o__n_o_47 / 8UL) * 32768UL) + ((fused_0m_o__n_o_47 % 8UL) * 64UL)) + ((_fuseiter215 * 512UL) + _fuseiter216))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_select_one_cast_mul__380_closure_9_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_select_one_cast_mul__380_closure_9(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_reorder__400_closure_10(uint64_t fused_0m_o__n_o_48, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_11 = *(void**)(__module_data + 24);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 4160UL);
  float* __origouts_370_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_11, &__ins_0[(fused_0m_o__n_o_48 * 32768UL)], &__ins_1[0UL], &__origouts_370_shr[0UL], 8, __stream);
  uint16_t* _cast_buf_0_shr = (uint16_t*)&__rescheduled_1[4096UL];
  for (uint64_t _fuseiter226 = 0UL; _fuseiter226 < 64UL; _fuseiter226 += 1UL) {
    vec_f32x16 __cached_0;
    __cached_0 = vec_f32x16::load(&__origouts_370_shr[(_fuseiter226 * 16UL)]);
    vec_u16x16 __cached_1;
    __cached_1 = tobf16(__cached_0);
    vec_u16x16::store(__cached_1, &_cast_buf_0_shr[0UL]);
    for (uint64_t _fuseiter_232 = 0UL; _fuseiter_232 < 16UL; _fuseiter_232 += 1UL) {
      if (((_fuseiter_232 < 13UL) && ((_fuseiter226 + (fused_0m_o__n_o_48 * 64UL)) < 131072UL))) {
        uint16_t __cached_2;
        __cached_2 = _cast_buf_0_shr[_fuseiter_232];
        uint16_t __cached_3;
        __cached_3 = __cached_2;
        __outs_0[(((_fuseiter226 + (fused_0m_o__n_o_48 * 64UL)) * 13UL) + _fuseiter_232)] = __cached_3;
      }
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_reorder__400_closure_10_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_reorder__400_closure_10(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__210_closure_11(uint64_t fused_0_fuseiter_235___fuseiter_236_49, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_237 = 0UL; _fuseiter_237 < 32UL; _fuseiter_237 += 1UL) {
    for (uint64_t _fuseiter_238 = 0UL; _fuseiter_238 < 32UL; _fuseiter_238 += 1UL) {
      for (uint64_t _fuseiter_239 = 0UL; _fuseiter_239 < 2UL; _fuseiter_239 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_239 + (_fuseiter_237 * 2UL)) + ((fused_0_fuseiter_235___fuseiter_236_49 % 2048UL) * 64UL)) * 512UL) + (_fuseiter_238 + ((fused_0_fuseiter_235___fuseiter_236_49 / 2048UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_235___fuseiter_236_49 / 2048UL) * 4194304UL) + (((fused_0_fuseiter_235___fuseiter_236_49 % 2048UL) * 2048UL) + ((_fuseiter_237 * 64UL) + ((_fuseiter_238 * 2UL) + _fuseiter_239))))] = __cached_1;
      }
    }
  }
}

static void reorder__210_closure_11_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__210_closure_11(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__200_closure_12(uint64_t _fuseiter_240_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_240_0inner = 0UL; _fuseiter_240_0inner < 32UL; _fuseiter_240_0inner += 1UL) {
    for (uint64_t _fuseiter_241 = 0UL; _fuseiter_241 < 13UL; _fuseiter_241 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[((((_fuseiter_240_0outer * 32UL) + _fuseiter_240_0inner) * 13UL) + _fuseiter_241)];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((_fuseiter_241 * 131072UL) + ((_fuseiter_240_0outer * 32UL) + _fuseiter_240_0inner))] = __cached_1;
    }
  }
}

static void reorder__200_closure_12_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__200_closure_12(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__390_closure_13(uint64_t fused_0m_o__n_o_50, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_12 = *(void**)(__module_data + 32);
  float* __origouts_380_shr = (float*)sc_thread_aligned_malloc(__stream, 1664UL);
  dnnl_brgemm_call(__sc_kernel_cache_12, &__ins_0[((fused_0m_o__n_o_50 / 16UL) * 1703936UL)], &__ins_1[((fused_0m_o__n_o_50 % 16UL) * 4194304UL)], &__origouts_380_shr[0UL], 2048, __stream);
  for (uint64_t _fuseiter244 = 0UL; _fuseiter244 < 13UL; _fuseiter244 += 1UL) {
    for (uint64_t _fuseiter245 = 0UL; _fuseiter245 < 32UL; _fuseiter245 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_380_shr[((_fuseiter244 * 32UL) + _fuseiter245)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_50 / 16UL) * 6656UL) + ((fused_0m_o__n_o_50 % 16UL) * 416UL)) + ((_fuseiter244 * 32UL) + _fuseiter245))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_380_shr);
}

static void matmul_core_cast__390_closure_13_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__390_closure_13(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__220_closure_14(uint64_t fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner = 0UL; fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner < 256UL; fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner += 1UL) {
    for (uint64_t _fuseiter_251 = 0UL; _fuseiter_251 < 2UL; _fuseiter_251 += 1UL) {
      if (((((_fuseiter_251 + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 2UL)) < 16UL) && ((_fuseiter_251 + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 2UL)) < 13UL)) && (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 512UL) * 64UL)) < 512UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_251 + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 2UL)) / 13UL) * 6656UL) + (((((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 512UL) * 64UL)) / 32UL) * 416UL) + ((((_fuseiter_251 + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 2UL)) % 13UL) * 32UL) + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 512UL) * 64UL)) % 32UL))))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) % 64UL) * 2UL) + _fuseiter_251)))] = __cached_1;
      } else {
        uint16_t __cached_2;
        __cached_2 = 0UL;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_247___fuseiter_248_51___fuseiter_249_52___fuseiter_250_53_0inner) % 64UL) * 2UL) + _fuseiter_251)))] = __cached_2;
      }
    }
  }
}

static void reorder__220_closure_14_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__220_closure_14(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__230_closure_15(uint64_t _fuseiter_252, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_253 = 0UL; _fuseiter_253 < 131072UL; _fuseiter_253 += 32UL) {
    vec_u16x32 row1_54;
    vec_u16x32 row2_55;
    vec_u16x32 row3_56;
    vec_u16x32 row4_57;
    vec_u16x32 row5_58;
    vec_u16x32 row6_59;
    vec_u16x32 row7_60;
    vec_u16x32 row8_61;
    vec_u16x32 row9_62;
    vec_u16x32 row10_63;
    vec_u16x32 row11_64;
    vec_u16x32 row12_65;
    vec_u16x32 row13_66;
    vec_u16x32 row14_67;
    vec_u16x32 row15_68;
    vec_u16x32 row16_69;
    vec_u16x8 __cached_0;
    __cached_0 = vec_u16x8::load(&__ins_0[((_fuseiter_253 * 512UL) + _fuseiter_252)]);
    row1_54 = vec_u16x32(__cached_0);
    vec_u16x8 __cached_1;
    __cached_1 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 8UL) * 512UL) + _fuseiter_252)]);
    row1_54 = sc_select(65280, vec_u16x32(__cached_1), row1_54);
    vec_u16x8 __cached_2;
    __cached_2 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 16UL) * 512UL) + _fuseiter_252)]);
    row1_54 = sc_select(16711680, vec_u16x32(__cached_2), row1_54);
    vec_u16x8 __cached_3;
    __cached_3 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 24UL) * 512UL) + _fuseiter_252)]);
    row1_54 = sc_select(-16777216, vec_u16x32(__cached_3), row1_54);
    vec_u16x8 __cached_4;
    __cached_4 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 1UL) * 512UL) + _fuseiter_252)]);
    row2_55 = vec_u16x32(__cached_4);
    vec_u16x8 __cached_5;
    __cached_5 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 9UL) * 512UL) + _fuseiter_252)]);
    row2_55 = sc_select(65280, vec_u16x32(__cached_5), row2_55);
    vec_u16x8 __cached_6;
    __cached_6 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 17UL) * 512UL) + _fuseiter_252)]);
    row2_55 = sc_select(16711680, vec_u16x32(__cached_6), row2_55);
    vec_u16x8 __cached_7;
    __cached_7 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 25UL) * 512UL) + _fuseiter_252)]);
    row2_55 = sc_select(-16777216, vec_u16x32(__cached_7), row2_55);
    vec_u16x8 __cached_8;
    __cached_8 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 2UL) * 512UL) + _fuseiter_252)]);
    row3_56 = vec_u16x32(__cached_8);
    vec_u16x8 __cached_9;
    __cached_9 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 10UL) * 512UL) + _fuseiter_252)]);
    row3_56 = sc_select(65280, vec_u16x32(__cached_9), row3_56);
    vec_u16x8 __cached_10;
    __cached_10 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 18UL) * 512UL) + _fuseiter_252)]);
    row3_56 = sc_select(16711680, vec_u16x32(__cached_10), row3_56);
    vec_u16x8 __cached_11;
    __cached_11 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 26UL) * 512UL) + _fuseiter_252)]);
    row3_56 = sc_select(-16777216, vec_u16x32(__cached_11), row3_56);
    vec_u16x8 __cached_12;
    __cached_12 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 3UL) * 512UL) + _fuseiter_252)]);
    row4_57 = vec_u16x32(__cached_12);
    vec_u16x8 __cached_13;
    __cached_13 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 11UL) * 512UL) + _fuseiter_252)]);
    row4_57 = sc_select(65280, vec_u16x32(__cached_13), row4_57);
    vec_u16x8 __cached_14;
    __cached_14 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 19UL) * 512UL) + _fuseiter_252)]);
    row4_57 = sc_select(16711680, vec_u16x32(__cached_14), row4_57);
    vec_u16x8 __cached_15;
    __cached_15 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 27UL) * 512UL) + _fuseiter_252)]);
    row4_57 = sc_select(-16777216, vec_u16x32(__cached_15), row4_57);
    vec_u16x8 __cached_16;
    __cached_16 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 4UL) * 512UL) + _fuseiter_252)]);
    row5_58 = vec_u16x32(__cached_16);
    vec_u16x8 __cached_17;
    __cached_17 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 12UL) * 512UL) + _fuseiter_252)]);
    row5_58 = sc_select(65280, vec_u16x32(__cached_17), row5_58);
    vec_u16x8 __cached_18;
    __cached_18 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 20UL) * 512UL) + _fuseiter_252)]);
    row5_58 = sc_select(16711680, vec_u16x32(__cached_18), row5_58);
    vec_u16x8 __cached_19;
    __cached_19 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 28UL) * 512UL) + _fuseiter_252)]);
    row5_58 = sc_select(-16777216, vec_u16x32(__cached_19), row5_58);
    vec_u16x8 __cached_20;
    __cached_20 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 5UL) * 512UL) + _fuseiter_252)]);
    row6_59 = vec_u16x32(__cached_20);
    vec_u16x8 __cached_21;
    __cached_21 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 13UL) * 512UL) + _fuseiter_252)]);
    row6_59 = sc_select(65280, vec_u16x32(__cached_21), row6_59);
    vec_u16x8 __cached_22;
    __cached_22 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 21UL) * 512UL) + _fuseiter_252)]);
    row6_59 = sc_select(16711680, vec_u16x32(__cached_22), row6_59);
    vec_u16x8 __cached_23;
    __cached_23 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 29UL) * 512UL) + _fuseiter_252)]);
    row6_59 = sc_select(-16777216, vec_u16x32(__cached_23), row6_59);
    vec_u16x8 __cached_24;
    __cached_24 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 6UL) * 512UL) + _fuseiter_252)]);
    row7_60 = vec_u16x32(__cached_24);
    vec_u16x8 __cached_25;
    __cached_25 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 14UL) * 512UL) + _fuseiter_252)]);
    row7_60 = sc_select(65280, vec_u16x32(__cached_25), row7_60);
    vec_u16x8 __cached_26;
    __cached_26 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 22UL) * 512UL) + _fuseiter_252)]);
    row7_60 = sc_select(16711680, vec_u16x32(__cached_26), row7_60);
    vec_u16x8 __cached_27;
    __cached_27 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 30UL) * 512UL) + _fuseiter_252)]);
    row7_60 = sc_select(-16777216, vec_u16x32(__cached_27), row7_60);
    vec_u16x8 __cached_28;
    __cached_28 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 7UL) * 512UL) + _fuseiter_252)]);
    row8_61 = vec_u16x32(__cached_28);
    vec_u16x8 __cached_29;
    __cached_29 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 15UL) * 512UL) + _fuseiter_252)]);
    row8_61 = sc_select(65280, vec_u16x32(__cached_29), row8_61);
    vec_u16x8 __cached_30;
    __cached_30 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 23UL) * 512UL) + _fuseiter_252)]);
    row8_61 = sc_select(16711680, vec_u16x32(__cached_30), row8_61);
    vec_u16x8 __cached_31;
    __cached_31 = vec_u16x8::load(&__ins_0[(((_fuseiter_253 + 31UL) * 512UL) + _fuseiter_252)]);
    row8_61 = sc_select(-16777216, vec_u16x32(__cached_31), row8_61);
    row9_62 = sc_unpack_low(row1_54, row2_55, 16);
    row10_63 = sc_unpack_high(row1_54, row2_55, 16);
    row11_64 = sc_unpack_low(row3_56, row4_57, 16);
    row12_65 = sc_unpack_high(row3_56, row4_57, 16);
    row13_66 = sc_unpack_low(row5_58, row6_59, 16);
    row14_67 = sc_unpack_high(row5_58, row6_59, 16);
    row15_68 = sc_unpack_low(row7_60, row8_61, 16);
    row16_69 = sc_unpack_high(row7_60, row8_61, 16);
    row1_54 = sc_unpack_low(row9_62, row11_64, 32);
    row2_55 = sc_unpack_high(row9_62, row11_64, 32);
    row3_56 = sc_unpack_low(row10_63, row12_65, 32);
    row4_57 = sc_unpack_high(row10_63, row12_65, 32);
    row5_58 = sc_unpack_low(row13_66, row15_68, 32);
    row6_59 = sc_unpack_high(row13_66, row15_68, 32);
    row7_60 = sc_unpack_low(row14_67, row16_69, 32);
    row8_61 = sc_unpack_high(row14_67, row16_69, 32);
    row9_62 = sc_unpack_low(row1_54, row5_58, 64);
    row10_63 = sc_unpack_high(row1_54, row5_58, 64);
    row11_64 = sc_unpack_low(row2_55, row6_59, 64);
    row12_65 = sc_unpack_high(row2_55, row6_59, 64);
    row13_66 = sc_unpack_low(row3_56, row7_60, 64);
    row14_67 = sc_unpack_high(row3_56, row7_60, 64);
    row15_68 = sc_unpack_low(row4_57, row8_61, 64);
    row16_69 = sc_unpack_high(row4_57, row8_61, 64);
    vec_u16x32 __cached_32;
    __cached_32 = row9_62;
    vec_u16x32::store(__cached_32, &__outs_0[((_fuseiter_252 * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_33;
    __cached_33 = row10_63;
    vec_u16x32::store(__cached_33, &__outs_0[(((_fuseiter_252 + 1UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_34;
    __cached_34 = row11_64;
    vec_u16x32::store(__cached_34, &__outs_0[(((_fuseiter_252 + 2UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_35;
    __cached_35 = row12_65;
    vec_u16x32::store(__cached_35, &__outs_0[(((_fuseiter_252 + 3UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_36;
    __cached_36 = row13_66;
    vec_u16x32::store(__cached_36, &__outs_0[(((_fuseiter_252 + 4UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_37;
    __cached_37 = row14_67;
    vec_u16x32::store(__cached_37, &__outs_0[(((_fuseiter_252 + 5UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_38;
    __cached_38 = row15_68;
    vec_u16x32::store(__cached_38, &__outs_0[(((_fuseiter_252 + 6UL) * 131072UL) + _fuseiter_253)]);
    vec_u16x32 __cached_39;
    __cached_39 = row16_69;
    vec_u16x32::store(__cached_39, &__outs_0[(((_fuseiter_252 + 7UL) * 131072UL) + _fuseiter_253)]);
  }
}

static void reorder__230_closure_15_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__230_closure_15(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__240_closure_16(uint64_t fused_0_fuseiter_254___fuseiter_255_54, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_256 = 0UL; _fuseiter_256 < 32UL; _fuseiter_256 += 1UL) {
    for (uint64_t _fuseiter_257 = 0UL; _fuseiter_257 < 32UL; _fuseiter_257 += 1UL) {
      for (uint64_t _fuseiter_258 = 0UL; _fuseiter_258 < 2UL; _fuseiter_258 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_258 + (_fuseiter_256 * 2UL)) + ((fused_0_fuseiter_254___fuseiter_255_54 % 2048UL) * 64UL)) * 256UL) + (_fuseiter_257 + ((fused_0_fuseiter_254___fuseiter_255_54 / 2048UL) * 32UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((fused_0_fuseiter_254___fuseiter_255_54 / 2048UL) * 4194304UL) + (((fused_0_fuseiter_254___fuseiter_255_54 % 2048UL) * 2048UL) + ((_fuseiter_256 * 64UL) + ((_fuseiter_257 * 2UL) + _fuseiter_258))))] = __cached_1;
      }
    }
  }
}

static void reorder__240_closure_16_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__240_closure_16(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast__370_closure_17(uint64_t fused_0m_o__n_o_55, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __outs_0){
  void*& __sc_kernel_cache_9 = *(void**)(__module_data + 8);
  float* __origouts_390_shr = (float*)sc_thread_aligned_malloc(__stream, 4096UL);
  dnnl_brgemm_call(__sc_kernel_cache_9, &__ins_0[((fused_0m_o__n_o_55 / 8UL) * 4194304UL)], &__ins_1[((fused_0m_o__n_o_55 % 8UL) * 4194304UL)], &__origouts_390_shr[0UL], 2048, __stream);
  for (uint64_t _fuseiter261 = 0UL; _fuseiter261 < 32UL; _fuseiter261 += 1UL) {
    for (uint64_t _fuseiter262 = 0UL; _fuseiter262 < 32UL; _fuseiter262 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_390_shr[((_fuseiter261 * 32UL) + _fuseiter262)]);
      vec_u16x16 __cached_1;
      __cached_1 = tobf16(__cached_0);
      vec_u16x16::store(__cached_1, &__outs_0[((((fused_0m_o__n_o_55 / 8UL) * 8192UL) + ((fused_0m_o__n_o_55 % 8UL) * 1024UL)) + ((_fuseiter261 * 32UL) + _fuseiter262))]);
    }
  }
  sc_thread_aligned_free(__stream, __origouts_390_shr);
}

static void matmul_core_cast__370_closure_17_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast__370_closure_17(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr));
}

static void reorder__250_closure_18(uint64_t fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner = 0UL; fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner < 16UL; fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner += 1UL) {
    for (uint64_t _fuseiter_267 = 0UL; _fuseiter_267 < 32UL; _fuseiter_267 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 256UL) * 8192UL) + ((((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 32UL) % 8UL) * 1024UL) + (((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) % 32UL) * 32UL) + _fuseiter_267)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_267 + (((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 32UL) % 8UL) * 32UL)) / 64UL) * 32768UL) + (((((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 256UL) * 32UL)) / 64UL) * 4096UL) + ((((((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 256UL) * 32UL)) % 64UL) / 2UL) * 128UL) + ((((_fuseiter_267 + (((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 32UL) % 8UL) * 32UL)) % 64UL) * 2UL) + ((((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) % 32UL) + ((((fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0outer * 16UL) + fused_0fused_0_fuseiter_264___fuseiter_265_56___fuseiter_266_57_0inner) / 256UL) * 32UL)) % 64UL) % 2UL)))))] = __cached_1;
    }
  }
}

static void reorder__250_closure_18_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__250_closure_18(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

