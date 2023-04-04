
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t mlp_training_forward_4k_data[64];
static constexpr int8_t* __module_data = mlp_training_forward_4k_data;
alignas(64) static int8_t __uninitialized_data[0UL];

static bool reorder__13(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_cast_add_relu__14(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool matmul_core_cast_add_relu__15(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool matmul_core_cast_add_relu__16(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static void reorder__130_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__140_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__150_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__160_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__130_closure_0(uint64_t fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__140_closure_1(uint64_t fused_0m_o__n_o_2, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__150_closure_2(uint64_t fused_0m_o__n_o_3, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__160_closure_3(uint64_t fused_0m_o__n_o_4, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);


extern "C" void mlp_training_forward_4k(uint16_t* relu_out0, uint16_t* relu_out1, uint16_t* relu_out2, uint16_t* input, uint16_t* weight0, uint16_t* bias0, uint16_t* weight1, uint16_t* bias1, uint16_t* weight2, uint16_t* bias2){
  uint16_t* buffer_7 = (uint16_t*)sc_aligned_malloc(__stream, 131072UL);
  reorder__13(buffer_7, input);
  matmul_core_cast_add_relu__14(relu_out0, buffer_7, weight0, &bias0[0]);
  matmul_core_cast_add_relu__15(relu_out1, relu_out0, weight1, &bias1[0]);
  matmul_core_cast_add_relu__16(relu_out2, relu_out1, weight2, &bias2[0]);
  sc_aligned_free(__stream, buffer_7);
}

static bool reorder__13(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs0[2UL];
  __tempargs0[0UL] = __ins_0;
  __tempargs0[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__130_closure_0_0wrapper, __stream, __module_data, 0UL, 128UL, 1UL, __tempargs0);
  return true;
}

static bool matmul_core_cast_add_relu__14(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs1[4UL];
  __tempargs1[0UL] = __ins_0;
  __tempargs1[1UL] = __ins_1;
  __tempargs1[2UL] = __ins_2;
  __tempargs1[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__140_closure_1_0wrapper, __stream, __module_data, 0UL, 512UL, 1UL, __tempargs1);
  return true;
}

static bool matmul_core_cast_add_relu__15(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs2[4UL];
  __tempargs2[0UL] = __ins_0;
  __tempargs2[1UL] = __ins_1;
  __tempargs2[2UL] = __ins_2;
  __tempargs2[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__150_closure_2_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs2);
  return true;
}

static bool matmul_core_cast_add_relu__16(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs3[4UL];
  __tempargs3[0UL] = __ins_0;
  __tempargs3[1UL] = __ins_1;
  __tempargs3[2UL] = __ins_2;
  __tempargs3[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__160_closure_3_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs3);
  return true;
}

extern "C" void sc_init_mlp_training_forward_4k() {
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  void*& __sc_kernel_cache_1 = *(void**)(__module_data + 8);
  void*& __sc_kernel_cache_2 = *(void**)(__module_data + 16);
  __sc_kernel_cache = dnnl_brgemm_func(64, 64, 16, 16, 64, 64, 1024, 1024, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_1 = dnnl_brgemm_func(64, 64, 64, 512, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_2 = dnnl_brgemm_func(32, 64, 64, 256, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
}

static void reorder__130_closure_0(uint64_t fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner = 0UL; fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner < 32UL; fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner += 1UL) {
    for (uint64_t _fuseiter_3 = 0UL; _fuseiter_3 < 16UL; _fuseiter_3 += 1UL) {
      if (((_fuseiter_3 < 13UL) && (((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) % 64UL) + ((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) / 64UL) * 64UL)) < 4096UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) % 64UL) + ((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) / 64UL) * 64UL)) * 13UL) + _fuseiter_3)];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) / 64UL) * 1024UL) + (((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) % 64UL) * 16UL) + _fuseiter_3))] = __cached_1;
      } else {
        uint16_t __cached_2;
        __cached_2 = 0UL;
        __outs_0[(((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) / 64UL) * 1024UL) + (((((fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0outer * 32UL) + fused_0fused_0_fuseiter_0___fuseiter_1_0___fuseiter_2_1_0inner) % 64UL) * 16UL) + _fuseiter_3))] = __cached_2;
      }
    }
  }
}

static void reorder__130_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__130_closure_0(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast_add_relu__140_closure_1(uint64_t fused_0m_o__n_o_2, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_30_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache, &__ins_0[((fused_0m_o__n_o_2 / 8UL) * 1024UL)], &__ins_1[((fused_0m_o__n_o_2 % 8UL) * 1024UL)], &__origouts_30_shr[0UL], 1, __stream);
  for (uint64_t _fuseiter4 = 0UL; _fuseiter4 < 64UL; _fuseiter4 += 1UL) {
    for (uint64_t _fuseiter5 = 0UL; _fuseiter5 < 64UL; _fuseiter5 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_30_shr[((_fuseiter4 * 64UL) + _fuseiter5)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_2 % 8UL) * 64UL) + _fuseiter5)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_0 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_0);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_2 / 8UL) * 32768UL) + ((fused_0m_o__n_o_2 % 8UL) * 64UL)) + ((_fuseiter4 * 512UL) + _fuseiter5))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__140_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__140_closure_1(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_add_relu__150_closure_2(uint64_t fused_0m_o__n_o_3, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_1 = *(void**)(__module_data + 8);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_40_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_1, &__ins_0[((fused_0m_o__n_o_3 / 4UL) * 32768UL)], &__ins_1[((fused_0m_o__n_o_3 % 4UL) * 32768UL)], &__origouts_40_shr[0UL], 8, __stream);
  for (uint64_t _fuseiter13 = 0UL; _fuseiter13 < 64UL; _fuseiter13 += 1UL) {
    for (uint64_t _fuseiter14 = 0UL; _fuseiter14 < 64UL; _fuseiter14 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_40_shr[((_fuseiter13 * 64UL) + _fuseiter14)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_3 % 4UL) * 64UL) + _fuseiter14)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_1 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_1);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_3 / 4UL) * 16384UL) + ((fused_0m_o__n_o_3 % 4UL) * 64UL)) + ((_fuseiter13 * 256UL) + _fuseiter14))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__150_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__150_closure_2(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_add_relu__160_closure_3(uint64_t fused_0m_o__n_o_4, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_2 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 8192UL);
  float* __origouts_50_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_2, &__ins_0[((fused_0m_o__n_o_4 / 2UL) * 8192UL)], &__ins_1[((fused_0m_o__n_o_4 % 2UL) * 16384UL)], &__origouts_50_shr[0UL], 4, __stream);
  for (uint64_t _fuseiter22 = 0UL; _fuseiter22 < 32UL; _fuseiter22 += 1UL) {
    for (uint64_t _fuseiter23 = 0UL; _fuseiter23 < 64UL; _fuseiter23 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_50_shr[((_fuseiter22 * 64UL) + _fuseiter23)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_4 % 2UL) * 64UL) + _fuseiter23)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_2 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_2);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_4 / 2UL) * 4096UL) + ((fused_0m_o__n_o_4 % 2UL) * 64UL)) + ((_fuseiter22 * 128UL) + _fuseiter23))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__160_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__160_closure_3(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

