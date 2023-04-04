
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t mlp_training_forward_128k_data[64];
static constexpr int8_t* __module_data = mlp_training_forward_128k_data;
alignas(64) static int8_t __uninitialized_data[0UL];

static bool reorder__13(uint16_t* __outs_0, uint16_t* __ins_0);
static bool matmul_core_cast_add_relu__14(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool matmul_core_cast_add_relu__15(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static bool matmul_core_cast_add_relu__16(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2);
static void reorder__130_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__140_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__150_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void matmul_core_cast_add_relu__160_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__130_closure_0(uint64_t _fuseiter_134, uint16_t* __ins_0, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__140_closure_1(uint64_t fused_0m_o__n_o_31, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__150_closure_2(uint64_t fused_0m_o__n_o_32, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);
static void matmul_core_cast_add_relu__160_closure_3(uint64_t fused_0m_o__n_o_33, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0);


extern "C" void mlp_training_forward_128k(uint16_t* relu_out0, uint16_t* relu_out1, uint16_t* relu_out2, uint16_t* input, uint16_t* weight0, uint16_t* bias0, uint16_t* weight1, uint16_t* bias1, uint16_t* weight2, uint16_t* bias2){
  uint16_t* buffer_7 = (uint16_t*)sc_aligned_malloc(__stream, 4194304UL);
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
  sc_parallel_call_cpu_with_env((void*)&reorder__130_closure_0_0wrapper, __stream, __module_data, 0UL, 2048UL, 1UL, __tempargs0);
  return true;
}

static bool matmul_core_cast_add_relu__14(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs1[4UL];
  __tempargs1[0UL] = __ins_0;
  __tempargs1[1UL] = __ins_1;
  __tempargs1[2UL] = __ins_2;
  __tempargs1[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__140_closure_1_0wrapper, __stream, __module_data, 0UL, 16384UL, 1UL, __tempargs1);
  return true;
}

static bool matmul_core_cast_add_relu__15(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs2[4UL];
  __tempargs2[0UL] = __ins_0;
  __tempargs2[1UL] = __ins_1;
  __tempargs2[2UL] = __ins_2;
  __tempargs2[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__150_closure_2_0wrapper, __stream, __module_data, 0UL, 8192UL, 1UL, __tempargs2);
  return true;
}

static bool matmul_core_cast_add_relu__16(uint16_t* __outs_0, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2){
  generic_val __tempargs3[4UL];
  __tempargs3[0UL] = __ins_0;
  __tempargs3[1UL] = __ins_1;
  __tempargs3[2UL] = __ins_2;
  __tempargs3[3UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&matmul_core_cast_add_relu__160_closure_3_0wrapper, __stream, __module_data, 0UL, 4096UL, 1UL, __tempargs3);
  return true;
}

extern "C" void sc_init_mlp_training_forward_128k() {
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 8);
  void*& __sc_kernel_cache_8 = *(void**)(__module_data + 16);
  __sc_kernel_cache = dnnl_brgemm_func(64, 64, 16, 16, 64, 64, 1024, 1024, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_7 = dnnl_brgemm_func(64, 64, 64, 512, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
  __sc_kernel_cache_8 = dnnl_brgemm_func(64, 64, 64, 256, 64, 64, 64, 4096, 0.f, 2, 2, ((void*)0), ((void*)0), ((void*)0));
}

static void reorder__130_closure_0(uint64_t _fuseiter_134, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t _fuseiter_136 = 0UL; _fuseiter_136 < 64UL; _fuseiter_136 += 1UL) {
    for (uint64_t _fuseiter_137 = 0UL; _fuseiter_137 < 16UL; _fuseiter_137 += 1UL) {
      if (((_fuseiter_137 < 13UL) && ((_fuseiter_136 + (_fuseiter_134 * 64UL)) < 131072UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_136 + (_fuseiter_134 * 64UL)) * 13UL) + _fuseiter_137)];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[((_fuseiter_134 * 1024UL) + ((_fuseiter_136 * 16UL) + _fuseiter_137))] = __cached_1;
      } else {
        uint16_t __cached_2;
        __cached_2 = 0UL;
        __outs_0[((_fuseiter_134 * 1024UL) + ((_fuseiter_136 * 16UL) + _fuseiter_137))] = __cached_2;
      }
    }
  }
}

static void reorder__130_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__130_closure_0(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void matmul_core_cast_add_relu__140_closure_1(uint64_t fused_0m_o__n_o_31, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache = *(void**)(__module_data + 0);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_230_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache, &__ins_0[((fused_0m_o__n_o_31 / 8UL) * 1024UL)], &__ins_1[((fused_0m_o__n_o_31 % 8UL) * 1024UL)], &__origouts_230_shr[0UL], 1, __stream);
  for (uint64_t _fuseiter138 = 0UL; _fuseiter138 < 64UL; _fuseiter138 += 1UL) {
    for (uint64_t _fuseiter139 = 0UL; _fuseiter139 < 64UL; _fuseiter139 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_230_shr[((_fuseiter138 * 64UL) + _fuseiter139)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_31 % 8UL) * 64UL) + _fuseiter139)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_0 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_0);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_31 / 8UL) * 32768UL) + ((fused_0m_o__n_o_31 % 8UL) * 64UL)) + ((_fuseiter138 * 512UL) + _fuseiter139))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__140_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__140_closure_1(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_add_relu__150_closure_2(uint64_t fused_0m_o__n_o_32, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_7 = *(void**)(__module_data + 8);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_240_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_7, &__ins_0[((fused_0m_o__n_o_32 / 4UL) * 32768UL)], &__ins_1[((fused_0m_o__n_o_32 % 4UL) * 32768UL)], &__origouts_240_shr[0UL], 8, __stream);
  for (uint64_t _fuseiter147 = 0UL; _fuseiter147 < 64UL; _fuseiter147 += 1UL) {
    for (uint64_t _fuseiter148 = 0UL; _fuseiter148 < 64UL; _fuseiter148 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_240_shr[((_fuseiter147 * 64UL) + _fuseiter148)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_32 % 4UL) * 64UL) + _fuseiter148)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_1 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_1);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_32 / 4UL) * 16384UL) + ((fused_0m_o__n_o_32 % 4UL) * 64UL)) + ((_fuseiter147 * 256UL) + _fuseiter148))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__150_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__150_closure_2(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

static void matmul_core_cast_add_relu__160_closure_3(uint64_t fused_0m_o__n_o_33, uint16_t* __ins_0, uint16_t* __ins_1, uint16_t* __ins_2, uint16_t* __outs_0){
  void*& __sc_kernel_cache_8 = *(void**)(__module_data + 16);
  int8_t* __rescheduled_1 = (int8_t*)sc_thread_aligned_malloc(__stream, 16384UL);
  float* __origouts_250_shr = (float*)&__rescheduled_1[0UL];
  dnnl_brgemm_call(__sc_kernel_cache_8, &__ins_0[((fused_0m_o__n_o_33 / 2UL) * 16384UL)], &__ins_1[((fused_0m_o__n_o_33 % 2UL) * 16384UL)], &__origouts_250_shr[0UL], 4, __stream);
  for (uint64_t _fuseiter156 = 0UL; _fuseiter156 < 64UL; _fuseiter156 += 1UL) {
    for (uint64_t _fuseiter157 = 0UL; _fuseiter157 < 64UL; _fuseiter157 += 16UL) {
      vec_f32x16 __cached_0;
      __cached_0 = vec_f32x16::load(&__origouts_250_shr[((_fuseiter156 * 64UL) + _fuseiter157)]);
      vec_f32x16 __cached_1;
      __cached_1 = __cached_0;
      vec_f32x16 __cached_2;
      __cached_2 = sc_reinterpret<vec_f32x16>(((vec_u32x16)(sc_reinterpret<vec_u16x16>(vec_u16x16::load(&__ins_2[(((fused_0m_o__n_o_33 % 2UL) * 64UL) + _fuseiter157)]))) << vec_u32x16(16UL)));
      __cached_1 = (__cached_1 + __cached_2);
      vec_u16x16 __cached_3;
      vec_f32x16 _arg_cache_2 = sc_max(__cached_1, vec_f32x16(0.f));
      __cached_3 = tobf16(_arg_cache_2);
      vec_u16x16::store(__cached_3, &__outs_0[((((fused_0m_o__n_o_33 / 2UL) * 8192UL) + ((fused_0m_o__n_o_33 % 2UL) * 64UL)) + ((_fuseiter156 * 128UL) + _fuseiter157))]);
    }
  }
  sc_thread_aligned_free(__stream, __rescheduled_1);
}

static void matmul_core_cast_add_relu__160_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  matmul_core_cast_add_relu__160_closure_3(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr), (uint16_t*)(args[2UL].v_ptr), (uint16_t*)(args[3UL].v_ptr));
}

