
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t reorder_data[0];
static constexpr int8_t* __module_data = reorder_data;
alignas(64) static int8_t __uninitialized_data[0UL];

extern "C" bool reorder_512x256_BA64a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0);
extern "C" bool reorder_512x256_AB_BA64a64b2a(uint16_t* __outs_0, uint16_t* __ins_0);
extern "C" bool reorder_256x128_BA64a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0);
extern "C" bool reorder_256x128_AB_BA64a64b2a(uint16_t* __outs_0, uint16_t* __ins_0);
extern "C" bool reorder_13x512_BA16a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0);
extern "C" bool reorder_13x512_AB_BA16a64b2a(uint16_t* __outs_0, uint16_t* __ins_0);
static void reorder__60_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__50_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__40_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__30_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__20_closure_4_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__10_closure_5_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args);
static void reorder__60_closure_0(uint64_t fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__50_closure_1(uint64_t fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__40_closure_2(uint64_t fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__30_closure_3(uint64_t fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__20_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer, uint16_t* __ins_0, uint16_t* __outs_0);
static void reorder__10_closure_5(uint64_t fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer, uint16_t* __ins_0, uint16_t* __outs_0);


static void main_entry(uint16_t* buffer_0, uint16_t* buffer_1, uint16_t* buffer_2, uint16_t* buffer_3, uint16_t* buffer_4, uint16_t* buffer_5, uint16_t* buffer_11, uint16_t* buffer_10, uint16_t* buffer_9, uint16_t* buffer_8, uint16_t* buffer_7, uint16_t* buffer_6){
  reorder_512x256_BA64a64b2a_AB(buffer_6, buffer_5);
  reorder_512x256_AB_BA64a64b2a(buffer_7, buffer_4);
  reorder_256x128_BA64a64b2a_AB(buffer_8, buffer_3);
  reorder_256x128_AB_BA64a64b2a(buffer_9, buffer_2);
  reorder_13x512_BA16a64b2a_AB(buffer_10, buffer_1);
  reorder_13x512_AB_BA16a64b2a(buffer_11, buffer_0);
}

extern "C" bool reorder_512x256_BA64a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs0[2UL];
  __tempargs0[0UL] = __ins_0;
  __tempargs0[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__60_closure_0_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs0);
  return true;
}

extern "C" bool reorder_512x256_AB_BA64a64b2a(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs1[2UL];
  __tempargs1[0UL] = __ins_0;
  __tempargs1[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__50_closure_1_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs1);
  return true;
}

extern "C" bool reorder_256x128_BA64a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs2[2UL];
  __tempargs2[0UL] = __ins_0;
  __tempargs2[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__40_closure_2_0wrapper, __stream, __module_data, 0UL, 64UL, 1UL, __tempargs2);
  return true;
}

extern "C" bool reorder_256x128_AB_BA64a64b2a(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs3[2UL];
  __tempargs3[0UL] = __ins_0;
  __tempargs3[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__30_closure_3_0wrapper, __stream, __module_data, 0UL, 64UL, 1UL, __tempargs3);
  return true;
}

extern "C" bool reorder_13x512_BA16a64b2a_AB(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs4[2UL];
  __tempargs4[0UL] = __ins_0;
  __tempargs4[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__20_closure_4_0wrapper, __stream, __module_data, 0UL, 16UL, 1UL, __tempargs4);
  return true;
}

extern "C" bool reorder_13x512_AB_BA16a64b2a(uint16_t* __outs_0, uint16_t* __ins_0){
  generic_val __tempargs5[2UL];
  __tempargs5[0UL] = __ins_0;
  __tempargs5[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__10_closure_5_0wrapper, __stream, __module_data, 0UL, 16UL, 1UL, __tempargs5);
  return true;
}

static void reorder__60_closure_0(uint64_t fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner = 0UL; fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner < 4UL; fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner += 1UL) {
    for (uint64_t _fuseiter_271 = 0UL; _fuseiter_271 < 64UL; _fuseiter_271 += 1UL) {
      for (uint64_t _fuseiter_272 = 0UL; _fuseiter_272 < 2UL; _fuseiter_272 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) / 256UL) * 32768UL) + ((((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) / 32UL) % 8UL) * 4096UL) + (((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) % 32UL) * 128UL) + ((_fuseiter_271 * 2UL) + _fuseiter_272))))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[((((_fuseiter_272 + ((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) / 32UL) % 8UL) * 64UL)) * 256UL) + (_fuseiter_271 + ((((fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0outer * 4UL) + fused_0fused_0_fuseiter_268___fuseiter_269_58___fuseiter_270_59_0inner) / 256UL) * 64UL)))] = __cached_1;
      }
    }
  }
}

static void reorder__60_closure_0_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__60_closure_0(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__50_closure_1(uint64_t fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner = 0UL; fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner < 4UL; fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner += 1UL) {
    for (uint64_t _fuseiter_276 = 0UL; _fuseiter_276 < 64UL; _fuseiter_276 += 1UL) {
      for (uint64_t _fuseiter_277 = 0UL; _fuseiter_277 < 2UL; _fuseiter_277 += 1UL) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[((((_fuseiter_277 + ((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) % 32UL) * 2UL)) + (((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) / 32UL) % 8UL) * 64UL)) * 256UL) + (_fuseiter_276 + ((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) / 256UL) * 64UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) / 256UL) * 32768UL) + ((((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) / 32UL) % 8UL) * 4096UL) + (((((fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0outer * 4UL) + fused_0fused_0_fuseiter_273___fuseiter_274_60___fuseiter_275_61_0inner) % 32UL) * 128UL) + ((_fuseiter_276 * 2UL) + _fuseiter_277))))] = __cached_1;
      }
    }
  }
}

static void reorder__50_closure_1_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__50_closure_1(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__40_closure_2(uint64_t fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner = 0UL; fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner < 256UL; fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner += 1UL) {
    for (uint64_t _fuseiter_282 = 0UL; _fuseiter_282 < 2UL; _fuseiter_282 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 8192UL) * 16384UL) + ((((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 2048UL) % 4UL) * 4096UL) + ((((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 64UL) % 32UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) % 64UL) * 2UL) + _fuseiter_282))))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[((((_fuseiter_282 + (((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 2048UL) % 4UL) * 64UL)) * 128UL) + ((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_278___fuseiter_279_62___fuseiter_280_63___fuseiter_281_64_0inner) / 8192UL) * 64UL)))] = __cached_1;
    }
  }
}

static void reorder__40_closure_2_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__40_closure_2(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__30_closure_3(uint64_t fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner = 0UL; fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner < 256UL; fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner += 1UL) {
    for (uint64_t _fuseiter_287 = 0UL; _fuseiter_287 < 2UL; _fuseiter_287 += 1UL) {
      uint16_t __cached_0;
      __cached_0 = __ins_0[((((_fuseiter_287 + (((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 64UL) % 32UL) * 2UL)) + (((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 2048UL) % 4UL) * 64UL)) * 128UL) + ((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 8192UL) * 64UL)))];
      uint16_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 8192UL) * 16384UL) + ((((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 2048UL) % 4UL) * 4096UL) + ((((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) / 64UL) % 32UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_283___fuseiter_284_65___fuseiter_285_66___fuseiter_286_67_0inner) % 64UL) * 2UL) + _fuseiter_287))))] = __cached_1;
    }
  }
}

static void reorder__30_closure_3_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__30_closure_3(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__20_closure_4(uint64_t fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner = 0UL; fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner < 256UL; fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner += 1UL) {
    for (uint64_t _fuseiter_292 = 0UL; _fuseiter_292 < 2UL; _fuseiter_292 += 1UL) {
      if (((((_fuseiter_292 + (((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 64UL) % 8UL) * 2UL)) < 16UL) && ((_fuseiter_292 + (((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 64UL) % 8UL) * 2UL)) < 13UL)) && (((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 512UL) * 64UL)) < 512UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) % 64UL) * 2UL) + _fuseiter_292)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((_fuseiter_292 + (((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 64UL) % 8UL) * 2UL)) * 512UL) + ((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_288___fuseiter_289_68___fuseiter_290_69___fuseiter_291_70_0inner) / 512UL) * 64UL)))] = __cached_1;
      }
    }
  }
}

static void reorder__20_closure_4_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__20_closure_4(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

static void reorder__10_closure_5(uint64_t fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer, uint16_t* __ins_0, uint16_t* __outs_0){
  for (uint64_t fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner = 0UL; fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner < 256UL; fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner += 1UL) {
    for (uint64_t _fuseiter_297 = 0UL; _fuseiter_297 < 2UL; _fuseiter_297 += 1UL) {
      if (((((_fuseiter_297 + (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 64UL) % 8UL) * 2UL)) < 16UL) && ((_fuseiter_297 + (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 64UL) % 8UL) * 2UL)) < 13UL)) && (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 512UL) * 64UL)) < 512UL))) {
        uint16_t __cached_0;
        __cached_0 = __ins_0[(((_fuseiter_297 + (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 64UL) % 8UL) * 2UL)) * 512UL) + ((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) % 64UL) + ((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 512UL) * 64UL)))];
        uint16_t __cached_1;
        __cached_1 = __cached_0;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) % 64UL) * 2UL) + _fuseiter_297)))] = __cached_1;
      } else {
        uint16_t __cached_2;
        __cached_2 = 0UL;
        __outs_0[(((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 512UL) * 1024UL) + ((((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) / 64UL) % 8UL) * 128UL) + (((((fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0outer * 256UL) + fused_0fused_0fused_0_fuseiter_293___fuseiter_294_71___fuseiter_295_72___fuseiter_296_73_0inner) % 64UL) * 2UL) + _fuseiter_297)))] = __cached_2;
      }
    }
  }
}

static void reorder__10_closure_5_0wrapper(void* __stream, int8_t* __module_data, uint64_t i, generic_val* args){
  reorder__10_closure_5(i, (uint16_t*)(args[0UL].v_ptr), (uint16_t*)(args[1UL].v_ptr));
}

