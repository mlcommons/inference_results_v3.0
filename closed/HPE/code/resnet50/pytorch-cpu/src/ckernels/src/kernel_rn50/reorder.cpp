
#include <kernel/kernel_includes.hpp>
static constexpr void *__stream = &sc::runtime::default_stream;

extern int8_t reorder_data[0];
static constexpr int8_t* __module_data = reorder_data;
alignas(64) static int8_t __uninitialized_data[0UL];

static bool reorder_8x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_8x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_4x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_4x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_256x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static bool reorder_256x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept __attribute__((nonnull (1,2)));
static void reorder__60_closure_0_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__50_closure_1_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__40_closure_2_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__30_closure_3_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__20_closure_4_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__10_closure_5_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept __attribute__((nonnull (2,4)));
static void reorder__60_closure_0(uint64_t fused_0_fuseiter_12258___fuseiter_12259_3539, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));
static void reorder__50_closure_1(uint64_t fused_0_fuseiter_12262___fuseiter_12263_3540, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));
static void reorder__40_closure_2(uint64_t fused_0_fuseiter_12266___fuseiter_12267_3541, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));
static void reorder__30_closure_3(uint64_t fused_0_fuseiter_12270___fuseiter_12271_3542, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));
static void reorder__20_closure_4(uint64_t fused_0_fuseiter_12274___fuseiter_12275_3543, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));
static void reorder__10_closure_5(uint64_t fused_0_fuseiter_12278___fuseiter_12279_3544, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept __attribute__((nonnull (2,3)));



static bool reorder_8x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs0[2UL];
  __tempargs0[0UL] = __ins_0;
  __tempargs0[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__60_closure_0_0wrapper, __stream, __module_data, 0UL, 512UL, 1UL, __tempargs0);
  return true;
}

static bool reorder_8x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs1[2UL];
  __tempargs1[0UL] = __ins_0;
  __tempargs1[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__50_closure_1_0wrapper, __stream, __module_data, 0UL, 56UL, 1UL, __tempargs1);
  return true;
}

static bool reorder_4x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs2[2UL];
  __tempargs2[0UL] = __ins_0;
  __tempargs2[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__40_closure_2_0wrapper, __stream, __module_data, 0UL, 256UL, 1UL, __tempargs2);
  return true;
}

static bool reorder_4x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs3[2UL];
  __tempargs3[0UL] = __ins_0;
  __tempargs3[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__30_closure_3_0wrapper, __stream, __module_data, 0UL, 28UL, 1UL, __tempargs3);
  return true;
}

static bool reorder_256x64x56x56_ABCD_ACDB(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs4[2UL];
  __tempargs4[0UL] = __ins_0;
  __tempargs4[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__20_closure_4_0wrapper, __stream, __module_data, 0UL, 16384UL, 1UL, __tempargs4);
  return true;
}

static bool reorder_256x2048x7x7_ACDB_ABCD(int8_t* __restrict__ __outs_0, int8_t* __restrict__ __ins_0) noexcept{
  generic_val __tempargs5[2UL];
  __tempargs5[0UL] = __ins_0;
  __tempargs5[1UL] = __outs_0;
  sc_parallel_call_cpu_with_env((void*)&reorder__10_closure_5_0wrapper, __stream, __module_data, 0UL, 1792UL, 1UL, __tempargs5);
  return true;
}

static void reorder__60_closure_0(uint64_t fused_0_fuseiter_12258___fuseiter_12259_3539, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12260 = 0UL; _fuseiter_12260 < 56UL; _fuseiter_12260 += 1UL) {
    for (uint64_t _fuseiter_12261 = 0UL; _fuseiter_12261 < 56UL; _fuseiter_12261 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12258___fuseiter_12259_3539 / 64UL) * 200704UL) + (((fused_0_fuseiter_12258___fuseiter_12259_3539 % 64UL) * 3136UL) + ((_fuseiter_12260 * 56UL) + _fuseiter_12261)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12258___fuseiter_12259_3539 / 64UL) * 200704UL) + ((_fuseiter_12260 * 3584UL) + ((_fuseiter_12261 * 64UL) + (fused_0_fuseiter_12258___fuseiter_12259_3539 % 64UL))))] = __cached_1;
    }
  }
}

static void reorder__60_closure_0_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__60_closure_0(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}

static void reorder__50_closure_1(uint64_t fused_0_fuseiter_12262___fuseiter_12263_3540, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12264 = 0UL; _fuseiter_12264 < 7UL; _fuseiter_12264 += 1UL) {
    for (uint64_t _fuseiter_12265 = 0UL; _fuseiter_12265 < 2048UL; _fuseiter_12265 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12262___fuseiter_12263_3540 / 7UL) * 100352UL) + (((fused_0_fuseiter_12262___fuseiter_12263_3540 % 7UL) * 14336UL) + ((_fuseiter_12264 * 2048UL) + _fuseiter_12265)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12262___fuseiter_12263_3540 / 7UL) * 100352UL) + ((_fuseiter_12265 * 49UL) + (((fused_0_fuseiter_12262___fuseiter_12263_3540 % 7UL) * 7UL) + _fuseiter_12264)))] = __cached_1;
    }
  }
}

static void reorder__50_closure_1_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__50_closure_1(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}

static void reorder__40_closure_2(uint64_t fused_0_fuseiter_12266___fuseiter_12267_3541, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12268 = 0UL; _fuseiter_12268 < 56UL; _fuseiter_12268 += 1UL) {
    for (uint64_t _fuseiter_12269 = 0UL; _fuseiter_12269 < 56UL; _fuseiter_12269 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12266___fuseiter_12267_3541 / 64UL) * 200704UL) + (((fused_0_fuseiter_12266___fuseiter_12267_3541 % 64UL) * 3136UL) + ((_fuseiter_12268 * 56UL) + _fuseiter_12269)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12266___fuseiter_12267_3541 / 64UL) * 200704UL) + ((_fuseiter_12268 * 3584UL) + ((_fuseiter_12269 * 64UL) + (fused_0_fuseiter_12266___fuseiter_12267_3541 % 64UL))))] = __cached_1;
    }
  }
}

static void reorder__40_closure_2_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__40_closure_2(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}

static void reorder__30_closure_3(uint64_t fused_0_fuseiter_12270___fuseiter_12271_3542, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12272 = 0UL; _fuseiter_12272 < 7UL; _fuseiter_12272 += 1UL) {
    for (uint64_t _fuseiter_12273 = 0UL; _fuseiter_12273 < 2048UL; _fuseiter_12273 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12270___fuseiter_12271_3542 / 7UL) * 100352UL) + (((fused_0_fuseiter_12270___fuseiter_12271_3542 % 7UL) * 14336UL) + ((_fuseiter_12272 * 2048UL) + _fuseiter_12273)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12270___fuseiter_12271_3542 / 7UL) * 100352UL) + ((_fuseiter_12273 * 49UL) + (((fused_0_fuseiter_12270___fuseiter_12271_3542 % 7UL) * 7UL) + _fuseiter_12272)))] = __cached_1;
    }
  }
}

static void reorder__30_closure_3_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__30_closure_3(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}

static void reorder__20_closure_4(uint64_t fused_0_fuseiter_12274___fuseiter_12275_3543, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12276 = 0UL; _fuseiter_12276 < 56UL; _fuseiter_12276 += 1UL) {
    for (uint64_t _fuseiter_12277 = 0UL; _fuseiter_12277 < 56UL; _fuseiter_12277 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12274___fuseiter_12275_3543 / 64UL) * 200704UL) + (((fused_0_fuseiter_12274___fuseiter_12275_3543 % 64UL) * 3136UL) + ((_fuseiter_12276 * 56UL) + _fuseiter_12277)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12274___fuseiter_12275_3543 / 64UL) * 200704UL) + ((_fuseiter_12276 * 3584UL) + ((_fuseiter_12277 * 64UL) + (fused_0_fuseiter_12274___fuseiter_12275_3543 % 64UL))))] = __cached_1;
    }
  }
}

static void reorder__20_closure_4_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__20_closure_4(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}

static void reorder__10_closure_5(uint64_t fused_0_fuseiter_12278___fuseiter_12279_3544, int8_t* __restrict__ __ins_0, int8_t* __restrict__ __outs_0) noexcept{
  for (uint64_t _fuseiter_12280 = 0UL; _fuseiter_12280 < 7UL; _fuseiter_12280 += 1UL) {
    for (uint64_t _fuseiter_12281 = 0UL; _fuseiter_12281 < 2048UL; _fuseiter_12281 += 1UL) {
      int8_t __cached_0;
      __cached_0 = __ins_0[(((fused_0_fuseiter_12278___fuseiter_12279_3544 / 7UL) * 100352UL) + (((fused_0_fuseiter_12278___fuseiter_12279_3544 % 7UL) * 14336UL) + ((_fuseiter_12280 * 2048UL) + _fuseiter_12281)))];
      int8_t __cached_1;
      __cached_1 = __cached_0;
      __outs_0[(((fused_0_fuseiter_12278___fuseiter_12279_3544 / 7UL) * 100352UL) + ((_fuseiter_12281 * 49UL) + (((fused_0_fuseiter_12278___fuseiter_12279_3544 % 7UL) * 7UL) + _fuseiter_12280)))] = __cached_1;
    }
  }
}

static void reorder__10_closure_5_0wrapper(void* __stream, int8_t* __restrict__ __module_data, uint64_t i, generic_val* __restrict__ args) noexcept{
  reorder__10_closure_5(i, (int8_t*)(args[0UL].v_ptr), (int8_t*)(args[1UL].v_ptr));
}
