#include <assert.h>
#include <cstdlib>

#include "amx_tdpbssd.hpp"
#include "el_common_intrin.hpp"
#include "i_mha_tpp.hpp"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"
#include "amx_config.hpp"

#include "helper.hpp"


using Time = std::chrono::high_resolution_clock;
namespace intel_mlperf {

static constexpr int max_tile_row = 16;
static constexpr int max_tile_colsb = 64;



static void reorder_k_to_buffer(int8_t *k_buffer, const int8_t *k_ptr, int row,
                                int col_tile, int stride) {
  /// k_buffer format : int8_t [col_tile][16][64]
  auto k_ptr_ = reinterpret_cast<const int8_t(*)[stride]>(k_ptr);
  auto k_buffer_ = reinterpret_cast<int8_t(*)[1024]>(k_buffer);

  int i = 0;
  for (; i < col_tile - 1; i++) {
    tr_vnni_x64<16>(k_buffer_[i], k_ptr_[i * 16], stride, 64);
  }

  decltype(tr_vnni_x64<1>)* tr_vnni_tbl [] = {
    tr_vnni_x64<1>, tr_vnni_x64<2>, tr_vnni_x64<3>, tr_vnni_x64<4>,
    tr_vnni_x64<5>, tr_vnni_x64<6>, tr_vnni_x64<7>, tr_vnni_x64<8>,
    tr_vnni_x64<9>, tr_vnni_x64<10>, tr_vnni_x64<11>, tr_vnni_x64<12>,
    tr_vnni_x64<13>, tr_vnni_x64<14>, tr_vnni_x64<15>, tr_vnni_x64<16>,
  };

  int k_tail = row - (col_tile - 1) * 16;
  tr_vnni_tbl[k_tail - 1](k_buffer_[i], k_ptr_[i * 16], stride, 64);
}

static void reorder_v_to_buffer_p64(int8_t *v_buffer, const int8_t *v_ptr, int row, int stride) {
  /// reorder v to v_buffer
  /// v_buffer format [4][col_tile*4][64]
  auto row_pad = (row + 63) / 64 * 64;
  int v_buffer_row = row_pad / 4;
  size_t v_stride = v_buffer_row * 64;
  int v_real_step = (row + 3) / 4;
  int v_tail = row - (v_real_step - 1) * 4;
  auto v_ptr_ = reinterpret_cast<const int8_t(*)[stride]>(v_ptr);
  auto v_buffer_ = reinterpret_cast<int8_t(*)[v_buffer_row][64]>(v_buffer);

  for (int i = 0; i < v_buffer_row; i++) {
    if (i >= v_real_step - 1) {
      switch (v_tail) {
      case (1):
        tr_vnni_4x<1>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (2):
        tr_vnni_4x<2>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (3):
        tr_vnni_4x<3>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (4):
        tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      default:
        tr_vnni_4x<0>(v_buffer_[0][i], v_ptr_[0], stride, v_stride);
        break;
      }
      v_tail = -1;
    } else {
      tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
    }
  }
}

// We limit row_tile 1 or 2, col_tile: 3, ..., 24 (384, could be more)
template <int row_tile, int col_tile> struct qk_gemm_impl {
  static constexpr int ldc = col_tile * 16;

  inline static void tile_loada(const void *a, size_t lda, int overlap) {
    auto a_ = reinterpret_cast<const int8_t(*)[lda]>(a);
    _tile_loadd(TMM4, a_[0], lda);
    if (row_tile == 2)
      _tile_loadd(TMM5, a_[16 - overlap], lda);
  }

  template <bool tail>
  inline static void tile_loadb(const void *b, int col_idx) {
    auto b_ = reinterpret_cast<const int8_t(*)[1024]>(b);
    _tile_loadd(TMM6, b_[col_idx * 2], 64);
    if (!tail)
      _tile_loadd(TMM7, b_[col_idx * 2 + 1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  template <bool tail> inline static void dot_prod(void *c, int col_idx) {
    auto c_ = reinterpret_cast<int (*)[col_tile][16][16]>(c);

    __tile_dpbssd<TMM0, TMM4, TMM6>();
    _tile_stored(TMM0, c_[0][col_idx * 2], 64);

    if (!tail) {
      __tile_dpbssd<TMM1, TMM4, TMM7>();
      _tile_stored(TMM1, c_[0][col_idx * 2 + 1], 64);
    }

    if (row_tile == 2) {
      __tile_dpbssd<TMM2, TMM5, TMM6>();
      _tile_stored(TMM2, c_[1][col_idx * 2], 64);

      if (!tail) {
        __tile_dpbssd<TMM3, TMM5, TMM7>();
        _tile_stored(TMM3, c_[1][col_idx * 2 + 1], 64);
      }
    }
  }

  // Preloaded A
  inline static void compute(void *c, const void *a, const void *b, size_t lda,
                             int overlap) {
    constexpr int col_tail = col_tile % 2;
    constexpr int col_loop = col_tile / 2;

    tile_loada(a, lda, overlap);

    int i = 0;
#pragma unroll(col_loop)
    for (; i < col_loop; ++i) {
      tile_loadb<false>(b, i);
      zero_accum();
      dot_prod<false>(c, i);
    }

    if (col_tail) {
      tile_loadb<true>(b, i);
      zero_accum();
      dot_prod<true>(c, i);
    }
  }

  inline static void softmax(int8_t *c_int8, int *c, int len, float M,
                             float oscale) {
    assert(len <= col_tile * 16);
    auto c_ = reinterpret_cast<int (*)[col_tile][16][16]>(c);

    // TODO: how many 16*64 tile does c_int8 have?
    // TODO: would col_tile must be divided by 4?
    int c_int8_p64_ctile = (col_tile * 16 + 63) / 64;
    auto c_len_pad64 = c_int8_p64_ctile * 64;
    auto c_int8_ = reinterpret_cast<int8_t (*)[c_int8_p64_ctile][16][64]>(c_int8);

    i32_scale_attlen_softmax_scale_fp16_i8_amx_tile_vnni<16>::run(c_int8_[0], c_[0], len, M, oscale, c_len_pad64);

    if (row_tile == 2) {
      i32_scale_attlen_softmax_scale_fp16_i8_amx_tile_vnni<16>::run(c_int8_[1], c_[1], len, M, oscale, c_len_pad64);
    }
  }
};

// n_tile: 1 or 2
// k_tile: {1, 2, 3, 4, 5, 6}
template <int n_tile, int k_step> struct av_gemm_impl {
  inline static void loada(void *a, size_t idx) {
    auto a_ = reinterpret_cast<int8_t (*)[k_step][16][64]>(a);

    _tile_loadd(TMM4, a_[0][idx], 64);
    if (n_tile == 2)
      _tile_loadd(TMM5, a_[1][idx], 64);
  }

  inline static void loadb(void *b_scratch, size_t ldb) {
    auto b_ = reinterpret_cast<int8_t(*)[ldb]>(b_scratch);
    _tile_loadd(TMM6, b_[0], 64);
    _tile_loadd(TMM7, b_[1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  inline static void dot_prod() {
    __tile_dpbssd<TMM0, TMM4, TMM6>();
    __tile_dpbssd<TMM1, TMM4, TMM7>();
    if (n_tile == 2) {
      __tile_dpbssd<TMM2, TMM5, TMM6>();
      __tile_dpbssd<TMM3, TMM5, TMM7>();
    }
  }

  inline static void store_quant(void *c, size_t ldc, size_t overlap, float m2) {
    alignas(64) int scratch[n_tile * 16 * 32];
    auto scratch_ = reinterpret_cast<int(*)[16][16]>(scratch);
    _tile_stored(TMM0, scratch_[0], 64);
    _tile_stored(TMM1, scratch_[1], 64);

    if (n_tile == 2) {
      _tile_stored(TMM2, scratch_[2], 64);
      _tile_stored(TMM3, scratch_[3], 64);
    }

    // quant out to c
    auto vscale = _mm512_set1_ps(m2);
    auto c_out = reinterpret_cast<int8_t (*)[ldc]>(c);

#pragma unroll(n_tile)
    for (int i = 0; i < n_tile; i++) {
#pragma unroll(16)
      for (int j = 0; j < 16; j++) {
        auto pr = _mm512_loadu_si512(scratch_[2 * i][j]);
        auto prf = _mm512_cvtepi32_ps(pr);
        auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
        int out_row = (i == n_tile - 1) ? (i * 16 - overlap + j) : (i * 16 + j);
        _mm512_mask_cvtepi32_storeu_epi8(&c_out[out_row][0], 0xffff, iout);
        pr = _mm512_loadu_si512(scratch_[2 * i + 1][j]);
        prf = _mm512_cvtepi32_ps(pr);
        iout = _mm512_scale_minmax_i8_ps(vscale, prf);
        _mm512_mask_cvtepi32_storeu_epi8(&c_out[out_row][16], 0xffff, iout);
      }
    }
  }

  inline static void compute(void *c, void *a, void *b_scratch, size_t ldc, size_t overlap, float m2) {

    zero_accum();

    auto b_ = reinterpret_cast<int8_t (*)[k_step][16][64]>(b_scratch);
    auto c_ = reinterpret_cast<int8_t (*)[ldc]>(c);

    size_t ldb = k_step * 16 * 64;

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(a, i);
      loadb(b_[0][i], ldb);
      dot_prod();
    }
    // quant only one step done
    store_quant(&c_[0][0], ldc, overlap, m2);
    zero_accum();

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(a, i);
      loadb(b_[2][i], ldb);
      dot_prod();
    }
    store_quant(&c_[0][32], ldc, overlap, m2);
  }
};

void amx_per_head(const void *qkv_ptr, size_t ldqkv, void *a_ptr, size_t ldatt, size_t sl,
                      float M, float oscale, int32_t att_mask, float M2) {

  int sl_pad = (sl + 15) / 16 * 16;
  int col_tile = sl_pad / 16;

  alignas(64) int8_t k_scrach[16 * sl_pad * 4];
  int sl_pad64 = (sl + 63) / 64 * 64;
  alignas(64) int8_t v_scrach_p64[sl_pad64 * 64];

  auto q = reinterpret_cast<const int8_t(*)[ldqkv]>(qkv_ptr);
  int qkv_dis = ldqkv / 3;
  reorder_k_to_buffer(k_scrach, &q[0][qkv_dis], sl, col_tile, ldqkv);
  reorder_v_to_buffer_p64(v_scrach_p64, &q[0][qkv_dis * 2], sl, ldqkv);

  int cur_r_pos = 0;
  int row_loop = col_tile / 2;
  int rollback =
      (sl % max_tile_row != 0) ? max_tile_row - (sl % max_tile_row) : 0;
  bool is_even = (col_tile % 2 == 0);
  alignas(64) int a_scrach[32 * sl_pad];
  alignas(64) int8_t apro_scrach[32 * sl_pad64];
  auto a = reinterpret_cast<int8_t(*)[ldatt]>(a_ptr);
  auto tile_config = Tilecfg();
  tile_config.set_config();
  for (int i = 0; i < row_loop; i++) {
    int overlap = (is_even && i == row_loop - 1) ? rollback : 0;
    cur_r_pos = i * 32;
    switch (col_tile) {
    case (3):
      qk_gemm_impl<2, 3>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 3>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (4):
      qk_gemm_impl<2, 4>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 4>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (5):
      qk_gemm_impl<2, 5>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 5>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (6):
      qk_gemm_impl<2, 6>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 6>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (7):
      qk_gemm_impl<2, 7>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 7>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (8):
      qk_gemm_impl<2, 8>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 8>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (9):
      qk_gemm_impl<2, 9>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 9>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (10):
      qk_gemm_impl<2, 10>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 10>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (11):
      qk_gemm_impl<2, 11>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 11>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (12):
      qk_gemm_impl<2, 12>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 12>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (13):
      qk_gemm_impl<2, 13>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 13>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (14):
      qk_gemm_impl<2, 14>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 14>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (15):
      qk_gemm_impl<2, 15>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 15>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (16):
      qk_gemm_impl<2, 16>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 16>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (17):
      qk_gemm_impl<2, 17>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 17>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, overlap, M2);
      break;
    case (18):
      qk_gemm_impl<2, 18>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 18>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (19):
      qk_gemm_impl<2, 19>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 19>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, overlap, M2);
      break;
    case (20):
      qk_gemm_impl<2, 20>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 20>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (21):
      qk_gemm_impl<2, 21>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 21>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (22):
      qk_gemm_impl<2, 22>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 22>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    case (23):
      qk_gemm_impl<2, 23>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 23>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, overlap, M2);
      break;
    case (24):
      qk_gemm_impl<2, 24>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 24>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, overlap, M2);
      break;
    }
  }
  cur_r_pos += 32 - rollback;
  if (!is_even) {
    switch (col_tile) {
    case (3):
      qk_gemm_impl<1, 3>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 3>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (5):
      qk_gemm_impl<1, 5>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 5>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (7):
      qk_gemm_impl<1, 7>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 7>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (9):
      qk_gemm_impl<1, 9>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 9>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (11):
      qk_gemm_impl<1, 11>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 11>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (13):
      qk_gemm_impl<1, 13>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 13>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (15):
      qk_gemm_impl<1, 15>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 15>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (17):
      qk_gemm_impl<1, 17>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 17>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, 0, M2);
      break;
    case (19):
      qk_gemm_impl<1, 19>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 19>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, 0, M2);
      break;
    case (21):
      qk_gemm_impl<1, 21>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 21>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, ldatt, 0, M2);
      break;
    case (23):
      qk_gemm_impl<1, 23>::compute(a_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 23>::softmax(apro_scrach, a_scrach, att_mask, M, oscale);
      av_gemm_impl<1, 6>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, ldatt, 0, M2);
      break;
    default:
      break;
    }
  }
}

} // namespace intel_mlperf
