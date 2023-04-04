#pragma once
#include <iostream>
#include <cstdlib>
#include <immintrin.h>
#include "el_common_intrin.hpp"

namespace intel_mlperf {

template <int vec_length> struct avx_type;

template <> struct avx_type<8> { typedef __m256 type; };

template <> struct avx_type<16> { typedef __m512 type; };

template <int vec_length> class i_softmax_tpp {
public:
  i_softmax_tpp(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
      : dim0(dim0), dim1(dim1), dim2(dim2), dim3(dim3) {}
  void ref(void *out, void *in, float *att_mask, float M, float oscale);
  void ref(void *out, void *in, int32_t *att_lens, float M, float oscale);

protected:
  using __mtype = typename avx_type<vec_length>::type;

  int64_t dim0, dim1, dim2, dim3;
};

template <int vec_length, int N> struct i32_scale_attlen_softmax_scale_i8 {
  inline static void run(void *out, void *in, int32_t att_len, float M,
                         float oscale, int64_t ld);
};

// For specific format of <int (*)[16][16]> to <int8_t (*)[16][64]>
template <int N = 16> struct i32_scale_attlen_softmax_scale_i8_amx_tile_vnni {
  constexpr static size_t i_tile_w = 16;
  constexpr static size_t i_tile_h = 16;
  inline static void run(void *out, void *in, int32_t att_len, float M,
                         float oscale, size_t len_pad64) {
    auto pin = reinterpret_cast<int32_t(*)[i_tile_h][i_tile_w]>(in);
    auto att_tile = att_len / i_tile_w;
    auto att_tail = att_len - att_tile * i_tile_w;

    // Scratch for max subtraction
    alignas(64) float dout[att_tile + (att_tail > 0)][N][i_tile_w];

    auto neg_large = _mm512_set1_epi32(-500000);
    auto vscale = _mm512_set1_ps(M);

    __m512i vmax[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_setzero_epi32();
    }

    size_t d2;
    for (d2 = 0; d2 < att_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_si512(pin[d2][i]);
        vmax[i] = _mm512_max_epi32(l, vmax[i]);
      }
    }

    if (att_tail) {
      __mmask16 mask = (1 << att_tail) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_mask_loadu_epi32(neg_large, mask, pin[d2][i]);
        vmax[i] = _mm512_max_epi32(l, vmax[i]);
      }
    }

    __m512 vsum[N];
    __m512 vmaxps[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      auto tem = _mm512_cvtepi32_ps(vmax[i]) * vscale;
      vmaxps[i] = _mm512_max_reduce_ps(tem);
      vsum[i] = _mm512_setzero_ps();
    }

    for (d2 = 0; d2 < att_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_si512(pin[d2][i]);
        auto f = _mm512_cvtepi32_ps(l) * vscale;
        auto d = f - vmaxps[i];
        auto e = exp_ps_0_1(d);
        _mm512_storeu_ps(dout[d2][i], e);
        vsum[i] += e;
      }
    }

    if (att_tail) {
      __mmask16 mask = (1 << att_tail) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_mask_loadu_epi32(neg_large, mask, pin[d2][i]);
        auto f = _mm512_cvtepi32_ps(l) * vscale;
        auto d = f - vmaxps[i];
        auto e = exp_ps_0_1(d);
        _mm512_storeu_ps(dout[d2][i], e);
        vsum[i] += e;
      }
    }

    auto voscale = _mm512_set1_ps(oscale);

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
#ifdef usercp
      vsum[i] = voscale * _mm512_rcp14_ps(_mm512_add_reduce_ps(vsum[i]));
#else
      vsum[i] = voscale / _mm512_add_reduce_ps(vsum[i]);
#endif
    }
    // if att_tile cannot be divided by 4?
    auto att_tile16_in_tile64 = att_tile / 4;
    auto att_tile16_in_tile64_tail = att_tile % 4;

    auto pout = reinterpret_cast<int8_t(*)[i_tile_h][4][i_tile_w]>(out);

    // Gather 4*16*16 int8_t tile into 16*64 tile
    for (d2 = 0; d2 < att_tile16_in_tile64; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l0 = _mm512_loadu_ps(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ps(l0, vsum[i]);

        auto l1 = _mm512_loadu_ps(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ps(l1, vsum[i]);

        auto l2 = _mm512_loadu_ps(dout[d2*4+2][i]);
        auto i2 = _mm512_scale_minmax_i8_ps(l2, vsum[i]);

        auto l3 = _mm512_loadu_ps(dout[d2*4+3][i]);
        auto i3 = _mm512_scale_minmax_i8_ps(l3, vsum[i]);

        // write combine?
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][0], 0xffff, i0);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][1], 0xffff, i1);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][2], 0xffff, i2);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][3], 0xffff, i3);
      }
    }
    // Tail process
    switch (att_tile16_in_tile64_tail) {
    case 1:
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l0 = _mm512_loadu_ps(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ps(l0, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][0], 0xffff, i0);

        if (att_tail) {
          auto l1 = _mm512_loadu_ps(dout[d2*4+1][i]);
          auto i1 = _mm512_scale_minmax_i8_ps(l1, vsum[i]);
          _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][1], 0xffff, i1);
        } else {
          auto i1 = _mm_set1_epi8(0);
          _mm_storeu_si128((__m128i *)pout[d2][i][1], i1);
        }

        auto i2 = _mm256_set1_epi8(0);
        _mm256_storeu_si256((__m256i *)pout[d2][i][2], i2);
      }
      break;
    case 2:
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l0 = _mm512_loadu_ps(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ps(l0, vsum[i]);
        auto l1 = _mm512_loadu_ps(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ps(l1, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][0], 0xffff, i0);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][1], 0xffff, i1);

        if (att_tail) {
          auto l2 = _mm512_loadu_ps(dout[d2*4+2][i]);
          auto i2 = _mm512_scale_minmax_i8_ps(l2, vsum[i]);
          _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][2], 0xffff, i2);
          auto i3 = _mm_set1_epi8(0);
        _mm_storeu_si128((__m128i *)pout[d2][i][3], i3);
        } else {
          auto i2 = _mm256_set1_epi8(0);
          _mm256_storeu_si256((__m256i *)pout[d2][i][2], i2);
        }
      }
      break;
    case 3:
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l0 = _mm512_loadu_ps(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ps(l0, vsum[i]);
        auto l1 = _mm512_loadu_ps(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ps(l1, vsum[i]);
        auto l2 = _mm512_loadu_ps(dout[d2*4+2][i]);
        auto i2 = _mm512_scale_minmax_i8_ps(l2, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][0], 0xffff, i0);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][1], 0xffff, i1);
        _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][2], 0xffff, i2);

        if (att_tail) {
          auto l3 = _mm512_loadu_ps(dout[d2*4+3][i]);
          auto i3 = _mm512_scale_minmax_i8_ps(l3, vsum[i]);
          _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][3], 0xffff, i3);
        } else {
          auto i3 = _mm_set1_epi8(0);
          _mm_storeu_si128((__m128i *)pout[d2][i][3], i3);
        }
      }
      break;
    default:
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        if (att_tail) {
          auto l0 = _mm512_loadu_ps(dout[d2*4][i]);
          auto i0 = _mm512_scale_minmax_i8_ps(l0, vsum[i]);
          _mm512_mask_cvtepi32_storeu_epi8(pout[d2][i][0], 0xffff, i0);

          auto i1 = _mm_set1_epi8(0);
          auto i2 = _mm256_set1_epi8(0);
          _mm_storeu_si128((__m128i *)pout[d2][i][1], i1);
          _mm256_storeu_si256((__m256i *)pout[d2][i][2], i2);
        } else {
          auto i0 = _mm512_setzero_si512();
          _mm512_storeu_si512(pout[d2][i], i0);
        }
      }
      break;
    }
    // set other position of len_pad out of att_len to zero
    auto len_pad64_tile = len_pad64 / 64;
    auto i0 = _mm512_setzero_si512();
    d2++;
    for ( ; d2 < len_pad64_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        _mm512_storeu_si512(pout[d2][i], i0);
      }
    }
  }
};

template <int N = 16> struct i32_scale_attlen_softmax_scale_fp16_i8_amx_tile_vnni {
  constexpr static size_t i_tile_w = 16;
  constexpr static size_t i_tile_h = 16;
  inline static void run(void *out, void *in, int32_t att_len, float M,
                         float oscale, size_t len_pad64) {
    auto pin = reinterpret_cast<int32_t(*)[i_tile_h][i_tile_w]>(in);

    auto att_tile = att_len / i_tile_w;
    auto att_tail = att_len - att_tile * i_tile_w;

    // Scratch for max subtraction
    alignas(64) _Float16 dout[att_tile + (att_tail > 0)][N][i_tile_w];

    auto neg_large = _mm512_set1_epi32(-100000);
    auto vscale = _mm512_set1_ps(M);

    __m512i vmax[N];
    

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_setzero_epi32();
    }

    size_t d2;
    for (d2 = 0; d2 < att_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_si512(pin[d2][i]);
        vmax[i] = _mm512_max_epi32(l, vmax[i]);
      }
    }

    if (att_tail) {
      __mmask16 mask = (1 << att_tail) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_mask_loadu_epi32(neg_large, mask, pin[d2][i]);
        vmax[i] = _mm512_max_epi32(l, vmax[i]);
      }
    }

    __m512h vsum[N / 2];
    __m512 vmaxps[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      auto tem = _mm512_cvtepi32_ps(vmax[i]) * vscale;
      vmaxps[i] = _mm512_max_reduce_ps(tem);
    }

#pragma unroll(N / 2)
    for (int i = 0; i < N / 2; ++i) {
      vsum[i] = _mm512_setzero_ph();
    }

    for (d2 = 0; d2 < att_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l1 = _mm512_loadu_si512(pin[d2][i]);
        auto f = _mm512_cvtepi32_ps(l1) * vscale;
        // d can be transposed to fp16
        auto d = f - vmaxps[i];
        // fp16d is _mm256h
        auto fp16d1 = _mm512_cvtps_ph(d, _MM_FROUND_NO_EXC);

        auto l2 = _mm512_loadu_si512(pin[d2][i + 1]);
        f = _mm512_cvtepi32_ps(l2) * vscale;
        d = f - vmaxps[i + 1];
        auto fp16d2 = _mm512_cvtps_ph(d, _MM_FROUND_NO_EXC);

        // cat two _m256h to _m512h
        auto fp16_512d1 = _mm512_zextsi256_si512(fp16d1);
        auto fp16_512d2 = _mm512_zextsi256_si512(fp16d2);

        auto cat_d = _mm512_castsi512_ph(_mm512_shuffle_i64x2(fp16_512d1, fp16_512d2, _MM_SHUFFLE(1, 0, 1, 0)));
        auto cat_e = exp_ph_0_1(cat_d);

        _mm512_store_ph(dout[d2][i], cat_e);
        vsum[i >> 1] += cat_e;
      }
    }

    if (att_tail) {
      __mmask16 mask = (1 << att_tail) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l1 = _mm512_mask_loadu_epi32(neg_large, mask, pin[d2][i]);
        auto f = _mm512_cvtepi32_ps(l1) * vscale;
        // d can be transposed to fp16
        auto d = f - vmaxps[i];
        // fp16d is _mm256h
        auto fp16d1 = _mm512_cvtps_ph(d, _MM_FROUND_NO_EXC);

        auto l2 = _mm512_mask_loadu_epi32(neg_large, mask, pin[d2][i + 1]);
        f = _mm512_cvtepi32_ps(l2) * vscale;
        d = f - vmaxps[i + 1];
        auto fp16d2 = _mm512_cvtps_ph(d, _MM_FROUND_NO_EXC);

        // cat two _m256h to _m512h
        auto fp16_512d1 = _mm512_zextsi256_si512(fp16d1);
        auto fp16_512d2 = _mm512_zextsi256_si512(fp16d2);

        auto cat_d = _mm512_castsi512_ph(_mm512_shuffle_i64x2(fp16_512d1, fp16_512d2, _MM_SHUFFLE(1, 0, 1, 0)));
        auto cat_e = exp_ph_0_1(cat_d);

        _mm512_store_ph(dout[d2][i], cat_e);
        vsum[i >> 1] += cat_e;
      }
    }
    
    auto voscale = _mm512_set1_ph(oscale);
    #pragma unroll(N / 2)
    for (int i = 0; i < N / 2; ++i) {
#ifdef usercp
      vsum[i] = voscale * _mm512_rcp_ph(_mm512_half_add_reduce_ph(vsum[i]));
#else
      vsum[i] = voscale / _mm512_half_add_reduce_ph(vsum[i]);
#endif
    }

    // if att_tile cannot be divided by 4?
    auto att_tile16_in_tile64 = att_tile / 4;
    auto att_tile16_in_tile64_tail = att_tile % 4;

    auto pout = reinterpret_cast<int8_t(*)[i_tile_h][4][i_tile_w]>(out);

    // Gather 4*16*16 int8_t tile into 16*64 tile
    for (d2 = 0; d2 < att_tile16_in_tile64; ++d2) {
    // TODO: N i += 2
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l0 = _mm512_loadu_ph(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ph(l0, vsum[i >> 1]);

        auto l1 = _mm512_loadu_ph(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ph(l1, vsum[i >> 1]);

        auto l2 = _mm512_loadu_ph(dout[d2*4+2][i]);
        auto i2 = _mm512_scale_minmax_i8_ph(l2, vsum[i >> 1]);

        auto l3 = _mm512_loadu_ph(dout[d2*4+3][i]);
        auto i3 = _mm512_scale_minmax_i8_ph(l3, vsum[i >> 1]);

        _mm512_cvtepi16_epi8_shuffle_storeu(pout[d2][i], 4 * i_tile_w, i0, i1, i2, i3);
      }
    }
    // Tail process
    switch (att_tile16_in_tile64_tail) {
    case 1:
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l0 = _mm512_loadu_ph(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ph(l0, vsum[i >> 1]);
        
        __m512i i1;
        if (att_tail) {
          auto l1 = _mm512_loadu_ph(dout[d2*4+1][i]);
          i1 = _mm512_scale_minmax_i8_ph(l1, vsum[i >> 1]);
        } else {
          i1 = _mm512_setzero_si512();
        }

        auto i2 = _mm512_setzero_si512();
        auto i3 = _mm512_setzero_si512();

        _mm512_cvtepi16_epi8_shuffle_storeu(pout[d2][i], 4 * i_tile_w, i0, i1, i2, i3);
      }
      break;
    case 2:
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l0 = _mm512_loadu_ph(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ph(l0, vsum[i >> 1]);
        auto l1 = _mm512_loadu_ph(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ph(l1, vsum[i >> 1]);

        __m512i i2;
        if (att_tail) {
          auto l2 = _mm512_loadu_ph(dout[d2*4+2][i]);
          i2 = _mm512_scale_minmax_i8_ph(l2, vsum[i >> 1]);
        } else {
          i2 = _mm512_setzero_si512();
        }
        auto i3 = _mm512_setzero_si512();
        _mm512_cvtepi16_epi8_shuffle_storeu(pout[d2][i], 4 * i_tile_w, i0, i1, i2, i3);
      }
      break;
    case 3:
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        auto l0 = _mm512_loadu_ph(dout[d2*4][i]);
        auto i0 = _mm512_scale_minmax_i8_ph(l0, vsum[i >> 1]);
        auto l1 = _mm512_loadu_ph(dout[d2*4+1][i]);
        auto i1 = _mm512_scale_minmax_i8_ph(l1, vsum[i >> 1]);
        auto l2 = _mm512_loadu_ph(dout[d2*4+2][i]);
        auto i2 = _mm512_scale_minmax_i8_ph(l2, vsum[i >> 1]);

        __m512i i3;
        if (att_tail) {
          auto l3 = _mm512_loadu_ph(dout[d2*4+3][i]);
          i3 = _mm512_scale_minmax_i8_ph(l3, vsum[i >> 1]);
        } else {
          i3 = _mm512_setzero_si512();
        }
        _mm512_cvtepi16_epi8_shuffle_storeu(pout[d2][i], 4 * i_tile_w, i0, i1, i2, i3);
      }
      break;
    default:
#pragma unroll(N)
      for (int i = 0; i < N; i += 2) {
        __m512i i0;
        if (att_tail) {
          auto l0 = _mm512_loadu_ph(dout[d2*4][i]);
          i0 = _mm512_scale_minmax_i8_ph(l0, vsum[i >> 1]);
        } else {
          i0 = _mm512_setzero_si512();
        }

        auto i1 = _mm512_setzero_si512();
        auto i2 = _mm512_setzero_si512();
        auto i3 = _mm512_setzero_si512();

        _mm512_cvtepi16_epi8_shuffle_storeu(pout[d2][i], 4 * i_tile_w, i0, i1, i2, i3);
      }
      break;
    }
    // set other position of len_pad out of att_len to zero
    auto len_pad64_tile = len_pad64 / 64;
    auto i0 = _mm512_setzero_si512();
    d2++;
    for ( ; d2 < len_pad64_tile; ++d2) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        _mm512_storeu_si512(pout[d2][i], i0);
      }
    }
  }
};

template <int N> struct i32_scale_attlen_softmax_scale_i8<16, N> {
  inline static void run(void *out, void *in, int32_t att_len, float M,
                         float oscale, int64_t ld) {
    auto pin = reinterpret_cast<int32_t(*)[ld]>(in);
    // assert(att_len <= ld);
    auto att_l16 = (att_len + 15) / 16 * 16;

    // Scratch for max subtraction
    alignas(64) float dout[N][att_l16];

    auto neg_large = _mm512_set1_epi32(-500000);
    auto vscale = _mm512_set1_ps(M);

    __m512 vmax[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_setzero_ps();
    }

    int d2;
    for (d2 = 0; d2 < att_len / 16 * 16; d2 += 16) {

#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_loadu_si512(&pin[i][d2]);
        auto f = _mm512_cvtepi32_ps(l) * vscale;
        vmax[i] = _mm512_max_ps(f, vmax[i]);
        _mm512_storeu_ps(&dout[i][d2], f);
      }
    }

    if (d2 < att_len) {
      int rem = att_len - d2;
      __mmask16 mask = (1 << rem) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = _mm512_mask_loadu_epi32(neg_large, mask, &pin[i][d2]);
        auto f = _mm512_cvtepi32_ps(l) * vscale;
        vmax[i] = _mm512_max_ps(f, vmax[i]);
        _mm512_storeu_ps(&dout[i][d2], f);
      }
    }

    __m512 vsum[N];

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
      vmax[i] = _mm512_max_reduce_ps(vmax[i]);
      vsum[i] = _mm512_setzero_ps();
    }

    for (d2 = 0; d2 < att_l16; d2 += 16) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto f = _mm512_loadu_ps(&dout[i][d2]);
        auto d = f - vmax[i];
        auto e = exp_ps_0_1(d);
        _mm512_storeu_ps(&dout[i][d2], e);
        vsum[i] += e;
      }
    }

    auto voscale = _mm512_set1_ps(oscale);

#pragma unroll(N)
    for (int i = 0; i < N; ++i) {
#ifdef usercp
      vsum[i] = voscale * _mm512_rcp14_ps(_mm512_add_reduce_ps(vsum[i]));
#else
      vsum[i] = voscale / _mm512_add_reduce_ps(vsum[i]);
#endif
    }

    auto pout = reinterpret_cast<int8_t(*)[ld]>(out);
    auto zero = _mm512_setzero_ps();

    for (d2 = 0; d2 < ld / 16 * 16; d2 += 16) {
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = d2 < att_l16 ? _mm512_loadu_ps(&dout[i][d2]) : zero;
        auto i_4 = _mm512_scale_minmax_i8_ps(l, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(&pout[i][d2], 0xffff, i_4);
      }
    }

    if (d2 < ld) {
      int rem = ld - d2;
      __mmask16 mask = (1 << rem) - 1;
#pragma unroll(N)
      for (int i = 0; i < N; ++i) {
        auto l = d2 < att_l16 ? _mm512_loadu_ps(&dout[i][d2]) : zero;
        auto i_4 = _mm512_scale_minmax_i8_ps(l, vsum[i]);
        _mm512_mask_cvtepi32_storeu_epi8(&pout[i][d2], mask, i_4);
      }
    }
  }
};

void f_i32_scale_softmax_scale_i8(int8_t *out, int32_t *in, float *att_mask,
                                  float M, float oscale, int64_t ld, int l);

void f_i32_scale_softmax_scale_i8(int8_t *out, int32_t *in, int32_t att_len,
                                  float M, float oscale, int64_t ld, int l);

} // namespace intel_mlperf
