#pragma once
#include <cmath>
#include <immintrin.h>
#include <omp.h>

inline void
intAdd1LogScale(int8_t *dst, const int32_t *src, const size_t len, const float s) {
    size_t i = 0;
    __m512 onef32 = _mm512_set1_ps(1.0f);
    __m512 scale = _mm512_set1_ps(s);
    for (; i < len - 15; i += 16) {
        // load, convert to float, add 1, log
        // load
        __m512i i32 = _mm512_loadu_epi32(src + i);
        // convert to float
        __m512 f32 = _mm512_cvt_roundepi32_ps(i32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // add 1
        f32 = _mm512_add_ps(f32, onef32);
        // log
        f32 = _mm512_log_ps(f32);
        // scale
        f32 = _mm512_mul_ps(f32, scale);
        // to int32
        __m512i r_s32 = _mm512_cvt_roundps_epi32(f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        // to int8
        __m128i r_s8 = _mm512_cvtsepi32_epi8(r_s32);
        // store
        _mm_storeu_si128((__m128i *)(dst + i), r_s8);
    }
    for (; i < len; i++) {
        float t = log((float)src[i] + 1.0f) * s;
        if (t > 127.5)
            t = 127;
        dst[i] = (int8_t)round(t);
    }
}

inline void
intAdd1LogScalePad32(int8_t *dst, const int32_t *src, const size_t batch, const float s) {
    __m512 onef32 = _mm512_set1_ps(1.0f);
    __m512 scale = _mm512_set1_ps(s);
    for (size_t i = 0; i < batch; i += 2) {
        // load, convert to float, add 1, log
        // load
        __m512i x0_s32 = _mm512_loadu_epi32(src + i * 13);
        __m512i x1_s32 = _mm512_loadu_epi32(src + (i + 1) * 13);
        // convert to float
        __m512 x0_f32 = _mm512_cvt_roundepi32_ps(x0_s32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 x1_f32 = _mm512_cvt_roundepi32_ps(x1_s32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // add 1
        x0_f32 = _mm512_add_ps(x0_f32, onef32);
        x1_f32 = _mm512_add_ps(x1_f32, onef32);
        // log
        x0_f32 = _mm512_log_ps(x0_f32);
        x1_f32 = _mm512_log_ps(x1_f32);
        // scale
        x0_f32 = _mm512_mul_ps(x0_f32, scale);
        x0_f32 = _mm512_maskz_mov_ps(8191, x0_f32);
        x1_f32 = _mm512_mul_ps(x1_f32, scale);
        x1_f32 = _mm512_maskz_mov_ps(8191, x1_f32);
        // to int32
        __m512i r0_s32 = _mm512_cvt_roundps_epi32(x0_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m512i r1_s32 = _mm512_cvt_roundps_epi32(x1_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        // to int8
        __m128i r0_s8 = _mm512_cvtsepi32_epi8(r0_s32);
        __m128i r1_s8 = _mm512_cvtsepi32_epi8(r1_s32);
        __m512i r_s8 = _mm512_setzero_si512();
        r_s8 = _mm512_inserti32x4(r_s8, r0_s8, 0);
        r_s8 = _mm512_inserti32x4(r_s8, r1_s8, 2);
        // store
        _mm512_store_si512(dst + i * 32, r_s8);
    }
}

inline void
cvtS32toF32(float *dst, const int32_t *src, const size_t len) {
    size_t i = 0;
    for (; i < len - 15; i += 16) {
        // load
        __m512i i32 = _mm512_loadu_epi32(src + i);
        // convert to float
        __m512 f32 = _mm512_cvt_roundepi32_ps(i32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm512_storeu_ps(dst + i, f32);
    }
    for (; i < len; i++ ) {
        dst[i] = (float)src[i];
    }
}

inline void
transpose(const char *src, char *dst) {
    /*
     * 8x8 transpose int32
     * Astride = 8 * 4 in byte
     * Bstride = 8 * 4 in byte
     */
    constexpr int I_stride = 64;
    constexpr int O_stride = 64;
    constexpr int M = 32;
    constexpr int N = 16;
    __mmask16 k1 = 0xf0;
    __mmask16 k2 = 0xf00;
    __mmask16 k3 = 0xf000;
    __m128i A0, A1, A2, A3, A4, A5, A6, A7;
    __m512i B0, B1, B2, B3, B4, B5, B6, B7;
    for (int nn = 0; nn < N / 8; ++nn) {
        for (int mm = 0; mm < M / 16; ++mm) {
            A0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
            B0 = _mm512_broadcast_i32x4(A0);
            A1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  4 * I_stride));
            B0 = _mm512_mask_broadcast_i32x4(B0, k1, A1);
            A2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  8 * I_stride));
            B0 = _mm512_mask_broadcast_i32x4(B0, k2, A2);
            A3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 12 * I_stride));
            B0 = _mm512_mask_broadcast_i32x4(B0, k3, A3);

            A4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + I_stride));
            B1 = _mm512_broadcast_i32x4(A4);
            A5 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  5 * I_stride));
            B1 = _mm512_mask_broadcast_i32x4(B1, k1, A5);
            A6 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  9 * I_stride));
            B1 = _mm512_mask_broadcast_i32x4(B1, k2, A6);
            A7 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 13 * I_stride));
            B1 = _mm512_mask_broadcast_i32x4(B1, k3, A7);

            A0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  2 * I_stride));
            B2 = _mm512_broadcast_i32x4(A0);
            A1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  6 * I_stride));
            B2 = _mm512_mask_broadcast_i32x4(B2, k1, A1);
            A2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 10 * I_stride));
            B2 = _mm512_mask_broadcast_i32x4(B2, k2, A2);
            A3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 14 * I_stride));
            B2 = _mm512_mask_broadcast_i32x4(B2, k3, A3);

            A4 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  3 * I_stride));
            B3 = _mm512_broadcast_i32x4(A4);
            A5 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src +  7 * I_stride));
            B3 = _mm512_mask_broadcast_i32x4(B3, k1, A5);
            A6 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 11 * I_stride));
            B3 = _mm512_mask_broadcast_i32x4(B3, k2, A6);
            A7 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 15 * I_stride));
            B3 = _mm512_mask_broadcast_i32x4(B3, k3, A7);

            B4 = _mm512_unpacklo_epi32(B0, B1);
            B5 = _mm512_unpackhi_epi32(B0, B1);
            B6 = _mm512_unpacklo_epi32(B2, B3);
            B7 = _mm512_unpackhi_epi32(B2, B3);

            B0 = _mm512_unpacklo_epi64(B4, B6);
            B1 = _mm512_unpackhi_epi64(B4, B6);
            B2 = _mm512_unpacklo_epi64(B5, B7);
            B3 = _mm512_unpackhi_epi64(B5, B7);

            _mm512_store_si512(dst, B0);
            _mm512_store_si512(dst + 1 * O_stride, B1);
            _mm512_store_si512(dst + 2 * O_stride, B2);
            _mm512_store_si512(dst + 3 * O_stride, B3);

            src += 16;
            dst += 64 * 4;
        }
    }
}

#define TileA0  0
#define TileA0t 1
#define TileA1  2
#define TileA1t 3
#define TileB0  4
#define TileB1  5
#define TileB2  6

typedef struct tileconfig {
    uint8_t  palette_id;
    uint8_t  startRow;
    uint8_t  reserved[14];
    uint16_t colb[16];
    uint8_t  rows[16];
} tileconfig_t;

static inline void scale_and_store_int8_64(char * __restrict__ out,
                                           const char * __restrict__ in,
                                           __m512& __restrict__ scale) {
    auto in0_0_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)in));
    auto in0_1_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 16)));
    auto in0_2_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 32)));
    auto in0_3_32i = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(in + 48)));
    auto in0_0_32f = _mm512_cvt_roundepi32_ps(in0_0_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    auto in0_1_32f = _mm512_cvt_roundepi32_ps(in0_1_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    auto in0_2_32f = _mm512_cvt_roundepi32_ps(in0_2_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    auto in0_3_32f = _mm512_cvt_roundepi32_ps(in0_3_32i, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_0_32f = _mm512_mul_round_ps(in0_0_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_1_32f = _mm512_mul_round_ps(in0_1_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_2_32f = _mm512_mul_round_ps(in0_2_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_3_32f = _mm512_mul_round_ps(in0_3_32f, scale, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_0_32i = _mm512_cvt_roundps_epi32(in0_0_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_1_32i = _mm512_cvt_roundps_epi32(in0_1_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_2_32i = _mm512_cvt_roundps_epi32(in0_2_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    in0_3_32i = _mm512_cvt_roundps_epi32(in0_3_32f, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*)out, _mm512_cvtsepi32_epi8(in0_0_32i));
    _mm_storeu_si128((__m128i*)(out + 16), _mm512_cvtsepi32_epi8(in0_1_32i));
    _mm_storeu_si128((__m128i*)(out + 32), _mm512_cvtsepi32_epi8(in0_2_32i));
    _mm_storeu_si128((__m128i*)(out + 48), _mm512_cvtsepi32_epi8(in0_3_32i));
}

static inline __attribute__((always_inline))
void scale_and_move_ker_64(int8_t * __restrict__ out, const int8_t * __restrict__ in, float scale) {
    __m512 scale_vec512 = _mm512_set1_ps(scale);
    scale_and_store_int8_64((char *)out, (const char*)in, scale_vec512);
}

static inline __attribute__((always_inline))
void load_s8x64_store_aligned_ker(int8_t * __restrict__ out, const int8_t * __restrict__ in) {
    auto in0 = _mm512_load_si512(in);
    _mm512_store_si512(out, in0);
}

#define GET_INDEX(N, feature_index, batch_index) (feature_index) + (N) * (batch_index)

template<int EMBDIM, int ROW, bool scalex>
inline void
fuse_emb_int_kernel(int64_t Batch,
                    const tileconfig_t *tc,
                    const int8_t *densex,
                    const int32_t *index,
                    const int8_t *weight[26],
                    int8_t *res,
                    float c_scale,
                    float scales[384]) {
    #pragma omp parallel
    {
        int num_thr = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t chunk = (Batch - 1) / num_thr + 1;
        int64_t start = tid * chunk;
        int64_t end = std::min(start + chunk, Batch);
        int32_t obuf[32*16] __attribute__((aligned(64)));
        int32_t flat_buf[384] __attribute__((aligned(64)));
        int8_t xbuf[32 * 64] __attribute__((aligned(64)));
        int8_t xtrbuf[16 * 64] __attribute__((aligned(64)));
        int8_t *emb_ptr[26];
        constexpr int stride = 64;
        constexpr int LOG2_K = 7;
        _tile_loadconfig((const void * )tc);

        const int8_t* input0_ptr = densex + (start << LOG2_K);
        _mm_prefetch(input0_ptr, _MM_HINT_T0);
        _mm_prefetch(input0_ptr + 64, _MM_HINT_T0);
        int8_t* output0_ptr = res + start * ROW;
        for (int j = 0; j < 26; ++j) {
            int64_t ii = index[GET_INDEX(26, j, start)];
            ii = ii << LOG2_K;
            int8_t *p = (int8_t *)&(weight[j][ii]);
            emb_ptr[j] = p;
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p + 64, _MM_HINT_T0);
        }
        for (int64_t i = start; i < end; ++i) {
            _tile_zero(TileB0);
            _tile_zero(TileB1);
            _tile_zero(TileB2);
            for (int k = 0; k < 128; k += stride) {
                if (scalex) {
                    scale_and_move_ker_64(output0_ptr + k, input0_ptr + k, c_scale);
                    load_s8x64_store_aligned_ker(xbuf, input0_ptr + k);
                } else {
                    __m512i x = _mm512_load_si512(input0_ptr + k);
                    _mm512_store_si512(output0_ptr + k, x);
                    _mm512_store_si512(xbuf, x);
                }
                for (int j = 0; j < 26; ++j) {
                    int8_t *p = emb_ptr[j];
                    _mm512_store_si512(xbuf + (j + 1) * stride,
                                       _mm512_load_si512(p + k));
                }
                _tile_loadd(TileA0, xbuf, stride);
                transpose((char *)xbuf, (char *)xtrbuf);
                _tile_loadd(TileA0t, xtrbuf, stride);
                _tile_dpbssd(TileB0, TileA0, TileA0t);
                _tile_loadd(TileA1, &xbuf[0x400], stride);
                _tile_dpbssd(TileB1, TileA1, TileA0t);
                transpose((char *)xbuf + 0x400, (char *)xtrbuf);
                _tile_loadd(TileA1t, xtrbuf, stride);
                _tile_dpbssd(TileB2, TileA1, TileA1t);
            }

            if (i < end - 1) {
#pragma unroll
                for (int j = 0; j < 26; ++j) {
                    int64_t ii = index[GET_INDEX(26, j, i + 1)];
                    ii = ii << LOG2_K;
                    int8_t *p = (int8_t *)&(weight[j][ii]);
                    emb_ptr[j] = p;
                    _mm_prefetch(p, _MM_HINT_T0);
                    _mm_prefetch(p + 64, _MM_HINT_T0);
                }
            }
            _tile_stored(TileB0, obuf, stride);

            __m128i t128;
            __m256i t256;
            __m512i t512;
            flat_buf[0] = obuf[1 * 16 + 0];
            flat_buf[1] = obuf[2 * 16 + 0];
            flat_buf[2] = obuf[2 * 16 + 1];

            t128 = _mm_loadu_epi32(&obuf[3 * 16 + 0]);
            _mm_storeu_epi32(&flat_buf[3], t128);
            t128 = _mm_loadu_epi32(&obuf[4 * 16 + 0]);
            _mm_storeu_epi32(&flat_buf[6], t128);
            t256 = _mm256_loadu_epi32(&obuf[5 * 16 + 0]);
            _mm256_storeu_epi32(&flat_buf[10], t256);
            t256 = _mm256_loadu_epi32(&obuf[6 * 16 + 0]);
            _mm256_storeu_epi32(&flat_buf[15], t256);
            t256 = _mm256_loadu_epi32(&obuf[7 * 16 + 0]);
            _mm256_storeu_epi32(&flat_buf[21], t256);
            t256 = _mm256_loadu_epi32(&obuf[8 * 16 + 0]);
            _mm256_storeu_epi32(&flat_buf[28], t256);
            t512 = _mm512_loadu_si512(&obuf[9 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[36], t512);
            t512 = _mm512_loadu_si512(&obuf[10 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[45], t512);
            t512 = _mm512_loadu_si512(&obuf[11 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[55], t512);
            t512 = _mm512_loadu_si512(&obuf[12 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[66], t512);
            t512 = _mm512_loadu_si512(&obuf[13 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[78], t512);
            t512 = _mm512_loadu_si512(&obuf[14 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[91], t512);
            t512 = _mm512_loadu_si512(&obuf[15 * 16 + 0]);
            _mm512_storeu_si512(&flat_buf[105], t512);

            _tile_stored(TileB1, &obuf[0], stride * 2);
            _tile_stored(TileB2, &obuf[16], stride * 2);

            t512 = _mm512_loadu_si512(&obuf[0 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[120], t512);
            t512 = _mm512_loadu_si512(&obuf[1 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[136], t512);
            t256 = _mm256_loadu_epi32(&obuf[1 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[136 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[2 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[153], t512);
            t256 = _mm256_loadu_epi32(&obuf[2 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[153 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[3 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[171], t512);
            t256 = _mm256_loadu_epi32(&obuf[3 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[171 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[4 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[190], t512);
            t256 = _mm256_loadu_epi32(&obuf[4 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[190 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[5 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[210], t512);
            t256 = _mm256_loadu_epi32(&obuf[5 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[210 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[6 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[231], t512);
            t256 = _mm256_loadu_epi32(&obuf[6 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[231 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[7 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[253], t512);
            t256 = _mm256_loadu_epi32(&obuf[7 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[253 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[8 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[276], t512);
            t256 = _mm256_loadu_epi32(&obuf[8 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[276 + 16], t256);
            t512 = _mm512_loadu_si512(&obuf[9 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[300], t512);
            t256 = _mm256_loadu_epi32(&obuf[9 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[300 + 16], t256);
            flat_buf[300 + 24] = obuf[9 * 32 + 24];
            t512 = _mm512_loadu_si512(&obuf[10 * 32 + 0]);
            _mm512_storeu_si512(&flat_buf[325], t512);
            t256 = _mm256_loadu_epi32(&obuf[10 * 32 + 16]);
            _mm256_storeu_epi32(&flat_buf[325 + 16], t256);
            flat_buf[325 + 24] = obuf[10 * 32 + 24];
            flat_buf[325 + 25] = obuf[10 * 32 + 25];
            flat_buf[351] = 0;

            int8_t* outp = output0_ptr + EMBDIM;
            int off;
#pragma unroll
            for (off = 0; off < 384; off += 64) {
                int32_t *in_ptr = flat_buf + off;
                float *scales_ptr = scales + off;
                int8_t *output_ptr = outp + off;
                __m512i in0_s32 = _mm512_load_si512((const void*)in_ptr);
                __m512i in1_s32 = _mm512_load_si512((const void*)(in_ptr + 16));
                __m512i in2_s32 = _mm512_load_si512((const void*)(in_ptr + 32));
                __m512i in3_s32 = _mm512_load_si512((const void*)(in_ptr + 48));
                __m512 scale0_f32 = _mm512_load_ps(scales_ptr);
                __m512 scale1_f32 = _mm512_load_ps(scales_ptr + 16);
                __m512 scale2_f32 = _mm512_load_ps(scales_ptr + 32);
                __m512 scale3_f32 = _mm512_load_ps(scales_ptr + 48);
                __m512 in0_f32 = _mm512_cvt_roundepi32_ps(in0_s32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                __m512 in1_f32 = _mm512_cvt_roundepi32_ps(in1_s32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                __m512 in2_f32 = _mm512_cvt_roundepi32_ps(in2_s32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                __m512 in3_f32 = _mm512_cvt_roundepi32_ps(in3_s32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                in0_f32 = _mm512_mul_round_ps(in0_f32, scale0_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                in1_f32 = _mm512_mul_round_ps(in1_f32, scale1_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                in2_f32 = _mm512_mul_round_ps(in2_f32, scale2_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                in3_f32 = _mm512_mul_round_ps(in3_f32, scale3_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
                __m128i out1_s8 = _mm512_cvtsepi32_epi8(_mm512_cvt_roundps_epi32(in0_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                __m128i out2_s8 = _mm512_cvtsepi32_epi8(_mm512_cvt_roundps_epi32(in1_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                __m128i out3_s8 = _mm512_cvtsepi32_epi8(_mm512_cvt_roundps_epi32(in2_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                __m128i out4_s8 = _mm512_cvtsepi32_epi8(_mm512_cvt_roundps_epi32(in3_f32, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
                __m512i out_s8;
                out_s8 = _mm512_inserti32x4(out_s8, out1_s8, 0);
                out_s8 = _mm512_inserti32x4(out_s8, out2_s8, 1);
                out_s8 = _mm512_inserti32x4(out_s8, out3_s8, 2);
                out_s8 = _mm512_inserti32x4(out_s8, out4_s8, 3);
                _mm512_store_si512((__m512i *)output_ptr, out_s8);
            }

            input0_ptr += EMBDIM;
            _mm_prefetch(input0_ptr, _MM_HINT_T0);
            _mm_prefetch(input0_ptr + 64, _MM_HINT_T0);
            output0_ptr += ROW;
        }
    }
}
