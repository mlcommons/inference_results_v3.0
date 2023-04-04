/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <stdio.h>

#include <algorithm>

#include "conv3d_3x3x3_c1_k32.h"

#include <mma.h>

using namespace nvcuda;

// <type, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
// WARP_ROW_TILES is determined as `c / 16`, where c is input feature maps, BLOCK_ROW_WARPS must be 1

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int ELEMENTS_PER_WARP_LOAD>
using Copy_int8_t = typename std::conditional<ELEMENTS_PER_WARP_LOAD == 32, int8_t,
    typename std::conditional<ELEMENTS_PER_WARP_LOAD == 64, uint16_t,
        typename std::conditional<ELEMENTS_PER_WARP_LOAD == 128, int,
            typename std::conditional<ELEMENTS_PER_WARP_LOAD == 256, int2, int4>::type>::type>::type>::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_int8_t<sizeof(T) / sizeof(int8_t) * ELEMENTS_PER_WARP_LOAD>;

template <int ELEMENTS_PER_THREAD>
using copy_int8_t = Copy_t<int8_t, kernel_params_int8::WARP_SIZE * ELEMENTS_PER_THREAD>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int ELEMENTS_PER_THREAD>
union Access_t {
    Copy_t<T, kernel_params_int8::WARP_SIZE * ELEMENTS_PER_THREAD> v;
    T x[ELEMENTS_PER_THREAD];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __host__ __device__ int div_up(int m, int n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
__global__ void __launch_bounds__(Kernel_params::THREADS_PER_BLOCK, 1)
    conv_3x3x3_c1_k32_linear_kernel(Conv3d3x3x3c1k32Params params)
{
    // Naive FP32 impplementation

    typedef typename Kernel_params::Input_Data_Type Input_Data_Type;
    typedef typename Kernel_params::Output_Data_Type Output_Data_Type;
    typedef float Math_Type;

    constexpr int PAD = 1;

    constexpr int FLT_T = Kernel_params::FLT_T;
    constexpr int FLT_R = Kernel_params::FLT_R;
    constexpr int FLT_S = Kernel_params::FLT_S;

    constexpr int D_PER_CTA = Kernel_params::D_PER_CTA;
    constexpr int H_PER_CTA = Kernel_params::H_PER_CTA;
    constexpr int W_PER_CTA = Kernel_params::W_PER_CTA;

    constexpr int K_PER_CTA = Kernel_params::K_PER_CTA;

    constexpr int WARP_SIZE = Kernel_params::WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = Kernel_params::THREADS_PER_BLOCK;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;

    constexpr int FLT_SIZE = FLT_T * FLT_R * FLT_S * K_PER_CTA;
    constexpr int SMEM_D_DIM = (D_PER_CTA + FLT_T - 1);
    constexpr int SMEM_H_DIM = (H_PER_CTA + FLT_R - 1);
    constexpr int SMEM_W_DIM = (W_PER_CTA + FLT_S - 1);
    __shared__ Input_Data_Type smem[SMEM_D_DIM * SMEM_H_DIM * SMEM_W_DIM];
    __shared__ Input_Data_Type smem_flt[FLT_SIZE];

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    int cta_d_begin = blockIdx.x % params.cta_per_d * D_PER_CTA;
    int cta_h_begin = blockIdx.x / params.cta_per_d % params.cta_per_h * H_PER_CTA;
    int cta_w_begin = blockIdx.x / (params.cta_per_d * params.cta_per_h) * W_PER_CTA;

    int cta_o_begin = cta_d_begin;
    int cta_p_begin = cta_h_begin;
    int cta_q_begin = cta_w_begin;

    Input_Data_Type* gmem_in = reinterpret_cast<Input_Data_Type*>(params.gmem_in) + n * params.img_stride_n
        + c * params.img_stride_c + cta_d_begin * params.img_stride_d + cta_h_begin * params.img_stride_h
        + cta_w_begin * params.img_stride_w;

    Input_Data_Type* gmem_flt = reinterpret_cast<Input_Data_Type*>(params.gmem_flt);

    Output_Data_Type* gmem_out = reinterpret_cast<Output_Data_Type*>(params.gmem_out) + n * params.out_stride_n
        + cta_o_begin * params.out_stride_o + cta_p_begin * params.out_stride_p + cta_q_begin * params.out_stride_q;

    bool is_valid = true;
    // input load is not efficient, but is much smaller than the output
    for (int w_index = lane_id; w_index < SMEM_W_DIM; w_index += WARP_SIZE)
    {
        int w = cta_w_begin - PAD + w_index;
        for (int dh_index = warp_id; dh_index < SMEM_D_DIM * SMEM_H_DIM; dh_index += NUM_WARPS)
        {
            is_valid = (w >= 0) && (w < params.img_w);
            int h_index = dh_index % SMEM_H_DIM;
            int d_index = dh_index / SMEM_H_DIM;
            int d = cta_d_begin - PAD + d_index;
            int h = cta_h_begin - PAD + h_index;
            is_valid = is_valid && (d >= 0) && (d < params.img_d);
            is_valid = is_valid && (h >= 0) && (h < params.img_h);
            smem[d_index * SMEM_H_DIM * SMEM_W_DIM + h_index * SMEM_W_DIM + w_index] = (is_valid)
                ? *(reinterpret_cast<Input_Data_Type*>(params.gmem_in) + n * params.img_stride_n
                      + d * params.img_stride_d + h * params.img_stride_h + w * params.img_stride_w)
                : Input_Data_Type(0);
        }
    }

    // Assume KCTRS format (does not matter for C == 1)
    for (int i = threadIdx.x; i < FLT_SIZE; i += THREADS_PER_BLOCK)
    {
        smem_flt[i] = gmem_flt[i];
    }

    __syncthreads();

    constexpr int Q_PER_CTA = W_PER_CTA;
    const int thread_in_cta_q = threadIdx.x % Q_PER_CTA;
    const int thread_k = threadIdx.x / Q_PER_CTA;

#pragma unroll
    for (int o_index = 0; o_index < D_PER_CTA; o_index++)
    {
#pragma unroll
        for (int p_index = 0; p_index < H_PER_CTA; p_index++)
        {
#pragma unroll
            for (int k = thread_k; k < K_PER_CTA; k += THREADS_PER_BLOCK / SMEM_W_DIM)
            {
                Math_Type sum = 0.0F;
#pragma unroll
                for (int t = 0; t < FLT_T; t++)
                {
#pragma unroll
                    for (int r = 0; r < FLT_R; r++)
                    {
#pragma unroll
                        for (int s = 0; s < FLT_S; s++)
                        {

                            Input_Data_Type val = smem[(o_index + t) * SMEM_H_DIM * SMEM_W_DIM
                                + (p_index + r) * SMEM_W_DIM + thread_in_cta_q + s];

                            sum += (float) val
                                * (float) smem_flt[k * 1 * FLT_T * FLT_R * FLT_S + t * FLT_R * FLT_S + r * FLT_S + s];
                        }
                    }
                }
                gmem_out[o_index * params.out_stride_o + p_index * params.out_stride_p
                    + thread_in_cta_q * params.out_stride_q + k * params.out_stride_k]
                    = sum;
            }
        }
    }
}

// The slicing to be implemented
// extern "C" __global__ void UNet3DKiTS19SliceKernelI8Linear(
//     const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams* p)
// {
//     int d = blockIdx.x;
//     int h = blockIdx.y;
//     int w = threadIdx.x;

//     d_out[p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w]
//         = d_in[p->image_w * p->image_h * (p->offset_d + d) + p->image_w * (p->offset_h + h) + (p->offset_w + w)];
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
__global__ void __launch_bounds__(Kernel_params::THREADS_PER_BLOCK, 1)
    conv_3x3x3_c1_k32_int8_kernel(Conv3d3x3x3c1k32Params params)
{
    /*
    x - spacial dim
    y - input c dim
    z - batch dim

    TILE_X - 32 or 64
    TILE_Y - 4
    TILE_Z - 4

    K_PER_CTA = 32

    */

    typedef typename Kernel_params::Input_Data_Type Input_Data_Type;
    typedef typename Kernel_params::Output_Data_Type Output_Data_Type;
    typedef int Math_Type;

    constexpr int PAD = 1;

    constexpr int FLT_T = Kernel_params::FLT_T;
    constexpr int FLT_R = Kernel_params::FLT_R;
    constexpr int FLT_S = Kernel_params::FLT_S;

    constexpr int D_PER_CTA = Kernel_params::D_PER_CTA;
    constexpr int H_PER_CTA = Kernel_params::H_PER_CTA;
    constexpr int W_PER_CTA = Kernel_params::W_PER_CTA;

    constexpr int K_PER_CTA = Kernel_params::K_PER_CTA;

    constexpr int WARP_SIZE = Kernel_params::WARP_SIZE;
    constexpr int THREADS_PER_BLOCK = Kernel_params::THREADS_PER_BLOCK;

    constexpr int THREADS_PER_PIXEL = Kernel_params::THREADS_PER_PIXEL;
    constexpr int ELEMENTS_PER_THREAD = WARP_SIZE / THREADS_PER_PIXEL;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;

    constexpr int FLT_S_PAD = (FLT_S + 4 - 1) / 4 * 4;
    constexpr int FLT_SIZE = FLT_T * FLT_R * FLT_S_PAD * K_PER_CTA;
    constexpr int SMEM_D_DIM = (D_PER_CTA + FLT_T - 1);
    constexpr int SMEM_H_DIM = (H_PER_CTA + FLT_R - 1);
    constexpr int SMEM_W_DIM = (W_PER_CTA + FLT_S - 1);

    constexpr int SMEM_W_STRIDE = 4;
    __shared__ Input_Data_Type smem[SMEM_D_DIM * SMEM_H_DIM * SMEM_W_DIM * SMEM_W_STRIDE];
    __shared__ Input_Data_Type smem_flt[FLT_SIZE];

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    int cta_d_begin = blockIdx.x % params.cta_per_d * D_PER_CTA;
    int cta_h_begin = blockIdx.x / params.cta_per_d % params.cta_per_h * H_PER_CTA;
    int cta_w_begin = blockIdx.x / (params.cta_per_d * params.cta_per_h) * W_PER_CTA;

    int cta_o_begin = cta_d_begin;
    int cta_p_begin = cta_h_begin;
    int cta_q_begin = cta_w_begin;

    Input_Data_Type* gmem_in = reinterpret_cast<Input_Data_Type*>(params.gmem_in) + n * params.img_stride_n
        + c * params.img_stride_c + cta_d_begin * params.img_stride_d + cta_h_begin * params.img_stride_h
        + cta_w_begin * params.img_stride_w;

    Input_Data_Type* gmem_flt = reinterpret_cast<Input_Data_Type*>(params.gmem_flt);

    Output_Data_Type* gmem_out = reinterpret_cast<Output_Data_Type*>(params.gmem_out) + n * params.out_stride_n
        + cta_o_begin * params.out_stride_o + cta_p_begin * params.out_stride_p + cta_q_begin * params.out_stride_q;

    bool is_valid = true;
    // input load is not efficient, but is much smaller than the output
    for (int w_index = lane_id; w_index < SMEM_W_DIM; w_index += WARP_SIZE)
    {
        for (int dh_index = warp_id; dh_index < SMEM_D_DIM * SMEM_H_DIM; dh_index += NUM_WARPS)
        {
            // 4th element is garbage, but would be zeroed out by filter zero-padding
            for (int iw = 0; iw < SMEM_W_STRIDE - 1; iw++)
            {
                int w = cta_w_begin - PAD + w_index + iw;
                is_valid = (w >= 0) && (w < params.img_w);
                int h_index = dh_index % SMEM_H_DIM;
                int d_index = dh_index / SMEM_H_DIM;
                int d = cta_d_begin - PAD + d_index;
                int h = cta_h_begin - PAD + h_index;
                is_valid = is_valid && (d >= 0) && (d < params.img_d);
                is_valid = is_valid && (h >= 0) && (h < params.img_h);
                smem[dh_index * SMEM_W_DIM * SMEM_W_STRIDE + w_index * SMEM_W_STRIDE + iw] = (is_valid)
                    ? *(reinterpret_cast<Input_Data_Type*>(params.gmem_in) + n * params.img_stride_n
                          + d * params.img_stride_d + h * params.img_stride_h + w * params.img_stride_w)
                    : Input_Data_Type(0);
            }
        }
    }

    // populate filter (KCTRS format C==1) properly for FLT_S_PAD
    for (int i = threadIdx.x; i < FLT_SIZE; i += THREADS_PER_BLOCK)
    {
        int ktr_index = i / FLT_S_PAD;
        int s = i % FLT_S_PAD;
        is_valid = s != 3;

        // convert smem_flt index into gmem_flt index
        // technically j == round(i * 0.75) if is_valid, but below should be faster
        int g = is_valid ? s : 0;
        int j = ktr_index * FLT_S + g;

        smem_flt[i] = is_valid ? gmem_flt[j] : Input_Data_Type(0);
    }

    __syncthreads();

    // whole CTA works in Q dimension
    const int thread_k = threadIdx.x % THREADS_PER_PIXEL;
    const int thread_in_cta_q = threadIdx.x / THREADS_PER_PIXEL;

    // NOTE/FIXME: assumption is LITTLE ENDIAN when INT8 values are packed in 4 Byte chunk
    int flt_int[ELEMENTS_PER_THREAD][FLT_T][FLT_R];
    int* smem_int_flt = reinterpret_cast<int*>(smem_flt);

    // load fliter to registers
    for (int k_index = 0; k_index < ELEMENTS_PER_THREAD; k_index++)
    {
        int k = thread_k * ELEMENTS_PER_THREAD + k_index;
        for (int t = 0; t < FLT_T; t++)
        {
            for (int r = 0; r < FLT_R; r++)
            {
                flt_int[k_index][t][r] = smem_int_flt[k * 1 * FLT_T * FLT_R * FLT_S_PAD / 4 + t * FLT_R * FLT_S_PAD / 4
                    + r * FLT_S_PAD / 4];
            }
        }
    }

    Access_t<int8_t, ELEMENTS_PER_THREAD> res;
    using copy_t = copy_int8_t<ELEMENTS_PER_THREAD>;

#pragma unroll
    for (int o_index = 0; o_index < D_PER_CTA; o_index++)
    {
#pragma unroll
        for (int p_index = 0; p_index < H_PER_CTA; p_index++)
        {
#pragma unroll
            for (int k_index = 0; k_index < ELEMENTS_PER_THREAD; k_index++)
            {
                Math_Type sum = 0;
#pragma unroll
                for (int t = 0; t < FLT_T; t++)
                {
#pragma unroll
                    for (int r = 0; r < FLT_R; r++)
                    {
                        auto vals
                            = *reinterpret_cast<int*>(&smem[(o_index + t) * SMEM_H_DIM * SMEM_W_DIM * SMEM_W_STRIDE
                                + (p_index + r) * SMEM_W_DIM * SMEM_W_STRIDE + SMEM_W_STRIDE * thread_in_cta_q]);
                        sum = __dp4a(vals, flt_int[k_index][t][r], sum);
                    }
                }
                // saturate value
                float x = __int2float_rn(sum) * params.scale;
                res.x[k_index] = __float_as_int(min(max(x + 12582912.0F, 12582785.0F), 12583039.0F));
            }
            *(reinterpret_cast<copy_t*>(&gmem_out[o_index * params.out_stride_o + p_index * params.out_stride_p
                  + thread_in_cta_q * params.out_stride_q])
                + thread_k)
                = res.v;
        }
    }
}

template <typename Kernel_params>
int conv_3x3x3_c1_k32_linear(
    const Conv3d3x3x3c1k32Context& context, Conv3d3x3x3c1k32Params& params, cudaStream_t stream)
{
    assert(Kernel_params::THREADS_PER_BLOCK == Kernel_params::THREADS_PER_PIXEL * Kernel_params::K_PER_CTA);

    assert(params.img_c == 1);
    assert(params.flt_k >= 32 && params.flt_k % 32 == 0);
    assert(params.img_d % Kernel_params::D_PER_CTA == 0);
    assert(params.img_h % Kernel_params::H_PER_CTA == 0);
    assert(params.img_w % Kernel_params::W_PER_CTA == 0);

    const int block_sz = Kernel_params::THREADS_PER_BLOCK;

    params.cta_per_d = div_up(params.img_d, Kernel_params::D_PER_CTA);
    params.cta_per_h = div_up(params.img_h, Kernel_params::H_PER_CTA);
    params.cta_per_w = div_up(params.img_w, Kernel_params::W_PER_CTA);

    dim3 grid = dim3(params.cta_per_d * params.cta_per_h * params.cta_per_w, params.img_c, params.img_n);
    // const int loops = div_up(div_up(params.m, block_sz), grid);

    conv_3x3x3_c1_k32_linear_kernel<Kernel_params><<<grid, block_sz, 0, stream>>>(params);

    return 0;
}

template <typename Kernel_params>
int conv_3x3x3_c1_k32_int8(const Conv3d3x3x3c1k32Context& context, Conv3d3x3x3c1k32Params& params, cudaStream_t stream)
{
    assert(Kernel_params::THREADS_PER_BLOCK
        == Kernel_params::THREADS_PER_PIXEL * Kernel_params::K_PER_CTA * Kernel_params::W_PER_CTA
            / Kernel_params::WARP_SIZE);

    assert(params.img_c == 1);
    assert(params.flt_k >= 32 && params.flt_k % 32 == 0);
    assert(params.img_d % Kernel_params::D_PER_CTA == 0);
    assert(params.img_h % Kernel_params::H_PER_CTA == 0);
    assert(params.img_w % Kernel_params::W_PER_CTA == 0);

    const int block_sz = Kernel_params::THREADS_PER_BLOCK;

    params.cta_per_d = div_up(params.img_d, Kernel_params::D_PER_CTA);
    params.cta_per_h = div_up(params.img_h, Kernel_params::H_PER_CTA);
    params.cta_per_w = div_up(params.img_w, Kernel_params::W_PER_CTA);

    dim3 grid = dim3(params.cta_per_d * params.cta_per_h * params.cta_per_w, params.img_c, params.img_n);
    // const int loops = div_up(div_up(params.m, block_sz), grid);

    conv_3x3x3_c1_k32_int8_kernel<Kernel_params><<<grid, block_sz, 0, stream>>>(params);

    return 0;
}

int conv_3x3x3_c1_k32_dispatch(
    const Conv3d3x3x3c1k32Context& context, Conv3d3x3x3c1k32Params& params, cudaStream_t stream)
{

    if (params.is_fp32)
    {
        conv_3x3x3_c1_k32_linear<kernel_params_fp32>(context, params, stream);
    }
    else
    {
        conv_3x3x3_c1_k32_int8<kernel_params_int8>(context, params, stream);
    }

    return 0;
}
