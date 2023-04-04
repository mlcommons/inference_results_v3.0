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
#ifndef CONV_3D_3X3X3_C1_K32_H
#define CONV_3D_3X3X3_C1_K32_H
#include <stdint.h>

#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Input_Data_Type_, typename Output_Data_Type_, int THREADS_PER_PIXEL_, int D_PER_CTA_, int H_PER_CTA_,
    int W_PER_CTA_>
struct Conv3d_3x3x3_c1_k32_kernel_params
{
    enum
    {
        THREADS_PER_PIXEL = THREADS_PER_PIXEL_
    };

    enum
    {
        FLT_T = 3
    };
    enum
    {
        FLT_R = 3
    };
    enum
    {
        FLT_S = 3
    };

    // Implementation constants.
    enum
    {
        D_PER_CTA = D_PER_CTA_
    };
    enum
    {
        H_PER_CTA = H_PER_CTA_
    };
    enum
    {
        W_PER_CTA = W_PER_CTA_
    };

    enum
    {
        K_PER_CTA = 32
    };

    typedef Input_Data_Type_ Input_Data_Type;
    typedef Output_Data_Type_ Output_Data_Type;

    enum
    {
        WARP_SIZE = 32
    };
    enum
    {
        THREADS_PER_BLOCK = THREADS_PER_PIXEL * K_PER_CTA * W_PER_CTA / WARP_SIZE
    };
};

using kernel_params_int8 = Conv3d_3x3x3_c1_k32_kernel_params<int8_t, int8_t, 8, 4, 4, 32>;
using kernel_params_fp32 = Conv3d_3x3x3_c1_k32_kernel_params<float, float, 32, 4, 4, 32>;

struct Conv3d3x3x3c1k32Context
{
    Conv3d3x3x3c1k32Context()
        : sm_count(0)
        , sm_shared_size(0)
        , sm_version(0){};
    int sm_count;
    int sm_shared_size;
    int sm_version;
};

struct Conv3d3x3x3c1k32Params
{
    float scale;
    // The images.
    int img_n, img_c, img_d, img_h, img_w;
    int out_k, out_o, out_p, out_q;

    int img_stride_n, img_stride_c, img_stride_d, img_stride_h, img_stride_w;
    int out_stride_n, out_stride_k, out_stride_o, out_stride_p, out_stride_q;

    void* gmem_in;
    void* gmem_flt;
    void* gmem_out;

    int cta_per_d, cta_per_h, cta_per_w;

    bool is_fp32;

    // The filter.
    int flt_k;
};

int conv_3x3x3_c1_k32_dispatch(
    const Conv3d3x3x3c1k32Context& context, Conv3d3x3x3c1k32Params& params, cudaStream_t stream);

#endif // CONV_3D_3X3X3_C1_K32_H
