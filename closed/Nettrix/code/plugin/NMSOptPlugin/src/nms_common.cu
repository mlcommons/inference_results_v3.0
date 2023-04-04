/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "nms_common.h"

__global__ void fp32_to_fp16_kernel(__half* dst, const float* src, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    dst[idx] = __float2half_rn(src[idx]);
}

__global__ void fp16_to_fp32_kernel(float* dst, const __half* src, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    dst[idx] = __half2float(src[idx]);
}

void fp32_to_fp16(__half* dst, const float* src, int count, cudaStream_t stream)
{

    const int BLOCK_SIZE = 256;
    int grid_dim = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fp32_to_fp16_kernel<<<grid_dim, BLOCK_SIZE, 0, stream>>>(dst, src, count);
}

void fp16_to_fp32(float* dst, const __half* src, int count, cudaStream_t stream)
{

    const int BLOCK_SIZE = 256;
    int grid_dim = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fp16_to_fp32_kernel<<<grid_dim, BLOCK_SIZE, 0, stream>>>(dst, src, count);
}
