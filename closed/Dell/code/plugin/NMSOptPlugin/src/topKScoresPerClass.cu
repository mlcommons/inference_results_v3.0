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

#include <cub/cub.cuh>
#include <vector>

#include <fstream>
#include <iostream>

#include "nms_common.h"
#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include "topK.h"

// C-API for the async copy
#include "nms_common.h"
#include <cuda_pipeline.h>

#define CUSTOM_TRANSPOSE 1
#define ENABLE_FUSED_TRANSPOSE 1

#if CUSTOM_TRANSPOSE == 0
#include <cublas_v2.h>
#endif

template <typename T>
void saveDeviceBuffer_topk(const T* buf, size_t count, std::ofstream& output)
{
    T* hBuf = new T[count];
    cudaDeviceSynchronize();
    cudaMemcpy(hBuf, buf, count * sizeof(T), cudaMemcpyDeviceToHost);
    output.write((char*) hBuf, count * sizeof(T));
    delete[] hBuf;
}

namespace nvinfer1
{
namespace plugin
{

template <int TILE_X, int TILE_Y, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void batched_transpose_fuse_sigmoid_topk_prepare_kernel(const float* data_in, float* data_out,
    const int num_classes, const int num_priors, int* active_counts, int* out_indices, int background_class_id,
    float threshold)
{
    // num_classes is "fast" dimension in data_in
    // BLOCK_DIM_X == TILE_X

    constexpr int SMEM_STRIDE = (TILE_X < TILE_Y) ? TILE_Y + 1 : TILE_X + 1;
    __shared__ float smem[SMEM_STRIDE * TILE_Y];
    __shared__ int smem_idx[SMEM_STRIDE * TILE_Y];

    __shared__ int smem_offset[BLOCK_DIM_X * (BLOCK_DIM_Y + 1)];
    __shared__ int smem_global_offset[TILE_X];

    // batch id
    const int n = blockIdx.z;
    const int n_offset = n * num_classes * num_priors;

    const int tile_x_in_offset = blockIdx.x * TILE_X;
    const int tile_y_in_offset = blockIdx.y * TILE_Y;

    const float* data_in_tile = data_in + n_offset + tile_x_in_offset + tile_y_in_offset * num_classes;

    if (tile_x_in_offset >= num_classes || tile_y_in_offset >= num_priors)
    {
        return;
    }

    int class_id = tile_x_in_offset + threadIdx.x;
    float is_not_background_class_id = (class_id != background_class_id) ? 1.0F : 0.0F;

    int active_count_per_y = 0;

    const int SCORES_PER_THREAD = TILE_Y / BLOCK_DIM_Y;
    float reg_scores[SCORES_PER_THREAD] = {0};

    //__syncthreads();

    // first element of the prefix sum
    smem_offset[threadIdx.y * BLOCK_DIM_X + threadIdx.x] = 0;

#pragma unroll
    for (int jj = 0; jj < SCORES_PER_THREAD; jj++)
    {
        int j = jj * BLOCK_DIM_Y;
        int x_idx = threadIdx.x;
        int y_idx = j + threadIdx.y;
        bool is_not_valid = tile_x_in_offset + x_idx >= num_classes || tile_y_in_offset + y_idx >= num_priors;
        // int tile_in_idx = threadIdx.x + tile_x_in_offset
        float score = (is_not_valid) ? -30.F : data_in_tile[y_idx * num_classes + x_idx];
        score = __expf(score) / (1.0f + __expf(score));
        // score = (score < threshold)? 0.0F : is_not_background_class_id * score;
        is_not_valid = is_not_valid || (class_id == background_class_id) || score < threshold;
        bool is_valid = !is_not_valid;
        active_count_per_y += is_valid;
        reg_scores[jj] = (is_valid) ? score : 0.0F;

        // smem[SMEM_STRIDE * y_idx + x_idx] = (is_not_valid)? 0.F : data_in_tile[y_idx * num_classes + x_idx];
        // if (is_valid) {
        //     smem[SMEM_STRIDE * (j + active_count_per_y) + x_idx] = score;
        // }
    }

    // prefix sum
    smem_offset[(threadIdx.y + 1) * BLOCK_DIM_X + threadIdx.x] = active_count_per_y;

    __syncthreads();

    if (threadIdx.y == 0)
    {
        for (int j = 2; j <= BLOCK_DIM_Y; j++)
        {
            smem_offset[j * BLOCK_DIM_X + threadIdx.x] += smem_offset[(j - 1) * BLOCK_DIM_X + threadIdx.x];
        }
    }

    __syncthreads();

    int partial_offset = smem_offset[threadIdx.y * BLOCK_DIM_X + threadIdx.x];
    int offset = smem_offset[BLOCK_DIM_Y * BLOCK_DIM_X + threadIdx.x];
    if (class_id < num_classes)
    {
        if (threadIdx.y == 0)
        {
            smem_global_offset[threadIdx.x] = atomicAdd(&active_counts[n * num_classes + class_id], offset);
        }
    }
    int cur = 0;
#pragma unroll
    for (int jj = 0; jj < SCORES_PER_THREAD; jj++)
    {
        int j = jj * BLOCK_DIM_Y;
        int offset = smem_offset[threadIdx.y * BLOCK_DIM_X + threadIdx.x];
        if (reg_scores[jj] != 0.0F)
        {
            smem[SMEM_STRIDE * threadIdx.x + partial_offset + cur] = reg_scores[jj];
            smem_idx[SMEM_STRIDE * threadIdx.x + partial_offset + cur]
                = class_id * num_priors + tile_y_in_offset + j + threadIdx.y;
            cur++;
        }
    }

    __syncthreads();

    int tile_x_out_offset = 0;
    int tile_y_out_offset = blockIdx.x * TILE_X;

#pragma unroll
    for (int j = 0; j < TILE_X; j += BLOCK_DIM_Y)
    {
        int y_idx = j + threadIdx.y;
        tile_x_out_offset = smem_global_offset[y_idx];
        float* data_out_tile = data_out + n_offset + tile_x_out_offset + tile_y_out_offset * num_priors;
        int* out_indices_tile = out_indices + n_offset + tile_x_out_offset + tile_y_out_offset * num_priors;
        // TILE_X == BLOCK_DIM_X
        offset = smem_offset[BLOCK_DIM_Y * BLOCK_DIM_X + y_idx];
        for (int x_idx = threadIdx.x; x_idx < offset; x_idx += BLOCK_DIM_X)
        {
            bool is_not_valid = tile_x_out_offset + x_idx >= num_priors // remove
                || tile_y_out_offset + y_idx >= num_classes;
            if (!is_not_valid)
            {
                // if (threadIdx.x == 0 && tile_x_out_offset != 0) {
                // if (smem[SMEM_STRIDE * x_idx + y_idx] > 0.89f) {
                //     printf("%f %d %d %d\n", smem[SMEM_STRIDE * x_idx + y_idx], smem_idx[SMEM_STRIDE * x_idx + y_idx],
                //     x_idx, y_idx);
                // }
                data_out_tile[y_idx * num_priors + x_idx] = smem[SMEM_STRIDE * y_idx + x_idx];
                out_indices_tile[y_idx * num_priors + x_idx]
                    = smem_idx[SMEM_STRIDE * y_idx + x_idx]; // tile_x_out_offset + x_idx;
            }
        }
    }
}

template <int BLOCK_THREADS>
__global__ void get_cub_offsets_kernel(
    int* in_active_counts, int* out_begin_offsets, int* out_end_offsets, int items, int segments, int num_top_k)
{
    // in_active_counts and out_end_offsets can be the same buffer

    int segment_id = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int stride = items / segments;
    if (segment_id >= segments)
        return;

    out_end_offsets[segment_id] = in_active_counts[segment_id] + segment_id * stride;

    out_begin_offsets[segment_id] = segment_id * stride;
}

template <int TILE_X, int TILE_Y, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void batched_transpose_kernel(
    const float* data_in, float* data_out, const int cols, const int rows, const int output_n_stride)
{
    // cols is "fast" dimension in data_in (i.e. row major)
    // BLOCK_DIM_X == TILE_X

    constexpr int SMEM_STRIDE = TILE_X + 1;
    __shared__ float smem[SMEM_STRIDE * TILE_Y];

    // batch id
    const int n = blockIdx.z;
    const int n_in_offset = n * cols * rows; // n * cols * rows;
    const int n_out_offset = n * output_n_stride;

    const int tile_x_in_offset = blockIdx.x * TILE_X;
    const int tile_y_in_offset = blockIdx.y * TILE_Y;

    const float* data_in_tile = data_in + n_in_offset + tile_x_in_offset + tile_y_in_offset * cols;

    if (tile_x_in_offset >= cols || tile_y_in_offset >= rows)
    {
        return;
    }

#pragma unroll
    for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y)
    {
        int x_idx = threadIdx.x;
        int y_idx = j + threadIdx.y;
        bool is_not_valid = tile_x_in_offset + x_idx >= cols || tile_y_in_offset + y_idx >= rows;
        // int tile_in_idx = threadIdx.x + tile_x_in_offset
        smem[SMEM_STRIDE * y_idx + x_idx] = (is_not_valid) ? 0.F : data_in_tile[y_idx * cols + x_idx];
    }

    __syncthreads();

    int tile_x_out_offset = blockIdx.y * TILE_Y;
    int tile_y_out_offset = blockIdx.x * TILE_X;
    float* data_out_tile = data_out + n_out_offset + tile_x_out_offset + tile_y_out_offset * rows;

#pragma unroll
    for (int x_idx = threadIdx.x; x_idx < TILE_Y; x_idx += BLOCK_DIM_X)
    {
#pragma unroll
        for (int j = 0; j < TILE_X; j += BLOCK_DIM_Y)
        {
            int y_idx = j + threadIdx.y;
            bool is_not_valid = tile_x_out_offset + x_idx >= rows || tile_y_out_offset + y_idx >= cols;
            if (!is_not_valid)
            {
                data_out_tile[y_idx * rows + x_idx] = smem[SMEM_STRIDE * x_idx + y_idx];
            }
        }
    }
}

#define ENABLE_LDGSTS 1
template <int TILE_X, int TILE_Y, int BLOCK_DIM_X, int BLOCK_DIM_Y, int N_TILES = 1>
__global__ void batched_transpose_pf_kernel(
    const float* data_in, float* data_out, const int cols, const int rows, const int output_n_stride)
{
    // cols is "fast" dimension in data_in
    // BLOCK_DIM_X == TILE_X
    // N_TILES = number of tiles in Y direction (we have a very narrow matrix)

    constexpr int SMEM_STRIDE = TILE_X + 1;
    constexpr int SMEM_FACTOR = (N_TILES > 1) ? 2 : 1;
    __shared__ float smem[SMEM_FACTOR * SMEM_STRIDE * TILE_Y];

    // batch id
    const int n = blockIdx.z;
    const int n_in_offset = n * cols * rows; // n * cols * rows;
    const int n_out_offset = n * output_n_stride;

    const int tile_x_in_offset = blockIdx.x * TILE_X;
    const int tile_y_in_offset = blockIdx.y * TILE_Y * N_TILES;

    if (tile_x_in_offset >= cols || tile_y_in_offset >= rows)
    {
        return;
    }

    const float* data_in_tile = data_in + n_in_offset + tile_x_in_offset + tile_y_in_offset * cols;
#pragma unroll
    for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y)
    {
        int x_idx = threadIdx.x;
        int y_idx = j + threadIdx.y;
        bool is_not_valid = tile_x_in_offset + x_idx >= cols || tile_y_in_offset + y_idx >= rows;
        // int tile_in_idx = threadIdx.x + tile_x_in_offset
#if ENABLE_LDGSTS == 1
        size_t zfill = (is_not_valid) ? sizeof(float) : 0;
        __pipeline_memcpy_async(
            &smem[SMEM_STRIDE * y_idx + x_idx], &data_in_tile[y_idx * cols + x_idx], sizeof(float), zfill);
#else
        smem[SMEM_STRIDE * y_idx + x_idx] = (is_not_valid) ? 0.F : data_in_tile[y_idx * cols + x_idx];
#endif
    }

#if ENABLE_LDGSTS == 1
    __pipeline_commit();
#endif

    int istage = 0;
    int istage_next = 1;
    for (int itile = 0; itile < N_TILES; itile++)
    {

#if ENABLE_LDGSTS == 1
        __pipeline_wait_prior(0);
#endif
        __syncthreads();

        istage_next = (istage + 1) % 2;

        const int tile_y_in_offset = blockIdx.y * TILE_Y * N_TILES + (itile + 1) * TILE_Y;
        const float* data_in_tile = data_in + n_in_offset + tile_x_in_offset + tile_y_in_offset * cols;
        if (tile_y_in_offset < rows && itile < N_TILES - 1)
        {
#pragma unroll
            for (int j = 0; j < TILE_Y; j += BLOCK_DIM_Y)
            {
                int x_idx = threadIdx.x;
                int y_idx = j + threadIdx.y;
                bool is_not_valid = tile_x_in_offset + x_idx >= cols || tile_y_in_offset + y_idx >= rows;
#if ENABLE_LDGSTS == 1
                size_t zfill = (is_not_valid) ? sizeof(float) : 0;
                __pipeline_memcpy_async(&smem[istage_next * SMEM_STRIDE * TILE_Y + SMEM_STRIDE * y_idx + x_idx],
                    &data_in_tile[y_idx * cols + x_idx], sizeof(float), zfill);
#else
                smem[istage_next * SMEM_STRIDE * TILE_Y + SMEM_STRIDE * y_idx + x_idx]
                    = (is_not_valid) ? 0.F : data_in_tile[y_idx * cols + x_idx];
#endif
            }
#if ENABLE_LDGSTS == 1
            __pipeline_commit();
#endif
        }

        // transpose the previous stage while loading the next one
        int tile_x_out_offset = blockIdx.y * TILE_Y * N_TILES + itile * TILE_Y;
        int tile_y_out_offset = blockIdx.x * TILE_X;
        float* data_out_tile = data_out + n_out_offset + tile_x_out_offset + tile_y_out_offset * rows;

#pragma unroll
        for (int x_idx = threadIdx.x; x_idx < TILE_Y; x_idx += BLOCK_DIM_X)
        {
#pragma unroll
            for (int j = 0; j < TILE_X; j += BLOCK_DIM_Y)
            {
                int y_idx = j + threadIdx.y;
                bool is_not_valid = tile_x_out_offset + x_idx >= rows || tile_y_out_offset + y_idx >= cols;
                if (!is_not_valid)
                {
                    data_out_tile[y_idx * rows + x_idx]
                        = smem[istage * SMEM_STRIDE * TILE_Y + SMEM_STRIDE * x_idx + y_idx];
                }
            }
        }

        istage = istage_next;
    }
}

// __global__ void sigmoid_kernel(float* data, int nthreads) {
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid >= nthreads) return;

//     float val = data[tid];
//     data[tid] = __expf(val) / (1.0f + __expf(val));
// }

#if CUSTOM_TRANSPOSE == 0
cublasHandle_t handle_;
#endif

void permuteConfDataFuseCubTopKPrepare(cudaStream_t stream, const int nthreads, const int num_classes,
    const int num_priors, int num_layers, bool confSigmoid, void* new_data, void* tmp_data,
    const void* const* conf_data, void* active_counts_per_class, const int* feature_size, const int* num_anchors,
    const bool packed32_nchw, void* out_indices, const int background_class_id, const float threshold)
{

    assert(packed32_nchw == 0);

#if CUSTOM_TRANSPOSE == 0
    static bool is_first_call = true;
    if (is_first_call)
    {
        cublasCreate(&handle_);
    }
    cublasSetStream(handle_, stream);
#endif

    float alpha = 1.f;
    float beta = 0.f;
    float* b_matrix = 0;

    int output_n_stride = num_classes * num_priors;
    int num_images = nthreads / output_n_stride;
    // printf("permuteConfData_transpose: num_images = %d, num_classes = %d, output_n_stride = %d, \n", num_images,
    // num_classes, output_n_stride);

    cudaMemsetAsync(active_counts_per_class, 0, sizeof(int), stream);

    // printf("enter cublas\n");

    constexpr int BLOCK_DIM_X = 32;
    constexpr int BLOCK_DIM_Y = 8;
    constexpr int TILE_X = BLOCK_DIM_X;
    constexpr int TILE_Y = 32;
    constexpr int N_TILES = 1;

#if CUSTOM_TRANSPOSE == 1

    int output_concat_offset = 0;
    for (int layer = 0; layer < num_layers; layer++)
    {
        int layer_hw_size = feature_size[layer] * feature_size[layer];
        int prev_layer_prior_size
            = (layer == 0) ? 0 : num_anchors[layer - 1] * feature_size[layer - 1] * feature_size[layer - 1];
        output_concat_offset += prev_layer_prior_size * num_classes;

        int m = num_anchors[layer] * num_classes; // number of conf channels
        int n = layer_hw_size;

        dim3 block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
        dim3 grid = dim3(div_up(n, TILE_X), div_up(m, TILE_Y * N_TILES), num_images);
        batched_transpose_pf_kernel<TILE_X, TILE_Y, BLOCK_DIM_X, BLOCK_DIM_Y, N_TILES>
            <<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(conf_data[layer]),
                reinterpret_cast<float*>(tmp_data) + output_concat_offset, n, m, output_n_stride);

        if (layer == num_layers - 1)
        {
            assert(output_concat_offset + num_anchors[layer] * layer_hw_size * num_classes == num_priors * num_classes);
        }
    }
#endif

#if CUSTOM_TRANSPOSE == 0
    for (int ib = 0; ib < num_images; ib++)
    {
        int output_n_offset = ib * output_n_stride;
        int output_concat_offset = output_n_offset;
        for (int layer = 0; layer < num_layers; layer++)
        {

            int layer_hw_size = feature_size[layer] * feature_size[layer];
            int prev_layer_prior_size
                = (layer == 0) ? 0 : num_anchors[layer - 1] * feature_size[layer - 1] * feature_size[layer - 1];
            output_concat_offset += prev_layer_prior_size * num_classes;
            // printf("permuteConfData_cublas: layer_hw_size = %d, num_anchors = %d, prev_layer_prior_size = %d,
            // output_concat_offset = %d\n",layer_hw_size,  num_anchors[layer], prev_layer_prior_size,
            // output_concat_offset);
            // cublas implies column major
            // number of rows of transposed input(in column major), i.e. output
            int m = num_anchors[layer] * num_classes; // number of conf channels
            // number of columns(in column major)
            int n = layer_hw_size;
            int lda = layer_hw_size;
            int ldb = m;
            int ldc = m;
            cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
                reinterpret_cast<const float*>(conf_data[layer]) + ib * m * n, lda, &beta, b_matrix, ldb,
                reinterpret_cast<float*>(tmp_data) + output_concat_offset, ldc);

            if (layer == num_layers - 1)
            {
                assert(output_concat_offset + num_anchors[layer] * layer_hw_size * num_classes - output_n_offset
                    == num_priors * num_classes);
            }
        }

#if (ENABLE_FUSED_TRANSPOSE == 0)
        // transpose
        int m = num_priors;
        int n = num_classes;
        int lda = n;
        int ldb = m;
        int ldc = m;
        cublasSgeam(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha,
            reinterpret_cast<float*>(tmp_data) + output_n_offset, lda, &beta, b_matrix, ldb,
            reinterpret_cast<float*>(new_data) + output_n_offset, ldc);
#endif
    }
#endif

#if (ENABLE_FUSED_TRANSPOSE == 1)

#if 0
    dim3 block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid = dim3(div_up(num_classes, TILE_X), div_up(num_priors,TILE_Y), num_images);
    batched_transpose_kernel<TILE_X, TILE_Y, BLOCK_DIM_X, BLOCK_DIM_Y><<<grid, block, 0, stream>>>
            (reinterpret_cast<float*>(tmp_data),
             reinterpret_cast<float*>(new_data),
             num_classes,
             num_priors);
#else
#if ENABLE_FUSED_TRANSPOSE == 1
    dim3 block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid = dim3(div_up(num_classes, TILE_X), div_up(num_priors, TILE_Y), num_images);
    cudaMemsetAsync(active_counts_per_class, 0, num_images * num_classes * sizeof(int), stream);
    batched_transpose_fuse_sigmoid_topk_prepare_kernel<TILE_X, TILE_Y, BLOCK_DIM_X, BLOCK_DIM_Y>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(tmp_data), reinterpret_cast<float*>(new_data),
            num_classes, num_priors, reinterpret_cast<int*>(active_counts_per_class),
            reinterpret_cast<int*>(out_indices), background_class_id, threshold);
#endif

#endif
#endif

#if SSD_CUBLAS_CONF_PERMUTE == 0
    int block_size_sigmoid = 256;
    int grid = div_up(nthreads, block_size_sigmoid);
    sigmoid_kernel<<<div_up(nthreads, block_size_sigmoid), block_size_sigmoid, 0, stream>>>(
        reinterpret_cast<float*>(new_data), nthreads);
#endif
}

//#ifdef SSD_STABLE_TOPK
struct BlockPrefixCallbackOp
{
    // Running prefix
    int running_total;
    // Constructor
    __device__ BlockPrefixCallbackOp(int running_total)
        : running_total(running_total)
    {
    }
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int block_aggregate)
    {
        int old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};
//#endif

#if USE_CUB_SEGMENTED_SORT == 1

// TODO: implement reduction of to get real actrive counts
template <typename T_SCORE, int BLOCK_THREADS>
__global__ void cub_top_k_prepare(T_SCORE* in_scores, T_SCORE* out_scores, int* out_indices, int* begin_offsets,
    int* end_offsets, int* active_count_per_batch, int items, int segments, int background_class_id, float threshold)
{

    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    if (threadIdx.x == 0)
    {
        // We have to initialize active_count_per_batch for the following allClassNMS kernel.
        // Do it here to avoid to avoid an extra memset launch.
        if (blockIdx.x == 0)
        {
            active_count_per_batch[blockIdx.y] = 0;
        }
    }

    const int class_id = blockIdx.x;
    const int segment = blockIdx.y * gridDim.x + blockIdx.x;
    const int stride = items / segments;
    const int begin_offset = segment * stride;

    in_scores += begin_offset;
    out_scores += begin_offset;
    out_indices += begin_offset;

    begin_offsets[segment] = segment * stride;
    // end_offsets[segment] = (segment + 1) * stride;

    if (class_id == background_class_id)
    {
        end_offsets[segment] = segment * stride;
        return;
    }

    float is_not_background_class_id = (class_id != background_class_id) ? 1.0F : 0.0F;
    int end = div_up(stride, BLOCK_THREADS) * BLOCK_THREADS;
    for (int idx = threadIdx.x; idx < end; idx += BLOCK_THREADS)
    {
#if SSD_CUBLAS_CONF_PERMUTE == 1
        T_SCORE score = (idx < stride) ? __expf(in_scores[idx]) / (1.0f + __expf(in_scores[idx])) : 0.0F;
#else
        T_SCORE score = (idx < stride) ? in_scores[idx] : 0.0F;
#endif
        score = (score < threshold) ? 0.0F : is_not_background_class_id * score;
        // debug
        // if (class_id == -1 && score > 0.0f) {
        //     printf("%d %f\n", idx, score);
        // }
        int selected = score > 0.0F;
        int offset;
        BlockScan(temp_storage).ExclusiveSum(selected, offset, prefix_op);

        if (selected)
        {
            out_scores[offset] = score;
            out_indices[offset] = idx + stride * blockIdx.x;
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        end_offsets[segment] = prefix_op.running_total + segment * stride;
    }

    // if (threadIdx.x == 0) {
    //     //printf("%d, %d, %d\n", segment, begin_offsets[segment], end_offsets[segment]);
    //     printf("%d, %d\n", segment, prefix_op.running_total);
    // }

    // if (threadIdx.x == 0)
    //     printf("%d, %d\n", segment, prefix_op.running_total);

    // assert(prefix_op.running_total <= stride);
}

template <int BLOCK_THREADS>
__global__ void get_active_counts(int* in_end_offsets, int* out_active_counts, int items, int segments, int num_top_k)
{

    int segment_id = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    const int stride = items / segments;

    if (segment_id >= segments)
        return;

    out_active_counts[segment_id] = min(max(0, in_end_offsets[segment_id] - segment_id * stride), num_top_k);

    DEBUG_PRINTF("%d, %d\n", segment_id, min(max(0, in_end_offsets[segment_id] - segment_id * stride), num_top_k));
}

#endif

constexpr int TOPK_PER_CLASS_BLOCK_THREADS = 512;

namespace
{
// sort one segment per cta
template <typename T_SCORE, int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void blockSortKernel(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in,
    int32_t* d_values_out, const int32_t* active_counts, int num_items_, int stride_items, int num_segments)
{
    // Specialize BlockRadixSort for a 1D block
    typedef cub::BlockRadixSort<T_SCORE, BLOCK_THREADS, ELEMENTS_PER_THREAD, int32_t> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    if (blockIdx.x >= num_segments)
        return;

    // if (threadIdx.x == 0)
    //     DEBUG_PRINTF("active counts[%d] = %d\n", blockIdx.x, active_counts[blockIdx.x]);

    int num_items = active_counts[blockIdx.x] > num_items_ ? num_items_ : active_counts[blockIdx.x];

    if (num_items == 0)
    {
        return;
    }

    // Obtain a segment of consecutive items that are blocked across threads
    T_SCORE thread_keys[ELEMENTS_PER_THREAD];
    int32_t thread_values[ELEMENTS_PER_THREAD];

    int32_t block_offset = blockIdx.x * stride_items;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items, 0);
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items, -1);
    __syncthreads();

    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items);
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items);
}

/// block sort kernel
template <typename T_SCORE>
void blockSort(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in, int32_t* d_values_out,
    const int32_t* active_counts, int num_items, int stride_items, int num_segments, cudaStream_t stream)
{
    if (num_items == 0)
        return;

    int kernel_index = div_up(num_items, 128) - 1;
    int warps_per_cta = (kernel_index + 1) * 128 / 32;
    assert(warps_per_cta <= 32);

    dim3 block(warps_per_cta * 32);
    dim3 grid(num_segments);

    using kernel_func = void (*)(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in,
        int32_t* d_values_out, const int32_t* active_counts, int num_items, int stride_items, int num_segments);

    static const kernel_func kernel_funcs[] = {
        &blockSortKernel<T_SCORE, 128, 1>,
        &blockSortKernel<T_SCORE, 256, 1>,
        &blockSortKernel<T_SCORE, 384, 1>,
        &blockSortKernel<T_SCORE, 512, 1>,
        &blockSortKernel<T_SCORE, 640, 1>,
        &blockSortKernel<T_SCORE, 768, 1>,
        &blockSortKernel<T_SCORE, 896, 1>,
        &blockSortKernel<T_SCORE, 1024, 1>,
    };
    kernel_funcs[kernel_index]<<<grid, block, 0, stream>>>(
        d_keys_in, d_keys_out, d_values_in, d_values_out, active_counts, num_items, stride_items, num_segments);
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS = 512>
__global__ void top_k_cuda_fused_prepare(int* in, int* out, int* out_indices, int* active_counts_per_class,
    int* active_count_per_batch, int items, unsigned int num_top_k, int segments, int num_sub_segments,
    int background_class_id, float threshold)
{

    extern __shared__ int2 dynamic_smem[];
    int2* selected_elements = dynamic_smem;
#ifdef SSD_STABLE_TOPK
    int active_count;
    __shared__ unsigned int selected_count;
#else
    __shared__ unsigned int selected_count;
    // stores the number of elements which are above the threshold
    __shared__ unsigned int active_count;
#endif

#ifdef SSD_STABLE_TOPK
    // Specialize BlockScan type for our thread block
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);
#endif

    // int num_sub_segments = gridDim.z;
    // int sub_segment_id = blockIdx.z;

    unsigned int old_selected_count;

    // this is a workaround
    int class_id = blockIdx.x / num_sub_segments;
    int segment = blockIdx.y * gridDim.x + blockIdx.x;
    int stride = items * num_sub_segments;
    int out_offset = (blockIdx.y * gridDim.x + blockIdx.x) * (stride / num_sub_segments)
        + blockIdx.x % num_sub_segments * num_top_k;

    if (threadIdx.x == 0)
    {
        // We have to initialize active_count_per_batch for the following allClassNMS kernel.
        // Do it here to avoid to avoid an extra memset launch.
        if (blockIdx.x == 0)
        {
            active_count_per_batch[blockIdx.y] = 0;
        }
        active_count = 0;
    }
    __syncthreads();

    int first_index = segment * items;
    in += first_index;
    out += out_offset;
    out_indices += out_offset;

    int index_limit = items;
    uint32_t thread_items[ITEMS_PER_THREAD];
    int local_filtered = 0;

    // number of items whose score is >0 int he current thread
    int thread_active = 0;
    // in case <= top_k are active, offset where to write the thread items to in the output
    int thread_offset = 0;

    if (background_class_id != class_id)
    {
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            int offset = threadIdx.x + i * blockDim.x;
            thread_items[i] = 0;
            if (offset < index_limit)
            {
                thread_items[i] = in[offset];
            }
        }

        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            if (__int_as_float(thread_items[i]) < threshold)
            {
                thread_items[i] = 0;

                // todo a bitmask + popc might be faster here
                int offset = threadIdx.x + i * blockDim.x;
                if (offset < index_limit)
                {
                    ++local_filtered;
                }
            }
            if (thread_items[i] > 0)
            {
                thread_active++;
            }
        }
#ifdef SSD_STABLE_TOPK
        BlockScan(temp_storage).ExclusiveSum(thread_active, thread_offset, active_count);
#else
        thread_offset = atomicAdd(&active_count, thread_active);
#endif
    }

    uint32_t select_mask = 0;
    uint32_t save_mask = 0;
    uint32_t save_bit = 0;

    if (threadIdx.x == 0)
    {
        selected_count = 0;
        old_selected_count = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        active_counts_per_class[segment] = active_count;
    }

    // !!! experiment
    // all elements are filtered, nothing to do
    //   if (active_count == 0) {
    //       return;
    //   }

    // we have at maximum top_k elements. there's no need to filter those, store them directly as result.
    if (active_count <= num_top_k)
    {
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            if (thread_items[i] != 0)
            {
                out_indices[thread_offset] = threadIdx.x + i * blockDim.x + items * blockIdx.x;
                out[thread_offset] = thread_items[i];
                ++thread_offset;
            }
        }
        return;
    }

    // iterate over bits.
    // skip the first two bits,
    // * bit 31 is the sign bit. all values are positive
    // * bit 30 is only set for values >= 2, but the input consists only of values in the range of [0,1]
    const int skip_bits = 2;
    int selected;
    for (int bit = 31 - skip_bits; true; --bit)
    {
        __syncthreads();
        uint32_t bit_mask = select_mask | (1u << bit);

        uint32_t enabled = 0;
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
            enabled |= (((thread_items[item] ^ bit_mask) & bit_mask) == 0) << item;
        }

        selected = __popc(enabled);
#ifdef SSD_STABLE_TOPK
        int offset;
        BlockScan(temp_storage).ExclusiveSum(selected, offset, prefix_op);
        if (threadIdx.x == 0)
        {
            selected_count = prefix_op.running_total;
        }
#else
        unsigned int offset = atomicAdd(&selected_count, selected);
#endif

        __syncthreads();
        int sc = selected_count;
        __syncthreads();

        if ((sc <= num_top_k && sc > 0) || (bit == 0 && sc > 0))
        {
            for (int item = 0; item < ITEMS_PER_THREAD; ++item)
            {
                if (enabled & (1u << item) && offset < num_top_k)
                {
                    selected_elements[offset]
                        = make_int2(thread_items[item], threadIdx.x + item * blockDim.x + items * blockIdx.x);
                    ++offset;
                    thread_items[item] = 0;
                }
            }
        }

        if (sc == num_top_k || bit == 0)
        {
            break;
        }
        else if (sc > num_top_k)
        {
            // There are too many bits in the current selection
            // Save the current state and go to the next bit
            // If there are not enough items left using the next bit
            // it's necessary to restart here with the current bit not set
            save_mask = bit_mask;
            save_bit = bit - 1;
            select_mask |= bit_mask;

            if (threadIdx.x == 0)
            {
                selected_count = old_selected_count;
#ifdef SSD_STABLE_TOPK
                prefix_op.running_total = old_selected_count;
#endif
            }
        }
        else
        {
            if (save_mask)
            {
                select_mask = save_mask;
                bit = save_bit;

                save_mask = 0;
            }
            if (threadIdx.x == 0)
            {
                old_selected_count = sc;
            }
        }
    }

    __syncthreads();

    // store data to global memory
    int sc = selected_count;
    for (int i = threadIdx.x; i < num_top_k; i += BLOCK_THREADS)
    {
        int2 selected_element = selected_elements[i];
        int out_element = i < sc ? selected_element.x : 0;
        out[i] = out_element;
        out_indices[i] = out_element > 0 ? selected_element.y : -1;
    }

    if (threadIdx.x == 0)
    {
        active_counts_per_class[segment] = num_top_k;
    }
}

} // namespace

template <typename T_SCORE>
ssdStatus_t topKScoresPerClass_gpu(cudaStream_t stream, const int num, const int num_classes,
    const int num_preds_per_class, const int num_top_k, const int background_label_id, const float confidence_threshold,
    void* conf_scores_gpu, void* index_array_gpu, void* active_counts_gpu, void* active_counts_per_batch_gpu,
    void* temp_scores, void* temp_indices, void* temp_active_counts, size_t temp_storage_bytes, void* workspace,
    const int num_priors, const int num_dim, bool confSigmoid, const void* const* conf_data, const int num_layers,
    const int* feature_size, const int* num_anchors, const int* box_channels, const bool packed32NCHW)
{

#if SSD_RETINA_NET == 1

#if USE_CUB_SEGMENTED_SORT == 1

    int segments = num * num_classes;
    int items = num * num_classes * num_preds_per_class;

#if SSD_CUBLAS_CONF_PERMUTE == 1

    // scores data flow:
    // conf_data->conf_scores_gpu(first transpose/concat)->temp_scores(output scores)
    permuteConfDataFuseCubTopKPrepare(stream, num * num_classes * num_preds_per_class, num_classes, num_preds_per_class,
        num_layers, confSigmoid,
#if ENABLE_FUSED_TRANSPOSE == 1
        temp_scores, conf_scores_gpu,
#else
        conf_scores_gpu, temp_scores,
#endif
        // temp_scores,
        conf_data, active_counts_gpu, feature_size, num_anchors, packed32NCHW, temp_indices, background_label_id,
        confidence_threshold);
    // TODO move inside the fuse kernel
    cudaMemsetAsync(active_counts_per_batch_gpu, 0, num * sizeof(int), stream);
#endif

#if ENABLE_FUSED_TRANSPOSE == 1 && SSD_CUBLAS_CONF_PERMUTE == 1
    {
        constexpr int BLOCK_SIZE = 256;
        int grid = div_up(segments, BLOCK_SIZE);

        get_cub_offsets_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(reinterpret_cast<int*>(active_counts_gpu),
            reinterpret_cast<int*>(active_counts_gpu), reinterpret_cast<int*>(temp_active_counts), items, segments,
            num_top_k);
    }

#else

    DEBUG_PRINTF("cub_top_k_prepare: items = %d, segments = %d, stride = %d\n", items, segments, items / segments);
    cub_top_k_prepare<T_SCORE, 1024><<<dim3(num_classes, num), 1024, 0, stream>>>(
        reinterpret_cast<T_SCORE*>(conf_scores_gpu), reinterpret_cast<T_SCORE*>(temp_scores),
        reinterpret_cast<int*>(temp_indices), reinterpret_cast<int*>(active_counts_gpu),
        reinterpret_cast<int*>(temp_active_counts), reinterpret_cast<int*>(active_counts_per_batch_gpu), items,
        segments, background_label_id, confidence_threshold);

#endif

    cub::DeviceSegmentedRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
        reinterpret_cast<T_SCORE*>(temp_scores), reinterpret_cast<T_SCORE*>(conf_scores_gpu),
        reinterpret_cast<int*>(temp_indices), reinterpret_cast<int*>(index_array_gpu), items, segments,
        (int*) active_counts_gpu, (int*) temp_active_counts, 0, sizeof(T_SCORE) * 8, stream);

    get_active_counts<128><<<div_up(segments, 128), 128, 0, stream>>>(reinterpret_cast<int*>(temp_active_counts),
        reinterpret_cast<int*>(active_counts_gpu), items, segments, num_top_k);

#endif // USE_CUB_SEGMENTED_SORT

#else
    const int BLOCK_THREADS = TOPK_PER_CLASS_BLOCK_THREADS;

    using top_k_kernel = void (*)(int* in, int* out, int* out_indices, int* active_counts_gpu,
        int* active_counts_per_batch_gpu, int items, unsigned int num_top_k, int segments, int num_sub_segments,
        int background_class_id, float threshold);
    top_k_kernel top_k_kernels[] = {
        top_k_cuda_fused_prepare<1, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<2, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<3, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<4, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<5, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<6, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<7, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<8, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<9, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<10, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<11, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<12, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<13, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<14, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<15, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<16, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<17, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<18, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<19, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<20, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<21, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<22, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<23, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<24, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<25, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<26, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<27, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<28, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<29, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<30, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<31, BLOCK_THREADS>,
        top_k_cuda_fused_prepare<32, BLOCK_THREADS>,
    };

    const int num_segments = num * num_classes;

    uint32_t smem_size = num_top_k * (sizeof(int) + sizeof(uint32_t));

    // TODO implement multi-stage topk
    int kernel_index = (num_preds_per_class + BLOCK_THREADS - 1) / BLOCK_THREADS;
    dim3 block(BLOCK_THREADS);

    bool do_n_pass = (kernel_index >= 32);

    DEBUG_PRINTF("num_preds_per_class = %d\n", num_preds_per_class);

    // do_n_pass = false;  // debug resnet34

    void* out_scores = (do_n_pass) ? temp_scores : conf_scores_gpu;
    void* out_indices = (do_n_pass) ? temp_indices : index_array_gpu;
    void* out_active_counts = (do_n_pass) ? temp_active_counts : active_counts_gpu;

    // number of segments each class is split into
    int num_sub_segments = 1; //&& num_preds_per_class % num_sub_segments
    while (kernel_index >= 32)
    {
        // introduce additional step
        num_sub_segments += 1;
        int items_per_sub_segment = div_up(num_preds_per_class, num_sub_segments);
        kernel_index = div_up(items_per_sub_segment, TOPK_PER_CLASS_BLOCK_THREADS);
    }

    //!!! hard_code for now:
    if (do_n_pass)
    {
        // num_sub_segments = 2;  // debug resnet34
        num_sub_segments = 9;
        int items_per_sub_segment = num_preds_per_class / num_sub_segments;
        kernel_index = div_up(items_per_sub_segment, TOPK_PER_CLASS_BLOCK_THREADS);
        DEBUG_PRINTF("Using kernel #%d\n", kernel_index);
    }

    assert(num_preds_per_class % num_sub_segments == 0);

    dim3 grid(num_classes * num_sub_segments, num);

    assert(kernel_index < 32);

    DEBUG_PRINTF("top_k Per Class\n");

    if (do_n_pass)
    {
        cudaMemcpyAsync(
            out_scores, conf_scores_gpu, num_preds_per_class * num_segments, cudaMemcpyDeviceToDevice, stream);
    }

    top_k_kernels[kernel_index]<<<grid, block, smem_size, stream>>>((int*) (conf_scores_gpu), (int*) (out_scores),
        (int*) out_indices, (int*) out_active_counts, (int*) active_counts_per_batch_gpu,
        num_preds_per_class / num_sub_segments, num_top_k, num_classes * num_sub_segments, num_sub_segments,
        background_label_id, confidence_threshold);

    if (do_n_pass)
    {
        top_k_multi_pass((int*) (out_scores), (int*) out_indices, (int*) (conf_scores_gpu), (int*) index_array_gpu,
            (int*) out_active_counts, (int*) active_counts_per_batch_gpu, num_top_k * num_sub_segments,
            num_preds_per_class, num_top_k, num_sub_segments, num_segments, stream);
    }

    blockSort<T_SCORE>((const T_SCORE*) (out_scores), (T_SCORE*) (conf_scores_gpu), (const int*) (out_indices),
        (int*) (index_array_gpu), (int*) out_active_counts, num_top_k, num_preds_per_class, num_segments, stream);

#endif // #if SSD_RETINA_NET

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerClass LAUNCH CONFIG {{{
typedef ssdStatus_t (*tkspcFunc)(cudaStream_t, const int, const int, const int, const int, const int, const float,
    void*, void*, void*, void*, void*, void*, void*, size_t, void*, const int, const int, bool, const void* const*,
    const int, const int*, const int*, const int*, const bool);
struct tkspcLaunchConfig
{
    DType_t t_score;
    tkspcFunc function;

    tkspcLaunchConfig(DType_t t_score)
        : t_score(t_score)
    {
    }
    tkspcLaunchConfig(DType_t t_score, tkspcFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const tkspcLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

static std::vector<tkspcLaunchConfig> tkspcFuncVec;
bool tkspcInit()
{
    tkspcFuncVec.push_back(tkspcLaunchConfig(DataType::kFLOAT, topKScoresPerClass_gpu<float>));
    return true;
}

static bool initialized = tkspcInit();
//}}}

ssdStatus_t topKScoresPerClass(cudaStream_t stream, const int num, const int num_classes, const int num_preds_per_class,
    const int num_top_k, const int background_label_id, const float confidence_threshold, const DType_t DT_SCORE,
    void* conf_scores_gpu, void* index_array_gpu, void* active_count_per_class, void* active_count_per_batch,
    void* temp_scores, void* temp_indices, void* temp_active_counts, size_t temp_storage_bytes, void* workspace,
    const int num_priors, const int num_dim, bool confSigmoid, const void* const* conf_data, const int num_layers,
    const int* feature_size, const int* num_anchors, const int* box_channels, const bool packed32NCHW)
{
    tkspcLaunchConfig lc = tkspcLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < tkspcFuncVec.size(); ++i)
    {
        if (lc == tkspcFuncVec[i])
        {
            DEBUG_PRINTF("sortScoresPerClass kernel %d\n", i);
            return tkspcFuncVec[i].function(stream, num, num_classes, num_preds_per_class, num_top_k,
                background_label_id, confidence_threshold, conf_scores_gpu, index_array_gpu, active_count_per_class,
                active_count_per_batch, temp_scores, temp_indices, temp_active_counts, temp_storage_bytes, workspace,
                num_priors, num_dim, confSigmoid, conf_data, num_layers, feature_size, num_anchors, box_channels,
                packed32NCHW);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t topKScoresPerClassWorkspaceSize(const int num, const int num_classes, const int num_preds_per_class,
    const int num_top_k, const DType_t DT_CONF, size_t& temp_active_counts_size, size_t& temp_sort_scores_size,
    size_t& temp_sort_indicies_size)
{
    int data_type_size = (DT_CONF == DataType::kFLOAT)
        ? sizeof(float)
        : (DT_CONF == DataType::kHALF) ? sizeof(uint16_t) : sizeof(uint8_t);

    int kernel_index = div_up(num_preds_per_class, TOPK_PER_CLASS_BLOCK_THREADS);
    temp_active_counts_size = temp_sort_scores_size = temp_sort_indicies_size = 0;

    // comment for debug
    // if (kernel_index < 32) return 0;

    // number of segments each class is split into
    int num_sub_segments = 1;

    size_t temp_cub_storage_bytes = 0;

#if USE_CUB_SEGMENTED_SORT == 1

    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_cub_storage_bytes, (float*) nullptr,
        (float*) nullptr, (int*) nullptr, (int*) nullptr, num * num_classes * num_preds_per_class, num * num_classes,
        (int*) nullptr, (int*) nullptr, 0, sizeof(int) * 8, 0);
#else
    while (kernel_index >= 32)
    {
        // introduce additional step
        num_sub_segments += 1;
        int items_per_sub_segment = div_up(num_preds_per_class, num_sub_segments);
        kernel_index = div_up(items_per_sub_segment, TOPK_PER_CLASS_BLOCK_THREADS);
    }
    //!!! hard_code for now:
    num_sub_segments = 10; // debug resnet34
                           // num_sub_segments = 9;
#endif

    temp_active_counts_size = num * num_classes * num_sub_segments * sizeof(int);
    size_t num0 = (size_t) num * (size_t) num_classes * (size_t) num_sub_segments * (size_t) num_top_k;
    size_t num1 = (size_t) num * (size_t) num_classes * (size_t) num_preds_per_class;
    temp_sort_scores_size = std::max(num0, num1) * data_type_size;
    temp_sort_indicies_size = std::max(num0, num1) * sizeof(int);
    DEBUG_PRINTF("Temp storage calculation: num_segments required = %d\n", num_sub_segments);
    DEBUG_PRINTF("Active counts temp arrays size = %d\n", temp_active_counts_size);
    DEBUG_PRINTF("Sorting temp arrays size = %d\n", temp_sort_scores_size);

    return temp_active_counts_size + temp_sort_scores_size + temp_sort_indicies_size + temp_cub_storage_bytes;
}

} // namespace plugin
} // namespace nvinfer1
