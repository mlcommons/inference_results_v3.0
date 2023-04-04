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

#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include "topK.h"

#include "nms_common.h"

//#undef USE_CUB_SEGMENTED_SORT

template <typename KeyT, typename ValueT>
size_t cubSortPairsWorkspaceSize(int num_items, int num_segments)
{
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending((void*) NULL, temp_storage_bytes, (const KeyT*) NULL,
        (KeyT*) NULL, (const ValueT*) NULL, (ValueT*) NULL,
        num_items,    // # items
        num_segments, // # segments
        (const int*) NULL, (const int*) NULL);
    return temp_storage_bytes;
}

namespace nvinfer1
{
namespace plugin
{

namespace
{

#if USE_CUB_SEGMENTED_SORT == 1

__global__ void get_cub_offsets_kernel(int* begin_offset, int* end_offset, int num, int stride)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num)
        return;
    int num_items = max(begin_offset[idx], 0);
    end_offset[idx] = idx * stride + num_items;
    begin_offset[idx] = idx * stride;
}

template <typename T_SCORE>
__global__ void top_k_score_per_image_prepare_outputs(T_SCORE* input_scores, int* input_indices, T_SCORE* output_scores,
    int* output_indices, int* in_end_offsets, int* out_active_counts, int items, int segments, int num_top_k)
{
    int segment_id = blockIdx.y;
    const int stride = items / segments;
    int elem_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (segment_id >= segments)
        return;
    int num_items = min(max(0, in_end_offsets[segment_id] - segment_id * stride), num_top_k);
    out_active_counts[segment_id] = num_items;

    if (elem_id >= num_top_k)
        return;

    int idx = elem_id + stride * segment_id;

    // make things consistent with the reference for now (pad up to top_k with 0/-1)
    output_scores[idx] = (elem_id < num_items) ? input_scores[idx] : 0.0F;
    output_indices[idx] = (elem_id < num_items) ? input_indices[idx] : -1;

    // output_scores[idx] = input_scores[idx];
    // output_indices[idx] = input_indices[idx];
}

template <int BLOCK_THREADS>
__global__ void get_active_counts(int* in_end_offsets, int* out_active_counts, int items, int segments, int num_top_k)
{

    int segment_id = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    const int stride = items / segments;

    if (segment_id >= segments)
        return;

    out_active_counts[segment_id] = min(max(0, in_end_offsets[segment_id] - segment_id * stride), num_top_k);

    // printf("%d, %d\n", segment_id, min(max(0, in_end_offsets[segment_id] - segment_id * stride), num_top_k));
}

#endif

// sort one segment per cta
template <typename T_SCORE, int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void blockSortKernel(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in,
    int32_t* d_values_out, const int32_t* active_counts, int num_items, int stride_items, int num_segments)
{
    // Specialize BlockRadixSort for a 1D block
    typedef cub::BlockRadixSort<T_SCORE, BLOCK_THREADS, ELEMENTS_PER_THREAD, int32_t> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    if (blockIdx.x >= num_segments)
        return;

    int num_active_items = active_counts[blockIdx.x];

    // Obtain a segment of consecutive items that are blocked across threads
    T_SCORE thread_keys[ELEMENTS_PER_THREAD];
    int32_t thread_values[ELEMENTS_PER_THREAD];

    int32_t block_offset = blockIdx.x * stride_items;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_active_items, 0);
    cub::LoadDirectStriped<BLOCK_THREADS>(
        threadIdx.x, d_values_out + block_offset, thread_values, num_active_items, -1);
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

} // namespace

template <typename T_SCORE>
ssdStatus_t topKScoresPerImage_gpu(cudaStream_t stream, const int num_images, const int num_items_per_image,
    const int num_top_k, void* unsorted_scores, void* unsorted_bbox_indices, void* sorted_scores,
    void* sorted_bbox_indices, void* active_count, void* active_count_per_batch, void* temp_active_count,
    size_t temp_storage_bytes, void* workspace)
{
    void* d_offsets = workspace;
    void* cubWorkspace = nextWorkspacePtr((int8_t*) d_offsets, (num_images + 1) * sizeof(int));

    const int num_classes = num_items_per_image / num_top_k;

#if USE_CUB_SEGMENTED_SORT == 1

    segmented_scan(
        (int*) active_count, (int*) active_count, (int*) active_count_per_batch, num_images, num_classes, stream);

    compact_segments(unsorted_scores, unsorted_bbox_indices, sorted_scores, sorted_bbox_indices, (int*) active_count,
        (int*) active_count_per_batch, num_classes, num_images, num_items_per_image, num_items_per_image, stream);

    // get offsets
    get_cub_offsets_kernel<<<div_up(num_images, 128), 128, 0, stream>>>(
        (int*) active_count_per_batch, (int*) temp_active_count, num_images, num_items_per_image);

    int items = num_images * num_items_per_image;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
        reinterpret_cast<T_SCORE*>(sorted_scores), // keys_in
        reinterpret_cast<T_SCORE*>(unsorted_scores),
        reinterpret_cast<int*>(sorted_bbox_indices), // indicies_in
        reinterpret_cast<int*>(unsorted_bbox_indices), items, num_images,
        //(int *)active_count,
        (int*) active_count_per_batch, (int*) temp_active_count, 0, sizeof(T_SCORE) * 8, stream);

    if (0)
    {
        // copy to output. write a smarter kernel based on the active counts
        cudaMemcpyAsync(sorted_scores, unsorted_scores, items * sizeof(T_SCORE), cudaMemcpyDefault, stream);
        cudaMemcpyAsync(sorted_bbox_indices, unsorted_bbox_indices, items * sizeof(int), cudaMemcpyDefault, stream);

        get_active_counts<128><<<div_up(num_images, 128), 128, 0, stream>>>(reinterpret_cast<int*>(temp_active_count),
            reinterpret_cast<int*>(active_count_per_batch), items, num_images, num_top_k);
    }
    else
    {

        top_k_score_per_image_prepare_outputs<<<dim3(div_up(num_top_k, 128), num_images, 1), 128, 0, stream>>>(
            reinterpret_cast<T_SCORE*>(unsorted_scores), reinterpret_cast<int*>(unsorted_bbox_indices),
            reinterpret_cast<T_SCORE*>(sorted_scores), reinterpret_cast<int*>(sorted_bbox_indices),
            reinterpret_cast<int*>(temp_active_count), reinterpret_cast<int*>(active_count_per_batch), items,
            num_images, num_top_k);
    }

#else

    uint32_t num_warps = (num_items_per_image > 1024) ? 32 : (num_items_per_image + 31) / 32;

    // const int WARP_SZ = 32;
    const int BLOCK_THREADS = 512;

    // printf("top_k Per Image\n");

    top_k_multi_pass((int*) (unsorted_scores), (int*) unsorted_bbox_indices, (int*) (sorted_scores),
        (int*) sorted_bbox_indices, (int*) active_count, (int*) active_count_per_batch, num_items_per_image,
        num_items_per_image, num_top_k, num_classes, num_images, stream);

    void* block_sort_scores = unsorted_scores;
    void* block_sort_indices = unsorted_bbox_indices;

    // dim3 block(num_warps * WARP_SZ);
    // dim3 grid(num_images);
    // block.x = num_warps * 32;

    blockSort<T_SCORE>((const T_SCORE*) (block_sort_scores), (T_SCORE*) (sorted_scores),
        (const int*) (block_sort_indices), (int*) (sorted_bbox_indices), (int*) active_count_per_batch, num_top_k,
        num_items_per_image, num_images, stream);

#endif

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// sortScoresPerImage LAUNCH CONFIG {{{
typedef ssdStatus_t (*tkspiFunc)(
    cudaStream_t, const int, const int, const int, void*, void*, void*, void*, void*, void*, void*, size_t, void*);
struct tkspiLaunchConfig
{
    DType_t t_score;
    tkspiFunc function;

    tkspiLaunchConfig(DType_t t_score)
        : t_score(t_score)
    {
    }
    tkspiLaunchConfig(DType_t t_score, tkspiFunc function)
        : t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const tkspiLaunchConfig& other)
    {
        return t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<tkspiLaunchConfig> tkspiFuncVec;
bool tkspiInit()
{
    tkspiFuncVec.push_back(tkspiLaunchConfig(DataType::kFLOAT, topKScoresPerImage_gpu<float>));
    return true;
}

static bool initialized = tkspiInit();
//}}}

ssdStatus_t topKScoresPerImage(cudaStream_t stream, const int num_images, const int num_items_per_image,
    const int num_top_k, const DType_t DT_SCORE, void* unsorted_scores, void* unsorted_bbox_indices,
    void* sorted_scores, void* sorted_bbox_indices, void* active_count, void* active_count_per_gpu,
    void* temp_active_count, size_t temp_storage_bytes, void* workspace)
{
    tkspiLaunchConfig lc = tkspiLaunchConfig(DT_SCORE);
    for (unsigned i = 0; i < tkspiFuncVec.size(); ++i)
    {
        if (lc == tkspiFuncVec[i])
        {
            DEBUG_PRINTF("topKScoresPerImage kernel %d\n", i);
            return tkspiFuncVec[i].function(stream, num_images, num_items_per_image, num_top_k, unsorted_scores,
                unsorted_bbox_indices, sorted_scores, sorted_bbox_indices, active_count, active_count_per_gpu,
                temp_active_count, temp_storage_bytes, workspace);
        }
    }
    return STATUS_BAD_PARAM;
}

size_t topKScoresPerImageWorkspaceSize(
    const int num_images, const int num_items_per_image, const int num_top_k, const DType_t DT_SCORE)
{
    const int arrayLen = num_images * num_items_per_image;
    size_t wss[2];
    wss[0] = (num_images + 1) * sizeof(int); // offsets
    if (DT_SCORE == DataType::kFLOAT)
    {
        wss[1] = cubSortPairsWorkspaceSize<float, int>(arrayLen, num_images); // cub workspace
    }
    else
    {
        printf("SCORE type not supported.\n");
        return (size_t) -1;
    }

    return calculateTotalWorkspaceSize(wss, 2);
}

} // namespace plugin
} // namespace nvinfer1
