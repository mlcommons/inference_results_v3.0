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

#include <stdint.h>

#include <cub/cub.cuh>

#include "nms_common.h"
#include "ssdOptMacros.h"
#include "topK.h"

constexpr int TOPK_BLOCK_THREADS = 512;

namespace
{

/*************************************************************************************************/

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

/*************************************************************************************************/

const int SEGMENTED_SCAN_BLOCK_THREADS = 128;

__global__ void segmented_scan_kernel(int* in, int* out, int* aggregate, int* max, int segments)
{
    // Specialize BlockScan type for our thread block
    // can be in-place
    typedef cub::BlockScan<int, SEGMENTED_SCAN_BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    const int offset = blockIdx.x * segments;
    in += offset;
    out += offset;

    int finish = div_up(segments, SEGMENTED_SCAN_BLOCK_THREADS) * SEGMENTED_SCAN_BLOCK_THREADS;
    for (int idx = threadIdx.x; idx < finish; idx += SEGMENTED_SCAN_BLOCK_THREADS)
    {
        int thread_count = (idx < segments) ? in[idx] : 0;

        int thread_offset;
        BlockScan(temp_storage).ExclusiveSum(thread_count, thread_offset, prefix_op);

        __syncthreads();

        if (idx < segments)
            out[idx] = thread_offset;
    }

    if (threadIdx.x == 0)
    {
        aggregate[blockIdx.x] = prefix_op.running_total;
        // TODO: we can detect num_items based on the outcome of reduction
        /*     if (max != NULL) {
              atomicMax(max, prefix_op.running_total);
            } */
    }
}

/*************************************************************************************************/

__global__ void compact_segments_kernel(const int* in, const int* in_indices, int* out, int* out_indices,
    int* active_count_scan, int* active_count_per_batch, int items, int stride)
{
    // Each batch contain "segment" pieces
    // The kernel compacts segments into the beginning of each batch
    // items = max number of items per batch
    // items % segment == 0
    //  number of segments per batch
    const int num_segment = gridDim.x;
    const int batch_id = blockIdx.y;

    active_count_scan += batch_id * num_segment;

    const int segment_id = blockIdx.x;
    int segment_size = items / num_segment;

    int offset_in = batch_id * stride + segment_id * segment_size;
    int offset_out = batch_id * stride + active_count_scan[segment_id];

    int items_per_segment = (segment_id < num_segment - 1)
        ? active_count_scan[segment_id + 1] - active_count_scan[segment_id]
        : active_count_per_batch[batch_id] - active_count_scan[segment_id];

    for (int idx = threadIdx.x; idx < items_per_segment; idx += blockDim.x)
    {
        out[offset_out + idx] = in[offset_in + idx];
        out_indices[offset_out + idx] = in_indices[offset_in + idx];
    }
}

/*************************************************************************************************/

template <int ITEMS_PER_THREAD, int BLOCK_THREADS>
__global__ void top_k_cuda(int* in, int* in_indices, int* out, int* out_indices, int* active_count,
    int* active_count_per_batch, int items, int stride, unsigned int num_top_k)
{
    extern __shared__ uint32_t dynamic_memory[];
    uint32_t* selected_items = dynamic_memory;
    int32_t* selected_indices = reinterpret_cast<int32_t*>(selected_items + num_top_k);
    __shared__ unsigned int selected_count;
    unsigned int old_selected_count;

// Specialize BlockScan type for our thread block
#ifdef SSD_STABLE_TOPK
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);
#endif

    int batch = blockIdx.x;
    int first_index = batch * stride;

    // segments per batch
    int num_segments = gridDim.y;
    int items_per_segment = div_up(items, num_segments);
    int segment_items_offset = blockIdx.y * items_per_segment;
    items = active_count_per_batch[batch];
    if (items < segment_items_offset)
    {
        items = 0;
    }

    items = min(items_per_segment, items - segment_items_offset);
    // items = (items < 0)? -items : items;
    int second_index = first_index + blockIdx.y * num_top_k;
    first_index += segment_items_offset;

    in += first_index;
    in_indices += first_index;

    out += second_index;
    out_indices += second_index;

    // Feed input
    uint32_t thread_items[ITEMS_PER_THREAD];
    int32_t thread_indices[ITEMS_PER_THREAD];

    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        int offset = threadIdx.x + i * blockDim.x;
        if (offset < items)
        {
            thread_items[i] = in[offset];
            thread_indices[i] = in_indices[offset];
        }
        else
        {
            thread_items[i] = 0;
            thread_indices[i] = -1;
        }
    }

    if (items <= num_top_k)
    {
        if (threadIdx.x == 0)
        {
            if (gridDim.y == 1)
            {
                active_count_per_batch[batch] = items;
            }
            else
            {
                // this is preliminary step, so we need to populate active_count
                active_count[blockIdx.x * num_segments + blockIdx.y] = items;
            }
        }

        // we know that the results are compact, so we can bail out early.
        for (int i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            int offset = threadIdx.x + i * blockDim.x;
            if (offset < num_top_k)
            {
                out[offset] = thread_items[i];
                out_indices[offset] = thread_indices[i];
            }
            else
            {
                return;
            }
        }
    }

    uint32_t select_mask = 0;
    uint32_t save_mask = 0;
    uint32_t save_bit = 0;

    if (threadIdx.x == 0)
    {
        selected_count = 0;
        old_selected_count = 0;
    }

#define MTA_D 0

    // iterate over bits.
    // skip the first two bits,
    // * bit 31 is the sign bit. all values are positive
    // * bit 30 is only set for values >= 2, but the input consists only of values in the range of [0,1]
    const int skip_bits = 0;
    int selected = 0;
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
                    selected_items[offset] = thread_items[item];
                    selected_indices[offset] = thread_indices[item];
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
    for (int i = threadIdx.x; i < num_top_k; i += blockDim.x)
    {
        out[i] = (i < sc) ? selected_items[i] : 1;
        out_indices[i] = (i < sc && selected_items[0] > 0) ? selected_indices[i] : -1;
    }

    if (threadIdx.x == 0)
    {
        if (gridDim.y == 1)
        {
            active_count_per_batch[batch] = num_top_k;
        }
        else
        {
            active_count[batch * num_segments + blockIdx.y] = num_top_k;
        }
    }
}

} // end namespace

/*************************************************************************************************/

void segmented_scan(int* active_count_in, int* active_count_out, int* active_count_per_batch, int num_images,
    int num_classes, cudaStream_t stream)
{

    // recomputes active_count_per_batch
    segmented_scan_kernel<<<num_images, SEGMENTED_SCAN_BLOCK_THREADS, 0, stream>>>(
        active_count_in, active_count_out, active_count_per_batch, NULL, num_classes);
}

/*************************************************************************************************/

void compact_segments(void* input_output_scores, void* input_output_indices, void* temp_scores, void* temp_indices,
    int* active_count, int* active_count_per_batch, int num_classes, int num_images, int num_items_per_image,
    int image_stride, cudaStream_t stream)
{

    compact_segments_kernel<<<dim3(num_classes, num_images, 1), 128, 0, stream>>>((int*) input_output_scores,
        (int*) input_output_indices, (int*) temp_scores, (int*) temp_indices, (int*) active_count,
        (int*) active_count_per_batch, num_items_per_image, image_stride);
}

/*************************************************************************************************/

using top_k_kernel = void (*)(int* in, int* in_indices, int* out, int* out_indices, int* active_count,
    int* active_count_per_batch, int items, int stride_items, unsigned int num_top_k);

/*************************************************************************************************/

template <int BLOCK_THREADS>
void top_k(top_k_kernel* kernel, int kernel_index, void* input_output_scores, void* input_output_indices,
    void* temp_scores, void* temp_indices, int* active_count, int* active_count_per_batch, int num_items_per_image,
    int image_stride, int num_top_k, int num_classes, int num_images, int num_segments, cudaStream_t& stream)
{
    // num_images - barch size
    // num_classes - here number of segments per batch(image)
    // num_items_per_image - total items per single batch

    DEBUG_PRINTF("top_k: num_segments = %d, num_classes = %d\n", num_segments, num_classes);

    if (num_items_per_image % num_classes)
    {
        DEBUG_PRINTF("topK Warning: num_items_per_image % num_classes != 0\n");
    }

    assert(num_items_per_image % num_classes == 0);

    // recomputes active_count_per_batch
    segmented_scan(active_count, active_count, active_count_per_batch, num_images, num_classes, stream);

    compact_segments(input_output_scores, input_output_indices, temp_scores, temp_indices, active_count,
        active_count_per_batch, num_classes, num_images, num_items_per_image, image_stride, stream);

    uint32_t smem_size = num_top_k * (sizeof(int) + sizeof(uint32_t));

    kernel[kernel_index]<<<dim3(num_images, num_segments, 1), BLOCK_THREADS, smem_size, stream>>>((int*) (temp_scores),
        (int*) temp_indices, (int*) (input_output_scores), (int*) input_output_indices, (int*) active_count,
        (int*) active_count_per_batch, num_items_per_image, image_stride, num_top_k);
}

/*************************************************************************************************/

template <int BLOCK_THREADS>
void top_k_n_pass(top_k_kernel* top_k_kernels, void* input_output_scores, void* input_output_indices, void* temp_scores,
    void* temp_indices, int* active_count, int* active_count_per_batch, int num_items_per_image, int image_stride,
    int num_top_k, int num_classes, int num_images, cudaStream_t& stream)
{

    // num_images - barch size
    // num_classes - here initial number of segments per batch
    // num_items_per_image - total items per single batch

    // printf("top_k_n_pass: num_top_k = %d, num_classes = %d, num_images = %d, num_items_per_image = %d, image_stride =
    // %d\n", num_top_k, num_classes, num_images, num_items_per_image, image_stride);

    int kernel_index = div_up(num_items_per_image, BLOCK_THREADS);

    int num_segments = 1;
    while (kernel_index >= 32)
    {
        // introduce additional step
        num_segments += 1;
        int items_per_sub_segment = div_up(num_items_per_image, num_segments);
        kernel_index = (items_per_sub_segment + BLOCK_THREADS - 1) / BLOCK_THREADS;
    }
    // num_segments = (num_segments == 8)? 1 : num_segments;

    top_k<BLOCK_THREADS>(top_k_kernels, kernel_index, input_output_scores, input_output_indices, temp_scores,
        temp_indices, active_count, active_count_per_batch, num_items_per_image, image_stride, num_top_k, num_classes,
        num_images, num_segments, stream);
    if (num_segments > 1)
    {
        int num_items_per_image_stage_next = num_segments * num_top_k;
        kernel_index = div_up(num_items_per_image_stage_next, BLOCK_THREADS);
        if (kernel_index >= 32)
        {
            top_k_n_pass<BLOCK_THREADS>(top_k_kernels, input_output_scores, input_output_indices, temp_scores,
                temp_indices, active_count, active_count_per_batch, num_items_per_image_stage_next, image_stride,
                num_top_k, num_segments, num_images, stream);
        }
        else
        {
            top_k<BLOCK_THREADS>(top_k_kernels, kernel_index, input_output_scores, input_output_indices, temp_scores,
                temp_indices, active_count, active_count_per_batch, num_items_per_image_stage_next, image_stride,
                num_top_k, num_segments, num_images, 1, stream);
        }
    }
}

/*************************************************************************************************/

void top_k_multi_pass(void* input_output_scores, void* input_output_indices, void* temp_scores, void* temp_indices,
    int* active_count, int* active_count_per_batch, int num_items_per_image, int image_stride, int num_top_k,
    int num_classes, int num_images, cudaStream_t& stream)
{

    // num_images - barch size
    // num_classes - here initial number of segments per batch
    // num_items_per_image - total items per single batch
    // active_count - counts per segment
    // active_count_per_batch - counts per batch, aggregated over the segments
    // IMPORTANT! input_output_scores/input_output_indices will be overwritten by output

    constexpr int BLOCK_THREADS = TOPK_BLOCK_THREADS;

    top_k_kernel top_k_kernels[] = {
        top_k_cuda<1, BLOCK_THREADS>,
        top_k_cuda<2, BLOCK_THREADS>,
        top_k_cuda<3, BLOCK_THREADS>,
        top_k_cuda<4, BLOCK_THREADS>,
        top_k_cuda<5, BLOCK_THREADS>,
        top_k_cuda<6, BLOCK_THREADS>,
        top_k_cuda<7, BLOCK_THREADS>,
        top_k_cuda<8, BLOCK_THREADS>,
        top_k_cuda<9, BLOCK_THREADS>,
        top_k_cuda<10, BLOCK_THREADS>,
        top_k_cuda<11, BLOCK_THREADS>,
        top_k_cuda<12, BLOCK_THREADS>,
        top_k_cuda<13, BLOCK_THREADS>,
        top_k_cuda<14, BLOCK_THREADS>,
        top_k_cuda<15, BLOCK_THREADS>,
        top_k_cuda<16, BLOCK_THREADS>,
        top_k_cuda<17, BLOCK_THREADS>,
        top_k_cuda<18, BLOCK_THREADS>,
        top_k_cuda<19, BLOCK_THREADS>,
        top_k_cuda<20, BLOCK_THREADS>,
        top_k_cuda<21, BLOCK_THREADS>,
        top_k_cuda<22, BLOCK_THREADS>,
        top_k_cuda<23, BLOCK_THREADS>,
        top_k_cuda<24, BLOCK_THREADS>,
        top_k_cuda<25, BLOCK_THREADS>,
        top_k_cuda<26, BLOCK_THREADS>,
        top_k_cuda<27, BLOCK_THREADS>,
        top_k_cuda<28, BLOCK_THREADS>,
        top_k_cuda<29, BLOCK_THREADS>,
        top_k_cuda<30, BLOCK_THREADS>,
        top_k_cuda<31, BLOCK_THREADS>,
        top_k_cuda<32, BLOCK_THREADS>,
    };

    top_k_n_pass<BLOCK_THREADS>(top_k_kernels, input_output_scores, input_output_indices, temp_scores, temp_indices,
        (int*) active_count, (int*) active_count_per_batch, num_items_per_image, image_stride, num_top_k, num_classes,
        num_images, stream);
}
