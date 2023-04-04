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

#pragma once

#include <cuda_runtime.h>

#define USE_CUB_SEGMENTED_SORT 1

void top_k_multi_pass(void* input_output_scores, void* input_output_indices, void* temp_scores, void* temp_indices,
    int* active_count, int* active_count_per_batch, int num_items_per_image, int image_stride, int num_top_k,
    int num_classes, int num_images, cudaStream_t& stream);

void segmented_scan(int* active_count_in, int* active_count_out, int* active_count_per_batch, int num_images,
    int num_classes, cudaStream_t stream);

void compact_segments(void* input_output_scores, void* input_output_indices, void* temp_scores, void* temp_indices,
    int* active_count, int* active_count_per_batch, int num_classes, int num_images, int num_items_per_image,
    int image_stride, cudaStream_t stream);
