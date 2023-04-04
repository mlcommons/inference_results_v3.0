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

/**********************************************************************
 * 3D UNet KiTS19 custom kernels required for sliding window inference
 * NOTE: currently assuming batchsize == 1 only
 **********************************************************************/

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_BATCH_SIZE 8
#define MAX_ROI_SIZE 128*128*128

namespace lwis {

struct UNet3DParams
{
    int image_d;
    int image_h;
    int image_w;
    int image_size;
    int actual_num_slices;
    int roi_dhw;
    int roi_size;
    int in_ch;
    int out_ch;

    int slice_to_off_d[MAX_BATCH_SIZE] = {};
    int slice_to_off_h[MAX_BATCH_SIZE] = {};
    int slice_to_off_w[MAX_BATCH_SIZE] = {};
    void* patches[MAX_BATCH_SIZE] = {};

    UNet3DParams()
    {
        image_d = 256;
        image_h = 256;
        image_w = 256;
        image_size = 256 * 256 * 256;
        actual_num_slices = 1;
        roi_dhw = 128;
        roi_size = 128 * 128 * 128;
        in_ch = 1;
        out_ch = 3;
    }
};

// Slicing kernel from INT8 LINEAR sample input to INT8 LINEAR Sliding Window input
void UNet3DKiTS19SliceKernelI8Linear_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream, const int deviceId);

// Patch&Accumulation kernel with pre-conditioned Gaussian Kernels
// FP16 3ch Sliding Window output is fed into this kernel
// Kernel produces FP16 3ch Sliding Window size output after weighting w/ Gaussian patch
// and then accumulates the output to proper location to input sample sized FP16 3ch output
// that is zeroed out before
void UNet3DKiTS19PatchKernel_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream, const int deviceId, const bool useCGImpl);

// ArgMax is done on the input sample sized FP16 3ch tensor (i.e. output of PatchKernel)
// and produces INT8 LINEAR 1ch output of input sample sized tensor as a final output of
// KiTS19 inference work (i.e. segmentation on background, normal kidney cells and tumor cells)
void UNet3DKiTS19ArgMaxKernel_wrapper(
    void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream, const int deviceId);

} // namespace lwis
