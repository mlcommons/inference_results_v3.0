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

#include "unet3d_sw.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace lwis
{

__global__ void UNet3DKiTS19SliceKernelI8Linear(const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    int slice_id = blockIdx.z;
    int slice_offset_d = p.slice_to_off_d[slice_id];
    int slice_offset_h = p.slice_to_off_h[slice_id];
    int slice_offset_w = p.slice_to_off_w[slice_id];

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw && slice_id < p.actual_num_slices)
    {
        d_out[(slice_id * p.roi_size) + (p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w)]
            = d_in[p.image_w * p.image_h * (slice_offset_d + d) + p.image_w * (slice_offset_h + h)
                + (slice_offset_w + w)];
    }
}

// This impl assumes there's no race condition in read-modify-write of d_out
__global__ void UNet3DKiTS19PatchKernelNoOverlap(const __half* __restrict__ d_in, __half* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    const int slice_id = blockIdx.z;

    const int slice_offset_d = p.slice_to_off_d[slice_id];
    const int slice_offset_h = p.slice_to_off_h[slice_id];
    const int slice_offset_w = p.slice_to_off_w[slice_id];

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw && slice_id < p.actual_num_slices)
    {
        #pragma unroll
        for (int c = 0; c < p.out_ch; ++c)
        {
            d_out[p.image_h * p.image_w * (slice_offset_d + d) + p.image_w * (slice_offset_h + h) + (slice_offset_w + w)
                  + p.image_size * c]
                += d_in[slice_id * p.roi_size * p.out_ch
                        + p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w + p.roi_size * c]
                * ((__half*)p.patches[slice_id])[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w];
        }
    }
}

// This impl uses Cooperative Group and performs device-wide sync, to address race condition on d_out
__global__ void UNet3DKiTS19PatchKernelOverlapCG(const __half* __restrict__ d_in, __half* __restrict__ d_out, const UNet3DParams p)
{
    const int total_d = p.roi_dhw;
    const int total_h = p.roi_dhw;
    const int total_w = p.roi_dhw;
    const int d_stride = gridDim.x;
    const int h_stride = gridDim.y;
    const int w_stride = blockDim.x;

    int num_slices = p.actual_num_slices;
    for (int slice_id = 0; slice_id < num_slices; ++slice_id)
    {
        const int slice_offset_d = p.slice_to_off_d[slice_id];
        const int slice_offset_h = p.slice_to_off_h[slice_id];
        const int slice_offset_w = p.slice_to_off_w[slice_id];

        for (int dd = 0; dd < total_d; dd += d_stride)
        {
            int d = dd + blockIdx.x;
            for (int hh = 0; hh < total_h; hh += h_stride)
            {
                int h = hh + blockIdx.y;
                for (int ww = 0; ww < total_w; ww += w_stride)
                {
                    int w = ww + threadIdx.x;
                    if (d < total_d && h < total_h && w < total_w)
                    {
                        #pragma unroll
                        for (int c = 0; c < p.out_ch; ++c)
                        {
                            d_out[p.image_h * p.image_w * (slice_offset_d + d) + p.image_w * (slice_offset_h + h) + (slice_offset_w + w)
                                + p.image_size * c]
                                += d_in[slice_id * p.roi_size * p.out_ch
                                        + p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w + p.roi_size * c]
                                * ((__half*)p.patches[slice_id])[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w];
                        }
                    }
                }
            }
        }
        cg::this_grid().sync();
    }
}

__global__ void UNet3DKiTS19ArgMaxKernel(
    const __half* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    __half a = d_in[p.image_h * p.image_w * d + p.image_w * h + w];
    __half b = d_in[p.image_h * p.image_w * d + p.image_w * h + w + p.image_size];
    __half c = d_in[p.image_h * p.image_w * d + p.image_w * h + w + 2 * p.image_size];
    __half m = b;
    uint8_t l = 1;
    if (a > b)
    {
        m = a;
        l = 0;
    }
    if (d < p.image_d && h < p.image_h && w < p.image_w)
    {
        d_out[p.image_h * p.image_w * d + p.image_w * h + w] = m > c ? l : 2;
    }
}

void UNet3DKiTS19SliceKernelI8Linear_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for slicing
    dim3 dimBlock_slice(p.roi_dhw, 1, 1);
    dim3 dimGrid_slice(p.roi_dhw, p.roi_dhw, p.actual_num_slices);    
    UNet3DKiTS19SliceKernelI8Linear<<<dimGrid_slice, dimBlock_slice, 0, stream>>>(
        static_cast<int8_t*>(d_in), static_cast<int8_t*>(d_out), p);
}

void UNet3DKiTS19PatchKernelNoOverlap_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for Gaussian patching & accumulating
    dim3 dimBlock_patch(p.roi_dhw, 1, 1);
    dim3 dimGrid_patch(p.roi_dhw, p.roi_dhw, p.actual_num_slices);
    UNet3DKiTS19PatchKernelNoOverlap<<<dimGrid_patch, dimBlock_patch, 0, stream>>>(static_cast<__half*>(d_in), static_cast<__half*>(d_out), p);
}

// Using Cooperative Group, and device-wide sync
void UNet3DKiTS19PatchKernelOverlapCG_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    
    int BLOCKS;
    int THREADS;
    cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, UNet3DKiTS19PatchKernelOverlapCG, 0, p.roi_dhw);

    int GridY = BLOCKS >= p.roi_dhw ? p.roi_dhw : BLOCKS;
    int GridX = BLOCKS < p.roi_dhw ? 1 : BLOCKS / GridY;

    void* kernel_args[] = { &d_in, &d_out, (void*)&p };
    // for Gaussian patching & accumulating
    dim3 dimBlock_patch(THREADS, 1, 1);
    dim3 dimGrid_patch(GridX, GridY, 1);

    cudaLaunchCooperativeKernel((void*)(UNet3DKiTS19PatchKernelOverlapCG), dimGrid_patch, dimBlock_patch, kernel_args, 0, stream);
}

// Using CPU implicit sync; launches no-overlap kernel one by one
void UNet3DKiTS19PatchKernelOverlapImplicitSync_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for Gaussian patching & accumulating
    dim3 dimBlock_patch(p.roi_dhw, 1, 1);
    dim3 dimGrid_patch(p.roi_dhw, p.roi_dhw, 1);
    // repackaging new UNet3DParams
    UNet3DParams u3p;
    u3p.image_d = p.image_d;
    u3p.image_h = p.image_h;
    u3p.image_w = p.image_w;
    u3p.image_size = p.image_size;
    auto roi_size = p.roi_size;
    auto out_ch = p.out_ch;
    for (int slice_id = 0; slice_id < p.actual_num_slices; slice_id++)
    {
        u3p.slice_to_off_d[0] = p.slice_to_off_d[slice_id];
        u3p.slice_to_off_h[0] = p.slice_to_off_h[slice_id];
        u3p.slice_to_off_w[0] = p.slice_to_off_w[slice_id];
        u3p.patches[0] = p.patches[slice_id];
        UNet3DKiTS19PatchKernelNoOverlap<<<dimGrid_patch, dimBlock_patch, 0, stream>>>(
            &(static_cast<__half*>(d_in)[slice_id * roi_size * out_ch]), 
            static_cast<__half*>(d_out), 
            u3p);
    }
}

void UNet3DKiTS19PatchKernel_wrapper(void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0, const bool useCGImpl = false)
{
    if (useCGImpl)
    {
        UNet3DKiTS19PatchKernelOverlapCG_wrapper(d_in, d_out, p, stream, deviceId);
    }
    else
    {
        UNet3DKiTS19PatchKernelOverlapImplicitSync_wrapper(d_in, d_out, p, stream, deviceId);
    }
}

void UNet3DKiTS19ArgMaxKernel_wrapper(
    void* d_in, void* d_out, const UNet3DParams& p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for final ArgMax
    dim3 dimBlock_argmax(p.image_w, 1, 1);
    dim3 dimGrid_argmax(p.image_d, p.image_h, 1);
    UNet3DKiTS19ArgMaxKernel<<<dimGrid_argmax, dimBlock_argmax, 0, stream>>>(
        static_cast<__half*>(d_in), static_cast<int8_t*>(d_out), p);
}

} // namespace lwis
