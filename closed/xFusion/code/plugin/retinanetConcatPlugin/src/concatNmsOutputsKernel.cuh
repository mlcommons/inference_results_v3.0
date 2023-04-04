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

#include "cuda_fp16.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ======================
// Common defines/tables
// ======================

#define WARP_SIZE 32


// ======================
// Cuda kernels
// ======================

// [BS, 1] [BS,1000,4] [BS,1000] [BS,1000] -> [BS,7001] (Duplicate one of the [BS,1000] twice)
// 
// inputs:
//   * (A) : TdetBox : bbox       [BS, 1000, 4] : swap the [0,1] and [2,3] dim so we change [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
//   * (B) : TdetSco : score      [BS, 1000]    : unsqueezed into [BS, 1000, 1]
//   * (C) : TdetCls : label      [BS, 1000]    : unsqueezed into [BS, 1000, 1]
//   * (D) : TnumDet : keep_count [BS, 1]
// 
// 1. Concat bbox,score,label in the order of [score, bbox, score, label] so they are together in one batch. [BS, 1000, 7]
// 2. Reshape to [BS, 7000], then concat with keep_count so it becomes [BS
// 
//  [ TdetSco ] 
//  [ TdetBox ] interleaved
//  [ TdetSco ]
//  [ TdetCls ]

template<int BLOCK_SIZE>
__global__ void concat_nms_outputs_gpu(float* d_To, int32_t* d_TnumDet, float* d_TdetBox, float* d_TdetSco, int32_t* d_TdetCls, int C0)
{
    // Notes about implementation: concat of 7 input sources are swizzled in blocks of 8 for simplicity (i.e 4 blocks per warp)
    int numSteps = C0*8;  
    int numB = ((numSteps+BLOCK_SIZE-1)/BLOCK_SIZE);

    int ni = blockIdx.x;
    int idx_in0 = ni*C0;        //  [BS, 1000, 1]
    int idx_in1 = ni*C0*4;      //  [BS, 1000, 4]
    int idx_out = ni*((7*C0)+1);  //  [BS, 7001]

    // Write C0 channels
    for(int b=0; b<numB; b++) 
    {
        int ci = threadIdx.x + (BLOCK_SIZE*b);
        int cOff = ci % 8;
        int cIdx = ci / 8;
        if (cIdx >= C0) break;

        // __syncthreads();
        // printf("    [%d,%d,%d,%d]\n",ni,hi,b,wi);
        // __syncthreads();

        switch(cOff) 
        {
            // 0  : detection scores
            // 1-4: swap the [0,1] and [2,3] dim so we change [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
            // 5  : detection scores (replication)
            // 6  : classification
            case 0 : d_To[idx_out + (cIdx*7) + 0] = d_TdetSco[cIdx*1 + idx_in0];
            case 1 : d_To[idx_out + (cIdx*7) + 1] = d_TdetBox[cIdx*4 + idx_in1 + 1];
            case 2 : d_To[idx_out + (cIdx*7) + 2] = d_TdetBox[cIdx*4 + idx_in1 + 0];
            case 3 : d_To[idx_out + (cIdx*7) + 3] = d_TdetBox[cIdx*4 + idx_in1 + 3];
            case 4 : d_To[idx_out + (cIdx*7) + 4] = d_TdetBox[cIdx*4 + idx_in1 + 2];
            case 5 : d_To[idx_out + (cIdx*7) + 5] = d_TdetSco[cIdx*1 + idx_in0];
            case 6 : d_To[idx_out + (cIdx*7) + 6] = (float) d_TdetCls[cIdx*1 + idx_in0];  // convert to fp32!
        }
    }

    // Write final num detections (convert to fp32)
    if (threadIdx.x == 0) {
         d_To[idx_out + 7*C0] = (float) d_TnumDet[ni];
    }
}



// ======================
// Launchers
// ======================


void launch_concat_nms_outputs_gpu(int N, int C0, float* d_To, float* d_TnumDet, float* d_TdetBox, float* d_TdetSco, float* d_TdetCls, cudaStream_t stream)
{
    // C0 = 7000
    // C1 = 1
    const int B = 128;
    dim3 block(B,1,1);     // C0 blocks
    dim3 grid(N,1,1);      // BS

    concat_nms_outputs_gpu<B><<<grid, block, 0, stream>>>(d_To, (int32_t*)d_TnumDet, d_TdetBox, d_TdetSco, (int32_t*)d_TdetCls, C0);
}

