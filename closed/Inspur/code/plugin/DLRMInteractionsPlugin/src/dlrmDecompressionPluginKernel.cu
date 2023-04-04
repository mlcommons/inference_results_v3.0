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

#include "dlrmHelper.h"
#include "dlrmInteractionsPluginKernel.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define gpuErrChk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

using namespace std;

__global__ void decompression_kernel_opt(
    const int* const __restrict__ com_data_gpu, int* const __restrict__ decom_data_gpu, const int num_ui_pairs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int max_per_thread = (num_ui_pairs + total_threads - 1) / total_threads;
    const int odd_offset = index > total_threads / 2 ? 1 : 0;
    index = index - total_threads / 2 * odd_offset;

#pragma unroll
    for (int i = 0; i < max_per_thread; i++)
    {
        // first half of threads work on even ui_pairs
        // second half of threds work on odd ui_pairs
        // this reduces warp divergence since adjacent threads execute
        // same block of switch statement
        long int ui_pair_idx = index * 2 + odd_offset + (total_threads * i);

        if (ui_pair_idx < num_ui_pairs)
        {

            int4 inp1, inp2, inp3, inp4;
            int4 outp1, outp2, outp3, outp4, outp5, outp6, outp7;

            // Reading the values from DRAM in 128 bit packets
            inp1 = *(int4*) &com_data_gpu[(ui_pair_idx * 16) + 0];
            inp2 = *(int4*) &com_data_gpu[(ui_pair_idx * 16) + 4];
            inp3 = *(int4*) &com_data_gpu[(ui_pair_idx * 16) + 8];
            inp4 = *(int4*) &com_data_gpu[(ui_pair_idx * 16) + 12];

            // Writing the values into DRAM in 128 bit packets
            int4* p_decom_data_gpu = NULL;

            // Every other ui_pair_idx is 16 byte aligned
            // 2 cases:
            //  1. when ui_pair_idx is 16 byte aligned, write int4s starting at base address
            //  2. when ui_pair_idx is not 16 byte aligned, write int4s starting at base address + 8 bytes
            switch (((ui_pair_idx * 26) % 4))
            {
            case 0:
                p_decom_data_gpu = (int4*) &decom_data_gpu[(ui_pair_idx * 26) + 0];
                outp1.x = inp1.x;
                outp1.y = (inp1.y & 0xFFFF'0000) >> 16;
                outp1.z = inp1.y & 0xFFFF;
                outp1.w = (inp1.z & 0xFFFF'0000) >> 16;
                p_decom_data_gpu[0] = outp1;

                outp2.x = inp1.z & 0xFFFF;
                outp2.y = (inp3.x & 0xFF00'0000) >> 24;
                outp2.z = (inp1.w & 0xFFFF'0000) >> 16;
                outp2.w = inp1.w & 0xFFFF;
                p_decom_data_gpu[1] = outp2;

                outp3.x = (inp3.x & 0xFF'0000) >> 16;
                outp3.y = inp2.x;
                outp3.z = inp2.y;
                outp3.w = inp2.z;
                p_decom_data_gpu[2] = outp3;

                outp4.x = (inp3.x & 0xFF00) >> 8;
                outp4.y = (inp2.w & 0xFFFF'0000) >> 16;
                outp4.z = inp2.w & 0xFFFF;
                outp4.w = inp3.x & 0xFF;
                p_decom_data_gpu[3] = outp4;

                outp5.x = (inp3.y & 0xFF00'0000) >> 24;
                outp5.y = (inp3.y & 0xFF'FF00) >> 8;
                outp5.z = inp3.y & 0xFF;
                outp5.w = inp3.z;
                p_decom_data_gpu[4] = outp5;

                outp6.x = inp3.w;
                outp6.y = inp4.x;
                outp6.z = inp4.y;
                outp6.w = (inp4.z & 0xFFFF'0000) >> 16;
                p_decom_data_gpu[5] = outp6;

                outp7.x = (inp4.z & 0xFF00) >> 8;
                outp7.y = inp4.z & 0xFF;
                decom_data_gpu[(ui_pair_idx * 26) + 24] = outp7.x;
                decom_data_gpu[(ui_pair_idx * 26) + 25] = outp7.y;
                break;
            case 1: break;

            case 2:
                p_decom_data_gpu = (int4*) &decom_data_gpu[(ui_pair_idx * 26) + 2];
                outp1.x = inp1.x;
                outp1.y = (inp1.y & 0xFFFF'0000) >> 16;
                decom_data_gpu[(ui_pair_idx * 26) + 0] = outp1.x;
                decom_data_gpu[(ui_pair_idx * 26) + 1] = outp1.y;

                outp2.x = inp1.y & 0xFFFF;
                outp2.y = (inp1.z & 0xFFFF'0000) >> 16;
                outp2.z = inp1.z & 0xFFFF;
                outp2.w = (inp3.x & 0xFF00'0000) >> 24;
                p_decom_data_gpu[0] = outp2;

                outp3.x = (inp1.w & 0xFFFF'0000) >> 16;
                outp3.y = inp1.w & 0xFFFF;
                outp3.z = (inp3.x & 0xFF'0000) >> 16;
                outp3.w = inp2.x;
                p_decom_data_gpu[1] = outp3;

                outp4.x = inp2.y;
                outp4.y = inp2.z;
                outp4.z = (inp3.x & 0xFF00) >> 8;
                outp4.w = (inp2.w & 0xFFFF'0000) >> 16;
                p_decom_data_gpu[2] = outp4;

                outp5.x = inp2.w & 0xFFFF;
                outp5.y = inp3.x & 0xFF;
                outp5.z = (inp3.y & 0xFF00'0000) >> 24;
                outp5.w = (inp3.y & 0xFF'FF00) >> 8;
                p_decom_data_gpu[3] = outp5;

                outp6.x = inp3.y & 0xFF;
                outp6.y = inp3.z;
                outp6.z = inp3.w;
                outp6.w = inp4.x;
                p_decom_data_gpu[4] = outp6;

                outp7.x = inp4.y;
                outp7.y = (inp4.z & 0xFFFF'0000) >> 16;
                outp7.z = (inp4.z & 0xFF00) >> 8;
                outp7.w = inp4.z & 0xFF;
                p_decom_data_gpu[5] = outp7;
                break;
            case 3: break;
            }
        }
    }
}

void run_decompression(int* com_data, int* decom_data, int num_ui_pairs, cudaStream_t stream)
{
    // TODO: how should launch config change for different architectures
    decompression_kernel_opt<<<128, 1024, 0, stream>>>(com_data, decom_data, num_ui_pairs);
}
