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

#include "pinned_memory_pool.hpp"
#include "cuda_runtime_api.h"

// NUMA utils
#include "utils.hpp"

namespace triton_frontend
{

PinnedMemoryPool::PinnedMemoryPool(
    const size_t element_count, const size_t element_byte_size, const TRITONSERVER_MemoryType mem_type)
    : m_Head(nullptr)
    , m_Tail(nullptr)
    , m_Blocks(element_count)
    , m_Buffer(nullptr)
    , m_MemType(mem_type)
{
    switch (m_MemType)
    {
    case TRITONSERVER_MEMORY_CPU:
        FAIL_IF_CUDA_ERR(cudaHostAlloc(&m_Buffer, element_count * element_byte_size, cudaHostAllocPortable),
            "failed to allocate pinned memory");
        break;
    case TRITONSERVER_MEMORY_GPU:
        FAIL_IF_CUDA_ERR(cudaMalloc(&m_Buffer, element_count * element_byte_size), "failed to allocate device memory");
        break;
    default: break;
    }

    char* next_buffer = m_Buffer;
    for (auto& block : m_Blocks)
    {
        if (m_Tail == nullptr)
        {
            m_Tail = &block;
        }
        if (m_Head != nullptr)
        {
            block.m_NextBlock = m_Head;
        }
        m_Head = &block;
        block.m_Data = next_buffer;
        next_buffer += element_byte_size;
    }
}

PinnedMemoryPool::~PinnedMemoryPool()
{
    if (m_Buffer != nullptr)
    {
        switch (m_MemType)
        {
        case TRITONSERVER_MEMORY_CPU: FAIL_IF_CUDA_ERR(cudaFreeHost(m_Buffer), "failed to free pinned memory"); break;
        case TRITONSERVER_MEMORY_GPU:
            FAIL_IF_CUDA_ERR(cudaFree(m_Buffer), "failed to free device memory");
            ;
            break;
        default: break;
        }
    }
}

PinnedMemoryPoolEnsemble::PinnedMemoryPoolEnsemble(const size_t per_instance_element_count,
    const size_t element_byte_size, const std::map<size_t, size_t>& gpu_instance_count, const NumaConfig& numa_config,
    const TRITONSERVER_MemoryType mem_type)
{
    size_t largest_gpu_idx = 0;
    for (const auto& gpu_instance : gpu_instance_count)
    {
        largest_gpu_idx = std::max(gpu_instance.first, largest_gpu_idx);
    }
    pools_ = std::vector<std::shared_ptr<PinnedMemoryPool>>(largest_gpu_idx + 1);

    auto gpu_numa_map = getGpuToNumaMap(numa_config);
    int dev;
    FAIL_IF_CUDA_ERR(cudaGetDevice(&dev), "failed to get device");
    for (const auto& gpu_instance : gpu_instance_count)
    {
        // if mem_type is TRITONSERVER_MEMORY_GPU, the PinnedMemoryPool will
        // actually contain GPU memory
        // set the cuda device here so that the GPU memory for Triton outputs
        // gets allocated on the same device on which Triton's model instance
        // is running
        FAIL_IF_CUDA_ERR(cudaSetDevice(gpu_instance.first), "failed to set device");
        if (!numa_config.empty())
        {
            bindNumaMemPolicy(gpu_numa_map[gpu_instance.first], numa_config.size());
        }
        // Use lockless version of the pool if we know that there will not
        // be shared access to the GPU specific pool
        if ((gpu_instance.second > 1))
        {
            pools_[gpu_instance.first] = std::make_shared<MutexPinnedMemoryPool>(
                per_instance_element_count * gpu_instance.second, element_byte_size, mem_type);
        }
        else
        {
            pools_[gpu_instance.first] = std::make_unique<PinnedMemoryPool>(
                per_instance_element_count * gpu_instance.second, element_byte_size, mem_type);
        }
        if (!numa_config.empty())
        {
            resetNumaMemPolicy();
        }
    }
    FAIL_IF_CUDA_ERR(cudaSetDevice(dev), "failed to set device");
}

} // namespace triton_frontend