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
    m_Buffer = (char*) malloc(element_count * element_byte_size);

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
        free(m_Buffer);
    }
}

} // namespace triton_frontend