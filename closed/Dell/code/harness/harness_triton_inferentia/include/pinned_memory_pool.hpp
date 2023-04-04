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

#ifndef __PINNED_MEMORY_POOL_HPP__
#define __PINNED_MEMORY_POOL_HPP__

// TRITON
#define TRITON_ENABLE_GPU 0

#include "src/common.h"
#include "src/tracer.h"
#include "triton/core/tritonserver.h"

// LoadGen
#include "system_under_test.h"

// General C++
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <thread>

namespace triton_frontend
{
// This mem pool can contain either CPU or GPU memory
// FIXME: rename (mem is not always pinned) and possibly template based on mem type
class PinnedMemoryPool
{
public:
    struct MemoryBlock
    {
        MemoryBlock()
            : m_NextBlock(nullptr)
            , m_Data(nullptr)
        {
        }
        MemoryBlock* m_NextBlock;
        char* m_Data;
    };

    PinnedMemoryPool(const size_t element_count, const size_t element_byte_size,
        const TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU);
    virtual ~PinnedMemoryPool();

    // Note that there is no checking for empty list
    virtual MemoryBlock* Obtain()
    {
        std::lock_guard<std::mutex> lk(m_ListMtx);
        MemoryBlock* res = m_Head;
        m_Head = m_Head->m_NextBlock;
        return res;
    }

    virtual void Release(MemoryBlock* block)
    {
        std::lock_guard<std::mutex> lk(m_ListMtx);
        block->m_NextBlock = nullptr;
        m_Tail->m_NextBlock = block;
        m_Tail = block;
        if (m_Head == nullptr)
        {
            m_Head = m_Tail;
        }
    }

    virtual TRITONSERVER_MemoryType getMemType()
    {
        return m_MemType;
    }

protected:
    MemoryBlock* m_Head;
    MemoryBlock* m_Tail;
    std::vector<MemoryBlock> m_Blocks;
    char* m_Buffer;
    TRITONSERVER_MemoryType m_MemType;
    std::mutex m_ListMtx;
};

} // namespace triton_frontend

#endif
