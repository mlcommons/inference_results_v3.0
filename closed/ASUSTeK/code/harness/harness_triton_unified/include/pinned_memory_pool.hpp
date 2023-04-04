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
#define TRITON_ENABLE_GPU 1

#include "src/common.h"
#include "src/tracer.h"
#include "triton/core/tritonserver.h"

// QSL
#include "qsl.hpp"

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

// NUMA utils
#include "utils.hpp"

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

    PinnedMemoryPool(const size_t nb_blocks, const size_t block_byte_size,
        const TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU);
    virtual ~PinnedMemoryPool();

    virtual MemoryBlock* Obtain()
    {
        CHECK(m_Size > 0);
        MemoryBlock* res = m_Head;
        m_Head = m_Head->m_NextBlock;
        m_Size -= 1;
        return res;
    }

    virtual void Release(MemoryBlock* block)
    {
        block->m_NextBlock = nullptr;
        m_Tail->m_NextBlock = block;
        m_Tail = block;
        m_Size += 1;
        CHECK(m_Size <= m_Capacity);
    }

    virtual TRITONSERVER_MemoryType getMemType()
    {
        return m_MemType;
    }

protected:
    MemoryBlock* m_Head;
    MemoryBlock* m_Tail;
    int32_t m_Size;
    int32_t m_Capacity;
    std::vector<MemoryBlock> m_Blocks;
    char* m_Buffer;
    TRITONSERVER_MemoryType m_MemType;
};

class MutexPinnedMemoryPool : public PinnedMemoryPool
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

    MutexPinnedMemoryPool(const size_t element_count, const size_t element_byte_size,
        const TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU)
        : PinnedMemoryPool(element_count, element_byte_size, mem_type)
    {
    }
    ~MutexPinnedMemoryPool() = default;

    PinnedMemoryPool::MemoryBlock* Obtain() override
    {
        std::lock_guard<std::mutex> lk(m_ListMtx);
        return PinnedMemoryPool::Obtain();
    }

    void Release(PinnedMemoryPool::MemoryBlock* block) override
    {
        std::lock_guard<std::mutex> lk(m_ListMtx);
        PinnedMemoryPool::Release(block);
    }

private:
    std::mutex m_ListMtx;
};

class PinnedMemoryPoolEnsemble
{
public:
    PinnedMemoryPoolEnsemble(const size_t per_instance_element_count, const size_t element_byte_size,
        const std::map<size_t, size_t>& gpu_instance_count, const NumaConfig& numa_config,
        const TRITONSERVER_MemoryType mem_type = TRITONSERVER_MEMORY_CPU);
    ~PinnedMemoryPoolEnsemble() {}

    std::shared_ptr<PinnedMemoryPool> operator[](int idx)
    {
        return pools_[idx];
    }

private:
    std::vector<std::shared_ptr<PinnedMemoryPool>> pools_;
};
} // namespace triton_frontend

#endif