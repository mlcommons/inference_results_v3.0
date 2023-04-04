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

#ifndef __TRITON_FRONTEND_HELPERS_HPP__
#define __TRITON_FRONTEND_HELPERS_HPP__

// QSL
#include "qsl_cpu.hpp"

// LoadGen
#include "system_under_test.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <thread>

// TRITON
#include "src/tracer.h"
#include "triton/core/tritonserver.h"
// Triton
#include "model_config.pb.h"
#include "src/common.h"

#include "pinned_memory_pool.hpp"

constexpr size_t BERT_MAX_SEQ_LENGTH{384};

namespace triton_frontend
{
namespace ni = triton::server;

class Triton_Server_SUT;
using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;
using PoolBlockPair = std::pair<PinnedMemoryPool*, PinnedMemoryPool::MemoryBlock*>;

size_t GetSampleLength(qsl::SampleLibraryPtr_t qsl_ptr, const mlperf::QuerySampleIndex idx);

TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id);

TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name, size_t byte_size,
    TRITONSERVER_MemoryType preferred_memory_type, int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id);

size_t GetDataTypeByteSize(const inference::DataType dtype);

TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype);

std::string MemoryTypeString(TRITONSERVER_MemoryType memory_type);

void WarmupResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flag, void* userp);

void BatchResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp);

void RequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp);

void ResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp);

struct ResponseMetaData
{
    ResponseMetaData()
        : m_ServerPtr(nullptr)
        , m_TracePtr(nullptr)
        , m_PaddingSize(0)
    {
    }
    ResponseMetaData(Triton_Server_SUT* server_ptr)
        : m_ServerPtr(server_ptr)
        , m_TracePtr(nullptr)
        , m_PaddingSize(0)
    {
    }
    Triton_Server_SUT* m_ServerPtr;
    mlperf::ResponseId m_ResponseId;
    mlperf::QuerySampleIndex m_QuerySampleIdx;
    TRITONSERVER_InferenceTrace* m_TracePtr;
    // FIXME assuming that there is only one output
    size_t m_PaddingSize;
};

// Use a separate metadata structure for triton requests that can contain multiple samples
struct BatchResponseMetaData
{
    BatchResponseMetaData()
        : m_ServerPtr(nullptr)
        , m_TracePtr(nullptr)
        , m_PaddingSize(0)
    {
    }
    BatchResponseMetaData(Triton_Server_SUT* server_ptr)
        : m_ServerPtr(server_ptr)
        , m_TracePtr(nullptr)
        , m_PaddingSize(0)
    {
    }
    Triton_Server_SUT* m_ServerPtr;
    std::vector<mlperf::ResponseId> m_ResponseId;
    std::vector<mlperf::QuerySampleIndex> m_QuerySampleIdxList;
    std::vector<size_t> m_DlrmNumPairsList;
    size_t m_DlrmTotalNumPairs;
    TRITONSERVER_InferenceTrace* m_TracePtr;
    // FIXME assuming that there is only one output
    size_t m_PaddingSize;
    size_t m_RequestBatchSize;
};

// A singlenton class of request pool for all inference requests,
// completed requests will be reused instead of being deleted.
class RequestPool
{
public:
    struct Block
    {
        Block()
            : m_AssignedPool(nullptr)
            , m_NextBlock(nullptr)
            , m_Data(nullptr)
        {
        }
        RequestPool* m_AssignedPool;
        Block* m_NextBlock;
        TRITONSERVER_InferenceRequest* m_Data;
        // Not a great place for holding response metadata as this imposes
        // release order, response then request. But it is handy as request
        // and response are not decoupled.
        ResponseMetaData m_ResponseMetadata;
        BatchResponseMetaData m_BatchResponseMetadata;
    };

    ~RequestPool();

    static void Create(const size_t initial_element_count, TRITONSERVER_Server* server, Triton_Server_SUT* server_sut,
        const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
        const std::vector<std::string>& outputs);

    static void Destroy();

    // Will create new block if the pool is currently empty, function caller
    // should check if the block is initialized (m_Data != nullptr) and call
    // InitInferenceRequest() if not.
    static Block* Obtain(size_t pool_idx)
    {
        auto& instance = instances_[pool_idx];
        std::lock_guard<std::mutex> lk(instance->m_ReleasedMtx);
        Block* res;
        // The implementation ensures that there is no concurrent obtain of the same instance
        if (instance->m_Head == nullptr)
        {
            instance->m_Head = instance->m_ReleasedHead;
            instance->m_ReleasedHead = nullptr;
        }
        res = instance->m_Head;
        if (res != nullptr)
        {
            instance->m_Head = instance->m_Head->m_NextBlock;
        }
        else
        {
            instance->m_Blocks.emplace_back();
            res = &instance->m_Blocks.back();
            instance->InternalInitInferenceRequest(res);
        }
        return res;
    }

    static void Release(Block* block)
    {
        auto instance = block->m_AssignedPool;
        std::lock_guard<std::mutex> lk(instance->m_ReleasedMtx);
        block->m_NextBlock = instance->m_ReleasedHead;
        instance->m_ReleasedHead = block;
    }

private:
    RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server, Triton_Server_SUT* server_sut,
        const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
        const std::vector<std::string>& outputs);

    void InternalInitInferenceRequest(RequestPool::Block* block);

    static std::vector<std::unique_ptr<RequestPool>> instances_;

    std::mutex m_ReleasedMtx;
    Block* m_Head;
    Block* m_ReleasedHead;
    // Use list so that we may add new blocks without invalidating
    // pointer / reference to existing blocks
    std::list<Block> m_Blocks;

    // Metadata to construct an TRTServerV2_InferenceRequest object
    TRITONSERVER_Server* m_Server;
    Triton_Server_SUT* m_ServerSUT;
    const std::string m_ModelName;
    const int64_t m_ModelVersion;

    std::vector<InputMetaData> m_Inputs;
    std::vector<std::string> m_Outputs;
};

template <typename T>
class SyncQueue
{
public:
    typedef typename std::deque<T>::iterator iterator;

    SyncQueue() {}

    bool empty()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        return m_Queue.empty();
    }

    void insert(const std::vector<T>& values)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin(), values.end());
        }
        m_Condition.notify_one();
    }
    void insert(const std::vector<T>& values, const size_t begin_idx, const size_t end_index)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin() + begin_idx, values.begin() + end_index);
        }
        m_Condition.notify_one();
    }
    void acquire(std::deque<T>& values, std::chrono::microseconds duration = 10000, size_t size = 1, bool limit = false)
    {
        size_t remaining = 0;

        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Condition.wait_for(l, duration, [=] { return m_Queue.size() >= size; });

            if (!limit || m_Queue.size() <= size)
            {
                values.swap(m_Queue);
            }
            else
            {
                auto beg = m_Queue.begin();
                auto end = beg + size;
                values.insert(values.end(), beg, end);
                m_Queue.erase(beg, end);
                remaining = m_Queue.size();
            }
        }

        // wake up any waiting threads
        if (remaining)
            m_Condition.notify_one();
    }

    void push_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.push_back(v);
        }
        m_Condition.notify_one();
    }
    void emplace_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.emplace_back(v);
        }
        m_Condition.notify_one();
    }
    T front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        return r;
    }
    T front_then_pop()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        m_Queue.pop_front();
        return r;
    }
    void pop_front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Queue.pop_front();
    }

private:
    mutable std::mutex m_Mutex;
    std::condition_variable m_Condition;
    std::deque<T> m_Queue;
};

} // namespace triton_frontend

#endif