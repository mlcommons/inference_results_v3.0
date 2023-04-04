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

#include "triton_callbacks.hpp"
#include "triton/core/tritonserver.h"
#include "triton_request_pool.hpp"
#include "triton_sut.hpp"

namespace triton_frontend
{

// Callback function for released request
void RequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    auto request_block = reinterpret_cast<triton_frontend::RequestPool::Block*>(userp);
    triton_frontend::RequestPool::Release(request_block);
}

// Callback function for completed batched response
void ResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    /* Pass the response (and various items from the userp field) to the member
     * Completion function
     */
    auto response_metadata = reinterpret_cast<triton_frontend::ResponseMetaData*>(userp);
    auto dev = response_metadata->m_DevPtr;
    dev->Completion(response, response_metadata);
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

// Callback function for inference requests during warmup phase
void WarmupResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flag, void* userp)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Increment the number of warmup responses received via member function call
     */
    auto response_metadata = reinterpret_cast<triton_frontend::ResponseMetaData*>(userp);
    auto dev = response_metadata->m_DevPtr;
    dev->IncrementWarmupResponses();
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name, size_t byte_size,
    TRITONSERVER_MemoryType preferred_memory_type, int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
    if (byte_size != 0)
    {
        // With knowledge that the 'preferred_memory_type_id' reflects the instance device ID
        auto pool = (*reinterpret_cast<triton_frontend::PinnedMemoryPoolEnsemble*>(userp))[preferred_memory_type_id];
        auto block = pool->Obtain();
        // Use CPU instead of CPU_PINNED to trigger internal pinned memory buffer
        // in Triton
        *actual_memory_type = pool->getMemType();
        // if mem type is not CPU or CPU_PINNED, the type id is 0 by convention
        *actual_memory_type_id = pool->getMemType() == TRITONSERVER_MEMORY_GPU ? preferred_memory_type_id : 0;
        *buffer = block->m_Data;
        *buffer_userp = new triton_frontend::PoolBlockPair(pool.get(), block);
    }
    // If 'byte_size' is zero just return 'buffer'==nullptr, we don't
    // need to do any other book-keeping.
    else
    {
        *buffer = nullptr;
        *buffer_userp = nullptr;
        // std::cerr << "allocated " << byte_size << " bytes for result tensor "
        // << tensor_name;
    }
    return nullptr; // Success
}

TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
{
    if (buffer_userp != nullptr)
    {
        auto pool_block = reinterpret_cast<triton_frontend::PoolBlockPair*>(buffer_userp);
        pool_block->first->Release(pool_block->second);
        delete pool_block;
    }
    return nullptr; // Success
}

} // namespace triton_frontend
