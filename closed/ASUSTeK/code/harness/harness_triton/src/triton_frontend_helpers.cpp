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

#include "triton_frontend_helpers.hpp"
#include "triton_frontend_server.hpp"

namespace triton_frontend
{
// Callback function for released request
void RequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    auto request_block = reinterpret_cast<triton_frontend::RequestPool::Block*>(userp);
    triton_frontend::RequestPool::Release(request_block);
}
// Callback function for completed response
void ResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    /* Pass the response (and various items from the userp field) to the member
     * Completion function
     */
    auto response_metadata = reinterpret_cast<triton_frontend::ResponseMetaData*>(userp);
    auto sut = response_metadata->m_ServerPtr;
    sut->Completion(response, response_metadata);
#if TRITON_FRONTEND_TRACE
    if (response_metadata->m_TracePtr != nullptr)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t trace_id = 0;
        TRITONSERVER_InferenceTraceId(response_metadata->m_TracePtr, &trace_id);
        sut->m_TraceManager->CaptureTimestamp(
            trace_id, TRITONSERVER_TRACE_LEVEL_MIN, "MLPerf Request Response RECV", TIMESPEC_TO_NANOS(ts));
    }
#endif // TRITON_FRONTEND_TRACE
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

// Callback function for completed batched response
void BatchResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    /* Pass the response (and various items from the userp field) to the member
     * Completion function
     */
    auto response_metadata = reinterpret_cast<triton_frontend::BatchResponseMetaData*>(userp);
    auto sut = response_metadata->m_ServerPtr;
    sut->BatchCompletion(response, response_metadata);
#if TRITON_FRONTEND_TRACE
    if (response_metadata->m_TracePtr != nullptr)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t trace_id = 0;
        TRITONSERVER_InferenceTraceId(response_metadata->m_TracePtr, &trace_id);
        sut->m_TraceManager->CaptureTimestamp(
            trace_id, TRITONSERVER_TRACE_LEVEL_MIN, "MLPerf Request Response RECV", TIMESPEC_TO_NANOS(ts));
    }
#endif // TRITON_FRONTEND_TRACE
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

// Callback function for inference requests during warmup phase
void WarmupResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flag, void* userp)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Increment the number of warmup responses received via member function call
     */
    auto sut = reinterpret_cast<triton_frontend::Triton_Server_SUT*>(userp);
    sut->IncrementWarmupResponses();
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

std::string MemoryTypeString(TRITONSERVER_MemoryType memory_type)
{
    return (memory_type == TRITONSERVER_MEMORY_CPU) ? "CPU memory" : "GPU memory";
}

TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype)
{
    switch (dtype)
    {
    case inference::DataType::TYPE_BOOL: return TRITONSERVER_TYPE_BOOL;
    case inference::DataType::TYPE_UINT8: return TRITONSERVER_TYPE_UINT8;
    case inference::DataType::TYPE_UINT16: return TRITONSERVER_TYPE_UINT16;
    case inference::DataType::TYPE_UINT32: return TRITONSERVER_TYPE_UINT32;
    case inference::DataType::TYPE_UINT64: return TRITONSERVER_TYPE_UINT64;
    case inference::DataType::TYPE_INT8: return TRITONSERVER_TYPE_INT8;
    case inference::DataType::TYPE_INT16: return TRITONSERVER_TYPE_INT16;
    case inference::DataType::TYPE_INT32: return TRITONSERVER_TYPE_INT32;
    case inference::DataType::TYPE_INT64: return TRITONSERVER_TYPE_INT64;
    case inference::DataType::TYPE_FP16: return TRITONSERVER_TYPE_FP16;
    case inference::DataType::TYPE_FP32: return TRITONSERVER_TYPE_FP32;
    case inference::DataType::TYPE_FP64: return TRITONSERVER_TYPE_FP64;
    case inference::DataType::TYPE_STRING: return TRITONSERVER_TYPE_BYTES;
    default: break;
    }

    return TRITONSERVER_TYPE_INVALID;
}

size_t GetDataTypeByteSize(const inference::DataType dtype)
{
    switch (dtype)
    {
    case inference::TYPE_BOOL: return 1;
    case inference::TYPE_UINT8: return 1;
    case inference::TYPE_UINT16: return 2;
    case inference::TYPE_UINT32: return 4;
    case inference::TYPE_UINT64: return 8;
    case inference::TYPE_INT8: return 1;
    case inference::TYPE_INT16: return 2;
    case inference::TYPE_INT32: return 4;
    case inference::TYPE_INT64: return 8;
    case inference::TYPE_FP16: return 2;
    case inference::TYPE_FP32: return 4;
    case inference::TYPE_FP64: return 8;
    case inference::TYPE_STRING: return 0;
    default: break;
    }

    return 0;
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
        // std::cout << "allocated " << byte_size << " bytes for result tensor "
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

size_t GetSampleLength(qsl::SampleLibraryPtr_t qsl_ptr, const mlperf::QuerySampleIndex idx)
{
    // Get sample length by checking where the input_mask change from 1 to 0
    size_t start{0};
    size_t end{BERT_MAX_SEQ_LENGTH};
    size_t cursor{(start + end) / 2};
    auto& input_mask = *static_cast<std::array<int32_t, BERT_MAX_SEQ_LENGTH>*>(qsl_ptr->GetSampleAddress(idx, 2));
    while (cursor != start)
    {
        if (input_mask[cursor])
        {
            start = cursor;
        }
        else
        {
            end = cursor;
        }
        cursor = (start + end) / 2;
    }
    return end;
}

std::vector<std::unique_ptr<RequestPool>> RequestPool::instances_;

void RequestPool::Create(const size_t initial_element_count, TRITONSERVER_Server* server, Triton_Server_SUT* server_sut,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
{
    for (size_t count = 0; count < 2 /*2 REQUEST_POOL_COUNT */; count++)
    {
        instances_.emplace_back(
            new RequestPool(initial_element_count, server, server_sut, model_name, model_version, inputs, outputs));
    }
}

void RequestPool::Destroy()
{
    for (auto& instance : instances_)
    {
        instance.reset(nullptr);
    }
}

RequestPool::RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server, Triton_Server_SUT* server_sut,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
    : m_Head(nullptr)
    , m_ReleasedHead(nullptr)
    , m_Blocks(initial_element_count)
    , m_Server(server)
    , m_ServerSUT(server_sut)
    , m_ModelName(model_name)
    , m_ModelVersion(model_version)
    , m_Inputs(inputs)
    , m_Outputs(outputs)
{
    Block* prev_block = nullptr;
    // Build free block list in the same order as in std list
    for (auto& block : m_Blocks)
    {
        if (m_Head == nullptr)
        {
            m_Head = &block;
        }
        if (prev_block != nullptr)
        {
            prev_block->m_NextBlock = &block;
        }
        prev_block = &block;
        InternalInitInferenceRequest(&block);
    }
}

RequestPool::~RequestPool()
{
    for (auto& block : m_Blocks)
    {
        if (block.m_Data != nullptr)
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(block.m_Data), "deleting inference request");
        }
    }
}

void RequestPool::InternalInitInferenceRequest(RequestPool::Block* block)
{
    block->m_AssignedPool = this;
    block->m_ResponseMetadata = ResponseMetaData(m_ServerSUT);
    block->m_BatchResponseMetadata = BatchResponseMetaData(m_ServerSUT);

    // Init m_Data (TRITONSERVER_InferenceRequest)
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&block->m_Data, m_Server, m_ModelName.c_str(), m_ModelVersion),
        "creating new inference request");
    for (const auto& input : m_Inputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(block->m_Data, std::get<0>(input).c_str(), std::get<1>(input),
                        std::get<2>(input).data(), std::get<2>(input).size()),
            "setting input meta-data for the request");
    }
    for (const auto& output : m_Outputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(block->m_Data, output.c_str()),
            "requesting output for the request");
    }
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(block->m_Data, RequestRelease, block),
        "setting request release callback");
}

} // namespace triton_frontend