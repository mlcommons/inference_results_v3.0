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

#ifndef __DLA_TRITON_FRONTEND_HELPERS_HPP__
#define __DLA_TRITON_FRONTEND_HELPERS_HPP__

#include "triton_frontend_helpers.hpp"

namespace triton_frontend
{
void DLARequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp);

// A singleton class of request pool for DLA inference requests,
// completed requests will be reused instead of being deleted.
class DLARequestPool
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
        DLARequestPool* m_AssignedPool;
        Block* m_NextBlock;
        TRITONSERVER_InferenceRequest* m_Data;
        // Not a great place for holding response metadata as this imposes
        // release order, response then request. But it is handy as request
        // and response are not decoupled.
        ResponseMetaData m_ResponseMetadata;
        BatchResponseMetaData m_BatchResponseMetadata;
    };

    ~DLARequestPool();

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
    DLARequestPool(const size_t initial_element_count, TRITONSERVER_Server* server, Triton_Server_SUT* server_sut,
        const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
        const std::vector<std::string>& outputs);

    void InternalInitInferenceRequest(DLARequestPool::Block* block);

    static std::vector<std::unique_ptr<DLARequestPool>> instances_;

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

} // namespace triton_frontend
#endif