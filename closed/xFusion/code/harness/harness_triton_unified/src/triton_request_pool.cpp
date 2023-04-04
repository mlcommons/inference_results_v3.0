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

#include "triton_request_pool.hpp"
#include "src/common.h"
#include "triton/core/tritonserver.h"
#include "triton_callbacks.hpp"

namespace triton_frontend
{

std::vector<std::unique_ptr<RequestPool>> RequestPool::instances_;

void RequestPool::Create(const size_t initial_element_count, TRITONSERVER_Server* server, ITritonDevice* dev,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
{
    for (size_t count = 0; count < 2 /*2 REQUEST_POOL_COUNT */; count++)
    {
        instances_.emplace_back(
            new RequestPool(initial_element_count, server, dev, model_name, model_version, inputs, outputs));
    }
}

void RequestPool::Destroy()
{
    for (auto& instance : instances_)
    {
        instance.reset(nullptr);
    }
}

RequestPool::RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server, ITritonDevice* dev,
    const std::string& model_name, const uint32_t model_version, const std::vector<InputMetaData>& inputs,
    const std::vector<std::string>& outputs)
    : m_Head(nullptr)
    , m_ReleasedHead(nullptr)
    , m_Blocks(initial_element_count)
    , m_Server(server)
    , m_Device(dev)
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
    block->m_ResponseMetadata = ResponseMetaData(m_Device);

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
