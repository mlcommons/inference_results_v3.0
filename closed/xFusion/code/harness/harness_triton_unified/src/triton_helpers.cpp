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

#include "triton_helpers.hpp"
#include "triton_callbacks.hpp"
#include "triton_sut.hpp"

namespace triton_frontend
{

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

std::string GenerateCPUStr(const std::vector<int32_t>& cpuIds)
{
    assert(std::is_sorted(std::begin(cpuIds), std::end(cpuIds)));

    std::string cpuStr;

    int32_t lower = -1;
    int32_t upper = -1;
    for (const auto& cpuId : cpuIds)
    {
        if (lower == -1)
        {
            upper = lower = cpuId;
        }
        else if (cpuId == (upper + 1))
        {
            upper = cpuId;
        }
        else
        {
            if (!cpuStr.empty())
            {
                cpuStr += ",";
            }
            cpuStr += (std::to_string(lower) + "-" + std::to_string(upper));
            lower = upper = -1;
        }
    }

    if (lower != -1)
    {
        if (!cpuStr.empty())
        {
            cpuStr += ",";
        }
        cpuStr += (std::to_string(lower) + "-" + std::to_string(upper));
    }

    return cpuStr;
}

} // namespace triton_frontend