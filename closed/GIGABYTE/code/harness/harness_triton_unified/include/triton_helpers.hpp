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

#ifndef MLPINF_TRITON_HELPERS_HPP
#define MLPINF_TRITON_HELPERS_HPP

// QSL
#include "qsl.hpp"

// LoadGen
#include "system_under_test.h"

#include <atomic>
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

class ITritonSUT;
using InputMetaData = std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>;
using PoolBlockPair = std::pair<PinnedMemoryPool*, PinnedMemoryPool::MemoryBlock*>;

size_t GetDataTypeByteSize(const inference::DataType dtype);

TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype);

std::string MemoryTypeString(TRITONSERVER_MemoryType memory_type);

/* Convert the NUMA config to form expected by Triton:
NUMA node id and CPU cores are set to a host policy which
is associated with particular Triton model instances based
on device placement.
CPU cores are specified in a range format, i.e. if the NUMA node
has CPU core 0, 2, 4, 5, 6, the following string should be passed to
Triton: "0-0,2-2,4-6"
Note that this function expects the input to be sorted */
std::string GenerateCPUStr(const std::vector<int>& cpuIds);

} // namespace triton_frontend

#endif