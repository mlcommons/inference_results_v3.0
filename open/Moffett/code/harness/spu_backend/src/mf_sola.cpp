/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
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

#include "mf_sola.h"

#include <vector>
#include <fstream>
#include <cstring>
#include <functional>
#include <queue>
#include <array>
#include <unordered_map>
#include <atomic>
#include <chrono>

#include <sola_runtime.h>
#include "multithreading.h"
#include "common.h"
#include "model_executor.h"

#define PERFORMANCE_TEST  0
#if PERFORMANCE_TEST
#include <algorithm>
#include <numeric>
#endif
#define NUM_CORE 4

namespace moffett {
namespace spu_backend {

enum ModelType {
  MODEL_TYPE_INVALID = 0,
  MODEL_TYPE_ONE_CORE = 1,
  MODEL_TYPE_FOUR_CORE_BROADCAST = 2,
  MODEL_TYPE_FOUR_CORE_SPLIT = 3,
};

struct ModelInfo {
  int id = 0;
  ModelType type = MODEL_TYPE_INVALID;
  std::string module_path;
  std::vector<std::unique_ptr<ModelExecutor>> executor;
};

std::atomic<int> g_model_id{0};

bool need_destroy = false;

#if PERFORMANCE_TEST
std::vector<double> inference_return_time;
std::vector<double> inference_finish_time;
#endif

std::mutex model_map_mutex;
std::unordered_map<int, ModelInfo> model_map;

MFSola& MFSola::GetInstance() {
  static MFSola instance;
  return instance;
}

MFSola::MFSola() {
//    ProfilerEnable(true);
  printf("sola runtime version: %s\n", mfrtGetVersionInternal());
}

MFSola::~MFSola() {
//    ProfilerEnable(false);
}

int MFSola::Init(const std::string &name,
                 int32_t device_id,
                 int32_t numa_node,
                 const std::string &module_path,
                 int32_t batch_size,
                 int32_t internal_batch_size,
                 int32_t user_batch_size,
                 const std::vector<size_t> &input_size_list,
                 const std::vector<size_t> &output_size_list,
                 bool four_core,
                 bool broadcast,
                 int stream_num) {
  printf("sola init device:%d, batch_size:%d, input_size:%d, output_size:%d, stream:%d, numa_node:%d\n",
         device_id, batch_size, input_size_list[0], output_size_list[0], stream_num, numa_node);
  int model_id = g_model_id++;
  {
    std::lock_guard<std::mutex> lock(model_map_mutex);
    model_map.emplace(model_id, ModelInfo{});
  }
  ModelInfo& info = model_map[model_id];
  info.id = model_id;
  if (!four_core) {
    // one core
    info.type = MODEL_TYPE_ONE_CORE;
    for (int i = 0; i < NUM_CORE; ++i) {
      info.executor.emplace_back(std::make_unique<ModelExecutor>(device_id,
                                                                 static_cast<MFMode>(MF_MODE_SINGLE_CORE_0 + i),
                                                                 batch_size / 4,
                                                                 input_size_list,
                                                                 output_size_list,
                                                                 user_batch_size,
                                                                 internal_batch_size,
                                                                 stream_num,
                                                                 numa_node
                                                                 ));
      info.executor[i]->Prepare(module_path.data());
    }
  } else {
    if (broadcast) {
      info.type = MODEL_TYPE_FOUR_CORE_BROADCAST;
      info.executor.emplace_back(std::make_unique<ModelExecutor>(
          device_id, MF_MODE_FOUR_CORE_BROADCAST, batch_size, input_size_list, output_size_list, user_batch_size, internal_batch_size, stream_num, numa_node));
    } else {
      info.type = MODEL_TYPE_FOUR_CORE_SPLIT;
      info.executor.emplace_back(std::make_unique<ModelExecutor>(
          device_id, MF_MODE_FOUR_CORE_SPLIT, batch_size, input_size_list, output_size_list, user_batch_size, internal_batch_size, stream_num, numa_node));
    }
    info.executor[0]->Prepare(module_path.data());
  }
  return model_id;
}

void MFSola::Inference(int model_id,
                       int core_id,
                       std::vector<std::vector<uint8_t *>>* input_list,
                       std::vector<std::vector<uint8_t *>>* output_list,
                       uint32_t start,
                       uint32_t samples,
                       std::function<void(void*)> cb,
                       void* cb_data) {
  const ModelInfo& info = model_map[model_id];
  info.executor[core_id]->Run(input_list, output_list, start, samples, cb, cb_data);
}

void MFSola::Destroy(int model_id) {
  const ModelInfo& info = model_map[model_id];
  if (info.type == MODEL_TYPE_ONE_CORE) {
    for (int i = 0; i < NUM_CORE; ++i) {
      info.executor[i]->Destroy();
    }
  } else {
    info.executor[0]->Destroy();
  }
}

void MFSola::Sync(int model_id) {
  const ModelInfo& info = model_map[model_id];
  for (auto& exec : info.executor) {
    exec->Sync();
  }
}

void MFSola::SetInferenceCount(int model_id, uint32_t count, uint32_t device_num) {
  const ModelInfo& info = model_map[model_id];
  for (auto& exec : info.executor) {
    exec->SetInferenceCount(count, device_num);
  }
}

void MFSola::ProfilerEnable(bool enable) {
//  printf("sola profiler enable: %d\n", enable);
  if (enable) {
    mfrtProfilerStart();
  } else {
    mfrtProfilerStop();
  }
}

}
}
