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

#ifndef SPU_BACKEND_MFSOLA_H
#define SPU_BACKEND_MFSOLA_H

#include <iostream>
#include <vector>
#include <pthread.h>
#include <memory>
#include <functional>

namespace moffett {
namespace spu_backend {

class MFSola {
 public:
  MFSola();
  ~MFSola();

  static MFSola& GetInstance();

  int Init(const std::string &name,
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
           int stream_num = 4);

  void Inference(int model_id,
                 int core_id,
                 std::vector<std::vector<uint8_t *>>* input_list,
                 std::vector<std::vector<uint8_t *>>* output_list,
                 uint32_t start_index,
                 uint32_t samples_count,
                 std::function<void(void*)> cb = {},
                 void* cb_data = nullptr);

  void Destroy(int model_id);

  void Sync(int model_id);

  void SetInferenceCount(int model_id, uint32_t count, uint32_t device_num);

  void ProfilerEnable(bool enable);
};

}
}
#endif // SPU_BACKEND_MFSOLA_H