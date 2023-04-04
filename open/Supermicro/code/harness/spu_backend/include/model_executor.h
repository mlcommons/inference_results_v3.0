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

#ifndef SPU_BACKEND_MODEL_EXECUTOR_H
#define SPU_BACKEND_MODEL_EXECUTOR_H

#include <vector>
#include <functional>

#include "common.h"

namespace moffett {
namespace spu_backend {

class ModelExecutor {
 public:
  ModelExecutor(int32_t device_id, MFMode mode, int32_t batch_size,
                const std::vector<size_t>& input_size, const std::vector<size_t>& output_size,
                int32_t batches, int32_t internal_batch_size, int stream_num = 4, int numa = 0);

  MFResult Prepare(const char* module_path);

  MFResult Run(std::vector<std::vector<uint8_t *>>* cur_input_list,
               std::vector<std::vector<uint8_t *>>* cur_output_list,
               uint32_t start,
               uint32_t samples,
               std::function<void(void*)> cb,
               void* cb_data);

  MFResult Sync();

  MFResult Destroy();

  int GetStreamNum() const { return stream_num_; }

  uint32_t SetInferenceCount(uint32_t inference_count, uint32_t device_num);

 private:
  int32_t device_id_ = 0;
  int32_t numa_ = 0;
  MFMode mode_;
  uint32_t inference_count_ = 0;
  int32_t input_num_;
  int32_t output_num_;
  int32_t batch_size_;
  int32_t internal_batch_size_;
  std::vector<size_t> input_size_; // input_num, notice input_size means a whole batch size
  std::vector<size_t> output_size_;

  MFModule module_;
  MFDims3 ph_;
  int32_t batches_ = 1;
  int32_t user_batch_size_;
  int32_t cur_batch = 0;
  std::vector<std::vector<std::vector<void *>>> args_list_;

  int stream_num_ = 4;
  std::vector<MFStream> streams_;
  std::vector<MFStream> memcpy_stream_;
  std::vector<MFEvent>  memcpy_event_;

  std::vector<std::vector<void *>> host_input_buffer_; // stream_num * input_num
  std::vector<std::vector<void *>> host_output_buffer_;
  std::vector<std::vector<void *>> device_input_buffer_; // stream_num * output_num
  std::vector<std::vector<void *>> device_output_buffer_;

  uint32_t last_stream_ = 0;

  std::vector<std::vector<uint8_t*>> batches_input_;
  std::vector<std::vector<uint8_t*>> batches_output_;
};

}
}

#endif // SPU_BACKEND_MODEL_EXECUTOR_H
