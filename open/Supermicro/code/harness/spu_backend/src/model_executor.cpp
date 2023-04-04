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

#include "model_executor.h"
#include "numa.h"

//#define DUMP_DATA
#ifdef DUMP_DATA
#include <cstdio>
FILE* output_file[4];
#endif

#define SOLA_NUMA 1

static uint64_t total_count = 0;
static uint32_t device_count = 0;

namespace moffett {
namespace spu_backend {

struct MemcpyParams {
  uint8_t *pin;
  std::vector<uint8_t *> cache;
  bool is_h2d;
  uint32_t start_idx;
  uint32_t copy_num;
  uint32_t single_size;
  int stream;
};

static void *pack_memcpy_params(uint8_t *pin,
                                std::vector<uint8_t *> cache,
                                size_t single_size,
                                bool is_h2d,
                                uint32_t start_idx,
                                uint32_t copy_num,
                                int stream = 0) {
  MemcpyParams *params = new MemcpyParams;
  params->pin = pin;
  params->cache = cache;
  params->is_h2d = is_h2d;
  params->start_idx = start_idx;
  params->copy_num = copy_num;
  params->single_size = single_size;
  params->stream = stream;

  return params;
}

static void host_memcpy(void *params) {
  auto memcpy_params = reinterpret_cast<MemcpyParams *>(params);
  for (uint32_t i = memcpy_params->start_idx; i < memcpy_params->start_idx + memcpy_params->copy_num; i++) {
    if (memcpy_params->is_h2d) {
      memcpy(memcpy_params->pin + (i - memcpy_params->start_idx) * memcpy_params->single_size,
             memcpy_params->cache[i],
             memcpy_params->single_size);
    } else {
      memcpy(memcpy_params->cache[i],
             memcpy_params->pin + (i - memcpy_params->start_idx) * memcpy_params->single_size,
             memcpy_params->single_size);
#ifdef DUMP_DATA
      fwrite(memcpy_params->cache[i], sizeof(char), memcpy_params->single_size, output_file[memcpy_params->stream]);
#endif
    }
  }
  delete memcpy_params;
}

ModelExecutor::ModelExecutor(int32_t device_id,
                             MFMode mode,
                             int32_t batch_size,
                             const std::vector<size_t> &input_size,
                             const std::vector<size_t> &output_size,
                             int32_t batches,
                             int32_t internal_batch_size,
                             int stream_num,
                             int numa

) {
  device_id_ = device_id;
  mode_ = mode;
  batch_size_ = batch_size;
  input_num_ = input_size.size();
  output_num_ = output_size.size();
  input_size_ = input_size;
  output_size_ = output_size;
  batches_ = batches;
  internal_batch_size_ = internal_batch_size;
  numa_ = numa;
  stream_num_ = stream_num;

  host_input_buffer_ = std::vector<std::vector<void *>>(stream_num_, std::vector<void *>(input_num_));
  device_input_buffer_ = std::vector<std::vector<void *>>(stream_num_, std::vector<void *>(input_num_));
  host_output_buffer_ = std::vector<std::vector<void *>>(stream_num_, std::vector<void *>(output_num_));
  device_output_buffer_ = std::vector<std::vector<void *>>(stream_num_, std::vector<void *>(output_num_));
  batches_input_ = std::vector<std::vector<uint8_t*>>(input_num_);
  batches_output_ = std::vector<std::vector<uint8_t*>>(output_num_);

  streams_ = std::vector<MFStream>(stream_num_);
#ifdef DUMP_DATA
  for (int i = 0; i < stream_num; ++i) {
    char file_name[100];
    snprintf(file_name, sizeof(file_name), "accu%d.bin", i);
    output_file[i] = fopen(file_name, "wb");
    if (output_file[i]) {
      printf("dump data to file: %s\n", file_name);
    } else {
      printf("opne file failed: %s\n", file_name);
    }
  }
#endif
}

MFResult ModelExecutor::Prepare(const char *module_path) {
  MFResult result = MF_SUCCESS;
  CheckError(mfrtSetDevice(device_id_));
  CheckError(mfrtSetDeviceMode(mode_));

  CheckError(mfrtModuleLoad(&module_, module_path));
  for (int i = 0; i < stream_num_; i++) {
    CheckError(mfrtStreamCreate(&streams_[i]));
  }

  for (int i = 0; i < stream_num_; i++) {
    for (int j = 0; j < input_num_; j++) {
#if SOLA_NUMA
      CheckError(mfrtMallocHost((void **) &host_input_buffer_[i][j], input_size_[j] * batches_, numa_));
#else
      host_input_buffer_[i][j] = MallocHostMemory(input_size_[j] * batches_, numa_);
      if (!host_input_buffer_[i][j]) {
        return MF_ERROR_OUT_OF_MEMORY;
      }
#endif
      CheckError(mfrtMalloc((void **) &device_input_buffer_[i][j], input_size_[j] * batches_));
    }
    for (int j = 0; j < output_num_; j++) {
#if SOLA_NUMA
      CheckError(mfrtMallocHost((void **) &host_output_buffer_[i][j], output_size_[j] * batches_, numa_));
#else
      host_output_buffer_[i][j] = MallocHostMemory(output_size_[j] * batches_, numa_);
      if (!host_output_buffer_[i][j]) {
        return MF_ERROR_OUT_OF_MEMORY;
      }
#endif
      CheckError(mfrtMalloc((void **) &device_output_buffer_[i][j], output_size_[j] * batches_));
    }
  }

  args_list_ = std::vector<std::vector<std::vector<void *>>>(stream_num_);
  for (int i = 0; i < stream_num_; i++) {
    args_list_[i] = std::vector<std::vector<void *>>(batches_);
    for (int bs = 0; bs < batches_; bs++) {
      args_list_[i][bs] = std::vector<void*>((input_num_ + output_num_)*2);
      for (int j = 0; j < input_num_; j++) {
        args_list_[i][bs][j] = (uint8_t*)device_input_buffer_[i][j];
      }
      for (int j = 0; j < output_num_; j++) {
        args_list_[i][bs][j + input_num_] = (uint8_t*)device_output_buffer_[i][j];
      }
      for (int j = 0; j < input_num_; j++) {
        args_list_[i][bs][j + input_num_ + output_num_] = (void*)(uint64_t)(bs * input_size_[j]);
      }
      for (int j = 0; j < output_num_; j++) {
        args_list_[i][bs][j + input_num_ + output_num_ + input_num_] = (void*)(uint64_t)(bs * output_size_[j]);
      }
    }
  }
  return result;
}

void progress(void* data) {
  uint64_t cur = (uint64_t)data;
  printf("Inference progress: %lu tasks\n",
         cur);
}

struct infer_param {
  std::function<void(void*)> cb;
  void* data;
};

MFResult ModelExecutor::Run(std::vector<std::vector<uint8_t *>>* input,
                            std::vector<std::vector<uint8_t *>>* output,
                            uint32_t start,
                            uint32_t samples,
                            std::function<void(void*)> cb,
                            void* cb_data) {
  static uint64_t count = 0;
  MFResult result = MF_SUCCESS;
  CheckError(mfrtSetDevice(device_id_));
  CheckError(mfrtSetDeviceMode(mode_));
  uint32_t infer_num = samples;
  uint32_t single_iter = batch_size_ * batches_;
  for (int i = 0; i < input_num_; i++) {
    for (int j = start; j < start + samples; j++) {
      batches_input_[i].emplace_back((*input)[i][j]);
    }
  }
  for (int i = 0; i < output_num_; i++) {
    for (int j = start; j < start + samples; j++) {
      batches_output_[i].emplace_back((*output)[i][j]);
    }
  }

  int real_batches = 1;
  if (mode_ >= MF_MODE_SINGLE_CORE_0 && mode_ <= MF_MODE_SINGLE_CORE_3) {
    real_batches = batches_input_[0].size() / (batch_size_ );
  }
  uint32_t iter = batches_input_[0].size() / (batch_size_ * batches_) >= 1 ? batches_input_[0].size() / (batch_size_ * batches_) : 1;
//    printf("device[%d]mode[%d] infer: num=%d, batch_size=%d, batches=%d, single_iter=%d, iter=%d\n",
//           device_id_, mode_, infer_num, batch_size_, batches_, single_iter, iter);
  for (int it = 0; it < iter; it++) {
    for (int j = 0; j < input_num_; j++) {
      CheckError(mfrtLaunchHostFunc(streams_[last_stream_], host_memcpy,
                                    pack_memcpy_params((uint8_t *) host_input_buffer_[last_stream_][j],
                                                       batches_input_[j],
                                                       input_size_[j] / batch_size_,
                                                       true,
                                                       it * batch_size_ * real_batches,
                                                       batch_size_ * real_batches)
      ));
      CheckError(mfrtMemcpyAsync(device_input_buffer_[last_stream_][j], host_input_buffer_[last_stream_][j],
                                 input_size_[j] * real_batches, MF_MEMCPY_HOST_TO_DEVICE, streams_[last_stream_]));
    }
    for (int batch = 0; batch < real_batches; ++batch) {
      CheckError(mfrtLaunchKernel(module_,
                                  ph_,
                                  ph_,
                                  (void **) (args_list_[last_stream_][batch].data()),
                                  0,
                                  streams_[last_stream_]));
    }
    for (int j = 0; j < output_num_; j++) {
      CheckError(mfrtMemcpyAsync(host_output_buffer_[last_stream_][j],
                                 device_output_buffer_[last_stream_][j],
                                 output_size_[j] * real_batches,
                                 MF_MEMCPY_DEVICE_TO_HOST,
                                 streams_[last_stream_]));
      CheckError(mfrtLaunchHostFunc(streams_[last_stream_], host_memcpy,
                                    pack_memcpy_params(
                                        (uint8_t *) host_output_buffer_[last_stream_][j],
                                        batches_output_[j],
                                        output_size_[j] / batch_size_,
                                        false,
                                        it * batch_size_ * real_batches,
                                        batch_size_ * real_batches,
                                        last_stream_)
      ));
    }
    if (it == iter - 1 && cb && cb_data) {
      auto param = new infer_param;
      param->cb = cb;
      param->data = cb_data;
      CheckError(mfrtLaunchHostFunc(streams_[last_stream_], [](void* data) {
        auto param = (infer_param*)data;
        if (param->cb) param->cb(param->data);
        delete param;
      }, param));
    }
    last_stream_ = (last_stream_ + 1) % stream_num_;
  }

  for (int i = 0; i < input_num_; i++) {
    batches_input_[i].clear();
  }
  for (int i = 0; i < output_num_; i++) {
    batches_output_[i].clear();
  }

//  uint32_t recently_used_stream = last_stream_ == 0 ? stream_num_ - 1 : last_stream_ - 1;
//  CheckError(mfrtLaunchHostFunc(streams_[recently_used_stream], progress, (void*)count++));

  return result;
}

MFResult ModelExecutor::Sync() {
  MFResult result = MF_SUCCESS;
  CheckError(mfrtSetDevice(device_id_));
  CheckError(mfrtDeviceSynchronize());
//  for (auto& s : streams_) {
//    mfrtStreamSynchronize(s);
//  }
}

MFResult ModelExecutor::Destroy() {
  MFResult result = MF_SUCCESS;
  CheckError(mfrtSetDevice(device_id_));
  CheckError(mfrtSetDeviceMode(mode_));
  CheckError(mfrtModuleUnload(module_));
  for (int i = 0; i < stream_num_; i++) {
    CheckError(mfrtStreamDestroy(streams_[i]));
    for (int j = 0; j < input_num_; j++) {
      CheckError(mfrtFree(device_input_buffer_[i][j]));
#if SOLA_NUMA
      CheckError(mfrtFreeHost(host_input_buffer_[i][j], numa_));
#else
      FreeHostMemory(host_input_buffer_[i][j], input_size_[j] * batches_, numa_);
#endif
    }
    for (int j = 0; j < output_num_; j++) {
      CheckError(mfrtFree(device_output_buffer_[i][j]));
#if SOLA_NUMA
      CheckError(mfrtFreeHost(host_output_buffer_[i][j], numa_));
#else
      FreeHostMemory(host_output_buffer_[i][j], output_size_[j] * batches_, numa_);
#endif
    }
  }
  return result;
}

uint32_t ModelExecutor::SetInferenceCount(uint32_t inference_count, uint32_t device_num) {
  inference_count_ = inference_count;
  total_count = inference_count;
  device_count = device_num;
}

}
}
