/*
 * Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
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

#include <algorithm>
#include <fstream>
#include <glog/logging.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <unistd.h>
#include <omp.h>
#include <hwloc.h>
#include "common.h"
#include "server.h"
#include "loadgen.h"
#include "query_sample_library.h"

#include "timer.h"

uint64_t g_samples_size = 0;
using namespace std::chrono_literals;

namespace moffett {
namespace spu_backend {

Batch::Batch() {
  responses.reset(std::make_unique<std::vector<mlperf::QuerySampleResponse>>().release());
  sampleIds.reset(std::make_unique<std::vector<mlperf::QuerySampleIndex>>().release());
}

Batch::Batch(const Batch& b)
    : responses(std::make_unique<std::vector<mlperf::QuerySampleResponse>>(*b.responses)),
      sampleIds(std::make_unique<std::vector<mlperf::QuerySampleIndex>>(*b.sampleIds)),
      datas(b.datas) {}
Batch::Batch(Batch&& b) : responses(b.responses.release()), sampleIds(b.sampleIds.release()), datas(std::move(b.datas)) {}
Batch::~Batch() {}

Device::~Device() {
  auto wait_thread_end = [](const std::unique_ptr<std::thread>& t) {
    if (!t) {
      return;
    }
    if (t->joinable()) {
      t->join();
    }
  };

  wait_thread_end(datacombine_thread_);
  wait_thread_end(infer_thread_);

  for (auto&&p : allocate_data_) {
    free(p);
    p = nullptr;
  }
  allocate_data_.clear();

  for (auto&& p : binding_buffers_) {
    delete p;
  }
  binding_buffers_.clear();
}

//----------------
// Device
//----------------
int Device::getNextCore() {
  if (four_core_) {
    return core_;
  }
  return core_++ % 4;
}

void Device::infer(BatchUniquePtr batch) noexcept {
  PUSH_QUEUE(request_queue_, std::move(batch));
}

void Device::OfflineInfer(BatchUniquePtr batch) noexcept {
  if (batch->sampleIds->empty()) {
    return;
  }
  InOutBindingHandle input_binding = nullptr;
  InOutBindingHandle output_bindings = nullptr;
  {
    TIME("_dataPrepare data prepare: ", device_id_);
    data_preparer_(input_binding, real_input_size_, batch.get(), batch_size_, device_id_,
                   server_settings_, sample_size_);
  }

  // allocate output data
  output_bindings = new InOutBindingType;
  binding_buffers_.push_back(output_bindings);
  output_bindings->resize(real_output_size_.size());
  size_t batch_offline_count = (*(batch->input_bindings))[0].size();
  for (int isize = 0; isize < real_output_size_.size(); ++isize) {
    (*output_bindings)[isize].resize(batch_offline_count);
    BindingHandle output_addrs;
    _allocateBuff(output_addrs, real_output_size_[isize] * batch_offline_count, false);
    for (int idx = 0; idx < batch_offline_count; ++idx) {
      (*output_bindings)[isize][idx] = output_addrs + idx * real_output_size_[isize];
    }
  }

  // begin infer
  printf("device %d OfflineInfer start time : %f s\n", device_id_, get_nanosecond() / 1000000000.0);
  if (!four_core_) {
    int step = 4 * user_batch_size_;
    uint32_t cur = 0;
    uint32_t iter_count = batch_offline_count / 4 / step;
    uint32_t total_count = iter_count * step * 4;
    uint32_t remain_count = batch_offline_count - total_count;
    auto start = std::chrono::high_resolution_clock::now();

    // four core do step count
    for (size_t iter = 0; iter < iter_count; ++iter) {
      for (int core = 0; core < 4; ++core) {
        spu_backend::MFSola::GetInstance().Inference(model_id_, core, batch->input_bindings, output_bindings, cur, step);
        cur += step;
      }
    }

    // some core do step count
    int core = 0;
    for (; core < 4 && remain_count > step; ++core) {
      spu_backend::MFSola::GetInstance().Inference(model_id_, core, batch->input_bindings, output_bindings, cur, step);
      cur += step;
      remain_count -= step;
    }

    // last core do remain count
    remain_count = (remain_count / 4) * 4;
    if (remain_count > 0) {
      spu_backend::MFSola::GetInstance().Inference(model_id_, core, batch->input_bindings, output_bindings, cur, remain_count);
    }
    spu_backend::MFSola::GetInstance().Sync(model_id_);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    printf("Device[%d] inference samples: %d, costs %f s, FPS = %f\n", device_id_, total_count, time_span.count(), total_count / time_span.count());
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    spu_backend::MFSola::GetInstance().Inference(model_id_, 0, batch->input_bindings, output_bindings, 0, batch_offline_count);
    spu_backend::MFSola::GetInstance().Sync(model_id_);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    uint32_t total_count = batch_offline_count;
    printf("Device[%d] inference samples: %d, costs %f s, FPS = %f\n", device_id_, total_count, time_span.count(), total_count / time_span.count());
  }

  {
    TIME("_completion result process:", device_idx_);
    if (result_processor_) {
      result_processor_(batch.get(), output_bindings, real_output_size_, batch_size_);
    }
  }

}

void Device::reset() noexcept {
}

void Device::_initInputOutputSizes() noexcept {
  real_input_size_ = inputs_size_;
  real_output_size_ = outputs_size_;
}

void Device::_allocateBuff(BindingHandle& bindings, std::size_t size, bool on_device) noexcept {
  bindings = (BindingHandle)malloc(size);
  allocate_data_.push_back(bindings);
}

void Device::_setup() noexcept {
  _initInputOutputSizes();

  std::vector<size_t> input_size;
  std::vector<size_t> output_size;
  bool four_core = false;
  bool broadcast = false;
  int32_t batch_size = 16;
  int32_t internal_batch = 1;
  user_batch_size_ = 1;
  if (name_.find("RN50") != std::string::npos) {
    // rn50
    for (auto && s : inputs_size_) {
      input_size.push_back(s * 4);
    }
    for (auto && s : outputs_size_) {
      output_size.push_back(s * 4);
    }
    user_batch_size_ = mlperf_mode_ == 0 ? 32 : 8;
  } else if (name_.find("Bert") != std::string::npos) {
    for (auto && s : real_inputs_size_) {
      input_size.push_back(s * 32);
    }
    for (auto && s : real_outputs_size_) {
      output_size.push_back(s * 32);
    }
    four_core = true;
    batch_size = 32;
  } else if (name_.find("Retinanet") != std::string::npos) {
    for (auto && s : inputs_size_) {
      input_size.push_back(s);
    }
    for (auto && s : outputs_size_) {
      output_size.push_back(s);
    }
    four_core = true;
    broadcast = true;
    batch_size = 1;
    user_batch_size_ = 1;
  }
  model_id_ = spu_backend::MFSola::GetInstance().Init(name_, device_id_, numa_node_, module_path_,
                                                      batch_size, internal_batch, user_batch_size_,
                                                      input_size, output_size, four_core, broadcast);

  datacombine_thread_ = std::make_unique<std::thread>(&Device::_dataPrepare, this);

  infer_thread_ = std::make_unique<std::thread>(&Device::_performInfer, this);

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  hwloc_topology_t topology;
  hwloc_obj_t obj;
  hwloc_cpuset_t hw_cpuset;
  // Initialize topology
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  // Get the NUMA node object
  obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NODE, numa_node_);
  if (obj) {
    // Get the cpuset of the NUMA node
    hw_cpuset = hwloc_bitmap_dup(obj->cpuset);
    // Print the cpuset
    for (int i = 0; i <= hwloc_bitmap_last(hw_cpuset); i++) {
      if (hwloc_bitmap_isset(hw_cpuset, i)) {
        CPU_SET(i, &cpuset);
      }
    }

    int res = pthread_setaffinity_np(datacombine_thread_->native_handle(), sizeof(cpu_set_t), &cpuset);
    if (res != 0) {
      printf("pthread_setaffinity_np error!\n");
    }
    res = pthread_setaffinity_np(infer_thread_->native_handle(), sizeof(cpu_set_t), &cpuset);
    if (res != 0) {
      printf("pthread_setaffinity_np error!\n");
    }
    hwloc_bitmap_free(hw_cpuset);
  }
  hwloc_topology_destroy(topology);
}

void Device::done() noexcept {
  // We can use nullptr to identify the end of work.
  PUSH_QUEUE(request_queue_, std::make_unique<Batch>());

  PUSH_QUEUE(copied_data_queue_, std::make_unique<Batch>());
}

std::string Device::_generateThreadNameWithCardId(const std::string& thread_key) {
  std::string t_name = thread_key + std::to_string(device_id_);
  return t_name;
}

void Device::_dataPrepare() {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("dPrep").c_str());

  auto batch = std::make_unique<Batch>();
  for (;;) {
    {
      TIME("_dataPrepare get request: ", device_id_);
      batch = POP_QUEUE(request_queue_);
      if (batch->sampleIds->empty()) {
        break;
      }
    }
    if (server_settings_.mode == 0 && name_.find("RN50") != std::string::npos) {
      OfflineInfer(std::move(batch));
      continue;
    }
    {
      TIME("_dataPrepare data prepare: ", device_id_);
      data_preparer_(nullptr, real_input_size_, batch.get(), batch_size_, device_id_,
                     server_settings_, sample_size_);
    }

    {
      TIME("_combineMempush batch back:", device_idx_);
      PUSH_QUEUE(copied_data_queue_, std::move(batch));
    }
  }
}

struct callback_param {
  std::unique_ptr<Batch> batch;
  InOutBindingType* output;
  ResultProcessFun func;
  std::vector<std::size_t> real_output_size;
  size_t batch_size;
};

callback_param* gen_cb_param(std::unique_ptr<Batch> batch,
                             InOutBindingType* output,
                             ResultProcessFun func,
                             std::vector<std::size_t> real_output_size,
                             size_t batch_size) {
  auto param = new callback_param;
  param->batch = std::move(batch);
  param->output = output;
  param->func = func;
  param->real_output_size = std::move(real_output_size);
  param->batch_size = batch_size;
  return param;
}

void Device::_performInfer() noexcept {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("Infer").c_str());

  auto batch = std::make_unique<Batch>();
  InOutBindingHandle output_bindings = nullptr;
  for (;;) {
    {
      TIME("_performInfer get infer data:", device_idx_);
      batch = POP_QUEUE(copied_data_queue_);
      if (!batch || batch->sampleIds->empty()) {
        break;
      }
    }

    {
      TIME("_performInfer infer:", device_idx_);

      // allocate output data
      output_bindings = new InOutBindingType;
      output_bindings->resize(real_output_size_.size());
      size_t batch_count = (*(batch->input_bindings))[0].size();
      for (int isize = 0; isize < real_output_size_.size(); ++isize) {
        (*output_bindings)[isize].resize(batch_count);
        BindingHandle output_addrs = (BindingHandle)malloc(real_output_size_[isize] * batch_count);
        for (int idx = 0; idx < batch_count; ++idx) {
          (*output_bindings)[isize][idx] = output_addrs + idx * real_output_size_[isize];
        }
      }
      batch->output_bindings = output_bindings;

       auto cb = [](void* param) {
        auto data = (callback_param*)param;
        if (data->func) {
          data->func(data->batch.get(), data->batch->output_bindings, data->real_output_size, data->batch_size);
          delete data->output;
        }
        delete data;
      };
      auto input_bindings = batch->input_bindings;
      spu_backend::MFSola::GetInstance().Inference(model_id_, core_, input_bindings, output_bindings, 0, batch_count,
                   cb, gen_cb_param(std::move(batch), output_bindings, result_processor_, real_output_size_, batch_size_));
      if (!four_core_) {
        core_ = (core_ + 1) % 4;
      }
    }
  }
}

void Server::Setup(const BackendSettings& settings, const ServerParams& params,
                   ResultProcessFun result_process, InputDataPrepareFun data_prepare) noexcept {
  server_settings_ = settings;
  scenario_ = params.scenario;
  device_idxs_ = server_settings_.devices;

  Reset();

  // We assume there will be no device failures during running.
  std::vector<std::thread> device_creators;
  valid_device_num_ = device_idxs_.size();

  size_t max_size_per_one_card = 0;

  device_creators.reserve(valid_device_num_ + 1);
  auto creator = [&](size_t device_idx) {
    devices_.emplace_back(std::make_shared<Device>(name_ + "_" + std::to_string(device_idx), device_idx,
                                                   server_settings_, result_process, data_prepare));
  };
  for (auto device_idx : device_idxs_) {
    device_creators.emplace_back(std::thread(creator, device_idx));
  }

  for (auto& t : device_creators) {
    if (t.joinable()) {
      t.join();
    }
  }

  // create batchers/dispatcher
  {
    threads_.emplace_back(
      std::thread(&Server::MergeQueryDispatcher<std::vector<mlperf::QuerySample*>>, this, &request_queue_));
  }
}

void Server::Done() noexcept {
  for (auto& device : devices_) {
    if (device) {
      device->done();
    }
  }

  std::size_t thread_idx = 0;
  for (auto& thread : threads_) {
    std::vector<mlperf::QuerySample*>* empty_samples = new std::vector<mlperf::QuerySample*>();
    empty_samples->push_back(std::make_unique<mlperf::QuerySample>().release());
    PUSH_QUEUE(request_queue_, empty_samples);
    ++thread_idx;
  }

  // join after we insert the dummy sample
  for (auto& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void Server::Reset() noexcept {
  device_index_ = 0;

  for (auto& device : devices_) {
    device->reset();
  }
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  {
    TIME("IssueQuery", 1);
    std::vector<mlperf::QuerySample*>* tmp_samples = new std::vector<mlperf::QuerySample*>();
    tmp_samples->reserve(samples.size());

    for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
      tmp_samples->push_back((mlperf::QuerySample*)&(*iter));
    }
    if (server_settings_.mode == 0 && (name_.find("RN50") != std::string::npos || name_.find("Retinanet") != std::string::npos))
    {  // Offline
      printf("[Offline] IssueQuery, samples size: %lu\n", samples.size());
      auto vir_devices = GetDevices();
      size_t begin = 0;
      size_t batch_size = samples.size() / vir_devices.size();
      size_t residue = samples.size() % vir_devices.size();
      size_t expand_batch = batch_size + 1;

      for (size_t i = 0; i < vir_devices.size(); ++i) {
        if (residue > 0 && i < residue) {
          IssueBatch(vir_devices[i], expand_batch, tmp_samples->begin() + begin, tmp_samples->begin() + begin + expand_batch);
          begin += expand_batch;
        } else {
          IssueBatch(vir_devices[i], batch_size, tmp_samples->begin() + begin, tmp_samples->begin() + begin + batch_size);
          begin += batch_size;
        }
      }
      delete tmp_samples;
    }
    else
      PUSH_QUEUE(request_queue_, tmp_samples);
  }
}

DevicePtr_t Server::GetNextDispatchDevice() noexcept {
  auto vir_devices = GetDevices();
  auto dev_index = device_index_;
  device_index_ = (device_index_ + 1) % vir_devices.size();
  return vir_devices[dev_index];
}

template<typename T>
void Server::MergeQueryDispatcher(QUEUE_TYPE<T*>* queue) noexcept {
  pthread_setname_np(pthread_self(), std::string("dispatcher").c_str());

  std::deque<T*> samples;
  auto device = GetNextDispatchDevice();
  std::vector<mlperf::QuerySample*> tmp_samples;
  tmp_samples.reserve(device->getBatchSize());
  bool quit{false};
  auto first_query_arrive_time = std::chrono::steady_clock::now();
  auto timeout = server_settings_.target_latency_ms - std::chrono::microseconds{8000us};

  for (; !quit;) {
    samples.clear();
    {
      ACQUIRE_QUEUE_TIMEOUT(request_queue_, samples, timeout);
    }

    if (samples.empty() && tmp_samples.empty()) {
      continue;
    }

    // merge samples
    {
      TIME("process samples", 99);
      auto now_time = std::chrono::steady_clock::now();
      // Use a null (0) id to represent the end of samples
      if (tmp_samples.empty()) {
        first_query_arrive_time = now_time;
      }

      auto batch_size = device->getBatchSize();

      for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
        std::vector<mlperf::QuerySample*>* p_sample_vec = *iter;
        for (auto p_sample : *p_sample_vec) {
          // Use a null (0) id to represent the end of samples
          if (!p_sample->id) {
            quit = true;
            break;
          }
          tmp_samples.push_back(p_sample);
          g_samples_size++;
          if (tmp_samples.size() >= batch_size) {
            {
              TIME("issue batch", 99);
              IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
            }
            device = GetNextDispatchDevice();
            batch_size = device->getBatchSize();
            tmp_samples.clear();
            // refresh timer
            first_query_arrive_time = now_time;
          }
        }
        delete p_sample_vec;
      }

      // dispatch when full or timedout
      if ((now_time - first_query_arrive_time >= timeout) || (tmp_samples.size() >= device->getBatchSize())) {
        auto batch_size = std::min(device->getBatchSize(), tmp_samples.size());
        if (batch_size > 0) {
          TIME("issue batch", 99);
          IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
        }
        device = GetNextDispatchDevice();
        tmp_samples.clear();
      }
    }
  }
}

void Server::IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end) noexcept {
  assert(!sample_libraries_.empty());
  std::size_t device_idx = device->getDeviceIdx();
  auto batch = std::make_unique<Batch>();

  // This can be member viriable which is set in setup.
  std::size_t input_num = device->getInputNum();
  if (input_num == 0) {
    return;
  }

  batch->datas.resize(input_num);
  batch->responses->reserve(batch_size);
  batch->sampleIds->reserve(batch_size);

  for (auto itr = begin; itr < end; ++itr) {
    batch->responses->push_back({(*itr)->id, 0, 0});
    batch->sampleIds->emplace_back((*itr)->index);
  }

  for (std::size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    BuildBatch(sample_libraries_[device_idx], input_idx, device_idx, begin, end, batch_size, *batch);
  }
  device->infer(std::move(batch));
}

void Server::BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                        std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end, std::size_t samples_cnt,
                        Batch& batch) noexcept {
  auto& data_ref = batch.datas[input_idx];
  data_ref.resize(samples_cnt);
  std::size_t sample_idx = 0;
  for (auto itr = begin; itr < end; ++itr, ++sample_idx) {
    auto data_adders = sl->GetSampleAddress((*itr)->index, input_idx, server_settings_.numa_map[device_idx], device_idx);
    data_ref[sample_idx] = data_adders;
  }
}

void Server::FlushQueries() {
  // Not used.
}

} // namespace spu_backend
}