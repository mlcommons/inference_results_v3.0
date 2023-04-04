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

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mf_sola.h>
#include "loadgen.h"
#include "samples_loader.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "utils.hpp"
#include "timer.h"

#include "syncqueue.h"

namespace moffett {
namespace spu_backend {

using namespace std::chrono_literals;
using namespace moffett;

class Server;
class Device;
struct BackendSettings;
struct Batch;

using BindingHandle = uint8_t *;
using InOutBindingType = std::vector<std::vector<uint8_t *>>;
using InOutBindingHandle = InOutBindingType *;
using BatchUniquePtr = std::unique_ptr<Batch>;
using DevicePtr_t = std::shared_ptr<Device>;
using ServerPtr_t = std::shared_ptr<Server>;
using NetSetupFun = std::function<bool(const BackendSettings &)>;
using InputDataPrepareFun = std::function<void(
    InOutBindingHandle input_bindings, const std::vector<std::size_t> &real_input_size, Batch *batch,
    std::size_t batch_size, std::size_t device_idx,
    const BackendSettings &server_settings, const std::vector<std::size_t> &sample_size)>;
using ResultProcessFun = std::function<void(Batch *batch, InOutBindingHandle infer_result_data,
                                            const std::vector<std::size_t> &output_data_size,
                                            std::size_t single_batch_size)>;

struct Batch {
  Batch();
  Batch(const Batch &b);
  Batch(Batch &&b);

  ~Batch();

  // We want to move this resource around without copying them.
  std::unique_ptr<std::vector<mlperf::QuerySampleResponse>> responses{nullptr};
  std::unique_ptr<std::vector<mlperf::QuerySampleIndex>> sampleIds{nullptr};

  // Input idx --> std::vector<void*>
  std::vector<std::vector<uint8_t *>> datas;

  std::vector<std::vector<uint8_t *>> *input_bindings{nullptr};
  std::vector<std::vector<uint8_t *>> *output_bindings{nullptr};

  void *padding_output_data;
  void *padding_output_mask;
  void *trunc_dict;
};

struct BackendSettings {
  int mode = 0;

  int32_t device_id;
  int32_t batch_size;
  std::string model_path;
  std::vector<std::size_t> devices;
  std::vector<int> numa_map;
  // one batch size
  std::vector<size_t> inputs_size;
  std::vector<size_t> outputs_size;

  std::vector<size_t> real_inputs_size;
  std::vector<size_t> real_outputs_size;

  std::chrono::microseconds Timeout{10000us};
  std::chrono::microseconds target_latency_ms{10000us};

  bool four_core = false;
};

struct ServerParams {
  std::string device_ids;
  std::string scenario;
  std::vector<std::vector<std::vector < std::string>>> EngineNames;
};

// captures execution engine for performing inference
class Device {
 public:
  Device(const std::string &name, int device_id, const BackendSettings &settings,
         const ResultProcessFun &result_process, const InputDataPrepareFun &data_process)
      : mlperf_mode_(settings.mode),
        name_(name),
        device_id_(device_id),
        numa_node_(settings.numa_map[device_id]),
        batch_size_(settings.batch_size),
        module_path_(settings.model_path),
        inputs_size_(settings.inputs_size),
        outputs_size_(settings.outputs_size),
        real_inputs_size_(settings.real_inputs_size),
        real_outputs_size_(settings.real_outputs_size),
        four_core_(settings.four_core),
        server_settings_(settings),
        result_processor_(result_process),
        data_preparer_(data_process) {
    _setup();
  }

  ~Device();

  void infer(BatchUniquePtr batch)
  noexcept;
  void OfflineInfer(BatchUniquePtr batch)
  noexcept;
  std::size_t getInputNum() const
  noexcept { return real_input_size_.size(); }
  std::size_t getOutputNum() const
  noexcept { return real_output_size_.size(); }

  std::size_t getBatchSize() const
  noexcept { return batch_size_; }

  uint32_t getDeviceIdx() const
  noexcept { return device_id_; }
  int getNextCore();
  void reset()
  noexcept;
  void done()
  noexcept;

 private:
  std::string _generateThreadNameWithCardId(const std::string &thread_key);

  // 1st step: fetch ready engine from queue and perfrom run on engine. Then add this batch to this engine
  //           and move this engine into completion queue.
  void _dataPrepare();

  // 2nd step: fetch ready engine from queue and perfrom run on engine. Then add this batch to this engine
  //           and move this engine into completion queue.
  void _performInfer()
  noexcept;

  // 3rd step: fetch engine from completion queue and sync stream and call QuerySamplesComplete with callback.
  void _setup()
  noexcept;
  void _initInputOutputSizes()
  noexcept;
  void _allocateBuff(BindingHandle &bindings, std::size_t size, bool on_device)
  noexcept;

 private:
  int mlperf_mode_ = 0;
  int32_t device_id_;
  int numa_node_ = 0;
  int32_t batch_size_;
  std::string module_path_;
  std::string name_;
  int32_t user_batch_size_;
  std::vector<size_t> inputs_size_;
  std::vector<size_t> outputs_size_;
  std::vector<size_t> real_inputs_size_;
  std::vector<size_t> real_outputs_size_;
  int core_ = 0;
  bool four_core_;
  bool broadcast_;
  int stream_num_ = 4;
  int model_id_;
  std::vector<BindingHandle> allocate_data_;

  std::unique_ptr<std::thread> datacombine_thread_;
  // Completion management
  std::unique_ptr<std::thread> infer_thread_;

  // input_idx -> data
  QUEUE_TYPE<BatchUniquePtr> request_queue_;
  QUEUE_TYPE<BatchUniquePtr> copied_data_queue_;

  std::vector<InOutBindingHandle> binding_buffers_; // used at destruct

  const BackendSettings &server_settings_;

  // input_idx -> sample size
  std::vector<std::size_t> sample_size_;
  std::vector<std::size_t> real_input_size_;
  std::vector<std::size_t> real_output_size_;
  // number of input
  ResultProcessFun result_processor_;
  InputDataPrepareFun data_preparer_;
};

// Create buffers and other execution resources.
// Perform queuing, batching, and manage execution resources.
class Server : public mlperf::SystemUnderTest {
 public:
  // Query management
  using BatchQueue = QUEUE_TYPE<std::vector<mlperf::QuerySample *>*>;

  Server(std::string name) : name_(name) {}
  ~Server() {
    for (auto &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void AddSampleLibrary(std::size_t device_idx, samplesLoader::SampleLoaderPtr sl)
  noexcept {
    if (device_idx >= sample_libraries_.size()) {
      sample_libraries_.resize(device_idx + 1);
    }
    sample_libraries_[device_idx] = sl;
  }

  void Setup(const BackendSettings &settings, const ServerParams &params,
             ResultProcessFun result_process, InputDataPrepareFun data_prepare)
  noexcept;
  void Done()
  noexcept;
  const std::vector<DevicePtr_t> &GetDevices() const
  noexcept { return devices_; }

  // SUT virtual interface
  const std::string &Name() const override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override;
  void FlushQueries() override;

 private:

  template<typename T>
  void MergeQueryDispatcher(QUEUE_TYPE<T *> *queue)
  noexcept;

  inline void BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                         std::vector<mlperf::QuerySample *>::iterator begin,
                         std::vector<mlperf::QuerySample *>::iterator end, std::size_t samples_cnt,
                         Batch &batch)
  noexcept;
  inline void IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample *>::iterator begin,
                         std::vector<mlperf::QuerySample *>::iterator end)
  noexcept;

  DevicePtr_t GetNextDispatchDevice()
  noexcept;
  void Reset()
  noexcept;

 private:
  const std::string name_;
  std::size_t device_index_;
  std::vector<std::size_t> device_idxs_;
  std::vector<DevicePtr_t> devices_;
  std::size_t valid_device_num_{0};
  std::size_t input_num_;
  std::vector<samplesLoader::SampleLoaderPtr> sample_libraries_;

  BackendSettings server_settings_;
  std::string scenario_;

  BatchQueue request_queue_;

  // Not use shared_ptr for performance considerations.
  std::vector<std::thread> threads_;
};

}
}
