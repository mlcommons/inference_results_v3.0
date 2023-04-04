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
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <cstdlib>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <map>
#include <unordered_map>
#include <numeric>
#include <deque>
#include <vector>

#include <glog/logging.h>

#include "query_sample_library.h"
#include "test_settings.h"
#include "utils.hpp"
#include "numa.h"
#include "sola_runtime.h"

namespace samplesLoader {

using DataTransformer = std::function<void(char*, std::size_t)>;

class PerfSampleLoader : public mlperf::QuerySampleLibrary {
public:
  virtual uint8_t* GetSampleAddress(mlperf::QuerySampleIndex sample_index, std::size_t input_idx, int numa,
                                              std::size_t device_idx = 0) = 0;
  virtual std::size_t GetSampleSize(std::size_t input_idx) const = 0;
  virtual std::size_t GetSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept = 0;
  virtual std::size_t GetOriginSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept = 0;
  virtual std::vector<std::size_t>& GetAllSampleLenOfInputRef(std::size_t input_idx) noexcept = 0;
  virtual std::vector<std::size_t> GetAllSampleLenOfInput(std::size_t input_idx) noexcept = 0;
  virtual std::size_t getInputSizes(std::size_t input_idx) const = 0;
  virtual void setDataTranformerForInput(std::size_t input_idx, DataTransformer tranformer) noexcept = 0;
//  virtual std::vector<char> readFile(const std::string& file_path) const = 0;
};

template<typename T>
std::size_t getDataLenWithoutLastZeros(void* data, std::size_t bytes) noexcept {
  T* data_ptr = reinterpret_cast<T*>(data);
  assert(bytes % sizeof(T) == 0);
  std::size_t total_data_len = bytes / sizeof(T) - 1;
  for (; total_data_len >= 0; --total_data_len) {
    if (*(data_ptr + total_data_len) != 0) {
      break;
    }
  }
  return total_data_len + 1;
}

class AppenedSampleLoader : public PerfSampleLoader {
public:
  AppenedSampleLoader(std::string name,
                      std::vector<std::string> inputs_path /*data folders path, one for each input*/,
                      std::size_t perf_sample_count /**/,
                      const std::vector<std::size_t>& input_size)
    : name_(name),
      perf_sample_count_(perf_sample_count),
      input_size_(input_size),
      inputs_path_(inputs_path) {
    // Get input size and allocate memory
    num_inputs_ = inputs_path_.size();
    sample_memory_.resize(num_inputs_);

    // as a safety, don't allow the perf_sample_count to be larger than sampleCount.
    sample_count_ = perf_sample_count_;
    printf("sample count: %lu, perform count: %lu\n", TotalSampleCount(), PerformanceSampleCount());
  }

  void setDataTranformerForInput(std::size_t input_idx, DataTransformer tranformer) noexcept {
    data_tranformers_[input_idx] = tranformer;
  }

  const std::string& Name() const override { return name_; }
  std::size_t TotalSampleCount() override { return sample_count_; }
  std::size_t PerformanceSampleCount() override { return perf_sample_count_; }
  std::size_t getInputSizes(std::size_t input_idx) const override { return input_size_[input_idx]; }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    std::cout << "LoadSamplesToRam samples size: " << samples.size() << std::endl;
    // copy the samples into pinned memory
    std::size_t current_idx = 0;
    sample_lens_.resize(num_inputs_);
    sample_origin_lens_.resize(num_inputs_);
    for (auto& sizes : sample_lens_) {
      sizes.resize(TotalSampleCount(), -1);
    }
    for (auto& sizes : sample_origin_lens_) {
      sizes.resize(TotalSampleCount(), -1);
    }

    // load input data
    if (input_data_.empty()) {
      for (std::size_t input_idx = 0; input_idx < num_inputs_; input_idx++) {
        std::string path = inputs_path_[input_idx];
        size_t size = 0;
        auto data_ptr = ReadBinary(path.c_str(), &size);
        if (!data_ptr) {
          printf("error: can't read file: %s\n", path.c_str());
          return;
        }
        void* pinned_memory = moffett::spu_backend::MallocHostMemory(size, 0);
        void* pinned_memory2 = moffett::spu_backend::MallocHostMemory(size, 1);
        if (!pinned_memory || !pinned_memory2) {
          printf("error: malloc host memory error!\n");
        }
        memcpy((char*)pinned_memory, data_ptr, size);
        memcpy((char*)pinned_memory2, data_ptr, size);
        input_data_.emplace_back((uint8_t *)pinned_memory);
        input_data2_.emplace_back((uint8_t *)pinned_memory2);
        printf("input file: %s, one sample size: %lu\n", path.c_str(), size / sample_count_);
      }
    }

    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
      auto& sampleId = samples[sampleIndex];
      samples_address_[sampleId].resize(num_inputs_);
      samples_address2_[sampleId].resize(num_inputs_);
      for (size_t input_idx = 0; input_idx < num_inputs_; ++input_idx) {
        sample_lens_[input_idx][sampleId] = sample_origin_lens_[input_idx][sampleId] = input_size_[input_idx];
        samples_address_[sampleId][input_idx] = input_data_[input_idx] + sampleId * input_size_[input_idx];
        samples_address2_[sampleId][input_idx] = input_data2_[input_idx] + sampleId * input_size_[input_idx];
      }
    }

    std::cout << "Loading data done!" << std::endl;
  }

  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
  }

  uint8_t *ReadBinary(const char *fname, size_t *size) {
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
      return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
      close(fd);
      return nullptr;
    }
    uint8_t *data = (uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
      close(fd);
      return nullptr;
    }
    close(fd);
    *size = st.st_size;
    return data;
  }


  uint8_t* GetSampleAddress(mlperf::QuerySampleIndex sample_index, std::size_t input_idx, int numa,
                         std::size_t device_idx = 0) {
    if (numa == 0)
      return reinterpret_cast<uint8_t*>(samples_address_[sample_index][input_idx]);
    else
      return reinterpret_cast<uint8_t*>(samples_address2_[sample_index][input_idx]);
  }

  std::size_t GetSampleSize(std::size_t input_idx) const {
    return input_size_[input_idx];
  }
  std::size_t GetSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept override {
    return sample_lens_[input_idx][sample_id];
  }
  std::size_t GetOriginSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept override {
    return sample_origin_lens_[input_idx][sample_id];
  }
  std::vector<std::size_t>& GetAllSampleLenOfInputRef(std::size_t input_idx) noexcept override {
    return sample_lens_[input_idx];
  }
  std::vector<std::size_t> GetAllSampleLenOfInput(std::size_t input_idx) noexcept override {
    return sample_lens_[input_idx];
  }

  ~AppenedSampleLoader() {
  }

private:
  std::size_t num_inputs_{0};
  int num_devices_{1};

  const std::string name_;
  std::size_t perf_sample_count_{0};

  // One for each input of model.
  std::vector<std::string> inputs_path_;

  std::size_t sample_count_{0};

  // maps sampleId to <fileName, label>
  std::map<mlperf::QuerySampleIndex, std::tuple<std::string, std::size_t>> file_label_map_;

  // One for each input of model.
  // dims of input.
  std::vector<std::size_t> input_size_;

  // datas of num_inputs of
  // InputData: [in_0, in_1, ...]
  using InputData = std::vector<void*>;
  InputData sample_memory_;
  std::vector<uint8_t*> input_data_;
  std::vector<uint8_t*> input_data2_;
  // Input -> sample len
  std::vector<std::vector<std::size_t>> sample_lens_;
  std::vector<std::vector<std::size_t>> sample_origin_lens_;

  // map[sampleId] ==> all inputs
  // map[sampleId][input_idx] ==> one sample
  std::unordered_map<mlperf::QuerySampleIndex, InputData> samples_address_;
  std::unordered_map<mlperf::QuerySampleIndex, InputData> samples_address2_;
  std::map<std::size_t, DataTransformer> data_tranformers_;
};

using SampleLoaderPtr = std::shared_ptr<PerfSampleLoader>;

class UniverseSampleLoader : public mlperf::QuerySampleLibrary {
public:
  UniverseSampleLoader(const std::vector<SampleLoaderPtr> loaders) : loaders_(loaders){};
  const std::string& Name() const override { return loaders_[0]->Name(); }
  std::size_t TotalSampleCount() override { return loaders_[0]->TotalSampleCount(); }
  std::size_t PerformanceSampleCount() override { return loaders_[0]->PerformanceSampleCount(); }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    loaders_[0]->LoadSamplesToRam(samples);
  }
  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto loader : loaders_) {
      loader->UnloadSamplesFromRam(samples);
    }
  }

private:
  std::vector<SampleLoaderPtr> loaders_;
};

} // namespace samplesLoader
