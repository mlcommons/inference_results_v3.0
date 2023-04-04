#pragma once

#include <query_sample_library.h>

#include <filesystem>
#include <unordered_map>

namespace rebel {

class QuerySampleLibrary : public mlperf::QuerySampleLibrary {
 public:
  virtual void** GetSample(const mlperf::QuerySampleIndex sample) const = 0;

  virtual size_t GetNumInputs() const = 0;
  virtual size_t GetInputSize(size_t index) const = 0;
};

class ImageNetQuerySampleLibrary : public QuerySampleLibrary {
  using ElemType = int8_t;

 public:
  ImageNetQuerySampleLibrary(size_t limit = 0);

  const std::string& Name() override { return name_; }

  size_t TotalSampleCount() override { return limit_ > 0 ? limit_ : 50000; }

  size_t PerformanceSampleCount() override { return 1024; }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

  void** GetSample(const mlperf::QuerySampleIndex sample) const override {
    return reinterpret_cast<void**>(cache_.at(sample));
  }

  size_t GetNumInputs() const override { return 1; }

  size_t GetInputSize(size_t index) const { return 224 * 224 * 4 * sizeof(ElemType); }

 private:
  std::string name_;
  size_t limit_;
  std::unordered_map<mlperf::QuerySampleIndex, uint8_t**> cache_;
};

class SQuADQuerySampleLibrary : public QuerySampleLibrary {
  using ElemType = int32_t;

 public:
  SQuADQuerySampleLibrary(size_t limit = 0);

  const std::string& Name() override { return name_; }

  size_t TotalSampleCount() override { return limit_ > 0 ? limit_ : input_ids_.size(); }

  size_t PerformanceSampleCount() override { return input_ids_.size(); }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

  void** GetSample(const mlperf::QuerySampleIndex sample) const override {
    return reinterpret_cast<void**>(cache_.at(sample));
  }

  size_t GetNumInputs() const override { return 3; }

  size_t GetInputSize(size_t index) const { return 384 * sizeof(ElemType); }

 private:
  std::string name_;
  std::unordered_map<mlperf::QuerySampleIndex, uint8_t**> cache_;
  size_t limit_;

  std::vector<std::vector<ElemType>> input_ids_;
  std::vector<std::vector<ElemType>> token_type_ids_;
  std::vector<std::vector<ElemType>> attention_masks_;
};

}  // namespace rebel
