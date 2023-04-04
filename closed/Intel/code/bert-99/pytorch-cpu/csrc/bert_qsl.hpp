#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <torch/csrc/jit/serialization/import_read.h>
#include <query_sample_library.h>

namespace qsl {
using namespace mlperf;
using TensorList = std::vector<at::Tensor>;
using Stack = std::vector<at::IValue>;
using Queue_t = std::list<mlperf::QuerySample>;

class SquadQuerySampleLibrary : public QuerySampleLibrary {
public:
  SquadQuerySampleLibrary(
      const std::string& filename,
      const char* input_ids_name = "input_ids_samples",
      const char* input_mask_name = "input_mask_samples",
      const char* segment_ids_name = "segment_ids_samples");

  SquadQuerySampleLibrary(
      const std::string& f_input_ids,
      const std::string& f_input_mask,
      const std::string& f_segment_ids);

  virtual ~SquadQuerySampleLibrary() = default;

  const std::string& Name() override {
    static const std::string name("BERT Squad QSL");
    return name;
  }

  size_t TotalSampleCount() override {
    return input_ids_set_.size();
  }

  void CheckSampleCount();

  size_t PerformanceSampleCount() override {
    return TotalSampleCount();
  }

  //
  // SQuAD is small enough to be in Memory
  //
  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {}
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {}

  static SquadQuerySampleLibrary Create(const std::string& filename);

  Stack GetSample(QuerySampleIndex index) const {
    return {
      input_ids_set_.at(index).expand({1,-1}),
      input_mask_set_.at(index).expand({1, -1}),
      segment_ids_set_.at(index).expand({1, -1})
    };
  }

  Stack AssembleSamples(std::vector<QuerySampleIndex> indices, int64_t max_length = 0) const;
  Stack ServerAssembleSamples(std::vector<QuerySampleIndex> indices, int64_t total_len) const;
  // create dummy samples to warm up
  // single_len: single sample length
  // total_len:  total samples length
  Stack CreateDummySamples(int64_t total_len);

  // List of tensor of 1d
  size_t GetFeatureLength(size_t index) const {
    return input_ids_set_[index].size(0);
  }

  //
  // Sort SQuAD data for batch behavior

  //
  Queue_t Sort(
      const std::vector<QuerySample>& samples, bool reverse = true,
      size_t minLength = 40, size_t maxLength = 384) const;

  c10::Dict<at::IValue, at::IValue> GetDictFrom(const std::string& filename);

  static TensorList GetTensorListFrom(at::IValue value);
  static TensorList GetTensorListFrom(const std::string& filename);
  static TensorList GetTensorListFrom(
      c10::Dict<at::IValue, at::IValue>& dict,
      const char* name);

private:
  TensorList input_ids_set_;
  TensorList input_mask_set_;
  TensorList segment_ids_set_;
};

}
