#pragma once

#include <query_sample_library.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "metadata.hpp"

namespace rnnt {
namespace qsl {
using namespace mlperf;
using TensorList = std::vector<at::Tensor>;
using Stack = std::vector<at::IValue>;
using Queue_t = std::list<mlperf::QuerySample>;

class RNNTQuerySampleLibrary : public QuerySampleLibrary {
public:
  RNNTQuerySampleLibrary(
      const std::string& filename, const char* x_name = "x",
      const char* x_lens_name = "x_lens");

  RNNTQuerySampleLibrary(const std::string& f_x, const std::string& f_x_lens);

  virtual ~RNNTQuerySampleLibrary() = default;

  const std::string& Name() override {
    static const std::string name("RNN-T LibriSpeech QSL");
    return name;
  }

  size_t TotalSampleCount() override { return x_set_.size(); }

  void CheckSampleCount();

  size_t PerformanceSampleCount() override { return TotalSampleCount(); }

  // LibriSpeech is small enough to be in Memory
  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
  }

  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {}

  static RNNTQuerySampleLibrary Create(const std::string& filename);

  Stack GetSample(QuerySampleIndex index) const {
    return {
        x_set_.at(index),
        x_lens_set_.at(index),
    };
  }

  std::tuple<at::Tensor, at::Tensor> GenerateDummySamples(
      long batch_size, bool processor = true);

  std::tuple<at::Tensor, at::Tensor> AssembleSamples(
      std::vector<QuerySampleIndex> indices, bool processor = true,
      int padded_batch_size = 1) const;

  // List of tensor of 1d
  size_t GetFeatureLength(QuerySampleIndex index) const {
    return x_lens_set_[index].item().toInt();
  }

  // Sort LibriSpeech data for batching
  Queue_t Sort(
      const std::vector<QuerySample>& samples, bool reverse = true) const;

  c10::Dict<at::IValue, at::IValue> GetDictFrom(const std::string& filename);

  static TensorList GetTensorListFrom(at::IValue value);
  static TensorList GetTensorListFrom(const std::string& filename);
  static TensorList GetTensorListFrom(
      c10::Dict<at::IValue, at::IValue>& dict, const char* name);
  static Stack GetIValueListFrom(at::IValue value);

private:
  TensorList x_set_;
  TensorList x_lens_set_;
  size_t minLength;
  size_t maxLength;
};

}  // namespace qsl
}  // namespace rnnt
