#include <ATen/ATen.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <cassert>
#include "bert_qsl.hpp"

#include <iostream>

namespace qsl {
TensorList SquadQuerySampleLibrary::GetTensorListFrom(
    at::IValue value) {
  std::vector<at::Tensor> tensor_list;
  auto toTensor = [](at::IValue item) {return item.toTensor();};

  if (value.isList()) {
    auto value_list = value.toList();

    std::transform(value_list.begin(), value_list.end(),
        std::back_inserter(tensor_list), toTensor);
  } else if (value.isTensorList()) {
    auto c10_list = value.toTensorList();
    tensor_list.insert(tensor_list.begin(), c10_list.begin(), c10_list.end());
  }else if (value.isTuple()) {
    auto value_list = value.toTuple()->elements();
    std::transform(value_list.begin(), value_list.end(),
        std::back_inserter(tensor_list), toTensor);
  } else {
    TORCH_CHECK(false, "Can't get TensorList from IValue type: ", value.tagKind());
  }

  return tensor_list;
}

TensorList SquadQuerySampleLibrary::GetTensorListFrom(
    const std::string& filename) {
  caffe2::serialize::PyTorchStreamReader reader(filename);
  auto stack = torch::jit::readArchiveAndTensors("data",
      "","",
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      reader);

  return GetTensorListFrom(stack);
}

TensorList SquadQuerySampleLibrary::GetTensorListFrom(
    c10::Dict<at::IValue, at::IValue>& dict,
    const char* name) {
  at::IValue ivname (name);

  auto tensor_list = dict.find(ivname);
  if ( tensor_list != dict.end() )
    return GetTensorListFrom(tensor_list->value());
  else
    return TensorList();
}

c10::Dict<at::IValue, at::IValue> SquadQuerySampleLibrary::GetDictFrom(
    const std::string& filename) {
  caffe2::serialize::PyTorchStreamReader reader(filename);
  auto stack = torch::jit::readArchiveAndTensors("data",
      "","",
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      reader);

  // Exception management
  return stack.toGenericDict();
}

SquadQuerySampleLibrary::SquadQuerySampleLibrary(
    const std::string& filename,
    const char* input_ids_name,
    const char* input_mask_name,
    const char* segment_ids_name) {
  auto datasets = GetDictFrom(filename);
  input_ids_set_ = GetTensorListFrom(datasets, input_ids_name);
  input_mask_set_ = GetTensorListFrom(datasets, input_mask_name);
  segment_ids_set_ = GetTensorListFrom(datasets, segment_ids_name);
  CheckSampleCount();
}

SquadQuerySampleLibrary::SquadQuerySampleLibrary(
    const std::string& f_input_ids,
    const std::string& f_input_mask,
    const std::string& f_segment_ids) {
  input_ids_set_ = GetTensorListFrom(f_input_ids);
  input_mask_set_ = GetTensorListFrom(f_input_mask);
  segment_ids_set_ = GetTensorListFrom(f_segment_ids);
  CheckSampleCount();
}

SquadQuerySampleLibrary SquadQuerySampleLibrary::Create(
    const std::string& filename) {
  return SquadQuerySampleLibrary(filename);
}

void SquadQuerySampleLibrary::CheckSampleCount() {
  /* throw if three sets have different sizes */
}

//
// Parallel bucket sort (unstable) would be the most efficient choice
// For length 40 ~ 384, each with a bucket of std::list
//
Queue_t SquadQuerySampleLibrary::Sort(
    const std::vector<QuerySample>& samples, bool reverse,
    size_t minLength, size_t maxLength) const {
  const auto lengthOffset = minLength;
  const auto nBucket = maxLength - lengthOffset +1;

  std::vector<Queue_t> Buckets(nBucket);
  std::vector<std::mutex> lks (nBucket);

  // (Parallel) sort
  // TODO: support other parallel library
# pragma omp parallel for
  for (size_t i = 0; i < samples.size(); ++ i) {
  // for (const auto &sample : samples) {
    auto& sample = samples[i];
    auto length = GetFeatureLength(sample.index);

    auto idx = reverse ? maxLength - length : length - lengthOffset;
    auto& bucket = Buckets[idx];
    auto& l = lks[idx];

    {
      std::unique_lock<std::mutex> guard(l);
      bucket.emplace_back(sample);
    }
  }

  // Splice them togather
  Queue_t result;
  for (auto &q : Buckets)
    result.splice(result.end(), std::move(q));

  return result;
}

//
// Assemble samples into larger batch
//
Stack SquadQuerySampleLibrary::AssembleSamples(
    std::vector<QuerySampleIndex> indices, int64_t max_length) const {
  TensorList ids_list, mask_list, segid_list;

  ids_list.reserve(indices.size());
  mask_list.reserve(indices.size());
  segid_list.reserve(indices.size());

  int64_t maxLength = max_length;

  for (auto index : indices) {
    auto input_ids = input_ids_set_[index];
    auto input_mask = input_mask_set_[index];
    auto segment_ids = segment_ids_set_[index];
 
    if (maxLength == 0)
      maxLength = input_ids.size(0);

    if (input_ids.size(0) < maxLength) {
      // Padding needed
      std::vector<int64_t> newShape {maxLength};

      auto len = input_ids.size(0);
      auto opts = at::TensorOptions().dtype<int>().memory_format(at::MemoryFormat::Contiguous);

      auto padded_ids = at::zeros(newShape, opts);
      auto padded_mask = at::zeros(newShape, opts);
      auto padded_segids = at::zeros(newShape, opts);

      padded_ids.narrow(0,0,len).copy_(input_ids);
      padded_mask.narrow(0,0,len).copy_(input_mask);
      padded_segids.narrow(0,0,len).copy_(segment_ids);

      ids_list.emplace_back(padded_ids);
      mask_list.emplace_back(padded_mask);
      segid_list.emplace_back(padded_segids);
    } else {
      ids_list.emplace_back(input_ids);
      mask_list.emplace_back(input_mask);
      segid_list.emplace_back(segment_ids);
    }
  }
  auto input_ids = at::vstack(ids_list);
  auto input_mask = at::vstack(mask_list);
  auto segment_ids = at::vstack(segid_list);
  return Stack { input_ids, input_mask, segment_ids };
}

Stack SquadQuerySampleLibrary::ServerAssembleSamples(
    std::vector<QuerySampleIndex> indices, int64_t total_len) const {
  auto opts = at::TensorOptions().dtype<int>().memory_format(at::MemoryFormat::Contiguous);
  // at::Tensor ids_cat, mask_cat, segid_cat, length_cat;
  auto ids_cat = at::zeros({total_len}, opts);
  auto mask_cat = at::zeros({total_len}, opts);
  auto segid_cat = at::zeros({total_len}, opts);
  auto length_cat = at::tensor({0}, opts);

  int64_t word_pos = 0;
  for (auto index : indices) {
    auto input_ids = input_ids_set_[index];
    auto input_mask = input_mask_set_[index];
    auto segment_ids = segment_ids_set_[index];

    auto len = input_ids.size(0);
    auto tlen = at::tensor({len+length_cat[-1].item<int>()}, opts);

    ids_cat.narrow(0, word_pos, len).copy_(input_ids);
    mask_cat.narrow(0, word_pos, len).copy_(input_mask);
    segid_cat.narrow(0, word_pos, len).copy_(segment_ids);
    length_cat = at::cat({length_cat, tlen}, 0);
    word_pos += len;
  }

  // for (int i = 0; i < ids_cat.sizes().size(); ++i) {
  //   std::cout << total_len << "<-------->" << ids_cat.sizes()[i] << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < ids_cat.sizes()[0]; ++i) {
  //   std::cout << ids_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < mask_cat.sizes()[0]; ++i) {
  //   std::cout << mask_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < segid_cat.sizes()[0]; ++i) {
  //   std::cout << segid_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < length_cat.sizes()[0]; ++i) {
  //   std::cout << length_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();
    
  length_cat.contiguous();
  auto input_ids = at::vstack({ids_cat});
  auto input_mask = at::vstack({mask_cat});
  auto segment_ids = at::vstack({segid_cat});
  // auto length_ids = at::vstack({length_cat});

  return Stack { input_ids, input_mask, segment_ids, length_cat };
}

Stack SquadQuerySampleLibrary::CreateDummySamples(int64_t total_len) {
  auto opts = at::TensorOptions().dtype<int>().memory_format(at::MemoryFormat::Contiguous);
  
  // construct single inputs
  auto single_ids_101 = at::tensor({101}, opts);
  auto single_ids_front = at::tensor({1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}, opts);
  auto single_ids_102 = at::tensor({102}, opts);
  auto single_ids_end = at::cat({single_ids_front, single_ids_front, single_ids_front, single_ids_front, single_ids_front, 
                                 single_ids_front, single_ids_front, single_ids_front, single_ids_front, single_ids_front, single_ids_102}, 0);
  auto single_ids = at::cat({single_ids_101, single_ids_front, single_ids_102, single_ids_end}, 0).contiguous();

  auto single_size = single_ids.size(0);
  auto q_size = single_ids_front.size(0) + 2;

  auto single_mask = at::ones({single_size}, opts);
  
  auto single_segid_1 = at::zeros({q_size}, opts);
  auto single_segid_2 = at::ones({single_size - q_size}, opts);
  auto single_segid = at::cat({single_segid_1, single_segid_2}, 0).contiguous();

  int64_t num = total_len / single_size + 1;
  // int64_t num = 1;
  int64_t real_size = num * single_size;

  auto ids_cat = at::zeros({real_size}, opts);
  auto mask_cat = at::zeros({real_size}, opts);
  auto segid_cat = at::zeros({real_size}, opts);
  auto length_cat = at::tensor({0}, opts);

  int64_t word_pos = 0;
  for (int i  = 0; i < num; ++i) {
    auto len = single_ids.size(0);
    auto tlen = at::tensor({len+length_cat[-1].item<int>()}, opts);

    ids_cat.narrow(0, word_pos, len).copy_(single_ids);
    mask_cat.narrow(0, word_pos, len).copy_(single_mask);
    segid_cat.narrow(0, word_pos, len).copy_(single_segid);

    length_cat = at::cat({length_cat, tlen}, 0);
    word_pos += len;
  }

  // for (int i = 0; i < ids_cat.sizes().size(); ++i) {
  //   std::cout << total_len << "<-------->" << ids_cat.sizes()[i] << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < ids_cat.sizes()[0]; ++i) {
  //   std::cout << ids_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < mask_cat.sizes()[0]; ++i) {
  //   std::cout << mask_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < segid_cat.sizes()[0]; ++i) {
  //   std::cout << segid_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  // for (int i = 0; i < length_cat.sizes()[0]; ++i) {
  //   std::cout << length_cat[i].item<int>() << "\t";
  // }
  // std::cout << std::endl;
  // getchar();

  length_cat.contiguous();
  auto input_ids = at::vstack({ids_cat});
  auto input_mask = at::vstack({mask_cat});
  auto segment_ids = at::vstack({segid_cat});
  // auto length_ids = at::vstack({length_cat});

  return Stack { input_ids, input_mask, segment_ids, length_cat };
}

}
