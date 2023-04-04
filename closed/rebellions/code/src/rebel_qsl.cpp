#include "rebel_qsl.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace rebel {

ImageNetQuerySampleLibrary::ImageNetQuerySampleLibrary(size_t limit)
    : name_("Rebellions-ImageNet"), limit_(limit) {}

void ImageNetQuerySampleLibrary::LoadSamplesToRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  for (const auto& sample_index : samples) {
    if (cache_.find(sample_index) != cache_.end()) continue;  // Skip if exists

    std::cout << "Loading sample " << sample_index << std::endl;

    std::ostringstream filename_stream;
    filename_stream << "scratch/preprocessed/imagenet/ILSVRC2012_val_" << std::setw(8)
                    << std::setfill('0') << sample_index + 1 << ".JPEG.npy";

    const std::filesystem::path image_path = filename_stream.str();
    if (!std::filesystem::exists(image_path)) {
      std::cerr << "File " << image_path.string() << " not exists" << std::endl;
      throw std::exception();
    }

    std::ifstream file(image_path.string(), std::ios::binary | std::ios::in);
    if (!file) {
      std::cerr << "Error: Unable to open file " << image_path << std::endl;
      exit(1);
    }

    // Skip Numpy Header
    file.seekg(8 * 16);

    size_t dst_bytes = 224 * 224 * 4 * sizeof(ElemType);
    uint8_t* sample_data = new uint8_t[dst_bytes];
    uint8_t** sample = new uint8_t*[1];
    sample[0] = sample_data;
    file.read(reinterpret_cast<char*>(sample[0]), dst_bytes);
    file.close();

    cache_.insert_or_assign(sample_index, sample);
  }
}

void ImageNetQuerySampleLibrary::UnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  for (const auto& sample_index : samples) {
    uint8_t** sample = cache_[sample_index];
    if (sample) {
      delete[] sample[0];
      delete[] sample;
    }
    cache_.erase(sample_index);
  }
}

SQuADQuerySampleLibrary::SQuADQuerySampleLibrary(size_t limit)
    : name_("Rebellions-SQuAD"), limit_(limit) {
  std::ostringstream filename_stream;
  filename_stream << "scratch/preprocessed/squad/squad.npy";

  const std::filesystem::path squad_path = filename_stream.str();
  if (!std::filesystem::exists(squad_path)) {
    std::cerr << "File " << squad_path.string() << " not exists" << std::endl;
    throw std::exception();
  }

  std::ifstream file(squad_path.string(), std::ios::binary | std::ios::in);
  if (!file) {
    std::cerr << "Error: Unable to open file " << squad_path << std::endl;
    exit(1);
  }

  // Skip Numpy Header
  // squad.npy : (10833, 3, 384) np.int32, [input_ids, segment_ids, input_mask]
  int num_sample = 10833;
  input_ids_.resize(num_sample);
  token_type_ids_.resize(num_sample);
  attention_masks_.resize(num_sample);

  file.seekg(8 * 16);
  auto n_bytes_per_sample = 384 * sizeof(ElemType);
  for (int i = 0; i < 10833; i++) {
    input_ids_[i].resize(384);
    token_type_ids_[i].resize(384);
    attention_masks_[i].resize(384);
    file.read(reinterpret_cast<char*>(input_ids_[i].data()), n_bytes_per_sample);
    file.read(reinterpret_cast<char*>(token_type_ids_[i].data()), n_bytes_per_sample);
    file.read(reinterpret_cast<char*>(attention_masks_[i].data()), n_bytes_per_sample);
  }

  file.close();
}

void SQuADQuerySampleLibrary::LoadSamplesToRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  for (const auto& sample_index : samples) {
    if (cache_.find(sample_index) != cache_.end()) continue;  // Skip if exists
    std::cout << "Loading sample " << sample_index << std::endl;
    auto sample = new uint8_t*[3];
    sample[0] = new uint8_t[GetInputSize(0)];
    memcpy(sample[0], input_ids_[sample_index].data(), GetInputSize(0));
    sample[1] = new uint8_t[GetInputSize(1)];
    memcpy(sample[1], token_type_ids_[sample_index].data(), GetInputSize(1));
    sample[2] = new uint8_t[GetInputSize(2)];
    memcpy(sample[2], attention_masks_[sample_index].data(), GetInputSize(2));
    cache_.insert_or_assign(sample_index, sample);
  }
}

void SQuADQuerySampleLibrary::UnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  for (const auto& sample_index : samples) {
    uint8_t** sample = cache_[sample_index];
    delete[] sample[0];
    delete[] sample[1];
    delete[] sample[2];
    delete[] sample;
    cache_.erase(sample_index);
  }
}

}  // namespace rebel