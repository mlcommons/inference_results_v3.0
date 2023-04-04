#include "rebel_sut.hpp"

#include <omp.h>

#include <thread>

namespace rebel {

rebel::SystemUnderTest::SystemUnderTest(std::string path_prefix, rebel::QuerySampleLibrary& qsl,
                                        PostProcessor& pp)
    : name_("Rebellions-SDK"), qsl_(qsl), pp_(pp) {}

rebel::SimpleSystemUnderTest::SimpleSystemUnderTest(std::string path_prefix,
                                                    rebel::QuerySampleLibrary& qsl,
                                                    PostProcessor& pp)
    : rebel::SystemUnderTest(path_prefix, qsl, pp) {
  RebelRetCode rc;
  ctx_holder_ = rebel_context_holder_alloc();
  rc = rebel_context_holder_init(ctx_holder_, path_prefix.c_str());
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_context_holder_init failed with code: " << rc << std::endl;
    exit(1);
  }
  runner_ = rebel_runner_alloc();
  rc = rebel_runner_init(runner_, ctx_holder_, 0);
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_runner_init failed with code: " << rc << std::endl;
    exit(2);
  }
  out_buffer_ = rebel_get_outputs(runner_);
}

rebel::SimpleSystemUnderTest::~SimpleSystemUnderTest() {
  delete[] out_buffer_;
  rebel_runner_free(runner_);
  rebel_context_holder_free(ctx_holder_);
};

void rebel::SimpleSystemUnderTest::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  for (const auto& sample : samples) {
    void** input_buffers = qsl_.GetSample(sample.index);
    rebel_set_inputs_simple(runner_, input_buffers);
    rebel_run(runner_);
    response_cache_ = std::move(mlperf::QuerySampleResponse{
      id : sample.id,
      data : reinterpret_cast<uintptr_t>(pp_.Get(out_buffer_)),
      size : pp_.GetSize()
    });
    // Report result
    mlperf::QuerySamplesComplete(&response_cache_, 1);
  }
}

rebel::BatchSystemUnderTest::BatchSystemUnderTest(std::string path_prefix,
                                                  rebel::QuerySampleLibrary& qsl, PostProcessor& pp,
                                                  size_t batch_size)
    : rebel::SystemUnderTest(path_prefix, qsl, pp), batch_size_(batch_size) {
  RebelRetCode rc;
  ctx_holder_ = rebel_context_holder_alloc();
  rc = rebel_context_holder_init(ctx_holder_, path_prefix.c_str(), 1);
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_context_holder_init failed with code: " << rc << std::endl;
    exit(1);
  }
  runner_ = rebel_runner_alloc();
  rc = rebel_runner_init(runner_, ctx_holder_, 0);
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_runner_init failed with code: " << rc << std::endl;
    exit(2);
  }

  input_mmaps_ = rebel_input_memory_map_alloc(qsl_.GetNumInputs());
  for (size_t i = 0; i < qsl_.GetNumInputs(); i++) {
    auto mmap = rebel_input_memory_map_get_index(input_mmaps_, i);
    rebel_input_memory_map_init(mmap, runner_, i, batch_size_);
  }

  out_buffer_ = rebel_get_outputs(runner_);

  response_cache_.resize(batch_size);
}

rebel::BatchSystemUnderTest::~BatchSystemUnderTest() {
  delete[] out_buffer_;
  rebel_input_memory_map_free(input_mmaps_);
  rebel_runner_free(runner_);
  rebel_context_holder_free(ctx_holder_);
}

void rebel::BatchSystemUnderTest::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  // Parse inputs
  for (size_t batch_index = 0; batch_index < samples.size(); batch_index++) {
    void** buffer_from_qsl = qsl_.GetSample(samples[batch_index].index);
    for (size_t input_index = 0; input_index < qsl_.GetNumInputs(); input_index++) {
      auto mmap = rebel_input_memory_map_get_index(input_mmaps_, input_index);
      rebel_input_memory_map_add_mapping(mmap, batch_index, buffer_from_qsl[input_index]);
    }
  }

  // Set inputs
  rebel_set_inputs(runner_, input_mmaps_);

  // Run
  rebel_run(runner_);

  // Create responses
  auto output = reinterpret_cast<const uint8_t* const*>(pp_.Get(out_buffer_));
  for (size_t i = 0; i < samples.size(); i++) {
    response_cache_[i] = mlperf::QuerySampleResponse{
      id : samples[i].id,
      data : reinterpret_cast<uintptr_t>(output[i]),
      size : pp_.GetSize()
    };
  }

  // Report result
  mlperf::QuerySamplesComplete(response_cache_.data(), response_cache_.size());
}

rebel::AsyncSystemUnderTest::AsyncSystemUnderTest(std::string path_prefix,
                                                  rebel::QuerySampleLibrary& qsl,
                                                  rebel::PostProcessor& pp, size_t batch_size,
                                                  size_t num_thread)
    : rebel::SystemUnderTest(path_prefix, qsl, pp),
      batch_size_(batch_size),
      num_thread_(num_thread) {
  RebelRetCode rc;
  ctx_holder_ = rebel_context_holder_alloc();
  rc = rebel_context_holder_init(ctx_holder_, path_prefix.c_str(), num_thread_);
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_context_holder_init failed with code: " << rc << std::endl;
    exit(1);
  }
  runner_ = rebel_async_runner_alloc();
  rc = rebel_async_runner_init(runner_, ctx_holder_, num_thread_);
  if (rc != RebelRetCode_OK) {
    std::cerr << "rebel_async_runner_init failed with code: " << rc << std::endl;
    exit(2);
  }

  for (auto i = 0; i < num_thread_; i++) {
    const auto& pp_downcasted = *dynamic_cast<rebel::BatchPostProcessor*>(&pp);
    pp_vector_.push_back(std::make_shared<rebel::BatchPostProcessor>(pp_downcasted));
  }
}

rebel::AsyncSystemUnderTest::~AsyncSystemUnderTest() {
  rebel_async_runner_free(runner_);
  rebel_context_holder_free(ctx_holder_);
};

void rebel::AsyncSystemUnderTest::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  for (size_t i = 0; i < samples.size(); i += batch_size_) {
    auto batch_size = std::min(i + batch_size_, samples.size()) - i;

    // Input memory map
    RebelInputMemoryMap* input_mmaps;
    input_mmaps = rebel_input_memory_map_alloc(qsl_.GetNumInputs());
    for (size_t i = 0; i < qsl_.GetNumInputs(); i++) {
      auto mmap = rebel_input_memory_map_get_index(input_mmaps, i);
      rebel_input_memory_map_init_async(mmap, runner_, i, batch_size_);
    }

    // Parse inputs
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
      void** buffer_from_qsl = qsl_.GetSample(samples[i + batch_index].index);
      for (size_t input_index = 0; input_index < qsl_.GetNumInputs(); input_index++) {
        auto mmap = rebel_input_memory_map_get_index(input_mmaps, input_index);
        rebel_input_memory_map_add_mapping(mmap, batch_index, buffer_from_qsl[input_index]);
      }
    }
    auto callback = [&samples, index = i, batch_size, &pp_vector_ = this->pp_vector_](
                        void* const* outputs, size_t runner_id) -> void {
      auto pp = pp_vector_[runner_id];
      // Create responses
      auto answers = reinterpret_cast<const uint8_t* const*>(pp->Get(outputs));
      std::vector<mlperf::QuerySampleResponse> responses;
      for (size_t i = 0; i < batch_size; i++) {
        auto response = mlperf::QuerySampleResponse{
          id : samples[index + i].id,
          data : reinterpret_cast<uintptr_t>(answers[i]),
          size : pp->GetSize()
        };
        // Report result
        mlperf::QuerySamplesComplete(&response, 1);
      }
    };

    // Run
    rebel_async_run(runner_, input_mmaps, callback);
  }
}

}  // namespace rebel
