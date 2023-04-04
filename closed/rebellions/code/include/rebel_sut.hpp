#pragma once

#include <loadgen.h>
#include <rebel_runtime.h>
#include <system_under_test.h>
#include <tvm/runtime/registry.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "postprocessor.hpp"
#include "rebel_qsl.hpp"

namespace rebel {

class SystemUnderTest : public mlperf::SystemUnderTest {
 public:
  SystemUnderTest(std::string path_prefix, rebel::QuerySampleLibrary& qsl, PostProcessor& pp);

  const std::string& Name() override { return name_; }

  // @jhlee: TODO
  void FlushQueries() override {}

 protected:
  std::string name_;
  rebel::QuerySampleLibrary& qsl_;
  rebel::PostProcessor& pp_;
};

class SimpleSystemUnderTest : public rebel::SystemUnderTest {
 public:
  SimpleSystemUnderTest(std::string path_prefix, rebel::QuerySampleLibrary& qsl, PostProcessor& pp);

  ~SimpleSystemUnderTest() override;

  const std::string& Name() { return name_; }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

  void FlushQueries() override {}

 private:
  RebelContextHolder* ctx_holder_;
  RebelRunner* runner_;

  mlperf::QuerySampleResponse response_cache_;
  void* const* out_buffer_;
};

class BatchSystemUnderTest : public rebel::SystemUnderTest {
 public:
  BatchSystemUnderTest(std::string path_prefix, rebel::QuerySampleLibrary& qsl, PostProcessor& pp,
                       size_t batch_size);

  ~BatchSystemUnderTest() override;

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

 private:
  RebelContextHolder* ctx_holder_;
  RebelRunner* runner_;

  RebelInputMemoryMap* input_mmaps_;

  std::vector<mlperf::QuerySampleResponse> response_cache_;
  void* const* out_buffer_;

  size_t batch_size_;
};

class AsyncSystemUnderTest : public rebel::SystemUnderTest {
 public:
  AsyncSystemUnderTest(std::string path_prefix, rebel::QuerySampleLibrary& qsl, PostProcessor& pp,
                       size_t batch_size, size_t num_thread);

  ~AsyncSystemUnderTest() override;

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

 private:
  RebelContextHolder* ctx_holder_;
  RebelAsyncRunner* runner_;

  size_t batch_size_;
  size_t num_thread_;
  std::vector<std::shared_ptr<rebel::BatchPostProcessor>> pp_vector_;
};

}  // namespace rebel
