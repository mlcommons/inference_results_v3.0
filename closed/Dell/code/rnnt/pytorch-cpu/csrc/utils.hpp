#pragma once

#include <issue_query_controller.h>
#include <loadgen.h>
#include <logging.h>
#include <query_sample.h>
#include <torch/csrc/autograd/profiler_legacy.h>

#include <chrono>

void sleep_thread(long duration);

long get_latency(const mlperf::QuerySample& sample);

long get_duration(const mlperf::PerfClock::time_point start);

long get_duration(
    const mlperf::PerfClock::time_point start,
    const mlperf::PerfClock::time_point end);

template <typename F, typename... Args>
double get_duration(F func, Args&&... args);

class ProfileRecord {
public:
  ProfileRecord(bool is_record, const std::string& profiler_file)
      : is_record_(is_record), profiler_file_(profiler_file) {
    if (is_record_)
      torch_profiler =
          std::make_unique<torch::autograd::profiler::RecordProfile>(
              profiler_file_);
  };

  virtual ~ProfileRecord(){};

private:
  bool is_record_;
  std::string profiler_file_;
  std::unique_ptr<torch::autograd::profiler::RecordProfile> torch_profiler;
};
