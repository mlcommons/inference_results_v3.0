#include "utils.hpp"

void sleep_thread(long duration) {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  return;
}

long get_latency(const mlperf::QuerySample& sample) {
  mlperf::PerfClock::time_point timestamp = mlperf::PerfClock::now();
  mlperf::loadgen::SampleMetadata* sample_metadata =
      reinterpret_cast<mlperf::loadgen::SampleMetadata*>(sample.id);
  mlperf::loadgen::QueryMetadata* query_metadata =
      sample_metadata->query_metadata;
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                      timestamp - query_metadata->scheduled_time)
                      .count();
  return duration;
}

long get_duration(mlperf::PerfClock::time_point start) {
  mlperf::PerfClock::time_point end = mlperf::PerfClock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return duration;
}

long get_duration(
    mlperf::PerfClock::time_point start, mlperf::PerfClock::time_point end) {
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return duration;
}

template <typename F, typename... Args>
double get_duration(F func, Args&&... args) {
  mlperf::PerfClock::time_point start = mlperf::PerfClock::now();
  func(std::forward<Args>(args)...);
  mlperf::PerfClock::time_point end = mlperf::PerfClock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return duration;
}
