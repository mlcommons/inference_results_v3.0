#pragma once

#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "blockingconcurrentqueue.h"
#include "kmp_launcher.hpp"
#include "rnnt_model.hpp"
#include "rnnt_processor.hpp"
#include "rnnt_qsl.hpp"

namespace rnnt {
class BaseSUT : public mlperf::SystemUnderTest {
public:
  using Queue_t = std::list<mlperf::QuerySample>;
  BaseSUT(
      const std::string& model_file, const std::string& samples_file,
      const std::string& processor_file, int batch_size, int split_len = -1,
      const std::string test_scenario = "Offline", bool processor = true,
      const std::string& profiler_foler = "", int profiler_iter = -1,
      int warmup_iter = -1);

  ~BaseSUT();

  const std::string& Name() override {
    static const std::string name("RNN-T_" + test_scenario_ + "_SUT");
    return name;
  }

  mlperf::QuerySampleLibrary* GetQSL() { return &qsl_; }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

  void FlushQueries() override {}

  // for Processor: test only
  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const at::Tensor& results);

  qsl::RNNTQuerySampleLibrary qsl_;
  models::TorchModel model_;
  models::AudioProcessor processor_;

  std::condition_variable ctrl_;
  std::mutex mtx_;

  Queue_t mQueue_;
  bool mStop_{false};

  std::vector<std::thread> mInstances_;
  // Control over max samples a instance will peek
  size_t mThreshold_;
  int split_len_;
  bool mHt_;
  int nMaxThread_;
  int nMaxProc_;
  int nSockets_;
  int nCoresPerSocket_;
  std::string test_scenario_;
  bool processor_flag_;

  // std::unique_ptr<ProfileRecord> guard_;
  std::string profiler_folder_;
  int profiler_iter_;
  int warmup_iter_;
  bool batch_sort_;
  bool pad_batch_size_;
};

class OfflineSUT : public BaseSUT {
public:
  OfflineSUT(
      const std::string& model_file, const std::string& samples_file,
      const std::string& processor_file, int inter_parallel, int intra_parallel,
      int batch_size, int split_len = -1,
      const std::string test_scenario = "Offline", bool processor = true,
      const std::string& profiler_foler = "", int profiler_iter = -1,
      int warmup_iter = -1);

private:
  void warmup(int which, int warmup_iter);

  void thInstance(int index, int root);

  // for Offline: batching response
  static void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const rnnt::State& state);

  int nInstances_;
  int nThreadsPerInstance_;
};

class ServerSUT : public BaseSUT {
public:
  ServerSUT(
      const std::string& samples_file, const std::string& model_file,
      const std::string& processor_file, int pro_inter_parallel,
      int pro_intra_parallel, int inter_parallel, int intra_parallel,
      int pro_batch_size, int batch_size, int split_len = -1,
      int response_size = -1, int qos_len = -1,
      const std::string test_scenario = "Server", bool processor = true,
      const std::string& profiler_foler = "", int profiler_iter = -1,
      int warmup_iter = -1);

private:
  void FlushQueries() override {
    lStop_ = true;
    std::cout << "Loadgen finish flushing" << std::flush;
  }

  void warmup(int which, int warmup_iter, int worker_type);

  void thProducer(int index, int root);

  void thConsumer(int index, int root);

  // for Pipeline: early response
  void QuerySamplesComplete(
      const std::vector<mlperf::QuerySample>& samples,
      const rnnt::PipelineState& state);

  enum WorkerType {
    Producer = 0,
    Consumer = 1,
  };

  int nProducers_;
  int nThreadsPerProducer_;
  int nConsumers_;
  int nThreadsPerConsumer_;
  // Control over max samples a instance will peek
  size_t mProThreshold_;
  size_t mResponseThreshold_;
  size_t mLengthThreshold_ = 230000;
  moodycamel::BlockingConcurrentQueue<
      std::tuple<mlperf::QuerySample, at::Tensor, at::Tensor>>
      mProcessedQueue_;
  bool lStop_ = false;
  Queue_t mQosQueue_;
};
}  // namespace rnnt
