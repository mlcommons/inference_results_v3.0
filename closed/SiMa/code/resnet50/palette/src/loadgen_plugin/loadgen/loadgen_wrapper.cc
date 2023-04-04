#include <future>

#include "image_loader.h"
#include "loadgen_wrapper.h"

#include "loadgen_wrapper_ctrl.h"

namespace simaai {
namespace mlperf_wrapper {
template<typename T>
LoadgenWrapper<T>::LoadgenWrapper (LoadgenConfig & cfg)
    :config(std::move(cfg)),
     offline_idx(0) {

  config.mlperf_test_settings.scenario = get_test_scenario(cfg.test_type);
  config.mlperf_test_settings.mode = get_test_mode(cfg.run_type);
  // Also load the defaults from the .conf files here
  std::string mode_str(get_scenario_string());
  config.mlperf_test_settings.FromConfig(config.mlperf_conf_fpath , config.model_name, mode_str);
  std::cout << "--- MLPerf Loaded config from " << config.mlperf_conf_fpath << "\n";
  config.mlperf_test_settings.FromConfig(config.user_conf_fpath, config.model_name, mode_str);
  std::cout << "--- MLPerf Loaded config from " << config.user_conf_fpath << "\n";

#if 0
  if (!validate_config())
    throw std::runtime_error("config validation failed");
#endif
};

template<typename T>
mlperf::TestMode LoadgenWrapper<T>::get_test_mode(simaai::mlperf_wrapper::MlperfRunType mode) {
  switch(mode) {
    case simaai::mlperf_wrapper::MlperfRunType::PERFORMANCE:
      return TestMode::PerformanceOnly;
    case simaai::mlperf_wrapper::MlperfRunType::ACCURACY:
      return TestMode::AccuracyOnly;
    case simaai::mlperf_wrapper::MlperfRunType::PEAK_PERFORMANCE:
      return TestMode::FindPeakPerformance;
    case simaai::mlperf_wrapper::MlperfRunType::SUBMISSION:
      return TestMode::SubmissionRun;
    default:
      return TestMode::PerformanceOnly;
  }
}

template<typename T>
mlperf::TestScenario LoadgenWrapper<T>::get_test_scenario(simaai::mlperf_wrapper::MlperfTestType scenario) {
  switch(scenario) {
    case simaai::mlperf_wrapper::MlperfTestType::SINGLE_STREAM_MODE:
      return TestScenario::SingleStream;
    case simaai::mlperf_wrapper::MlperfTestType::MULTI_STREAM_MODE:
      return TestScenario::MultiStream;
    case simaai::mlperf_wrapper::MlperfTestType::OFFLINE_MODE:
      return TestScenario::Offline;
    case simaai::mlperf_wrapper::MlperfTestType::SERVER:
      return TestScenario::Server;
    default:
      return TestScenario::SingleStream;
  }
}

template<typename T>
bool LoadgenWrapper<T>::validate_config(void) {
  return true;
};

template<typename T>
std::string LoadgenWrapper<T>::get_scenario_string(void) {
  switch (config.mlperf_test_settings.scenario) {
    case TestScenario::SingleStream:
      return "SingleStream";
    case TestScenario::MultiStream:
      return "MultiStream ";
    case TestScenario::Offline:
      return "Offline";
    case TestScenario::Server:
      return "Server";
    default:
      return "SingleStream";
  }
};

template<typename T>
std::string LoadgenWrapper<T>::get_mode_string(void) {
  switch (config.mlperf_test_settings.mode) {
    case TestMode::PerformanceOnly:
      return " PERFORMANCE ";
    case TestMode::AccuracyOnly:
      return " ACCURACY ";
    case TestMode::FindPeakPerformance:
      return " PEAK PERFORMANCE ";
    case TestMode::SubmissionRun:
      return " SUBMISSION RUN ";
    default:
      return " DEFAULT PERFORMANCE ";
  }
};

template<typename T>
int LoadgenWrapper<T>::loadgen_worker () {
  std::cout << "--- MLPerf Configuration for the current test\n";
  std::cout << "--- MLPerf is scneario: " << get_scenario_string() << "\n" ;
  std::cout << "--- MLPerf is mode: " << get_mode_string() << "\n" ;
  // std::cout << "--- MLPerf is min query count: " << config.mlperf_test_settings.min_query_count << "\n" ;
  // std::cout << "--- MLPerf is min duration ms " << config.mlperf_test_settings.min_duration_ms << "\n" ;

  std::cout << "--- MLPerf Clock Start\n";
  std::cout << "--- MLPerf StartTest\n";

  MLPERF_QSL.num_images = config.num_of_images;

  mlperf::StartTest(&MLPERF_SUT,
                    &MLPERF_QSL,
                    config.mlperf_test_settings,
                    config.mlperf_log_settings);

  std::unique_lock<std::mutex> lock(LoadgenWrapperCtrl::query_mutex);
  LoadgenWrapperCtrl::is_mlperf_done.store(1, std::memory_order_relaxed);
  std::cout << "--- MLPerf Clock Stop - Done\n";
  std::cout << ":::: StartTest(....) Returned\n";
  LoadgenWrapperCtrl::query_cv.notify_one();
  return 0;
}

template<typename T>
void LoadgenWrapper<T>::loadgen_start() {
  if (config.sample_data_fpath.empty()) {
    std::cout << "Inside loadgen_start 2\n";
    throw std::runtime_error("Unable to start loadgen since file was emtpy");
  }
  std::cout << "--- MLPerf is fpath: " << config.sample_data_fpath << "\n" ;
  std::cout << "--- MLPerf is num_of_images: " << config.num_of_images << "\n" ;

  img_loader =
      ImageLoader<T>(config.sample_data_fpath,
                     config.num_of_images);

  // img_loader.load();
  img_loader.load_batch (config.cur_batch_size);
  std::cout << "--- MLPerf Image Loading done\n";
  LoadgenWrapperCtrl::future = std::async(std::launch::async,
                                    &LoadgenWrapper::loadgen_worker,
                                    this);
  return;
};

template<typename T>
void LoadgenWrapper<T>::loadgen_stop() {
  if (LoadgenWrapperCtrl::is_mlperf_done.load() != 0) {
    if (LoadgenWrapperCtrl::future.get() == 0)
      std::cout << "--- MLPerf thread exit success\n";
    LoadgenWrapperCtrl::is_mlperf_done.store(0, std::memory_order_relaxed);
  }
  return;
};

#if 0    
// TODO:FIXEM
template<typename T>
void LoadgenWrapper<T>::loadgen_get_image_data(int batch_size, int8_t * data) {
  int img_sz = img_loader.get_cur_image_size();
  int _switch = 0;

  if ((config.test_type == simaai::mlperf_wrapper::MlperfTestType::SINGLE_STREAM_MODE)
      || (config.test_type == simaai::mlperf_wrapper::MlperfTestType::MULTI_STREAM_MODE))
    _switch = 1;

  // std::cout << "---MLPerf:: Switch " << _switch << "\n";
  
  switch(_switch) {
    case 1:
      for (int i = 0 ; i < batch_size; i++) {
        int8_t * src_data = static_cast<int8_t *>(img_loader.get_image_at_pos(LoadgenWrapperCtrl::cur_ids[i]));
        memcpy(data + i * img_sz,
               src_data,
               img_sz);
      }
      break;
    default: {
      if (batch_size <= 0)
        return;

      // size_t img_sz = img_loader.get_cur_image_size();
      // for (size_t i = offline_idx; i < offline_idx + batch_size ; i++) {
      //   if (i < LoadgenWrapperCtrl::cur_ids.size()) {
      //     int8_t * src_data = static_cast<int8_t *>(img_loader.get_image_at_pos(LoadgenWrapperCtrl::cur_ids[i]));
      //     memcpy(data + (i % batch_size) * img_sz,
      //            src_data,
      //            img_sz);

      //   } else {
      //     // Zero-pad
      //     // std::cout << "--- MLPerf::: Zero padding" << i << "\n";
      //     std::memset(data + (i % batch_size) * img_sz, 0, img_sz);
      //   }
      // }
      offline_idx += batch_size;
      // std::cout << "offline idx "  << offline_idx << std::endl;
      break;
    }
  }
}
#endif
    
template<typename T>
void LoadgenWrapper<T>::loadgen_get_image_phys (uint64_t phys_addr_arr[]) {
  if (!phys_addr_arr)
    throw std::runtime_error("Allocated physical address memory");

  int start_idx = 0, end_idx = 0;
  uint64_t * in_addr_list = reinterpret_cast<uint64_t *>(phys_addr_arr);
  
  if ((config.test_type == simaai::mlperf_wrapper::MlperfTestType::SINGLE_STREAM_MODE)
      || (config.test_type == simaai::mlperf_wrapper::MlperfTestType::MULTI_STREAM_MODE)) {
      start_idx = 0; end_idx = LoadgenWrapperCtrl::cur_ids.size();
  } else {
      start_idx = offline_idx;
      end_idx = offline_idx + config.cur_batch_size;
  }

  for (int i = start_idx ; i < end_idx; i++) {
      int idx = i % config.cur_batch_size;
      if (config.test_type == simaai::mlperf_wrapper::MlperfTestType::OFFLINE_MODE) {
          if (idx >= LoadgenWrapperCtrl::cur_ids.size())
              idx = idx % LoadgenWrapperCtrl::cur_ids.size();
      }
      in_addr_list[idx] = img_loader.get_image_phys_addr(LoadgenWrapperCtrl::cur_ids[i]);
      // fprintf(stderr, "--- MLPerf:Loadgen::Inaddr: [%d]:[%016lx]\n", idx, in_addr_list[idx]);
  }

  if (config.test_type == simaai::mlperf_wrapper::MlperfTestType::OFFLINE_MODE)
      offline_idx += config.cur_batch_size;
  
}

template<typename T>
void LoadgenWrapper<T>::loadgen_get_sample(size_t * bytes, uint64_t phys_addr_arr[]) {
  std::unique_lock<std::mutex> lock(LoadgenWrapperCtrl::query_mutex);

  LoadgenWrapperCtrl::query_cv.wait(lock, [this] {
    return (((LoadgenWrapperCtrl::next_example_ready ==
              LoadgenWrapperCtrl::last_seen + 1) &&
             (LoadgenWrapperCtrl::next_example_ready != 0)) ||
            (LoadgenWrapperCtrl::is_mlperf_done.load(std::memory_order_relaxed) == 1) ||
            (offline_idx != 0));
  });

  LoadgenWrapperCtrl::last_seen = LoadgenWrapperCtrl::next_example_ready;

  if ((LoadgenWrapperCtrl::is_mlperf_done.load(std::memory_order_relaxed) == 1) ||
      ((offline_idx + 1) > LoadgenWrapperCtrl::cur_ids.size())) {
    std::cout << "--- MLPerf REACHED END OF STREAM.\n";;
    *bytes = 0;
    sleep(1);
    loadgen_stop();
    lock.unlock();
    return;
  }

  loadgen_get_image_phys(phys_addr_arr);
  *bytes = static_cast<size_t>(img_loader.get_cur_image_size()) * config.cur_batch_size;
  lock.unlock();
  return;
};

template<typename T>
void LoadgenWrapper<T>::loadgen_complete_sample(const std::vector<int32_t> & arg_max) {
  QuerySampleResponse resp[config.cur_batch_size];
  int iter_size = config.cur_batch_size;

  int examples_idx = -1;

  for (size_t i = 0; i < iter_size; i++) {
    if (config.mlperf_test_settings.scenario == TestScenario::Offline) {
      examples_idx = i + offline_idx - config.cur_batch_size;
      if (examples_idx > LoadgenWrapperCtrl::next_example_ids.size())
        break;
    } else {
      examples_idx = i;
    }
    resp[i].id = LoadgenWrapperCtrl::next_example_ids[examples_idx];
    resp[i].data = reinterpret_cast<uintptr_t>(&arg_max[i]);
    resp[i].size = sizeof(int32_t);
    // printf("Issuing query complete %ld.\n", resp[i].id);
  }

  // At times the number of responses do not match teh cur_batch_size because the samples have exhausted
  // Then please 'clip' the response back to mlperf to the number of completed samples here.
  int completed = 0;
  completed = config.cur_batch_size;

  if (config.mlperf_test_settings.scenario == TestScenario::Offline) {
    if (offline_idx > LoadgenWrapperCtrl::cur_ids.size())
      completed = LoadgenWrapperCtrl::cur_ids.size() - offline_idx + config.cur_batch_size;
    // std::cout << "offline_idx " << offline_idx <<
    //     " LoadgenWrapperCtrl::cur_ids.size() " << LoadgenWrapperCtrl::cur_ids.size() << "\n";
  }

  // std::cout << "Completed " << completed << " cur_batch_size " << config.cur_batch_size << "\n";
  QuerySamplesComplete(resp, completed);
  return;
};

}
}
