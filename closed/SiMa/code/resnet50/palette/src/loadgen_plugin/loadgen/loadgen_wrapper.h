#ifndef LOADGEN_WRAPPER_H_
#define LOADGEN_WRAPPER_H_

#include <mutex>
#include <thread>
#include <vector>

#include <loadgen.h>
#include <test_settings.h>

#include "image_loader.h"
#include "loadgen_query_intf.h"
#include "loadgen_wrapper_ctrl.h"

using namespace mlperf;

#define SINGLE_STREAM_BS (1)
#define MULTI_STREAM_BS (8)
#define OFFLINE_BS (24)

namespace simaai {
namespace mlperf_wrapper {

/**
 * @brief MlperfTestType enum, representing the mlperf testtype in loadgen wrapper
 */
enum class MlperfTestType {
  SINGLE_STREAM_MODE = 0,
  MULTI_STREAM_MODE,
  OFFLINE_MODE,
  SERVER
};

/**
 * @brief MlperfRunType enum, representing the mlperf runtype in loadgen wrapper
 */
enum class MlperfRunType {
  PERFORMANCE = 0,
  ACCURACY,
  PEAK_PERFORMANCE,
  SUBMISSION
};

/**
 * @brief Dimension of the input image
 */
struct Dims {
  int stride;
  int width;
  int height;
};

/**
 * @brief LoadgenConfiguration used by mlperf loadgen
 */
struct LoadgenConfig {
  std::string sut_name; ///< Readable name for SystemUnderTest
  std::string qsl_name; ///< Readable name for QSL

  std::string sample_data_fpath; ///< Samples to be used by the SUT
  std::string mlperf_conf_fpath; ///< mlperf.conf path
  std::string user_conf_fpath; ///< user.conf path

  std::string model_name; ///< Readable name for the model under test

  mlperf::TestSettings mlperf_test_settings; ///< MLPerf Loadgen settings
  mlperf::LogSettings mlperf_log_settings; ///< Mlperf Log settings

  MlperfTestType test_type; ///< SiMa Mlperf test type
  MlperfRunType run_type; ///< SiMa mlperf run type

  int64_t cur_batch_size; ///< Batch size settings valid values are 1,8,24 for SiMa models
  int64_t cur_buffer_size; ///< The buffer size for memeory allocation
  int64_t num_of_images; ///< Number of sample images used by the framework for evaluation

  struct Dims dims; ///< Input dimensions of the sample image
};

using LoadgenConfig = simaai::mlperf_wrapper::LoadgenConfig;

/**
 * @brief LoagenWrapper class used to communicate back from loadgen and used by the gstreamer plugins
 */
template <typename T>
class LoadgenWrapper {
 public:
  LoadgenWrapper() = default;

  // Create loadgen config from input
  LoadgenWrapper (LoadgenConfig & config);
  ~LoadgenWrapper() {};

  /**
   * @brief loadgen_start API to pre-load the image data to dram
   */
  void loadgen_start();
  /**
   * @brief loadgen_stop cleanup for loadgen wrapper, DO NOT call from gstreamer.
   */
  void loadgen_stop();
  /**
   * @brief the worker API to call the loadgen(mlperf) StartTest
   */
  int loadgen_worker();
  bool validate_config (void);

  /**
   * @brief helper api to get current test mode
   * @return Accuray , Performance, Submission, PeakPerformance
   */
  mlperf::TestMode get_test_mode(simaai::mlperf_wrapper::MlperfRunType mode);
  /**
   * @brief Helper API to get current test scenario
   * @return SingleStream, MultiStream, Offline etc
   */
  mlperf::TestScenario get_test_scenario(simaai::mlperf_wrapper::MlperfTestType scenario);

  /**
   * @brief Helper API to get current scenario string
   */
  std::string get_scenario_string(void);
  std::string get_mode_string(void);
  size_t get_current_batch_size() { return config.cur_batch_size; };
  int64_t get_current_buffer_size() { return config.cur_buffer_size; };
  int64_t get_current_sample_index(int i) {
    // std::cout << "--- MLPerf:::Current sample idx " << i << "\n";
    // std::cout << "--- MLPerf:::Offline idx " << offline_idx << "\n";
    // std::cout << "--- MLPerf:::Sample idx " << LoadgenWrapperCtrl::cur_ids.size() << "\n";
    // std::cout << "--- MLPerf:::Sample idx " << LoadgenWrapperCtrl::cur_ids[i + offline_idx] << "\n";
    if ((config.cur_batch_size != SINGLE_STREAM_BS) ||
        (config.cur_batch_size != MULTI_STREAM_BS))
      return LoadgenWrapperCtrl::cur_ids[i + offline_idx];
    return LoadgenWrapperCtrl::cur_ids[i];
  }

  int8_t * get_image_at_pos(int pos) {
    // std::cout << "--- MLPerf::Pos pos" << pos << "\n";
    return static_cast<int8_t *>(img_loader.get_image_at_pos(pos));
  }
  /**
   * @brief Get the sample from the idx returned from IssueQuery, from loadgen
   * @param[out]  bytes read from samples
   * @return int8_t image data or NULL
   */
  void loadgen_get_sample(size_t * bytes, uint64_t phys_addr_ptr[]);

  /**
   * @brief Complete the current sample by calling complete of loadgen
   * @param[in]  arg_max, argmax for 'n' samples from next_examples_ids
   */
  void loadgen_complete_sample(const std::vector<int32_t> & arg_max);

  // Getter functions
  void loadgen_get_image_data(int batch_size, int8_t * data);
  void loadgen_get_image_phys(uint64_t phys_addr_arr[]);
    
 private:
  struct LoadgenConfig config;

  std::vector<mlperf::ResponseId> next_example_ids;
  std::vector<mlperf::QuerySampleResponse> responses;

  bool is_toy_mode;
  int64_t offline_idx;
  int64_t offline_queries_completed;
  int8_t * multistream_buffer;
  int8_t * offline_buffer;

  SiMaSUT MLPERF_SUT;
  SiMaQSL MLPERF_QSL;

  size_t last_seen = 1;
  simaai::mlperf_wrapper::ImageLoader<T> img_loader;
};
}
}

// Foward declaration
template class simaai::mlperf_wrapper::LoadgenWrapper<int8_t>;

#endif // LOADGEN_WRAPPER_H_
