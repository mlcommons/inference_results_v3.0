#ifndef LOADGEN_WRAPPER_INTF_H_
#define LOADGEN_WRAPPER_INTF_H_

#include <future>
#include "loadgen_wrapper.h"

using namespace simaai::mlperf_wrapper;

/**
 * @brief LoadgenWrapper Singleton class called by gstreamer src/sink plugins
 */
class LoadgenSingleton {
 public:
 private:
  /**
   * Singleton instance of Loadgen singleton wrapper
   */
  static LoadgenSingleton * loadgen;
  /**
   * Mutex to protect access to singleton class
   */
  static std::mutex m;

  LoadgenSingleton(){};
  virtual ~LoadgenSingleton() = default;
  LoadgenSingleton(const LoadgenSingleton&) = delete;

  std::once_flag loadgen_flag;

 public:
  LoadgenWrapper<int8_t> loader;
  LoadgenConfig cfg;

  /**
   * @brief Initialize loadenwrapper from the singleton
   * This api is thread safe to be called from multiple threads
   * @param[in] LoadgenConfiguration passed to loadgenwrapper
   */
  void loadgen_init_impl (LoadgenConfig & cfg) {
    std::call_once(loadgen_flag, [&cfg, this]() {
      loader = LoadgenWrapper<int8_t>(cfg);
      if (!loadgen_start_impl())
        std::cout << "--- MLPerf:Loadgen Error\n";
      std::cout << "Done with start\n";
    });
  }

  /**
   * @brief Start the loadgen, loads the image data into dram
   */
  bool loadgen_start_impl () {
    loader.loadgen_start();
    std::cout << "Returning from loadgen start\n";
    return true;
  }

  /**
   * @brief Helper API to get the current batchsize
   */
  size_t loadgen_get_cur_bs () {
    return loader.get_current_batch_size();
  }

  /**
   * @brief Helper API wrapper to get the current sample based on loaded samples
   * @param[out] out param of writing backt he number of bytes loaded
   */
  void loadgen_get_sample(size_t * bytes, int8_t * data) {
    uint64_t addr[8];
    loader.loadgen_get_sample(bytes, addr);
    if (data == NULL)
      std::cout << "--- MLPerf::ERROR Data is null please check loader\n";

    return;
  }

  /**
   * @brief Helpder API to get the current sample idx
   * @return Returns the id for bs=1
   */
  int loadgen_get_current_sample_index(int idx) {
    return loader.get_current_sample_index(idx);
  }

  /**
   * @brief Wrapper to loadgen_complete sample, which completes the current sample so that we get the next one
   * @param arg_max the arg_max calculted from the sink plugin
   */
  void loadgen_complete_sample(const std::vector<int32_t> & arg_max) {
    loader.loadgen_complete_sample(arg_max);
  }

  /**
   * Default delete copy and get instance operator
   */
  void operator=(const LoadgenSingleton &) = delete;

  static LoadgenSingleton *get_instance();
};

#endif // LOADGEN_WRAPPER_INTF_H_
