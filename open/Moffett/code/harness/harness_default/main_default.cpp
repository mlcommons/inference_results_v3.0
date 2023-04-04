/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <glog/logging.h>
#include <cassert>
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <thread>
#if !SERVER_MODE
#include <omp.h>
#endif
#include "server.h"
#include "loadgen.h"
#include "timer.h"
#include "utils.hpp"

#include "harness_flags.hpp"

extern uint64_t g_samples_size;

auto RN50DataProcess = [](moffett::spu_backend::InOutBindingHandle input_binding, const std::vector<std::size_t>& real_input_size,
                          moffett::spu_backend::Batch* batch, std::size_t batch_size, std::size_t device_idx,
                          const moffett::spu_backend::BackendSettings& server_settings,
                          const std::vector<std::size_t>& sample_size) {
  {
    size_t count = batch->datas[0].size();
    size_t residue = count % 4;
    if (residue > 0) {
      for (size_t i=0; i < 4 - residue; ++i) {
        batch->datas[0].emplace_back(batch->datas[0][0]);
      }
    }
    batch->input_bindings = &(batch->datas);
  }
};

auto RN50ResponseCallback = [](moffett::spu_backend::Batch* batch, moffett::spu_backend::InOutBindingHandle infer_result_data, const std::vector<std::size_t>& output_data_size,
                               std::size_t batch_size) {
//  auto start = std::chrono::high_resolution_clock::now();
  static size_t count = 0;
  auto& responses = batch->responses;
  auto size = responses->size();

  // get max value
  std::vector<int32_t> max_indexs(size);
#if !SERVER_MODE
#pragma omp parallel for
#endif
  for (int batch_idx = 0; batch_idx < size; ++batch_idx) {
    int32_t max_index = 0;
    int8_t* sample = (int8_t*)(*infer_result_data)[0][batch_idx];
    for (size_t i = 0; i < output_data_size[0] && i < 1000; ++i) {
      if ( *(sample + i) > *(sample + max_index)) {
        max_index = i;
      }
    }
    max_indexs[batch_idx] = max_index;
    (*responses)[batch_idx].data = (uintptr_t)(&max_indexs[batch_idx]);
    (*responses)[batch_idx].size = sizeof(uint32_t);
  }
//  auto end = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//  printf("[post process] costs %f s\n", time_span.count());

  // Parse the index in callback.
  mlperf::QuerySamplesComplete(
    &(*responses)[0], responses->size(),
    [](mlperf::QuerySampleResponse* response) {
    });
};

static void InitTestSettings(mlperf::TestSettings& testSettings) {
  testSettings.scenario = scenarioMap[FLAGS_scenario];
  testSettings.mode = testModeMap[FLAGS_test_mode];

  testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
  testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
  testSettings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
  testSettings.server_coalesce_queries = true;
  testSettings.performance_sample_count_override =
    std::min(testSettings.performance_sample_count_override, FLAGS_performance_sample_count);
}

static void InitLogSettings(mlperf::LogSettings& logSettings) {
  logSettings.log_output.outdir = FLAGS_logfile_outdir;
  logSettings.log_output.prefix = FLAGS_logfile_prefix;
  logSettings.log_output.suffix = FLAGS_logfile_suffix;
  logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
  logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
  logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
  logSettings.log_mode = logModeMap[FLAGS_log_mode];
  logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
  logSettings.enable_trace = FLAGS_log_enable_trace;
}

static void InitBackendSettings(moffett::spu_backend::BackendSettings& backendSettings, const mlperf::TestSettings& testSettings) {
  backendSettings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
  if (FLAGS_scenario == "Server") {
    backendSettings.mode = 1;
    backendSettings.target_latency_ms = std::chrono::microseconds(testSettings.server_target_latency_ns / 1000);
  }
  backendSettings.numa_map = GetDevicesIdsFromString(FLAGS_numa);
  backendSettings.batch_size = FLAGS_gpu_batch_size;
  backendSettings.model_path = FLAGS_model_path;
//  backendSettings.verbose = FLAGS_verbose;
  backendSettings.inputs_size = {224*224*3};
  backendSettings.outputs_size = {1024};
}

/* Helper function to actually perform inference using MLPerf Loadgen */
void inference() {
  // Configure the test settings
  mlperf::TestSettings testSettings;
  InitTestSettings(testSettings);

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  InitLogSettings(logSettings);

  moffett::spu_backend::ServerPtr_t server = std::make_shared<moffett::spu_backend::Server>("RN50");

  moffett::spu_backend::BackendSettings backendSettings;
  InitBackendSettings(backendSettings, testSettings);

  moffett::spu_backend::ServerParams sut_params;
  sut_params.device_ids = FLAGS_devices;
  sut_params.scenario = FLAGS_scenario;

  std::cout << "Creating samples loader." << std::endl;
  std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
  const std::size_t padding = 0;

  std::vector<samplesLoader::SampleLoaderPtr> loaders;
  std::vector<int32_t> devices_list = GetDevicesIdsFromString(sut_params.device_ids);
  auto oneQsl = std::make_shared<samplesLoader::AppenedSampleLoader>(
      "suPerf_SampleLibrary", splitString(FLAGS_tensor_path, ","),
      FLAGS_performance_sample_count ? FLAGS_performance_sample_count : FLAGS_gpu_batch_size,
      backendSettings.inputs_size);
  for (int32_t device_idx : devices_list) {
    auto constructQsl = [&]() {
      std::this_thread::sleep_for(std::chrono::microseconds(200));

      backendSettings.devices.push_back(device_idx);
      server->AddSampleLibrary(device_idx, oneQsl);
      loaders.emplace_back(oneQsl);
    };
    std::thread th(constructQsl);
    th.join();
  }
  std::shared_ptr<mlperf::QuerySampleLibrary> aggregated_loader =
    std::shared_ptr<samplesLoader::UniverseSampleLoader>(new samplesLoader::UniverseSampleLoader(loaders));

  std::cout << "Finished Creating samples loader." << std::endl;

  std::cout << "Setting up SUT." << std::endl;
  server->Setup(backendSettings, sut_params, RN50ResponseCallback,
                RN50DataProcess); // Pass the requested server settings and params to our SUT

  std::cout << "Finished setting up SUT." << std::endl;

  std::cout << "Starting running actual test." << std::endl;
  mlperf::StartTest(server.get(), aggregated_loader.get(), testSettings, logSettings);
  std::cout << "Finished running actual test." << std::endl;

  server->Done();

  aggregated_loader.reset();
  server.reset();
}

int main(int argc, char* argv[]) {
  // Initialize logging
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging("Moffett mlperf");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  // Perform inference
  inference();

  return 0;
}
