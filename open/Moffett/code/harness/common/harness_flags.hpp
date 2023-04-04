/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <gflags/gflags.h>
#include <map>

// settings
DEFINE_uint32(num_complete_threads, 1, "num_complete_threads");
DEFINE_string(model_path, "", "the path to model to load");
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");

DEFINE_string(devices, "all", "Enable comma separated numbered devices");
DEFINE_string(numa, "all", "numa node for devices");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
              "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated "
              "list if there are more than one input.");
DEFINE_uint32(gpu_batch_size, 256, "Max Batch size to use for all devices and engines");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

// Loadgen test settings
DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "resnet50", "Model name");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(single_stream_expected_latency_ns, 100000, "Inverse of desired target QPS");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
                                                       {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
                                                       {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
                                                       {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
                                                         {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
                                                         {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {{"Offline", mlperf::TestScenario::Offline},
                                                           {"SingleStream", mlperf::TestScenario::SingleStream},
                                                           {"Server", mlperf::TestScenario::Server},
                                                           {"MultiStream", mlperf::TestScenario::MultiStream}};
