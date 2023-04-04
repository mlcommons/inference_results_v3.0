/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
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
 *
 * Modified by NEUCHIPS on 2023.
 *
 */

#include "config.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

#include "loadgen.h"
#include "test_settings.h"

#include "dlrm_qsl.hpp"
#include "dlrm_server.h"
#include "numpy.hpp"
#include "qsl.hpp"
#include "utils.hpp"

#include <chrono>
#include <dlfcn.h>
#include <thread>

DEFINE_string(gpu_engines, "", "Engine");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "dlrm", "Model name");
DEFINE_uint32(server_num_issue_query_threads, 0, "Number of IssueQuery threads used in Server scenario");
DEFINE_uint32(gpu_batch_size, N3000_MAX_BATCH_SIZE, "Max Batch size to use for all devices and engines");
DEFINE_bool(use_graphs, false, "Enable cudaGraphs for TensorRT engines");
DEFINE_bool(verbose, false, "Use verbose logging");

DEFINE_uint32(gpu_copy_streams, 1, "[CURRENTLY NOT USED] Number of copy streams");
DEFINE_uint32(gpu_num_bundles, 2, "Number of event-buffer bundles per GPU");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");
DEFINE_uint32(gpu_inference_streams, 1, "Number of inference streams");

DEFINE_double(warmup_duration, 1.0, "Minimum duration to run warmup for");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");

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

// QSL arguments
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(sample_partition_path, "", "Path to sample partition file in npy format.");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than "
    "one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");
DEFINE_bool(start_from_device, false, "Assuming that inputs start from device memory in QSL");

// Dataset arguments
DEFINE_uint32(min_sample_size, 100, "Minimum number of pairs a sample can contain.");
DEFINE_uint32(max_sample_size, 700, "Maximum number of pairs a sample can contain.");

// BatchMaker arguments
DEFINE_uint32(num_staging_threads, 8, "Number of staging threads in DLRM BatchMaker");
DEFINE_uint32(num_staging_batches, 8, "Number of staging batches in DLRM BatchMaker");

DEFINE_uint32(
    max_pairs_per_staging_thread, N3000_MAX_BATCH_SIZE, "Maximum pairs to copy in one BatchMaker staging thread (0 = use default"); //neuchips
DEFINE_uint32(split_threshold, 500, "Sample size threshold to start round-robin on BatchMakers");
DEFINE_bool(check_contiguity, false,
    "Whether to use contiguity checking in BatchMaker (default: false, recommended: true for Offline)");

DEFINE_string(numa_config, "1:32-47&0:48-63", "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices.");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"Server", mlperf::TestScenario::Server},
};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1;  // Log to console
#ifdef DEBUG_GLOG_INFO
    FLAGS_minloglevel  = 0;    // 0: INFO, WARNING, ERROR, and FATAL
#else
    FLAGS_minloglevel  = 1;    // 0: INFO, WARNING, ERROR, and FATAL
#endif
    ::google::InitGoogleLogging("Neuchips N3000 mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "DLRM_HARNESS";

    LOG(INFO) << "main() running...";

    {
        int num_gpu = 0;
        LOG(INFO) << "numa_config " << FLAGS_numa_config;
        char ch = ':';
        for (int i = 0; (i = FLAGS_numa_config.find(ch, i)) !=
                std::string::npos; i++) {
            num_gpu++;
        }
        LOG(INFO) << "num_gpu: " << num_gpu;

        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_test_mode];

        LOG(INFO) << "testSettings.mode " << FLAGS_test_mode;
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model,
                    FLAGS_scenario);
        LOG(INFO) << "testSettings.FromConfig mlperf_conf " <<
                    FLAGS_mlperf_conf_path << ", FLAGS_model " << FLAGS_model
                    << " FLAGS_scenario " << FLAGS_scenario;
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model,
                    FLAGS_scenario);
        LOG(INFO) << "testSettings.FromConfig user_conf" << FLAGS_user_conf_path
                  << ", FLAGS_model " << FLAGS_model
                  << " FLAGS_scenario " << FLAGS_scenario;

        LOG(INFO) << "FLAGS_server_num_issue_query_threads "
                  << FLAGS_server_num_issue_query_threads;

        if (0 == FLAGS_scenario.compare("Server")) {
            LOG(INFO) << "FLAGS_scenario is Server.";
            testSettings.server_num_issue_query_threads =
                    FLAGS_server_num_issue_query_threads;
        }
        testSettings.server_coalesce_queries = true;

        // Configure the logging settings
        mlperf::LogSettings logSettings;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;
        logSettings.log_output.prefix_with_datetime =
                                        FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.copy_detail_to_stdout =
                                        FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout =
                                    !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms =
                                    FLAGS_log_mode_async_poll_interval_ms;
        logSettings.enable_trace = FLAGS_log_enable_trace;

        std::vector<int> gpus(num_gpu);
        std::iota(gpus.begin(), gpus.end(), 0);

        std::vector<int> originalPartition;

        // Scope to automatically close the file
        {
            npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
            CHECK_EQ(samplePartitionFile.getDims().size(), 1);

            LOG(INFO) << "samplePartitionFile.getDims()[0] "
                      << samplePartitionFile.getDims()[0];

            size_t numPartitions = samplePartitionFile.getDims()[0];
            LOG(INFO) << "numPartitions " << numPartitions
                      << ", FLAGS_performance_sample_count "
                      << FLAGS_performance_sample_count;

            std::vector<char> tmp(samplePartitionFile.getTensorSize());
            samplePartitionFile.loadAll(tmp);

            originalPartition.resize(numPartitions);
            memcpy(originalPartition.data(), tmp.data(), tmp.size());
            LOG(INFO) << "Loaded " << originalPartition.size() - 1
                      << " sample partitions. (" << tmp.size()
                      << ") bytes.";
        }

        // Force underlying QSL to load all samples,
        // since we want to be able to grab any partition given the sample
        // index.
        size_t perfPairCount = originalPartition.back();
        const auto numaConfig = parseNumaConfig(FLAGS_numa_config);
        std::vector<DLRMSampleLibraryPtr_t> qsls;

        LOG(INFO) << "originalPartition.back() perfPairCount " << perfPairCount;
        LOG(INFO) << "FLAGS_tensor_path " << FLAGS_tensor_path;
        if (numaConfig.empty()) {
            auto oneQsl = std::make_shared<DLRMSampleLibrary>("DLRM QSL", FLAGS_map_path,
                splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count, perfPairCount,
                0, true, FLAGS_start_from_device, -1, 0);
            qsls.emplace_back(oneQsl);
        } else {
            const int32_t nbNumas = numaConfig.size();
            for (int32_t numaIdx = 0; numaIdx < nbNumas; numaIdx++) {
                LOG(INFO) << "constructQsl bindNumaMemPolicy target numa: "
                          << numaConfig[numaIdx].first[0];
                bindNumaMemPolicy(numaConfig[numaIdx].first[0],
                                    NUM_NUMA_ZONES);
                auto constructQsl = [&]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    auto oneQsl =
                        std::make_shared<DLRMSampleLibrary>("DLRM QSL",
                        FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
                        originalPartition, FLAGS_performance_sample_count,
                        perfPairCount, 0, true, FLAGS_start_from_device,
                        numaIdx, numaConfig.size(), numaConfig[numaIdx].first);
                    qsls.emplace_back(oneQsl);
                };
                std::thread th(constructQsl);
                bindThreadToCpus(th, numaConfig[numaIdx].second);
                th.join();
            }
            resetNumaMemPolicy();
        }

        LOG(INFO) << "Running batch size FLAGS_max_pairs_per_staging_thread: "
                  << FLAGS_max_pairs_per_staging_thread
                  << " FLAGS_gpu_batch_size: " << FLAGS_gpu_batch_size;

        auto dlrm_server
            = std::make_shared<DLRMServer>("DLRM SERVER",
                FLAGS_gpu_engines, qsls, gpus, FLAGS_gpu_batch_size,
                FLAGS_gpu_num_bundles, FLAGS_complete_threads,
                FLAGS_gpu_inference_streams, FLAGS_warmup_duration,
                FLAGS_num_staging_threads, FLAGS_num_staging_batches,
                FLAGS_max_pairs_per_staging_thread, FLAGS_split_threshold,
                FLAGS_check_contiguity, FLAGS_start_from_device, numaConfig,
                FLAGS_scenario, FLAGS_server_num_issue_query_threads);

        std::shared_ptr<DLRMSampleLibraryEnsemble> qslEnsemble(
                new DLRMSampleLibraryEnsemble(qsls));

        LOG(INFO) << "Starting running actual test.";
        StartTest(dlrm_server.get(), qslEnsemble.get(),
            testSettings, logSettings);
        LOG(INFO) << "Finished running actual test.";
    }

    LOG(INFO) << "main() finished ...";

    return 0;
}
