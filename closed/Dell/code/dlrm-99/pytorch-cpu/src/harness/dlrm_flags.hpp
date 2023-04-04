#pragma once
#include <gflags/gflags.h>
#include <map>
#include "test_settings.h"

DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_string(model_path, "", "Path to saved model");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in bin format (<full_image_name>.bin). Comma-separated "
    "list if there are more than one input.");

DEFINE_uint64(batch_size, 1000, "batch size for each query");
DEFINE_uint64(warmup_iteration, 5, "Minimum iteration to run warmup for");

// Loadgen test settings
DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "dlrm", "Model name");
DEFINE_uint64(performance_sample_count, 1, "Number of samples to load in performance set.  0=use default");

// Hardware settings
DEFINE_uint64(num_sockets, 2, "number of sockets");
DEFINE_uint64(cores_per_socket, 56, "core per sockets");

// SUT settings
DEFINE_uint64(num_producers, 2, "number of producers");
DEFINE_uint64(consumers_per_producer, 56, "consumer per producers");
DEFINE_uint64(start_consumer_core, 1, "start consumer core");

// configuration files
DEFINE_string(mlperf_conf_path, "./mlperf.conf", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "./user.conf", "Path to user.conf");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");

// QSL
DEFINE_string(sample_partition_path, "", "Path to sample partition file in npy format.");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server},
    {"Offline", mlperf::TestScenario::Offline}};
