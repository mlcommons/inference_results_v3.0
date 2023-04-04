#include <glog/logging.h>
#include <chrono>
#include <memory>
#include <thread>
#include "loadgen.h"
#include "dlrm_server.hpp"
#include "dlrm_flags.hpp"
#include "numpy.hpp"

void initTestSettings(mlperf::TestSettings &test) {
    test.scenario = scenarioMap[FLAGS_scenario];
    test.mode = testModeMap[FLAGS_test_mode];
    test.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    test.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
    // test.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
    // this flag will hang whole program, requires investigation here
    test.server_coalesce_queries = true;
    // test.server_target_qps = 600000.0f;
    // test.server_target_latency_ns = 30000000.0f;
    // test.min_duration_ms = 600000;
    // test.min_query_count = 270336;
    test.performance_sample_count_override = FLAGS_performance_sample_count;
    // test.server_coalesce_queries = true;
    LOG(INFO) << "target_qps " << test.server_target_qps;
}

void initLogSettings(mlperf::LogSettings &log) {
    log.log_output.outdir = FLAGS_logfile_outdir;
    log.log_output.prefix = FLAGS_logfile_prefix;
    log.log_output.suffix = FLAGS_logfile_suffix;
    log.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    log.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    log.log_mode = logModeMap[FLAGS_log_mode];
}

void doInference() {
    // Configure the test settings
    mlperf::TestSettings test_settings;
    initTestSettings(test_settings);

    // Configure the logging settings
    mlperf::LogSettings log_settings;
    initLogSettings(log_settings);

    std::vector<int> originalPartition;
    {
        npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
        CHECK_EQ(samplePartitionFile.getDims().size(), 1);

        size_t numPartitions = samplePartitionFile.getDims()[0];
        CHECK_EQ(numPartitions > FLAGS_performance_sample_count, true);

        std::vector<char> tmp(samplePartitionFile.getTensorSize());
        samplePartitionFile.loadAll(tmp);
        originalPartition.resize(numPartitions);
        memcpy(originalPartition.data(), tmp.data(), tmp.size());
        LOG(INFO) << "Loaded " << originalPartition.size() - 1
                  << " sample partitions. (" << tmp.size()
                  << ") bytes.";
    }
    size_t perfPairCount = originalPartition.back(); // get length of day23
    std::vector<DLRMSampleLibraryPtr_t> qsls;
    // numa aware
    int nbNums = FLAGS_num_sockets;
    for (int i = 0; i < nbNums; ++i) {
        qsls.emplace_back(new DLRMSampleLibrary(
                              "DLRM QSL",
                              FLAGS_map_path,
                              splitString(FLAGS_tensor_path, ","),
                              originalPartition,
                              FLAGS_performance_sample_count,
                              perfPairCount, 0, true, false,
                              i, nbNums));
    }
    std::vector<std::vector<float>> scales = {
        {8.556900024414062, 8.357013702392578}, // bot1
        {8.357013702392578, 16.566802978515625}, // bot2
        {16.566802978515625, 25.050329208374023}, // bot3
        {25.050329208374023, 25.050329208374023}, // fuseembint
        {25.050329208374023, 24.18439483642578}, // top1
        {24.18439483642578, 29.01119613647461}, // top2
        {29.01119613647461, 34.23401641845703}, // top3
        {34.23401641845703, 35.30814743041992}, // top4
        {35.30814743041992, 10.125483512878418}, // top5
    };

    std::shared_ptr<DLRMSampleLibraryEnsemble> qslEnsembly(new DLRMSampleLibraryEnsemble(qsls));
    LOG(INFO) << "creating SampleLibraryEnsemble " << qsls.size();
    // dlrm server
    DLRMServerPtr_t sut = std::make_shared<DLRMServer>("DLRM Server",
                                                       qsls,
                                                       scales,
                                                       FLAGS_model_path,
                                                       FLAGS_logfile_outdir,
                                                       FLAGS_batch_size,
                                                       FLAGS_num_sockets,
                                                       FLAGS_cores_per_socket,
                                                       FLAGS_num_producers,
                                                       FLAGS_consumers_per_producer,
                                                       FLAGS_start_consumer_core,
                                                       test_settings.mode == mlperf::TestMode::AccuracyOnly,
                                                       test_settings.scenario == mlperf::TestScenario::Server);
    LOG(INFO) << "Start mlperf test for " << FLAGS_test_mode << " Scenario " << FLAGS_scenario;
    mlperf::StartTest(sut.get(), qslEnsembly.get(), test_settings, log_settings);
    LOG(INFO) << "End mlperf test for " << FLAGS_test_mode << " Scenario " << FLAGS_scenario;
}

int main(int argc, char* argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    doInference();
    return 0;
}
