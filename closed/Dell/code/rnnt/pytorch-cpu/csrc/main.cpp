#include <loadgen.h>
#include <omp.h>
#include <unistd.h>

#include <iostream>
#include <thread>
#include <vector>

#include "cxxopts.hpp"
#include "test_settings.h"
#include "torch_sut.hpp"

std::map<std::string, mlperf::TestScenario> scenario_map = {
    {"Offline", mlperf::TestScenario::Offline},
    {"Server", mlperf::TestScenario::Server}};

int main(int argc, char **argv) {
  cxxopts::Options opts("rnnt_inference", "MLPerf Benchmark, RNN-T Inference");
  // opts.allow_unrecognised_options();
  opts.add_options(
      "", {{"s,sample_file", "LibriSpeech Sample File",
            cxxopts::value<std::string>()},

           {"m,model_file", "Torch Model File", cxxopts::value<std::string>()},

           {"processor_file", "Audio Processor File",
            cxxopts::value<std::string>()},

           {"pro_inter_parallel", "Instance Number of Producer(Server)",
            cxxopts::value<int>()->default_value("1")},

           {"pro_intra_parallel", "Thread Number per Producer(Server)",
            cxxopts::value<int>()->default_value("8")},

           {"n,inter_parallel", "Instance Number",
            cxxopts::value<int>()->default_value("1")},

           {"j,intra_parallel", "Thread Number per Instance/Consumer(Server)",
            cxxopts::value<int>()->default_value("4")},

           {"pro_batch_size", "Producer batch size(Server)",
            cxxopts::value<int>()->default_value("32")},

           {"b,batch_size", "Model Batch Size",
            cxxopts::value<int>()->default_value("1")},

           {"split_len", "Sequence split len",
            cxxopts::value<int>()->default_value("-1")},

           {"response_size", "Minimum response size for early-response(Server)",
            cxxopts::value<int>()->default_value("-1")},

           {"qos_len", "Minimum sequence length for QoS(Server)",
            cxxopts::value<int>()->default_value("-1")},

           {"k,test_scenario", "Test scenario [Offline, Server]",
            cxxopts::value<std::string>()->default_value("Offline")},

           {"processor", "Whether enbale audio preprocess or not",
            cxxopts::value<bool>()->default_value("false")},

           {"f,profiler_folder",
            "If profiler is True, output json in profiler_folder",
            cxxopts::value<std::string>()->default_value("logs")},

           {"profiler_iter", "Profile iteration number",
            cxxopts::value<int>()->default_value("-1")},

           {"warmup_iter", "Warmup iteration number",
            cxxopts::value<int>()->default_value("-1")},

           {"c,mlperf_config", "Configuration File for LoadGen",
            cxxopts::value<std::string>()->default_value("mlperf.conf")},

           {"u,user_config", "User Configuration for LoadGen",
            cxxopts::value<std::string>()->default_value("user.conf")},

           {"o,output_dir", "Test Output Directory",
            cxxopts::value<std::string>()->default_value("mlperf_output")},

           {"a,accuracy", "Run test in accuracy mode instead of performance",
            cxxopts::value<bool>()->default_value("false")}});

  auto parsed_opts = opts.parse(argc, argv);

  auto sample_file = parsed_opts["sample_file"].as<std::string>();
  auto model_file = parsed_opts["model_file"].as<std::string>();
  auto processor_file = parsed_opts["processor_file"].as<std::string>();
  auto pro_inter_parallel = parsed_opts["pro_inter_parallel"].as<int>();
  auto pro_intra_parallel = parsed_opts["pro_intra_parallel"].as<int>();
  auto inter_parallel = parsed_opts["inter_parallel"].as<int>();
  auto intra_parallel = parsed_opts["intra_parallel"].as<int>();
  auto pro_batch_size = parsed_opts["pro_batch_size"].as<int>();
  auto batch_size = parsed_opts["batch_size"].as<int>();
  auto split_len = parsed_opts["split_len"].as<int>();
  auto response_size = parsed_opts["response_size"].as<int>();
  auto qos_len = parsed_opts["qos_len"].as<int>();
  auto test_scenario = parsed_opts["test_scenario"].as<std::string>();
  auto processor_flag = parsed_opts["processor"].as<bool>();
  auto profiler_folder = parsed_opts["profiler_folder"].as<std::string>();
  auto profiler_iter = parsed_opts["profiler_iter"].as<int>();
  auto warmup_iter = parsed_opts["warmup_iter"].as<int>();
  auto mlperf_conf = parsed_opts["mlperf_config"].as<std::string>();
  auto user_conf = parsed_opts["user_config"].as<std::string>();
  auto output_dir = parsed_opts["output_dir"].as<std::string>();
  auto accuracy_mode = parsed_opts["accuracy"].as<bool>();

  mlperf::TestSettings testSettings;
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = output_dir;

  testSettings.scenario = scenario_map[test_scenario];
  testSettings.FromConfig(mlperf_conf, "rnnt", test_scenario);
  testSettings.FromConfig(user_conf, "rnnt", test_scenario);
  if (accuracy_mode) testSettings.mode = mlperf::TestMode::AccuracyOnly;

  if (test_scenario == "Offline") {
    rnnt::OfflineSUT sut(
        sample_file, model_file, processor_file, inter_parallel, intra_parallel,
        batch_size, split_len, test_scenario, processor_flag, profiler_folder,
        profiler_iter, warmup_iter);

    if (warmup_iter > 0) {
      std::cout << "Start warmup..." << std::endl;
      sleep(15);
      std::cout << "Warmup done." << std::endl;
    }

    std::cout << "Start " << test_scenario << " testing..." << std::endl;
    mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
    std::cout << "Testing done." << std::endl;
  } else {
    rnnt::ServerSUT sut(
        sample_file, model_file, processor_file, pro_inter_parallel,
        pro_intra_parallel, inter_parallel, intra_parallel, pro_batch_size,
        batch_size, split_len, response_size, qos_len, test_scenario,
        processor_flag, profiler_folder, profiler_iter, warmup_iter);

    if (warmup_iter > 0) {
      sleep(15);
      std::cout << "Warmup done." << std::endl;
    }

    std::cout << "Start " << test_scenario << " testing..." << std::endl;
    mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
    std::cout << "Testing done." << std::endl;
  }
  return 0;
}
