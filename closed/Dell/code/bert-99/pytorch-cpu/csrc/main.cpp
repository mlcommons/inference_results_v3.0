#include <loadgen.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>

#include "cxxopts.hpp"
#include "torch_sut.hpp"
#include "test_settings.h"

int main(int argc, char **argv) {
  cxxopts::Options opts (
    "bert_inference", "MLPerf Benchmark, BERT Inference");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("m,model_file", "Torch Model File",
     cxxopts::value<std::string>())

    ("s,sample_file", "SQuAD Sample File",
     cxxopts::value<std::string>())

    ("k,test_scenario", "Test scenario [Offline, Server]",
     cxxopts::value<std::string>()->default_value("Offline"))

    ("n,inter_parallel", "Instance Number",
     cxxopts::value<int>()->default_value("1"))

    ("j,intra_parallel", "Thread Number Per-Instance",
     cxxopts::value<int>()->default_value("4"))

    ("c,mlperf_config", "Configuration File for LoadGen",
     cxxopts::value<std::string>()->default_value("mlperf.conf"))

    ("user_config", "User Configuration for LoadGen",
     cxxopts::value<std::string>()->default_value("user.conf"))

    ("o,output_dir", "Test Output Directory",
     cxxopts::value<std::string>()->default_value("mlperf_output"))

    ("b,batch", "Offline Model Batch Size",
     cxxopts::value<int>()->default_value("1"))

    ("w,watermark", "Model sequence length watermark",
     cxxopts::value<int>()->default_value("1290"))

    ("u,upper_watermark", "Model sequence length upper watermark",
     cxxopts::value<int>()->default_value("1600"))

    ("warmup", "Whether system enabled hyper-threading or not",
     cxxopts::value<bool>()->default_value("false"))

    ("a,accuracy", "Run test in accuracy mode instead of performance",
     cxxopts::value<bool>()->default_value("false"))

    ("p,profiler", "Whether output trace json or not",
     cxxopts::value<bool>()->default_value("false"))

    ("f,profiler_folder", "If profiler is True, output json in profiler_folder",
     cxxopts::value<std::string>()->default_value("logs"))

    ;

  auto parsed_opts = opts.parse(argc, argv);

  auto model_file = parsed_opts["model_file"].as<std::string>();
  auto sample_file = parsed_opts["sample_file"].as<std::string>();
  auto inter_parallel = parsed_opts["inter_parallel"].as<int>();
  auto intra_parallel = parsed_opts["intra_parallel"].as<int>();
  auto output_dir = parsed_opts["output_dir"].as<std::string>();
  auto mlperf_conf = parsed_opts["mlperf_config"].as<std::string>();
  auto user_conf = parsed_opts["user_config"].as<std::string>();
  auto batch_size = parsed_opts["batch"].as<int>();
  auto watermark = parsed_opts["watermark"].as<int>();
  auto upper_watermark = parsed_opts["upper_watermark"].as<int>();
  auto warmup = parsed_opts["warmup"].as<bool>();
  auto test_scenario = parsed_opts["test_scenario"].as<std::string>();
  auto accuracy_mode = parsed_opts["accuracy"].as<bool>();
  auto profiler_flag = parsed_opts["profiler"].as<bool>();
  auto profiler_folder = parsed_opts["profiler_folder"].as<std::string>();


  mlperf::TestSettings testSettings;
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = output_dir;

  if (test_scenario == "Offline") {
    BertOfflineSUT sut(
        model_file, sample_file, inter_parallel,
        intra_parallel, batch_size, watermark, warmup, profiler_flag, profiler_folder);
  
    testSettings.scenario = mlperf::TestScenario::Offline;
    testSettings.FromConfig(mlperf_conf, "bert", "Offline");
    testSettings.FromConfig(user_conf, "bert", "Offline");

    if (accuracy_mode)
      testSettings.mode = mlperf::TestMode::AccuracyOnly;

    if (warmup) sleep(5);

    std::cout<<"Start Offline testing..."<<std::endl;
    mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
    std::cout<<"Testing done."<<std::endl;

  } else if (test_scenario == "Server") {
    BertServerSUT server_sut(
        model_file, sample_file, inter_parallel,
        intra_parallel, batch_size, watermark, upper_watermark, warmup, profiler_flag, profiler_folder);

    testSettings.scenario = mlperf::TestScenario::Server;
    testSettings.FromConfig(mlperf_conf, "bert", "Server");
    testSettings.FromConfig(user_conf, "bert", "Server");

    if (accuracy_mode)
      testSettings.mode = mlperf::TestMode::AccuracyOnly;

    sleep(5);
    std::cout<<"Start Server testing..."<<std::endl;
    mlperf::StartTest(&server_sut, server_sut.GetQSL(), testSettings, logSettings);
    std::cout<<"Testing done."<<std::endl;
  }

  return 0;
}
