#include <loadgen.h>
#include <test_settings.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <vector>

#include "postprocessor.hpp"
#include "rebel_qsl.hpp"
#include "rebel_sut.hpp"

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("rebel_mlperf");
  program.add_argument(std::string{"task"}).action([](const std::string& value) {
    const std::vector<std::string> choices = {"vision/classification", "language/bert"};
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
      std::cerr << "Unrecognized task: " << value << std::endl;
      exit(1);
    }
    return value;
  });
  program.add_argument(std::string{"scenario"}).action([](const std::string& value) {
    const std::vector<std::string> choices = {"SingleStream", "MultiStream", "Offline"};
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
      std::cerr << "Unrecognized scenario: " << value << std::endl;
      exit(1);
    }
    return value;
  });
  program.add_argument(std::string{"mode"}).action([](const std::string& value) {
    const std::vector<std::string> choices = {"AccuracyOnly", "PerformanceOnly",
                                              "FindPeakPerformance", "SubmissionRun"};
    if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
      std::cerr << "Unrecognized mode: " << value << std::endl;
      exit(1);
    }
    return value;
  });
  program.add_argument("--config")
      .help("Path to mlperf.conf")
      .default_value(std::string("./mlperf.conf"));
  program.add_argument("--user-config")
      .help("Path to user.conf")
      .default_value(std::string("./user.conf"));
  program.add_argument("-n", "--count")
      .help("Limit number of samples for debugging")
      .default_value(0)
      .scan<'i', int>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  std::string system_desc_id = "ATOM";
  std::string task = program.get<std::string>("task");
  std::string scenario = program.get<std::string>("scenario");
  std::string mode = program.get<std::string>("mode");
  std::string config_path = program.get<std::string>("config");
  std::string user_config_path = program.get<std::string>("user-config");
  int count = program.get<int>("count");

  std::shared_ptr<rebel::QuerySampleLibrary> qsl;
  std::shared_ptr<rebel::SystemUnderTest> sut;
  std::shared_ptr<rebel::PostProcessor> pp;
  mlperf::TestSettings test_settings;
  std::string model;
  if (task == "vision/classification" && scenario == "SingleStream") {
    qsl = std::make_shared<rebel::ImageNetQuerySampleLibrary>(count);
    auto pp_func = rebel::ArgmaxFunc<uint16_t, int32_t, 1000>;
    pp = std::make_shared<rebel::SimplePostProcessor>(pp_func, 1, sizeof(int32_t));
    sut = std::make_shared<rebel::SimpleSystemUnderTest>("scratch/models/resnet50_ss/RebelRuntime",
                                                         *qsl, *pp);
    test_settings.scenario = mlperf::TestScenario::SingleStream;
    model = "resnet50";
  } else if (task == "vision/classification" && scenario == "MultiStream") {
    qsl = std::make_shared<rebel::ImageNetQuerySampleLibrary>(count);
    auto pp_func = rebel::ArgmaxFunc<uint16_t, int32_t, 1000>;
    pp = std::make_shared<rebel::BatchPostProcessor>(pp_func, 1, 8, 1024 * sizeof(int16_t),
                                                     sizeof(int32_t));
    sut = std::make_shared<rebel::BatchSystemUnderTest>("scratch/models/resnet50_ms/RebelRuntime",
                                                        *qsl, *pp, 8);
    test_settings.scenario = mlperf::TestScenario::MultiStream;
    model = "resnet50";
  } else if (task == "vision/classification" && scenario == "Offline") {
    qsl = std::make_shared<rebel::ImageNetQuerySampleLibrary>(count);
    auto pp_func = rebel::ArgmaxFunc<uint16_t, int32_t, 1000>;
    pp = std::make_shared<rebel::BatchPostProcessor>(pp_func, 1, 8, 1024 * sizeof(int16_t),
                                                     sizeof(int32_t));
    sut = std::make_shared<rebel::AsyncSystemUnderTest>("scratch/models/resnet50_ms/RebelRuntime",
                                                        *qsl, *pp, 8, 2);
    test_settings.scenario = mlperf::TestScenario::Offline;
    model = "resnet50";
  } else if (task == "language/bert" && scenario == "SingleStream") {
    qsl = std::make_shared<rebel::SQuADQuerySampleLibrary>(count);
    pp = std::make_shared<rebel::SimplePostProcessor>(rebel::IdentityFunc, 3,
                                                      sizeof(uint16_t) * 384 * 2);
    sut = std::make_shared<rebel::SimpleSystemUnderTest>(
        "scratch/models/bert_large_ss/RebelRuntime", *qsl, *pp);
    test_settings.scenario = mlperf::TestScenario::SingleStream;
    model = "bert_large";
  } else {
    throw std::exception();
  }

  if (program.get("mode") == "AccuracyOnly")
    test_settings.mode = mlperf::TestMode::AccuracyOnly;
  else if (program.get("mode") == "PerformanceOnly")
    test_settings.mode = mlperf::TestMode::PerformanceOnly;
  else if (program.get("mode") == "FindPeakPerformance")
    test_settings.mode = mlperf::TestMode::FindPeakPerformance;
  else if (program.get("mode") == "SubmissionRun")
    test_settings.mode = mlperf::TestMode::SubmissionRun;
  else
    throw std::exception();

  test_settings.FromConfig(config_path, model, scenario);
  test_settings.FromConfig(user_config_path, model, scenario);

  // Configure the logging settings
  mlperf::LogSettings log_settings;
  if (program.get("task") == "vision/classification" && scenario == "SingleStream")
    log_settings.log_output.outdir = "./logs/vision_classification_SingleStream";
  else if (program.get("task") == "vision/classification" && scenario == "MultiStream")
    log_settings.log_output.outdir = "./logs/vision_classification_MultiStream";
  else if (program.get("task") == "vision/classification" && scenario == "Offline")
    log_settings.log_output.outdir = "./logs/vision_classification_Offline";
  else if (program.get("task") == "language/bert" && scenario == "SingleStream")
    log_settings.log_output.outdir = "./logs/language_bert_SingleStream";
  else if (program.get("task") == "language/bert" && scenario == "Offline")
    log_settings.log_output.outdir = "./logs/language_bert_Offline";
  log_settings.log_output.prefix_with_datetime = false;
  log_settings.log_output.copy_detail_to_stdout = false;
  log_settings.log_output.copy_summary_to_stdout = true;
  log_settings.log_mode = mlperf::LoggingMode::AsyncPoll;
  log_settings.log_mode_async_poll_interval_ms = 1000;
  log_settings.enable_trace = false;

  // Create log dir
  std::filesystem::create_directories(log_settings.log_output.outdir);

  // Start test
  mlperf::StartTest(static_cast<mlperf::SystemUnderTest*>(sut.get()),
                    static_cast<mlperf::QuerySampleLibrary*>(qsl.get()), test_settings,
                    log_settings);

  // Clean-up
  return 0;
}
