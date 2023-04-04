#include <loadgen.h>
#include <test_settings.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <map>

//#include <glog/logging.h>

#include "sut.hpp"
#include "sut_server.hpp"
#include "dataset.hpp"
#include "input_flags.hpp"

std::map<std::string, mlperf::TestScenario> test_scenarios;
/*
test_scenarios["SingleStream"] = mlperf::TestScenario::SingleStream;
test_scenarios["Offline"] = mlperf::TestScenario::Offline;
test_scenarios["Server"] = mlperf::TestScenario::Server;
*/

int main(int argc, char **argv){
    //google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    test_scenarios["SingleStream"] = mlperf::TestScenario::SingleStream;
    test_scenarios["Offline"] = mlperf::TestScenario::Offline;
    test_scenarios["Server"] = mlperf::TestScenario::Server;

 
    mlperf::TestSettings settings;
    mlperf::LogSettings log_settings;
    log_settings.enable_trace = false;

    settings.FromConfig(FLAGS_mlperf_conf, FLAGS_model_name, FLAGS_scenario);
    settings.FromConfig(FLAGS_user_conf, FLAGS_model_name, FLAGS_scenario);

    if (FLAGS_mode.compare("Accuracy") == 0) {
        settings.mode = mlperf::TestMode::AccuracyOnly;
    }
    if (FLAGS_mode.compare("Performance") == 0) {
        settings.mode = mlperf::TestMode::PerformanceOnly;
    }
    if (FLAGS_mode.compare("Submission") == 0) {
        settings.mode = mlperf::TestMode::SubmissionRun;
    }

    //settings.mode = test_modes[FLAGS_scenario];
    
    if (FLAGS_mode.compare("FindPeakPerformance") == 0) {
        settings.mode = mlperf::TestMode::FindPeakPerformance;
    }
    

    settings.scenario = test_scenarios[FLAGS_scenario];

    if (FLAGS_scenario.compare("Server")==0){
        SUTServer sut_server(FLAGS_model_path, FLAGS_data_path, FLAGS_total_sample_count, FLAGS_num_instance, FLAGS_cpus_per_instance, FLAGS_user_conf, FLAGS_warmup_iters, FLAGS_batch_size);
        sut_server.doWarmup();
        std::cout << " STARTING SERVER TEST\n";
        mlperf::StartTest(&sut_server, sut_server.GetQsl(), settings, log_settings);
    } else if(FLAGS_scenario.compare("Offline")==0){
        SUT sut_offline(FLAGS_model_path, FLAGS_data_path, FLAGS_total_sample_count, FLAGS_num_instance, FLAGS_cpus_per_instance, FLAGS_user_conf, FLAGS_warmup_iters, FLAGS_batch_size);
        sut_offline.doWarmup();
        std::cout << " STARTING OFFLINE TEST\n";
        mlperf::StartTest(&sut_offline, sut_offline.GetQsl(), settings, log_settings);
    } else {
        std::cout << " Unsupported scenario\n";
    }

    return 0;
}
