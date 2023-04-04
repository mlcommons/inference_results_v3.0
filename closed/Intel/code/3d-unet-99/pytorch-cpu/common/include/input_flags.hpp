#include <string>
#include <vector>
#include <gflags/gflags.h>
//#include <boost/filesystem.hpp>
#include <fstream>

/** ========================== MLPerf flags ==========================**/
bool FileExists(const std::string file_path){

    std::ifstream infile(file_path.c_str());
    return infile.good();
}


// Scenario flag
static const char scenario_message[] = "MLPerf scenario: one of SingleStream, Offline, Server, MultiStream";

static bool validate_scenario(const char* scenario, const std::string& value){
    std::list<std::string> scenarios = {"SingleStream", "Offline", "Server", "MultiStream"};

    for (std::list<std::string>::iterator it = scenarios.begin(); it != scenarios.end(); ++it){
        if (value.compare(*it) == 0) return true;
    }
    std::cout << value << " is not a valid/supported scenario. " << scenario_message << std::endl;
    return false;
}

DEFINE_string(scenario, "SingleStream", scenario_message);
DEFINE_validator(scenario, &validate_scenario);


// Mode flag
static const char mode_message[] = "MLPerf mode: Performance, Accuracy";

static bool validate_mode(const char* mode, const std::string& value){
    std::list<std::string> modes = {"Performance", "Accuracy"};

    for (std::list<std::string>::iterator it = modes.begin(); it != modes.end(); ++it){
        if (value.compare(*it) == 0) return true;
    }
    std::cout << value << " is not a valid/supported mode. " << mode_message << std::endl;
    return false;
}

DEFINE_string(mode, "Performance", mode_message);
DEFINE_validator(mode, &validate_mode);

// Config file
static const char mlperf_conf_message[] = "MLPerf default config file (mlperf.conf)";

static bool validate_mlperf_config(const char* mlperf_conf, const std::string& mlperf_conf_value){

    if (mlperf_conf_value.length() == 0){
        //std::cout << " --mlperf_conf not provided." << std::endl;
        return false;
    }

    if (!FileExists( mlperf_conf_value ) ){
        std::cout << " [ERROR]: Provided MLPerf config '" << mlperf_conf_value << "' could not be found. Please provide </path/to/mlperf.conf>" << std::endl;
        return false;
    }

    return true;
}

DEFINE_string(mlperf_conf, "", mlperf_conf_message);
DEFINE_validator(mlperf_conf, &validate_mlperf_config);

// User config file
static const char user_conf_message[] = "SUT User config file";

static bool validate_user_config(const char* user_conf, const std::string& user_conf_value){

    if (user_conf_value.length() == 0){
        std::cout << "[WARNING]: --user_conf not provided. Will use use mlperf default config (if --mlperf_conf provided)" << std::endl;
        return true;
    }

    if (!FileExists(user_conf_value) ){
        std::cout << "[WARNING]: Provided user config '" << user_conf_value << "' could not be found. Please" << std::endl;
    }

    return true;
}

DEFINE_string(user_conf, "", user_conf_message);
DEFINE_validator(user_conf, &validate_user_config);


// Total-Sample-Count
static const char total_sample_count_message[] = "Total Number of samples available for benchmark";
DEFINE_int32(total_sample_count, 5000, total_sample_count_message);

// Performance-Sample-Count
static const char perf_sample_count_message[] = "Number of samples to load for benchmarking";
DEFINE_int32(perf_sample_count, 256, perf_sample_count_message);


/**========================== Workload flags ==========================**/


// Dataset directory
static const char data_path_message[] = "Path to workload dataset";
/*
static bool validate_data_path(const char* data_path, const std::string& data_path_value){
    
    if (data_path_value.length()==0) return false;

    boost::filesystem::path p( data_path_value );
    if (!(boost::filesystem::exists( p ) ) ) {
	std::cout <<"Dataset path <" << data_path_value << "> not found." << std::endl;
	return false;
    }

    return true;
}
*/

DEFINE_string(data_path, "", data_path_message);
//DEFINE_validator(data_path, &validate_data_path);

// Model (workload) name
static const char model_name_message[] = "Supported workloads:\n\tresnet50, \n\tmobilenet, \n\tssd-mobilenet, \n\tssd-resnet34, \n\tdeeplabv3, \n\tbert, \n\tmobilebert, \n\tmobilenet-edge, \n\tssd-mobilenet-v2";
static bool validate_model_name( const char*, const std::string& model_name_value){

    if (model_name_value.length()==0) return false;

    std::list<std::string> model_names = {"resnet50", "mobilenet", "ssd-mobilenet", "ssd-resnet34", "deeplabv3", "bert", "mobilebert", "mobilenet-edge", "ssd-mobilenet-v2"};

    for (std::list< std::string >::iterator it = model_names.begin(); it != model_names.end(); ++it){
    if ( model_name_value.compare(*it) == 0 ) return true;
    }
    std::cout << "Unknown model name '" << model_name_value << "'. " << model_name_message << std::endl;

    return false;
}
DEFINE_string(model_name, "resnet50", model_name_message);
DEFINE_validator(model_name, &validate_model_name);

/**========================== Workload Flags ==========================**/

static const char model_path_message[] = "Path to model xml file";
static bool validate_model_path(const char* model_path, const std::string& model_path_value){

    if (model_path_value.length()==0){
        return false;
    }

    if ( !FileExists(model_path_value ) ){
        std::cout << " Model path <" << model_path_value << "> Not FOUND." << std::endl;
    return false;
    }
    return true;
}

DEFINE_string(model_path, "", model_path_message);
DEFINE_validator(model_path, &validate_model_path);


static const char instances_message[] = "Number of instances.";
DEFINE_uint64(num_instance, 1, instances_message);


static const char cpus_per_ins_message[] = "Optional. Number of cores assigned to each instance";
DEFINE_uint64(cpus_per_instance, 0, cpus_per_ins_message);


static const char batch_size_message[] = "Optional. Batch size value. Defaults to 1";
DEFINE_uint64(batch_size, 1, batch_size_message);


static const char warmup_message[] = "Number of warmup iterations. Defaults to 10.";
DEFINE_uint64(warmup_iters, 0, warmup_message);

static const char start_core_message[] = "Number of samples to load for benchmarking";
DEFINE_int32(start_core, 0, start_core_message);

