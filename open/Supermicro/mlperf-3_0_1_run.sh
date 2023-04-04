#! /bin/sh
# Script that will run the steps to generate a valid result for the Nvidia Submission

# Generate measurements files
make generate_conf_files

# resnet50 server
## test01
#make run RUN_ARGS="--benchmarks=resnet50 --scenarios=server --config_ver=default,high_accuracy"                  
## test02
#make run RUN_ARGS="--benchmarks=resnet50 --scenarios=server --config_ver=default,high_accuracy --test_mode=AccuracyOnly"       

# resnet50 offline
## test03
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=default,high_accuracy"                               
## test04
make run RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=default,high_accuracy --test_mode=AccuracyOnly"      

# Stage the results
make stage_results

# Update the results
make update_results

# Run the audit
# resnet50 server
## test01
#make run_audit_test01 RUN_ARGS="--benchmarks=resnet50 --scenarios=server --config_ver=default,high_accuracy"
## test04
#make run_audit_test04 RUN_ARGS="--benchmarks=resnet50 --scenarios=server --config_ver=default,high_accuracy"
## test05
#make run_audit_test05 RUN_ARGS="--benchmarks=resnet50 --scenarios=server --config_ver=default,high_accuracy"

# resnet50 offline
## test01
make run_audit_test01 RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=default,high_accuracy"
## test04
make run_audit_test04 RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=default,high_accuracy"
## test05
make run_audit_test05 RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=default,high_accuracy"

# Stage Compliance
make stage_compliance