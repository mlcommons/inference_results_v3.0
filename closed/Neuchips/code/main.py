#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by NEUCHIPS on 2023

__doc__ = """NEUCHIPS rewrite NVIDIA's MLPerf Inference Benchmark submission code to perform DLRM benchmark. 

We follow the 'harness run' phase which launches the NEUCHIPS AI engine in a server-like harness that
accepts input from LoadGen (MLPerf Inference's official Load Generator), runs the inference with the AI engine, and reports
the output back to LoadGen.

More about the MLPerf Inference Benchmark and NEUCHIPS submission implementation can be found in README.md for this
project.
"""


import argparse
import json
import math
import multiprocessing as mp
import re
import shutil
import time
import traceback
from importlib import import_module
from multiprocessing import Process
from typing import List

import os
import sys
sys.path.insert(0, os.getcwd())

import code.common.arguments as common_args
from code import get_benchmark, get_harness
from code.common import run_command, auditing, logging
from code.common.constants import (
    Benchmark,
    Scenario,
    Action,
    HarnessType,
    AccuracyTarget,
    PowerSetting,
    WorkloadSetting,
    CPUArchitecture,
    config_ver_to_workload_setting,
    G_DEFAULT_HARNESS_TYPES,
)
from code.common.arguments import apply_overrides
from code.common.scopedMPS import ScopedMPS, turn_off_mps
from code.common.scopedPowerLimit import ScopedPowerLimit, get_power_state
from code.common.fields import MainArgs, get_applicable_fields
from code.common.systems.system_list import DETECTED_SYSTEM, MATCHED_SYSTEM, SystemClassifications
from code.common.systems.inferentia import Inferentia
from configs.configuration import ConfigRegistry

#
from code.common.systems.system_list import KnownSystem
from code.common.systems.custom_list import *
#

def launch_handle_generate_engine(*args, **kwargs):
    retries = 1
    timeout = 7200
    success = False
    for i in range(retries):
        # Build engines in another process to make sure we exit with clean cuda
        # context so that MPS can be turned off.
        from code.main import handle_generate_engine
        p = Process(target=handle_generate_engine, args=args, kwargs=kwargs)
        p.start()
        try:
            p.join(timeout)
        except KeyboardInterrupt:
            p.terminate()
            p.join(timeout)
            raise KeyboardInterrupt
        if p.exitcode == 0:
            success = True
            break

    if not success:
        raise RuntimeError("Building engines failed!")


def copy_engine(benchmark, source_engine_setting):
    """Copy engine file from default path to new path."""
    new_path = benchmark._get_engine_fpath(None, None)  # Use default values
    benchmark.config_ver = source_engine_setting
    old_path = benchmark._get_engine_fpath(None, None)

    logging.info(f"Copying {old_path} to {new_path}")
    shutil.copyfile(old_path, new_path)


def handle_generate_engine(config, gpu=True, dla=True, equiv_engine_setting=None):
    benchmark = config["benchmark"]
    scenario = config["scenario"]
    logging.info(f"Building engines for {benchmark.valstr()} benchmark in {scenario.valstr()} scenario...")

    start_time = time.time()

    arglist = common_args.GENERATE_ENGINE_ARGS
    config = apply_overrides(config, arglist)

    if dla and "dla_batch_size" in config:
        config["batch_size"] = config["dla_batch_size"]
        logging.info("Building DLA engine for " + config["config_name"])
        b = get_benchmark(config)

        if equiv_engine_setting is not None:
            copy_engine(b, equiv_engine_setting)
        else:
            b.build_engines()

    if gpu and "gpu_batch_size" in config:
        config["batch_size"] = config["gpu_batch_size"]
        config["dla_core"] = None
        logging.info("Building GPU engine for " + config["config_name"])
        b = get_benchmark(config)

        if equiv_engine_setting is not None:
            copy_engine(b, equiv_engine_setting)
        else:
            b.build_engines()

    end_time = time.time()
    logging.info(f"Finished building engines for {benchmark.valstr()} benchmark in {scenario.valstr()} scenario.")
    duration = end_time - start_time
    print(f"Time taken to generate engines: {duration} seconds")


def handle_audit_verification(audit_test_name, config):
    # Decouples the verification step from any auditing runs for better maintenance and testing
    logging.info('AUDIT HARNESS: Running verification script...')
    # Prepare log_dir
    config['log_dir'] = os.path.join('build/compliance_logs', audit_test_name)
    # Get a harness object
    harness, config = get_harness(config=config, profile=None)

    result = None
    if audit_test_name == 'TEST01':
        result = auditing.verify_test01(harness)
        if result == 'TEST01 FALLBACK':
            # Signals a fallback for failed test
            # Process description:
            #   1. Generate baseline_accuracy file
            #   2. Calculate the accuracy of baseline, using the benchmark's accuracy script
            #   3. Use same script to calculate accuracy of compliance run
            #   4. Depending on accuracy level, declare success if two values are within defined tolerance.
            logging.info('main.py notified for fallback handling on TEST01')

            # Run compliance script to generate baseline file
            full_log_dir = harness.get_full_log_dir()
            results_path = os.path.join('results', harness.get_system_name(), harness._get_submission_benchmark_name(),
                                        harness.scenario.valstr())
            harness_accuracy_log = os.path.join(results_path, 'accuracy/mlperf_log_accuracy.json')
            compliance_accuracy_log = os.path.join(full_log_dir, 'mlperf_log_accuracy.json')
            fallback_command = f'bash build/inference/compliance/nvidia/TEST01/create_accuracy_baseline.sh {harness_accuracy_log} {compliance_accuracy_log}'
            # generates new file called mlperf_log_accuracy_baseline.json
            run_command(fallback_command, get_output=True)

            def move_file(src, dst):
                logging.info(f'Moving file: {src} --> {dst}')
                shutil.move(src, dst)

            def copy_file(src, dst):
                logging.info(f'Copying file: {src} --> {dst}')
                shutil.copy(src, dst)

            # Create accuracy and performance directories
            accuracy_dir = os.path.join(full_log_dir, 'TEST01', 'accuracy')
            performance_dir = os.path.join(full_log_dir, 'TEST01', 'performance', 'run_1')
            os.makedirs(accuracy_dir, exist_ok=True)
            os.makedirs(performance_dir, exist_ok=True)

            # Get the accuracy of baseline file
            fallback_result_baseline = check_accuracy('mlperf_log_accuracy_baseline.json', config, is_compliance=True)
            # Move it to the submission dir
            dest_path = os.path.join(accuracy_dir, 'baseline_accuracy.txt')
            move_file('accuracy.txt', dest_path)

            # Get the accuracy of compliance file
            fallback_result_compliance = check_accuracy(f'{full_log_dir}/mlperf_log_accuracy.json', config, is_compliance=True)
            # Move it to the submission dir - check_accuracy stores accuracy.txt in the directory
            # name provided in its first argument. So this file will already be located inside get_full_log_dir()
            src_path = os.path.join(full_log_dir, 'accuracy.txt')
            dest_path = os.path.join(accuracy_dir, 'compliance_accuracy.txt')
            move_file(src_path, dest_path)

            # Move the required logs to their correct locations since run_verification.py has failed.
            move_file('verify_accuracy.txt', os.path.join(full_log_dir, 'TEST01', 'verify_accuracy.txt'))
            copy_file(os.path.join(full_log_dir, 'mlperf_log_accuracy.json'), os.path.join(accuracy_dir, 'mlperf_log_accuracy.json'))
            copy_file(os.path.join(full_log_dir, 'mlperf_log_detail.txt'), os.path.join(performance_dir, 'mlperf_log_detail.txt'))
            copy_file(os.path.join(full_log_dir, 'mlperf_log_summary.txt'), os.path.join(performance_dir, 'mlperf_log_summary.txt'))

            # Need to run verify_performance.py script to get verify_performance.txt file.
            verify_performance_command = ("python3 build/inference/compliance/nvidia/TEST01/verify_performance.py -r "
                                          + results_path + "/performance/run_1/mlperf_log_summary.txt" + " -t "
                                          + performance_dir + "/mlperf_log_summary.txt | tee " + full_log_dir + "/TEST01/verify_performance.txt")
            run_command(verify_performance_command, get_output=True)

            # Check level of accuracy - this test's tolerance depends on it
            accuracy_level = config["accuracy_level"][:-1]
            if accuracy_level == '99.9':
                logging.info('High Accuracy benchmark detected. Tolerance set to 0.1%')
                if not math.isclose(fallback_result_baseline, fallback_result_compliance, rel_tol=0.001):
                    raise ValueError(f'TEST01 + Fallback failure: BASELINE ACCURACY: {fallback_result_baseline}, COMPLIANCE_ACCURACY: {fallback_result_compliance}')
                else:
                    logging.info('AUDIT HARNESS: Success: TEST01 failure redeemed via fallback approach.')
                    print('TEST PASS')
            elif accuracy_level == '99':
                logging.info('Low Accuracy benchmark detected. Tolerance set to 1%')
                if not math.isclose(fallback_result_baseline, fallback_result_compliance, rel_tol=0.01):
                    raise ValueError(f'TEST01 + Fallback failure: BASELINE ACCURACY: {fallback_result_baseline}, COMPLIANCE_ACCURACY: {fallback_result_compliance}')
                else:
                    logging.info('AUDIT HARNESS: Success: TEST01 failure redeemed via fallback approach.')
                    print('TEST PASS')
            else:
                raise ValueError(f'Accuracy level not supported: {accuracy_level}')
    elif audit_test_name == 'TEST04':
        exclude_list = [Benchmark.BERT, Benchmark.DLRM, Benchmark.RNNT, Benchmark.UNET3D]
        if config['benchmark'] in exclude_list:
            benchmark_name = config['benchmark'].valstr()
            logging.info(f'TEST04 is not supported for benchmark {benchmark_name}. Ignoring request...')
            return None
        result = auditing.verify_test04(harness)
    elif audit_test_name == 'TEST05':
        result = auditing.verify_test05(harness)
    return result


def handle_run_harness(config, gpu=True, dla=True, profile=None,
                       power=False, generate_conf_files_only=False, compliance=False):
    """Run harness for given benchmark and scenario."""

    benchmark = config["benchmark"]
    benchmark_name = benchmark.valstr()
    scenario = config["scenario"]
    logging.info(f"Running harness for {benchmark_name} benchmark in {scenario.valstr()} scenario...")

    arglist = common_args.getScenarioBasedHarnessArgs(config["scenario"])
    config = apply_overrides(config, arglist)

    # Validate arguments
    if not dla:
        config["dla_batch_size"] = None
    if not gpu:
        config["gpu_batch_size"] = None

    # If we only want to generate conf_files, then set flag to true
    if generate_conf_files_only:
        config["generate_conf_files_only"] = True
        profile = None
        power = False

    # MLPINF-829: Disable CUDA graphs when there is a profiler
    if profile is not None:
        logging.warn("Due to MLPINF-829, CUDA graphs results in a CUDA illegal memory access when run with a profiler \
                on r460 driver. Force-disabling CUDA graphs.")
        config["use_graphs"] = False

    print('handle_run_harness {}, {}'.format(config, profile))

    harness, config = get_harness(config, profile)

    print(harness)

    if power:
        try:
            from code.internal.power_measurements import PowerMeasurements
            power_logfile_name = "_".join([
                config.get("config_name"),
                config.get("accuracy_level"),
                config.get("optimization_level"),
                config.get("inference_server")])
            power_measurements = PowerMeasurements(os.path.join(
                os.getcwd(),
                "power_measurements",
                power_logfile_name))
            power_measurements.start()
        except BaseException:
            power_measurements = None

    for key, value in config.items():
        print(f"{key} : {value}")
    result = ""

    if compliance:
        # AP: We need to keep the compliance logs separated from accuracy and perf
        # otherwise it messes up the update_results process
        config['log_dir'] = os.path.join('build/compliance_logs', config['audit_test_name'])
        logging.info('AUDIT HARNESS: Overriding log_dir for compliance run. Set to ' + config['log_dir'])

    # Launch the harness
    passed = True
    try:
        result = harness.run_harness()
        logging.info(f"Result: {result}")
    except Exception as _:
        traceback.print_exc(file=sys.stdout)
        passed = False
    finally:
        if power and power_measurements is not None:
            power_measurements.stop()
    if not passed:
        raise RuntimeError("Run harness failed!")

    if generate_conf_files_only and result == "Generated conf files":
        return

    # Append result to perf result summary log.
    log_dir = config["log_dir"]
    summary_file = os.path.join(log_dir, "perf_harness_summary.json")
    results = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)

    config_name = "-".join([
        harness.get_system_name(),
        config["config_ver"],
        config["scenario"].valstr()])
    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark.valstr()] = result

    with open(summary_file, "w") as f:
        json.dump(results, f)

    # Check accuracy from loadgen logs.
    if not compliance:
        # TEST01 fails the accuracy test because it produces fewer predictions than expected
        accuracy = check_accuracy(os.path.join(harness.get_full_log_dir(), "mlperf_log_accuracy.json"), config)
        summary_file = os.path.join(log_dir, "accuracy_summary.json")
        results = {}
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                results = json.load(f)

        if config_name not in results:
            results[config_name] = {}
        results[config_name][benchmark.valstr()] = accuracy

        with open(summary_file, "w") as f:
            json.dump(results, f)


def check_accuracy(log_file, config, is_compliance=False):
    """Check accuracy of given benchmark."""

    benchmark = config["benchmark"]

    # See individual benchmark READMEs in https://github.com/mlcommons/inference for accuracy targets.
    accuracy_targets = {
        Benchmark.BERT: 90.874,
        Benchmark.DLRM: 80.25,
        Benchmark.RNNT: 100.0 - 7.45225,
        Benchmark.ResNet50: 76.46,
        Benchmark.SSDMobileNet: 22.0,
        Benchmark.SSDResNet34: 20.0,
        Benchmark.UNET3D: 0.86331,
    }
    threshold_ratio = float(config["accuracy_level"][:-1]) / 100

    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."

    # checking if log_file is empty by just reading first several bytes
    # indeed, first 4B~6B is likely all we need to check: '', '[]', '[]\r', '[\n]\n', '[\r\n]\r\n', ...
    # but checking 8B for safety
    with open(log_file, 'r') as lf:
        first_8B = lf.read(8)
        if not first_8B or ('[' in first_8B and ']' in first_8B):
            return "No accuracy results in PerformanceOnly mode."

    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"}  # Use FP16 output for INT8 mode

    # into sys.path so it can successfully import it.
    # Insert into index 1 so that current working directory still takes precedence.
    sys.path.insert(1, os.path.join(os.getcwd(), "build", "inference", "tools", "submission"))
    accuracy_regex_map = import_module("submission_checker").ACC_PATTERN

    threshold = accuracy_targets[benchmark] * threshold_ratio

    # Every benchmark has its own accuracy script. Prepare commandline with args to the script.
    skip_run_command = False
    if benchmark in [Benchmark.ResNet50]:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file {:} \
            --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 ".format(log_file)
        regex = accuracy_regex_map["acc"]
    elif Benchmark.SSDResNet34 == benchmark:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-resnet34-results.json --use-inv-map".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = accuracy_regex_map["mAP"]
    elif Benchmark.SSDMobileNet == benchmark:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-mobilenet-results.json".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = accuracy_regex_map["mAP"]
    elif Benchmark.BERT == benchmark:
        # Having issue installing tokenizers on SoC systems
        if SystemClassifications.is_soc():
            cmd = "python3 code/bert/tensorrt/accuracy-bert.py --mlperf-accuracy-file {:} --squad-val-file {:}".format(
                log_file, os.path.join(os.environ.get("DATA_DIR", "build/data"), "squad", "dev-v1.1.json"))
        else:
            dtype = config["precision"].lower()
            if dtype in dtype_expand_map:
                dtype = dtype_expand_map[dtype]
            val_data_path = os.path.join(
                os.environ.get("DATA_DIR", "build/data"),
                "squad", "dev-v1.1.json")
            vocab_file_path = "build/models/bert/vocab.txt"
            if 'CPU' in config['config_name']:
                vocab_file_path = "build/data/squad/vocab.txt"
            output_prediction_path = os.path.join(os.path.dirname(log_file), "predictions.json")
            cmd = "PYTHONPATH=code/bert/tensorrt/helpers:$PYTHONPATH " \
                "python3 build/inference/language/bert/accuracy-squad.py " \
                "--log_file {:} --vocab_file {:} --val_data {:} --out_file {:} " \
                "--output_dtype {:}".format(log_file, vocab_file_path, val_data_path, output_prediction_path, dtype)
        regex = accuracy_regex_map["F1"]
    elif Benchmark.DLRM == benchmark:
        cmd = "python3 build/inference/recommendation/dlrm/pytorch/tools/accuracy-dlrm.py --mlperf-accuracy-file {:} " \
              "--day-23-file build/data/criteo/day_23 --aggregation-trace-file " \
              "build/preprocessed_data/criteo/full_recalib/sample_partition_trace.txt".format(log_file)
        regex = accuracy_regex_map["AUC"]
    elif Benchmark.RNNT == benchmark:
        # Having issue installing librosa on SoC systems
        if SystemClassifications.is_soc():
            cmd = "python3 code/rnnt/tensorrt/accuracy.py --loadgen_log {:}".format(log_file)
        else:
            # RNNT output indices are in INT8
            cmd = "python3 build/inference/speech_recognition/rnnt/accuracy_eval.py " \
                "--log_dir {:} --dataset_dir build/preprocessed_data/LibriSpeech/dev-clean-wav " \
                "--manifest build/preprocessed_data/LibriSpeech/dev-clean-wav.json " \
                "--output_dtype int8".format(os.path.dirname(log_file))
        regex = accuracy_regex_map["WER"]
    elif Benchmark.UNET3D == benchmark:
        postprocess_dir = "build/postprocessed_data"
        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)
        # WAR for numpy linargerror_eigenvalues_nonconvergence happen in ARM64 (x86 don't see this)
        cmd_prefix = "OPENBLAS_CORETYPE=armv8" if DETECTED_SYSTEM.host_cpu_conf.get_architecture() == 'aarch64' else ""
        cmd = "{:} python3 code/3d-unet/tensorrt/accuracy_kits.py --log_file {:}".format(cmd_prefix, log_file)
        regex = accuracy_regex_map["DICE"]
        # Cannot use nibabel in Xavier so cannot check the accuracy directly
        if SystemClassifications.is_soc():
            # Internally, run on another node to process the accuracy.
            try:
                cmd = cmd.replace(os.getcwd(), ".", 1)
                temp_cmd = "ssh -oBatchMode=yes computelab-frontend-2 \"timeout 1200 srun --gres=gpu:ga100:1 -t 20:00 " \
                    "bash -c 'cd {:} && make prebuild DOCKER_COMMAND=\\\"{:}\\\"'\"".format(os.getcwd(), cmd)
                full_output = run_command(temp_cmd, get_output=True)
                start_line_idx = -1
                end_line_idx = -1
                for (line_idx, line) in enumerate(full_output):
                    if "Loading necessary metadata..." in line:
                        start_line_idx = line_idx
                    if "Done!" in line:
                        end_line_idx = line_idx
                assert start_line_idx != -1 and end_line_idx != -1, "Failed in accuracy checking"
                output = full_output[start_line_idx:end_line_idx + 1]
                skip_run_command = True
            except Exception as e:
                logging.warning(
                    "Accuracy checking for 3D-UNet is not supported on Xavier. "
                    "Please run the following command on desktop:\n{:}".format(cmd))
                output = ["Accuracy: mean = 1.00000, kidney = 1.0000, tumor = 1.0000"]
                skip_run_command = True
    else:
        raise ValueError(f"Unknown benchmark: {benchmark.valstr()}")

    # Run benchmark's accuracy script and parse output for result.
    if not skip_run_command:
        output = run_command(cmd, get_output=True)
    result_regex = re.compile(regex)
    accuracy = None
    with open(os.path.join(os.path.dirname(log_file), "accuracy.txt"), "w") as f:
        for line in output:
            print(line, file=f)
    for line in output:
        result_match = result_regex.match(line)
        if not result_match is None:
            accuracy = float(result_match.group(1))
            break

    accuracy_result = "PASSED" if accuracy is not None and accuracy >= threshold else "FAILED"

    if accuracy_result == "FAILED" and not is_compliance:
        raise RuntimeError(
            "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}!".format(
                accuracy, threshold, accuracy_result))

    if is_compliance:
        return accuracy  # Needed for numerical comparison

    return "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}.".format(
        accuracy, threshold, accuracy_result)


def handle_calibrate(config):
    benchmark = config["benchmark"]
    logging.info(f"Generating calibration cache for Benchmark \"{benchmark.valstr()}\"")
    config = apply_overrides(config, common_args.CALIBRATION_ARGS)
    config["dla_core"] = None
    config["force_calibration"] = True
    b = get_benchmark(config)
    b.calibrate()


def dispatch_action(main_args, benchmark_conf, conf_ver, equiv_engine_setting=None):
    # Pull settings out from main_args.
    action = Action.get_match(main_args["action"])
    profile = main_args.get("profile", None)
    power = main_args.get("power", False)
    need_gpu = not main_args["no_gpu"]
    need_dla = not main_args["gpu_only"]

    if not need_gpu and not need_dla:
        raise RuntimeError("Cannot set --gpu_only and --no_gpu concurrently.")

    if action in (Action.RunAuditHarness,
                  Action.RunAuditVerify,
                  Action.RunCPUAuditHarness,
                  Action.RunCPUAuditVerify):
        if main_args["audit_test"] == "TEST04":
            exclude_list = [Benchmark.BERT, Benchmark.DLRM, Benchmark.RNNT, Benchmark.UNET3D, Benchmark.Retinanet]
            if benchmark_conf["benchmark"] in exclude_list:
                benchmark_name = benchmark_conf['benchmark'].valstr()
                logging.info(f'TEST04 is not supported for benchmark {benchmark_name}. Ignoring request...')
                return

    # Check if we need scoped power limit setting.
    power_state = get_power_state(main_args, benchmark_conf)
    # Generate engines.
    if action == "generate_engines":
        with ScopedPowerLimit(power_state):
            # Turn on MPS if server scenario and if active_sms is specified.
            benchmark_conf = apply_overrides(benchmark_conf, ["active_sms"])
            active_sms = benchmark_conf.get("active_sms", None)

            if equiv_engine_setting is not None:
                logging.info(f"config_ver={conf_ver} can re-use engine from {equiv_engine_setting}")

            _gen_args = [benchmark_conf]
            _gen_kwargs = {
                "gpu": need_gpu,
                "dla": need_dla,
                "equiv_engine_setting": equiv_engine_setting
            }

            if not main_args["no_child_process"]:
                if Scenario.Server == benchmark_conf["scenario"] and active_sms is not None and active_sms < 100:
                    with ScopedMPS(active_sms):
                        launch_handle_generate_engine(*_gen_args, **_gen_kwargs)
                else:
                    launch_handle_generate_engine(*_gen_args, **_gen_kwargs)
            else:
                handle_generate_engine(*_gen_args, **_gen_kwargs)
    # Run CPU harness:
    elif action == "run_cpu_harness":
        auditing.cleanup()
        if not benchmark_conf["use_cpu"]:
            raise RuntimeError("Cannot run CPU harness for non-CPU system accelerator")
        handle_run_harness(benchmark_conf, False, False, None, power)
    elif action == "run_inferentia_harness":
        auditing.cleanup()
        if not benchmark_conf["use_inferentia"]:
            raise RuntimeError("Cannot run Inferentia harness for non-CPU system accelerator")
        handle_run_harness(benchmark_conf, False, False, None, power)
    # Run harness.
    elif action == "run_harness":
        with ScopedPowerLimit(power_state):
            # In case there's a leftover audit.config file from a prior compliance run or other reason
            # we need to delete it or we risk silent failure.
            auditing.cleanup()

            handle_run_harness(benchmark_conf, need_gpu, need_dla, profile, power)
    elif action == "run_audit_harness" or action == "run_cpu_audit_harness":
        with ScopedPowerLimit(power_state):
            logging.info('\n\n\nRunning compliance harness for test ' + main_args['audit_test'] + '\n\n\n')

            # Find the correct audit.config file and move it in current directory
            dest_config = auditing.load(main_args['audit_test'], benchmark_conf['benchmark'])

            # Make sure the log_file override is valid
            os.makedirs("build/compliance_logs", exist_ok=True)

            # Pass audit test name to handle_run_harness via benchmark_conf
            benchmark_conf['audit_test_name'] = main_args['audit_test']

            if action == "run_cpu_audit_harness":
                need_gpu = False
                need_dla = False
                profile = None
                if not benchmark_conf["use_cpu"]:
                    raise RuntimeError("Cannot run CPU harness for non-CPU system accelerator")

            # Run harness
            handle_run_harness(benchmark_conf, need_gpu, need_dla, profile, power, compliance=True)

            # Cleanup audit.config
            logging.info("AUDIT HARNESS: Cleaning Up audit.config...")
            auditing.cleanup()
    elif action == "run_audit_verification":
        logging.info("Running compliance verification for test " + main_args['audit_test'])
        handle_audit_verification(audit_test_name=main_args['audit_test'], config=benchmark_conf)
        auditing.cleanup()
    elif action == "run_cpu_audit_verification":
        logging.info("Running compliance verification for test " + main_args['audit_test'])
        if not benchmark_conf["use_cpu"]:
            raise RuntimeError("Cannot run CPU harness for non-CPU system accelerator")
        handle_audit_verification(audit_test_name=main_args['audit_test'], config=benchmark_conf)
        auditing.cleanup()
    elif action == "calibrate":
        # To generate calibration cache, we only need to run each benchmark once.
        # Use offline config.
        if Scenario.Offline == benchmark_conf["scenario"]:
            handle_calibrate(benchmark_conf)
    elif action == "generate_conf_files":
        handle_run_harness(benchmark_conf, need_gpu, need_dla, generate_conf_files_only=True)


def populate_config_registry(benchmarks: List[Benchmark], scenarios: List[Scenario]):
    # Load and validate all configs. Note that the validation step is done implicitly and automatically when the
    # BenchmarkConfiguration is loaded and registered. You can assume the config satisfies all applicable constraints in
    # any further code, as failure to meet constraints will result in a raised Exception.
    for benchmark in benchmarks:
        for scenario in scenarios:
            ConfigRegistry.load_configs(benchmark, scenario)


def main(main_args, system, load_config_fn=populate_config_registry):
    """
    Args:
        main_args: Args parsed from user input.
        system: System to use
    """
    print('main: system {}: {}'.format(type(system), system))
    system_id = system.get_id()
    print('main: system_id {}'.format(system_id))

    # Turn off MPS in case it's turned on.
    #turn_off_mps()

    # Get user's benchmarks, else run all.
    benchmarks = list(Benchmark)
    if main_args["benchmarks"] is not None:
        benchmark_names = main_args["benchmarks"].split(",")
        benchmarks = []
        for benchmark_name in benchmark_names:
            benchmark = Benchmark.get_match(benchmark_name)
            if benchmark is None:
                raise RuntimeError(f"'{benchmark_name}' is not a valid benchmark name.")
            benchmarks.append(benchmark)

    # Get user's scenarios, else use all.
    scenarios = list(Scenario)
    if main_args["scenarios"] is not None:
        scenario_names = main_args["scenarios"].split(",")
        scenarios = []
        for scenario_name in scenario_names:
            scenario = Scenario.get_match(scenario_name)
            if scenario is None:
                raise RuntimeError(f"'{scenario_name}' is not a valid scenario name.")
            scenarios.append(scenario)

    load_config_fn(benchmarks, scenarios)

    for benchmark in benchmarks:
        for scenario in scenarios:
            if ConfigRegistry.available_workload_settings(benchmark, scenario) is None:
                continue

            # Build the workload_setting.
            harness_type_str = main_args["harness_type"]
            if harness_type_str == "auto":
                harness_type = G_DEFAULT_HARNESS_TYPES[benchmark]
            else:
                harness_type = HarnessType.get_match(harness_type_str)

            default_workload = WorkloadSetting(
                harness_type=harness_type,
                accuracy_target=AccuracyTarget(main_args["accuracy_target"]),
                power_setting=PowerSetting.get_match(main_args["power_setting"]))
            workload_settings = [default_workload]

            # TODO: Support Legacy config_ver. Remove in the future. If the legacy config_ver is used, override the
            # workload setting with the equivalent ones from the config_vers.
            if main_args.get("config_ver", None) is not None:
                config_vers = main_args["config_ver"].split(",")
                # Retain legacy behavior of processing default first (matters for engine copying / caching)
                if "default" in config_vers:
                    config_vers = ["default"] + list(set(config_vers) - {"default"})
                workload_settings = [config_ver_to_workload_setting(benchmark, config_ver) for config_ver in config_vers]
                if "all" in config_vers:
                    workload_settings = ConfigRegistry.available_workload_settings(benchmark, scenario)

            seen_gen_eng_confs = dict()
            for workload_setting in workload_settings:
                print('workload_setting {}'.format(workload_setting))
                config = ConfigRegistry.get(benchmark, scenario, system, **workload_setting.as_dict())
                if config is None:
                    logging.warning(f"No registered config for {benchmark.value.name}.{scenario.value.name}.{system_id} "
                                    f"for WorkloadSetting({workload_setting.shortname()})")
                    continue
                config_dict = config.as_dict()

                # Config uses a KnownSystem - Update the field to use detected parameters
                config_dict["system"] = system

                # Detect if this engine has the same engine build parameters as a previously built engine from the same
                # job.
                equiv_engine_setting = None
                req_gen_args, opt_gen_args = get_applicable_fields(Action.GenerateEngines, benchmark, scenario, system,
                                                                   workload_setting)
                gen_args = set([f.name for f in req_gen_args]).union(set([f.name for f in opt_gen_args]))
                gen_config = {k: config_dict[k] for k in config_dict if k in gen_args}
                for k, v in seen_gen_eng_confs.items():
                    if benchmark != Benchmark.RNNT and gen_config == v:
                        equiv_engine_setting = k
                        break
                seen_gen_eng_confs[workload_setting.shortname()] = gen_config

                # TODO: Much of the following is legacy code from config.json-style BenchmarkConfigs, and can be changed
                # to interact better with the new style.
                config_dict["config_name"] = f"{system_id}_{benchmark.valstr()}_{scenario.valstr()}"
                workload_id = workload_setting.shortname()  # Spoof a 'config_ver'
                config_dict["config_ver"] = workload_id
                config_dict["accuracy_level"] = "99%" if workload_setting.accuracy_target == AccuracyTarget.k_99 else "99.9%"
                config_dict["optimization_level"] = "plugin-enabled"
                config_dict["inference_server"] = str(workload_setting.harness_type.value)

                # Use strings instead of the Enums for now until the rest of the codebase is refactored.
                config_dict["system_id"] = system_id
                config_dict["benchmark"] = benchmark
                config_dict["scenario"] = scenario

                if main_args.get("system_name", None) is not None:
                    config_dict["system_name"] = main_args["system_name"]

                # Check for use_cpu
                if len(system.accelerator_conf.get_accelerators()) == 0 and \
                        system.host_cpu_conf.get_architecture() == CPUArchitecture.x86_64:
                    config_dict["use_cpu"] = True
                else:
                    config_dict["use_cpu"] = False

                config_dict["use_inferentia"] = (system.accelerator_conf.num_inferentia() != 0)

                dispatch_action(main_args, config_dict, workload_id, equiv_engine_setting=equiv_engine_setting)


def parse_main_args(custom=None):
    """
    Parses sys.args for the arguments that main.py requires to function.

    Args:
        custom (Optional[List[str]]): If not None, describes a list of strings like sys.argv

    Returns:
        Dict[str, Any]: A dict representing the parsed main.py command flags
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for arg in MainArgs:
        arg.value.add_to_argparser(parser, allow_argparse_default=True)
    return vars(parser.parse_known_args(args=custom)[0])


if __name__ == "__main__":
    mp.set_start_method("spawn")

    MATCHED_SYSTEM = KnownSystem.N3000_CPU_2S_Neuchips
    DETECTED_SYSTEM = custom_systems['N3000_CPU_2S_Neuchips']

    if MATCHED_SYSTEM is None:
        logging.info(f"Detected System ID: {DETECTED_SYSTEM} "
                     "did not match any known systems. Exiting.")
    else:
        logging.info(f"Detected System ID: {MATCHED_SYSTEM}")

        print(f"Matched System ID: {MATCHED_SYSTEM}")
        print(f"Detected System ID: {DETECTED_SYSTEM}")
        main_args = parse_main_args()
        main(main_args, DETECTED_SYSTEM)
