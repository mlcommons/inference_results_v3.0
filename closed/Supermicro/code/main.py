#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


__doc__ = """NVIDIA's MLPerf Inference Benchmark submission code. NVIDIA's implementation runs in 2 phases.

The first phase is 'engine generation', which builds a TensorRT Engine using TensorRT, a Deep Learning Inference
performance optimization SDK by NVIDIA. This only applies to NVIDIA accelerator-based workloads. This step is skipped
for systems such as Intel CPU or AWS Inferentia.

The second phase is a 'harness run', which launches the generated TensorRT engine in a server-like harness that
accepts input from LoadGen (MLPerf Inference's official Load Generator), runs the inference with the engine, and reports
the output back to LoadGen.

More about the MLPerf Inference Benchmark and NVIDIA's submission implementation can be found in the README.md for this
project.
"""


import argparse
import multiprocessing as mp
from typing import List

import code.common.auditing as auditing

from code.actionhandler import *
from code.common import logging
from code.common.constants import *
from code.common.fields import MainArgs, apply_helper_and_legacy_fields, get_effective_values
from code.common.mps import turn_off_mps
from code.common.power_limit import get_power_controller
from code.common.systems.system_list import DETECTED_SYSTEM, MATCHED_SYSTEM, SystemClassifications
from configs.configuration import ConfigRegistry


def populate_config_registry(benchmarks: List[Benchmark], scenarios: List[Scenario]):
    # Load and validate all configs. Note that the validation step is done implicitly and automatically when the
    # BenchmarkConfiguration is loaded and registered. You can assume the config satisfies all applicable constraints in
    # any further code, as failure to meet constraints will result in a raised Exception.
    for benchmark in benchmarks:
        for scenario in scenarios:
            ConfigRegistry.load_configs(benchmark, scenario)


def legacy_config_ver_override(benchmark, config_vers):
    config_vers = config_vers.split(",")
    # Retain legacy behavior of processing default first (matters for engine copying / caching)
    if "default" in config_vers:
        config_vers = ["default"] + list(set(config_vers) - {"default"})
    workload_settings = [config_ver_to_workload_setting(benchmark, config_ver) for config_ver in config_vers]
    if "all" in config_vers:
        workload_settings = ConfigRegistry.available_workload_settings(benchmark, scenario)
    return workload_settings


def main(main_args, system, load_config_fn=populate_config_registry):
    """
    Args:
        main_args: Arguments parsed from the command line. If run via Make, this is taken from the RUN_ARGS environment
                   variable.
        system: The System configuration that is assumed to be running on. This is usually
                code.common.systems.system_list.DETECTED_SYSTEM, but can be changed to other values for testing.
        load_config_fn: The function to call to populate the ConfigRegistry. (Default: populate_config_registry)
    """
    system_id = system.get_id()

    # Turn off MPS in case it's turned on.
    turn_off_mps()

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

            # Cannot copy engines across different benchmarks/scenarios, only workload settings. Reset the
            # GenerateEngines session
            GenerateEngineHandler.reset_session()

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
                workload_settings = legacy_config_ver_override(benchmark, main_args.get("config_ver"))

            for workload_setting in workload_settings:
                config = ConfigRegistry.get(benchmark, scenario, system, **workload_setting.as_dict())
                if config is None:
                    logging.warning(f"No registered config for {benchmark.value.name}.{scenario.value.name}.{system_id} "
                                    f"for WorkloadSetting({workload_setting.shortname()})")
                    continue

                # Config uses a KnownSystem - Update the field to use detected parameters
                config_dict = config.as_dict()
                config_dict["system"] = system
                assert config_dict["benchmark"] == benchmark, f"Registered config for benchmark {benchmark} instead has benchmark={config_dict['benchmark']}"
                assert config_dict["scenario"] == scenario, f"Registered config for scenario {scenario} instead has scenario={config_dict['scenario']}"

                dispatch_action(main_args, config_dict, workload_setting)


def dispatch_action(main_args, benchmark_conf, workload_setting):
    # Pull settings out of main_args
    action = Action.get_match(main_args["action"])
    profile = main_args.get("profile", None)
    power = main_args.get("power", False)
    need_gpu = not main_args["no_gpu"]
    need_dla = not main_args["gpu_only"]
    use_child_process = not main_args["no_child_process"]
    logging.debug(f"{main_args=}")
    if not need_gpu and not need_dla:
        raise RuntimeError("Cannot set --gpu_only and --no_gpu concurrently.")

    # Filter the benchmark configuration for applicable fields and apply command line overrides
    filtered = get_effective_values(benchmark_conf, action, workload_setting)
    benchmark_conf = apply_helper_and_legacy_fields(filtered,
                                                    workload_setting,
                                                    system_name_override=main_args.get("system_name", None))

    # Clean up run environment
    auditing.cleanup()

    # Check if we need scoped power limit setting.
    power_controller = get_power_controller(main_args, benchmark_conf)
    handler = None
    if action == Action.GenerateEngines:
        handler = GenerateEngineHandler(benchmark_conf,
                                        power_controller,
                                        allow_mps=use_child_process,
                                        build_gpu=need_gpu,
                                        build_dla=need_dla)
        if use_child_process:
            handler = SubprocessActionHandlerWrapper(handler)
    elif action == Action.GenerateConfFiles:
        handler = GenerateConfFilesHandler(benchmark_conf,
                                           use_gpu=need_gpu,
                                           use_dla=need_dla)
    elif action == Action.RunHarness:
        handler = RunHarnessHandler(benchmark_conf,
                                    power_controller,
                                    use_gpu=need_gpu,
                                    use_dla=need_dla,
                                    profiler=profile,
                                    measure_power=power)
    elif action == Action.Calibrate:
        handler = CalibrateHandler(benchmark_conf)
    elif action == Action.RunAuditHarness:
        handler = RunAuditHandler(main_args["audit_test"],
                                  benchmark_conf,
                                  power_controller,
                                  use_gpu=need_gpu,
                                  use_dla=need_dla,
                                  profiler=profile,
                                  verify=not main_args["no_audit_verify"])
    else:
        logging.info(f"Action {action} is not currently supported")
    handler.run()


def parse_main_args(custom=None):
    """
    Parses sys.args for the arguments that main.py requires to function.

    Args:
        custom (Optional[List[str]]): If not None, describes a list of strings like sys.argv

    Returns:
        Dict[str, Any]: A dict representing the parsed main.py command flags
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    for arg in MainArgs:
        arg.value.add_to_argparser(parser, allow_argparse_default=True)
    return vars(parser.parse_known_args(args=custom)[0])


if __name__ == "__main__":
    mp.set_start_method("spawn")

    if MATCHED_SYSTEM is None:
        logging.info(f"Detected system did not match any known systems. Exiting. {DETECTED_SYSTEM}")
    else:
        logging.info(f"Detected system ID: {MATCHED_SYSTEM}")
        main_args = parse_main_args()
        main(main_args, DETECTED_SYSTEM)
