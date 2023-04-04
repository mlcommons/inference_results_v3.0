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

from __future__ import annotations

import glob
import importlib
import json
import os
import sys
import traceback

import code.common.arguments as common_args

from code import get_harness, G_HARNESS_CLASS_MAP
from code.actionhandler.base import ActionHandler
from code.actionhandler.generate_conf_files import GenerateConfFilesHandler
from code.common import logging
from code.common.accuracy_checker import check_accuracy
from code.common.constants import *
from code.common.log_parser import read_loadgen_result_by_key, get_perf_regression_ratio, extract_mlperf_result_from_log
from code.common.power_limit import ScopedPowerStateController
from code.common.protected_super import ProtectedSuper


class RunHarnessHandler(GenerateConfFilesHandler):
    """Handles the RunHarness action. This calls the underlying GenerateConfFilesHandler first to generate the
    measurements/ file required for LoadGen.
    """

    def __init__(self, benchmark_conf, power_controller, use_gpu=True, use_dla=True, profiler=None, measure_power=False, skip_postprocess=False):
        """Creates a new ActionHandler for RunHarness

        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
            power_controller (PowerController): The PowerController to control to power settings of the system
            use_gpu (bool): Whether or not GPUs are used for this configuration.
            use_dla (bool): Whether or not DLAs are used for this configuration
            profiler (str): INTERNAL ONLY. Name of the profiler to use. (Default: None)
            measure_power (bool): INTERNAL ONLY. If True, measures the power usage during the harness run in a
                                  background thread. (Default: False)
            skip_postprocess (bool): If True, skips accuracy log parsing and statistics reporting in cleanup().
                                     (Default: False)
        """
        # Force use_gpu and use_dla to False for non-GPU based harnesses.
        if benchmark_conf["use_cpu"] or benchmark_conf["use_inferentia"]:
            use_gpu, use_dla, profiler = (False, False, None)
        super().__init__(benchmark_conf, use_gpu=use_gpu, use_dla=use_dla, skip_file_checks=False)
        self.action = Action.RunHarness

        self.power_controller = ScopedPowerStateController(power_controller)
        self.profiler = profiler
        self.measure_power = measure_power
        self.power_monitor = None
        self.skip_postprocess = skip_postprocess

        # MLPINF-829: Disable CUDA graphs when there is a profiler
        if self.profiler is not None:
            logging.warn("MLPINF-829: CUDA graphs results in a CUDA illegal memory access when run with a profiler. Force-disabling CUDA graphs.")
            self.benchmark_conf["use_graphs"] = False

        self.config_name = None

    def setup(self):
        """Called once before handle().
        """
        self.power_controller.set_power_state()

        # Generates the measurements/ entries and handle any failures
        with ProtectedSuper(self) as duper:
            duper.run()

        # After measurements/ is generated, set profiler if necessary
        if self.profiler is not None:
            profiler_module_path = G_HARNESS_CLASS_MAP["profiler_harness"].module_path
            if importlib.util.find_spec(profiler_module_path) is None:
                logging.info("Could not load profiler: Are you an internal user?")
            else:
                ProfilerHarness = importlib.import_module(profiler_module_path).ProfilerHarness
                self.harness = ProfilerHarness(self.harness, self.profiler)

        self.log_dir = self.benchmark_conf["log_dir"]

        # If measure_power is requested, start the power monitor
        if self.measure_power:
            if importlib.util.find_spec("code.internal.power_measurements") is None:
                logging.info("Could not load power monitor: Are you an internal user?")
            else:
                PowerMeasurements = importlib.import_module("code.internal.power_measurements").PowerMeasurements
                power_logfile_name = "_".join([
                    self.benchmark_conf.get("config_name"),
                    self.benchmark_conf.get("accuracy_level"),
                    self.benchmark_conf.get("optimization_level"),
                    self.benchmark_conf.get("inference_server")])
                self.power_monitor = PowerMeasurements(os.path.join(self.log_dir,
                                                                    "power_measurements",
                                                                    power_logfile_name))
                self.power_monitor.start()

        self.config_name = "-".join([self.harness.get_system_name(),
                                     self.workload_setting.shortname(),
                                     self.scenario.valstr()])

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.
        """
        self.power_controller.reset_power_state()
        if self.measure_power and self.power_monitor is not None:
            self.power_monitor.stop()

        if success and not self.skip_postprocess:
            self.check_accuracy_log()
            if not self.measure_power and self.benchmark_conf.get("test_mode", None) != "AccuracyOnly":
                self.report_stats()
            elif self.measure_power and self.power_monitor is not None:
                self.power_monitor.report_stats(self.log_dir)

    def _update_json(self, summary_file: str, val: str):
        """Helper method to update the summary JSON with a value for the current config and benchmark

        Args:
            summary_file (str): The path to the summary JSON file
            val (str): The value to set for results[self.config_name][self.benchmark.valstr()]
        """
        results = {}
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                results = json.load(f)

        if self.config_name not in results:
            results[self.config_name] = {}
        results[self.config_name][self.benchmark.valstr()] = val

        with open(summary_file, "w") as f:
            json.dump(results, f)

    def check_accuracy_log(self):
        """For each session, we store a summary of results in the log directory. Append the current accuracy result to it.
        """
        accuracy = check_accuracy(os.path.join(self.harness.get_full_log_dir(), "mlperf_log_accuracy.json"),
                                  self.benchmark_conf)
        summary_file = os.path.join(self.log_dir, "accuracy_summary.json")
        self._update_json(summary_file, accuracy)

    def report_stats(self):
        print(f"\n======================= Extra Perf Stats: =======================\n")
        print(f"{self.harness.get_system_name()}-{self.benchmark_conf['config_ver']}:")
        log_paths = glob.glob(os.path.join(self.log_dir, "**", "mlperf_log_detail.txt"), recursive=True)
        if len(log_paths) > 1:
            print(f"    More than one result found under {self.log_dir}, Skipping the extra stats...")
        else:
            log_path = log_paths[0]
            parts = log_path.split("/")
            # Ex: /work/build/logs/2022.03.23-06.05.20/DGX-A100_A100-SXM-80GBx1_TRT/bert-99/Server/mlperf_log_detail.txt
            system_id, benchmark, scenario = parts[-4:-1]
            log_dir = os.path.dirname(log_path)

            try:
                perf_number, metric = extract_mlperf_result_from_log(log_dir, system_id, benchmark, scenario)
                regression_ratio, current_perf = get_perf_regression_ratio(perf_number, "results", system_id, benchmark, scenario)

                if current_perf == 0:
                    print(f"    WARNING: Perf value of 0 found in results.")
                else:
                    print(f"    {metric}: {perf_number:.2f} is {regression_ratio:.2f} of the current results {current_perf:.2f}.")
            except FileNotFoundError:
                print(f"    No results found under results/ folder.")

            # Extra stats for Server
            if scenario == Scenario.Server:
                latency_99 = read_loadgen_result_by_key(log_dir, "result_99.00_percentile_latency_ns", None)
                target_latency = read_loadgen_result_by_key(log_dir, "requested_server_target_latency_ns", None)
                latency_ratio = latency_99 / target_latency
                print(f"    Server 99-percentile latency {latency_99:.0f} ns is {latency_ratio:.2f} of the target_latency {target_latency:.0f} ns")

                # If the offline result exists, print how much percentage
                try:
                    server_offline_ratio, offline_perf = get_perf_regression_ratio(perf_number, "results", system_id, benchmark, "Offline")
                    print(f"    Server QPS {perf_number:.0f} is {server_offline_ratio:.2f} of the current Offline QPS {offline_perf:.0f} under results/")
                except FileNotFoundError:
                    pass

    def handle(self) -> bool:
        """Run the action.

        Returns:
            bool: True if handle() succeeded, False otherwise.
        """
        # TODO: Communicate with @garvitk on how to handle this better. Currently, this is necessary because CI/CD
        # parses this to retrieve metadata about the run.
        for key, value in self.benchmark_conf.items():
            print(f"{key} : {value}")

        result = ""
        try:
            result = self.harness.run_harness(flag_dict=self.harness_flag_dict, skip_generate_measurements=True,
                                              skip_postprocess=self.skip_postprocess)
            logging.info(f"Result: {result}")
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            return False

        # We have to update the summary file here with `result` rather than cleanup() since if this ActionHandler is
        # wrapped inside a SubprocessActionHandler, `result` will only be visible in the child process and can't get
        # surfaced to the parent which calls cleanup().
        # RunHarness isn't supposed to be called in a subprocess, but this is here to prevent bugs in case it is.
        summary_file = os.path.join(self.log_dir, "perf_harness_summary.json")
        self._update_json(summary_file, result)
        return True

    def handle_failure(self):
        """Called after handle() if it errors.
        """
        raise RuntimeError("Run harness failed!")
