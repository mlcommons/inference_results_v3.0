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

import code.common.arguments as common_args
from code import get_harness
from code.common import logging
from code.actionhandler.base import ActionHandler
from code.common.constants import *
from code.common.submission import generate_measurements_entry


class GenerateConfFilesHandler(ActionHandler):
    """Handles the GenerateConfFiles action. The .conf files are the loadgen config files located in measurements/. This
    ActionHandler is responsible for generating them based on the BenchmarkConfiguration and user-provided CLI flags.

    This is also used as the base class for RunHarness, since running the harness requires the .conf files to be
    generated.
    """

    def __init__(self, benchmark_conf, use_gpu=True, use_dla=True, skip_file_checks=True):
        """Creates a new ActionHandler for GenerateConfFiles

        Note that use_gpu and use_dla are still necessary parameters, since these can change values in the config,
        depending on if these .conf files are meant for a GPU-only run, a DLA-only run, or a concurrent GPU+DLA run.

        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
            use_gpu (bool): Whether or not GPUs are used for this configuration.
            use_dla (bool): Whether or not DLAs are used for this configuration
            skip_file_checks (bool): If True, tells the internal harness object to not check for files. This is
                                     usually the desired behavior unless you are doing verification in preparation
                                     for a harness run. (Default: True)
        """
        super().__init__(Action.GenerateConfFiles)

        self.benchmark_conf = benchmark_conf
        self.system = benchmark_conf["system"]
        self.benchmark = benchmark_conf["benchmark"]
        self.scenario = benchmark_conf["scenario"]
        self.workload_setting = benchmark_conf["workload_setting"]
        self.use_gpu = use_gpu
        self.use_dla = use_dla
        self.benchmark_conf["skip_file_checks"] = skip_file_checks

        # Initialized during setup()
        self.harness = None
        self.harness_flag_dict = None

    def setup(self):
        """Called once before handle().
        """
        if not self.use_dla:
            self.benchmark_conf["dla_batch_size"] = None
        if not self.use_gpu:
            self.benchmark_conf["gpu_batch_size"] = None

        self.harness, self.benchmark_conf = get_harness(self.benchmark_conf, None)
        self.harness_flag_dict = self.harness.build_non_custom_flags()

    def handle(self) -> bool:
        """Run the action.

        Returns:
            bool: True if handle() succeeded, False otherwise.
        """
        generate_measurements_entry(self.harness.get_system_name(),
                                    self.harness.name,
                                    self.harness._get_submission_benchmark_name(),
                                    self.scenario,
                                    self.benchmark_conf["input_dtype"],
                                    self.benchmark_conf["precision"],
                                    self.harness_flag_dict)
        return True

    def handle_failure(self):
        """Called after handle() if it errors.
        """
        system_name = self.harness.get_system_name()
        benchmark_name = self.harness._get_submission_benchmark_name()
        scenario_name = self.scenario.valstr()
        raise RuntimeError(f"Could not generate measurements/ entries for {system_name}/{benchmark_name}/{scenario_name}")

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.

        success (bool): Indicates whether or not self.handle() executed successfully.  This is useful when cleanup
                        behaves differently when handle fails, or the cleanup code depends on something that is only
                        done on successful runs.
        """
        if success:
            system_name = self.harness.get_system_name()
            benchmark_name = self.harness._get_submission_benchmark_name()
            scenario_name = self.scenario.valstr()
            logging.info(f"Generated measurements/ entries for {system_name}/{benchmark_name}/{scenario_name}")
