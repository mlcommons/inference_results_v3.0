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

from typing import Any, Dict, List

from code import get_benchmark
from code.actionhandler.base import ActionHandler
from code.common import logging, dict_eq
from code.common.constants import *


class CalibrateHandler(ActionHandler):
    """Handles the Calibrate action."""

    def __init__(self, benchmark_conf):
        """Creates a new ActionHandler for Calibrate

        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
        """
        super().__init__(Action.Calibrate)

        self.benchmark_conf = benchmark_conf
        self.system = benchmark_conf["system"]
        self.benchmark = benchmark_conf["benchmark"]
        self.scenario = benchmark_conf["scenario"]
        self.workload_setting = benchmark_conf["workload_setting"]

    def setup(self):
        """Called once before handle().
        """
        logging.info(f"Generating calibration cache for Benchmark \"{self.benchmark.valstr()}\"")
        self.benchmark_conf["dla_core"] = None  # Cannot calibrate on DLA
        self.benchmark_conf["force_calibration"] = True

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.
        """
        return

    def handle(self):
        """Run the action.
        """
        b = get_benchmark(self.benchmark_conf)
        b.calibrate()
        return True

    def handle_failure(self):
        """Called after handle() if it errors.
        """
        raise RuntimeError("Calibration failed!")
