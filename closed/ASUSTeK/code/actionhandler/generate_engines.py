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

import shutil
import time
from dataclasses import dataclass
from enum import Enum, unique, auto
from typing import Any, Dict, List

from code import get_benchmark
from code.actionhandler.base import ActionHandler
from code.common import logging, dict_eq
from code.common.constants import *
from code.common.fields import get_applicable_fields
from code.common.mps import turn_on_mps, turn_off_mps
from code.common.power_limit import ScopedPowerStateController


@dataclass
class _GenerateEngineJob:
    """Contains metadata for a GenerateEngine job"""

    id: str
    """str: The identifier for this job"""

    config: Dict[str, Any]
    """Dict[str, Any]: The benchmark configuration to generate an engine for, in dictionary form"""

    cached_engine_identifier: Optional[str] = None
    """str: If set, denotes the identifier of an already built, cached engine that can be copied from. In this case, a
    sufficient unique identifier is the config_ver string."""


class GenerateEngineHandler(ActionHandler):
    """Handles the GenerateEngine action."""

    SESSION_BUILT_ENGINES = dict()
    """Dict[str, Dict]: Cache of engine settings that have been generated this session. Maps the workload setting to a
    dictionary of configuration settings.

    The NVIDIA MLPerf Inference framework supports a feature that allows a TRT engine to be copied from an identical
    configuration. However, to ensure that the engine configuration is exactly the same, it does not copy from
    previously built engines, and only ones built during the same session.

    By default, each process is its own separate session. However, you can call 'GenerateEngines.reset_session()' to
    start a new session within the same process.
    """

    def __init__(self, benchmark_conf, power_controller, allow_mps=False, build_gpu=True, build_dla=True):
        """Creates a new ActionHandler for GenerateEngines

        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
            power_controller (PowerController): The PowerController to control the power settings of the system
            allow_mps (bool): Whether or not MPS can be enabled. (Default: False)
            build_gpu (bool): Whether or not to build the GPU engine. (Default: True)
            build_dla (bool): Whether or not to build the DLA engine, if applicable. Ignored if DLA is not used for this
                              benchmark. (Default: True)
        """
        super().__init__(Action.GenerateEngines)

        self.benchmark_conf = benchmark_conf
        self.system = benchmark_conf["system"]
        self.benchmark = benchmark_conf["benchmark"]
        self.scenario = benchmark_conf["scenario"]
        self.workload_setting = benchmark_conf["workload_setting"]
        self.power_controller = ScopedPowerStateController(power_controller)
        self.allow_mps = allow_mps
        self.use_mps = False
        self.build_gpu = build_gpu
        self.build_dla = build_dla

        self.jobs = []

    def setup(self):
        """Called once before handle().
        """
        self.power_controller.set_power_state()

        active_sms = self.benchmark_conf.get("active_sms", None)
        if self.allow_mps:
            # Turn on MPS if server scenario and if active_sms is specified.
            self.use_mps = (self.scenario == Scenario.Server and active_sms is not None and active_sms < 100)
            if self.use_mps:
                turn_on_mps(active_sms)

        # Preprocess the configs for the GPU and DLA engines in setup(), rather than handle(), due to handle() being
        # called in a subprocess when wrapped in a SubprocessActionHandler. Any changes done to
        # GenerateEngineHandler.SESSION_BUILT_ENGINES will only affect the instance in the child processes and not
        # propagate to the parent.
        engine_setting = self.workload_setting.shortname()
        engine_id = f"{self.system.get_id()}.{self.benchmark.valstr()}.{self.scenario.valstr()}.{engine_setting}"
        if self.build_dla and "dla_batch_size" in self.benchmark_conf:
            dla_config = self.benchmark_conf.copy()
            dla_config["batch_size"] = dla_config["dla_batch_size"]
            self.jobs.append(_GenerateEngineJob(engine_id + ".dla", dla_config))
        if self.build_gpu and "gpu_batch_size" in self.benchmark_conf:
            gpu_config = self.benchmark_conf.copy()
            gpu_config["batch_size"] = gpu_config["gpu_batch_size"]
            gpu_config["dla_core"] = None
            self.jobs.append(_GenerateEngineJob(engine_id + ".gpu", gpu_config))
        logging.debug(f"Queued {len(self.jobs)} engines to generate")

        # Get the keys relevant keys to engine generation and see if we can use a cached engine
        # Cannot reuse cached engines for RNNT since it is a multi-engine benchmark and the naming of the engine files
        # currently disallows this.
        if self.benchmark != Benchmark.RNNT:
            for job in self.jobs:
                for k, v in GenerateEngineHandler.SESSION_BUILT_ENGINES.items():
                    logging.debug(f"Checking new job {job.id} against cached engine {k}")
                    conf_equiv = dict_eq(job.config, v.config, ignore_keys={"workload_setting",
                                                                            "config_ver",
                                                                            "inference_server",
                                                                            "accuracy_level"})
                    if conf_equiv:
                        logging.info(f"GenerateEnginesJob({job.id}) can re-use engine {k}")
                        job.cached_engine_identifier = v.config["workload_setting"].shortname()
                        break

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.
        """
        self.power_controller.reset_power_state()
        if self.use_mps:
            turn_off_mps()

        # If success, add jobs to session cache
        if success:
            for job in self.jobs:
                GenerateEngineHandler.SESSION_BUILT_ENGINES[job.id] = job

    def build_engine(self, job: _GenerateEngineJob) -> float:
        """Build engine for a given job.

        Args:
            job (_GenerateEngineJob): The job to build the engine for

        Returns:
            float: The time taken to build the engine
        """
        logging.debug(f"{job.id=} {job.config=}")
        start_time = time.time()

        # Create the benchmark builder and see if we can use a cached engine
        builder = get_benchmark(job.config)
        if job.cached_engine_identifier is not None:
            dst_path = builder._get_engine_fpath(None, None)
            builder.config_ver = job.cached_engine_identifier
            src_path = builder._get_engine_fpath(None, None)
            logging.info(f"Copying {src_path} to {dst_path}")
            shutil.copyfile(src_path, dst_path)
        else:
            builder.build_engines()
        return time.time() - start_time

    def handle(self):
        """Run the action.
        """
        logging.info(f"Building engines for {self.benchmark.valstr()} benchmark in {self.scenario.valstr()} scenario...")
        total_engine_build_time = 0
        for job in self.jobs:
            total_engine_build_time += self.build_engine(job)
        logging.info(f"Finished building engines for {self.benchmark.valstr()} benchmark in {self.scenario.valstr()} scenario.")
        print(f"Time taken to generate engines: {total_engine_build_time} seconds")
        return True

    def handle_failure(self):
        """Called after handle() if it errors.
        """
        raise RuntimeError("Building engines failed!")

    @classmethod
    def reset_session(cls):
        cls.SESSION_BUILT_ENGINES.clear()
