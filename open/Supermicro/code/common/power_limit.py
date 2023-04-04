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


import subprocess, time
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass

from code.common import logging, run_command
from code.common.constants import AcceleratorType
from code.common.systems.accelerator import GPU
from code.common.systems.system_list import SystemClassifications
from code.common.nvpmodel_orin import nvpmodel_template_orin, cpu_clock_str


@dataclass
class OrinPowerState:
    gpu_freq: int
    dla_freq: int
    cpu_freq: int
    emc_freq: int
    num_cpu_cores: int


@dataclass
class ServerPowerState:
    power_limit: Union[int, List[int], None]
    cpu_freq: Optional[int]


PowerState = Union[OrinPowerState, ServerPowerState]


def extract_field(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any], field_name: str) -> Any:
    """Extracts a field from the given parameters, preferring main_args if it is supplied."""
    field = main_args.get(field_name, None)
    if field is None:
        field = benchmark_conf.get(field_name, None)
    return field


def set_cpufreq(cpu_freq: int) -> List[float]:
    # Record current cpu governor
    cmd = "sudo cpupower -c all frequency-set -g userspace"
    logging.info(f"Set cpu power governor: userspace")
    run_command(cmd)

    # Set cpu freq
    cmd = f"sudo cpupower -c all frequency-set -f {cpu_freq}"
    logging.info(f"Setting cpu frequency: {cmd}")
    run_command(cmd)


def reset_cpufreq():
    # Record current cpu governor
    cmd = "sudo cpupower -c all frequency-set -g ondemand"
    logging.info(f"Set cpu power governor: ondemand")
    run_command(cmd)


class PowerStateController(ABC):
    """Allows for the system power setting to be set and reset
    """

    def __init__(self, main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
        self.main_args = main_args
        self.benchmark_conf = benchmark_conf

    @abstractmethod
    def power_state_target(self):
        """Computes the PowerState that the system should be set to for benchmarks to run"""
        pass

    @abstractmethod
    def set_power_state(self, target: PowerState) -> PowerState:
        """Sets the system power settings to the settings in `target`.

        Args:
            target (PowerState): Power settings the system should be set to.

        Returns:
            PowerState: The power setting of the system before setting to the new state
        """
        pass

    @abstractmethod
    def reset_power_state(self, target: PowerState):
        """Resets the system power settings to the system default.

        Args:
            target (PowerState): The PowerState before self.set_power_state was called
        """
        pass


class NoopPowerStateController(PowerStateController):
    """Dummy PowerStateController. Used for tests and cases where no power settings are needed."""

    def power_state_target(self):
        return None

    def set_power_state(self, target: PowerState):
        return None

    def reset_power_state(self, target: PowerState):
        return None


class ServerPowerStateController(PowerStateController):
    """PowerStateController for NVIDIA Server systems with dGPUs"""

    def power_state_target(self):
        power_limit = self.benchmark_conf["power_limit"] = extract_field(self.main_args, self.benchmark_conf, "power_limit")
        cpu_freq = self.benchmark_conf["cpu_freq"] = extract_field(self.main_args, self.benchmark_conf, "cpu_freq")
        if power_limit or cpu_freq:
            return ServerPowerState(power_limit, cpu_freq)
        return None

    def set_power_state(self, target: PowerState) -> PowerState:
        # Record current power limits.
        if target.power_limit:
            cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"
            logging.info(f"Getting current GPU power limits: {cmd}")
            output = run_command(cmd, get_output=True, tee=False)
            current_limits = [float(line) for line in output]

            # Set power limit to the specified value.
            cmd = f"sudo nvidia-smi -pl {target.power_limit}"
            logging.info(f"Setting current GPU power limits: {cmd}")
            run_command(cmd)

        if target.cpu_freq:
            set_cpufreq(target.cpu_freq)

        return ServerPowerState(current_limits, None)

    def reset_power_state(self, target: PowerState):
        # Reset power limit to the specified value.
        power_limits = target.power_limit
        for i in range(len(power_limits)):
            cmd = f"sudo nvidia-smi -i {i} -pl {power_limits[i]}"
            logging.info(f"Resetting power limit for GPU {i}: {cmd}")
            run_command(cmd)
            time.sleep(1)


class OrinPowerStateController(PowerStateController):
    """PowerStateController for Orin systems"""

    def power_state_target(self):
        gpu_freq = self.benchmark_conf["soc_gpu_freq"] = extract_field(self.main_args, self.benchmark_conf, "soc_gpu_freq")
        dla_freq = self.benchmark_conf["soc_dla_freq"] = extract_field(self.main_args, self.benchmark_conf, "soc_dla_freq")
        cpu_freq = self.benchmark_conf["soc_cpu_freq"] = extract_field(self.main_args, self.benchmark_conf, "soc_cpu_freq")
        emc_freq = self.benchmark_conf["soc_emc_freq"] = extract_field(self.main_args, self.benchmark_conf, "soc_emc_freq")
        num_cores = self.benchmark_conf["orin_num_cores"] = extract_field(self.main_args, self.benchmark_conf, "orin_num_cores")

        # if any of these flags are set, all of them should be
        frequencies = [gpu_freq, dla_freq, cpu_freq, emc_freq, num_cores]
        if None in frequencies:
            assert not any(frequencies), f"All frequencies must be set ({frequencies})"
            return None
        return OrinPowerState(*frequencies)

    def set_power_state(self, target: PowerState) -> PowerState:
        logging.info(f"Setting power state on Orin")

        num_cores_total = 12
        if not target.num_cpu_cores:
            cores_map = [1] * num_cores_total  # all cores on by default
        else:
            num_cores_offline = num_cores_total - target.num_cpu_cores
            cores_map = [1] * target.num_cpu_cores + [0] * num_cores_offline

        with open("build/nvpmodel.temp.conf", "w") as f:
            f.write(nvpmodel_template_orin.format(gpu_clock=target.gpu_freq,
                                                  dla_clock=target.dla_freq,
                                                  cpu_clock_str=cpu_clock_str(target.num_cpu_cores, target.cpu_freq),
                                                  emc_clock=target.emc_freq,
                                                  cpu_core=cores_map))
        cmd = "sudo /usr/sbin/nvpmodel -f build/nvpmodel.temp.conf -m 0"
        #logging.error(f"Setting current nvpmodel conf: {cmd}")
        run_command(cmd)
        # ignore error because disabling CPU cores makes nvpmodel command return 255, even though they
        # do get disabled
   
        return None #FIXME: Return type incorrect shouldnt this be an empty PowerState object or omit return type declearation?

    def reset_power_state(self, target: PowerState):
        cmd = "sudo /usr/sbin/nvpmodel -f /etc/nvpmodel.conf -m 0"
        logging.info(f"Resetting nvpmodel conf: {cmd}")
        run_command(cmd)

def get_power_controller(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    """Returns an instance of the PowerStateController that corresponds to the current system"""
    primary_accelerator = benchmark_conf["system"].accelerator_conf.get_primary_accelerator()
    if primary_accelerator is None or benchmark_conf["use_cpu"]:
        return NoopPowerStateController(main_args, benchmark_conf)
    elif SystemClassifications.is_orin():
        return OrinPowerStateController(main_args, benchmark_conf)
    elif isinstance(primary_accelerator, GPU) and primary_accelerator.accelerator_type == AcceleratorType.Discrete:
        return ServerPowerStateController(main_args, benchmark_conf)

    return NoopPowerStateController(main_args, benchmark_conf)


class ScopedPowerStateController:
    """Wraps a PowerStateController and allows it to be used as a scoped context.

    Calls set_power_state on __enter__ and reset_power_state on __exit__.
    """

    def __init__(self, power_state_controller: PowerStateController):
        self.power_state_controller = power_state_controller
        self.power_state_target = power_state_controller.power_state_target()
        self.original_power_state = None

    def set_power_state(self):
        if self.power_state_target is not None:
            self.original_power_state = self.power_state_controller.set_power_state(self.power_state_target)

    def reset_power_state(self):
        if self.power_state_target is not None:
            self.power_state_controller.reset_power_state(self.original_power_state)

    def __enter__(self):
        self.set_power_state()

    def __exit__(self, type, value, traceback):
        self.reset_power_state()
