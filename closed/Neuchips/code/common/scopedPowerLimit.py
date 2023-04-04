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

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging, run_command
from code.common.systems.system_list import SystemClassifications
import subprocess
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from code.common.nvpmodel_orin import nvpmodel_template_orin
from code.common.nvpmodel_xavier import nvpmodel_template_xavier_nx, nvpmodel_template_xavier_agx

@dataclass
class XavierPowerState:
    gpu_freq: int
    dla_freq: int
    cpu_freq: int
    emc_freq: int

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


PowerState = Union[XavierPowerState, OrinPowerState, ServerPowerState]


def extract_field(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any], field_name: str) -> Any:
    """Extracts a field from the given parameters, preferring main_args if it is supplied."""
    field = main_args.get(field_name, None)
    if field is None:
        field = benchmark_conf.get(field_name, None)
    return field


def get_power_state_server(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    # override bencmark conf if arg was also supplied to main
    power_limit = benchmark_conf["power_limit"] = extract_field(main_args, benchmark_conf, "power_limit")
    cpu_freq = benchmark_conf["cpu_freq"] = extract_field(main_args, benchmark_conf, "cpu_freq")

    if power_limit or cpu_freq:
        return ServerPowerState(power_limit, cpu_freq)

    return None


def get_power_state_xavier(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    # override bencmark conf if arg was also supplied to main
    gpu_freq = benchmark_conf["soc_gpu_freq"] = extract_field(main_args, benchmark_conf, "soc_gpu_freq")
    dla_freq = benchmark_conf["soc_dla_freq"] = extract_field(main_args, benchmark_conf, "soc_dla_freq")
    cpu_freq = benchmark_conf["soc_cpu_freq"] = extract_field(main_args, benchmark_conf, "soc_cpu_freq")
    emc_freq = benchmark_conf["soc_emc_freq"] = extract_field(main_args, benchmark_conf, "soc_emc_freq")

    # if any of these flags are set, all of them should be
    frequencies = [gpu_freq, dla_freq, cpu_freq, emc_freq]
    if None in frequencies:
        assert not any(frequencies), f"All frequencies must be set ({frequencies})"
        return None
    return XavierPowerState(*frequencies)


def get_power_state_orin(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    # override bencmark conf if arg was also supplied to main
    gpu_freq = benchmark_conf["soc_gpu_freq"] = extract_field(main_args, benchmark_conf, "soc_gpu_freq")
    dla_freq = benchmark_conf["soc_dla_freq"] = extract_field(main_args, benchmark_conf, "soc_dla_freq")
    cpu_freq = benchmark_conf["soc_cpu_freq"] = extract_field(main_args, benchmark_conf, "soc_cpu_freq")
    emc_freq = benchmark_conf["soc_emc_freq"] = extract_field(main_args, benchmark_conf, "soc_emc_freq")
    num_cores = benchmark_conf["orin_num_cores"] = extract_field(main_args, benchmark_conf, "orin_num_cores")

    # if any of these flags are set, all of them should be
    #frequencies = [gpu_freq, dla_freq, cpu_freq, emc_freq, num_cores]
    frequencies = [gpu_freq, dla_freq, cpu_freq, emc_freq, num_cores]
    if None in frequencies:
        assert not any(frequencies), f"All frequencies must be set ({frequencies})"
        return None
    return OrinPowerState(*frequencies)



def get_power_state(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    """Parse args and get target power state"""

    if benchmark_conf["use_cpu"]:
        return None

    if SystemClassifications.is_xavier():
        return get_power_state_xavier(main_args, benchmark_conf)
    if SystemClassifications.is_orin():
        return get_power_state_orin(main_args, benchmark_conf)
    else:
        return get_power_state_server(main_args, benchmark_conf)


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


def set_power_state_server(power_state: ServerPowerState) -> List[float]:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Record current power limits.
    if power_state.power_limit:
        cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"
        logging.info(f"Getting current GPU power limits: {cmd}")
        output = run_command(cmd, get_output=True, tee=False)
        current_limits = [float(line) for line in output]

        # Set power limit to the specified value.
        cmd = f"sudo nvidia-smi -pl {power_state.power_limit}"
        logging.info(f"Setting current GPU power limits: {cmd}")
        run_command(cmd)

    if power_state.cpu_freq:
        set_cpufreq(power_state.cpu_freq)

    return ServerPowerState(current_limits, None)


def reset_power_state_server(power_state: ServerPowerState) -> None:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Reset power limit to the specified value.
    power_limits = power_state.power_limit
    for i in range(len(power_limits)):
        cmd = f"sudo nvidia-smi -i {i} -pl {power_limits[i]}"
        logging.info(f"Resetting power limit for GPU {i}: {cmd}")
        run_command(cmd)


def set_power_state_xavier(nvpmodel_template, power_state: XavierPowerState) -> None:
    """Record the current power state and set power limit using nvpmodel."""

    with open("build/nvpmodel.temp.conf", "w") as f:
        f.write(nvpmodel_template.format(gpu_clock=power_state.gpu_freq,
                                         dla_clock=power_state.dla_freq,
                                         cpu_clock=power_state.cpu_freq,
                                         emc_clock=power_state.emc_freq))
    cmd = "sudo /usr/sbin/nvpmodel -f build/nvpmodel.temp.conf -m 8 && sudo /usr/sbin/nvpmodel -d cool"
    logging.info(f"Setting current nvpmodel conf: {cmd}")
    run_command(cmd)

    return None


def set_power_state_xavier_agx(power_state: XavierPowerState) -> None:
    logging.info(f"Setting power state on Xavier AGX")
    return set_power_state_xavier(nvpmodel_template_xavier_agx, power_state)


def set_power_state_xavier_nx(power_state: XavierPowerState) -> None:
    logging.info(f"Setting power state on Xavier NX")
    return set_power_state_xavier(nvpmodel_template_xavier_nx, power_state)


def set_power_state_orin(power_state: OrinPowerState) -> None:
    """Record the current power state and set power limit using nvpmodel."""
    logging.info(f"Setting power state on Orin")

    num_cores_total = 12
    if not power_state.num_cpu_cores:
        cores_map = [1] * num_cores_total # all cores on by default
    else:
        num_cores_offline = num_cores_total - power_state.num_cpu_cores
        cores_map = [1] * power_state.num_cpu_cores + [0] * num_cores_offline 

    with open("build/nvpmodel.temp.conf", "w") as f:
        f.write(nvpmodel_template_orin.format(gpu_clock=power_state.gpu_freq,
                                         dla_clock=power_state.dla_freq,
                                         cpu_clock=power_state.cpu_freq,
                                         emc_clock=power_state.emc_freq,
                                         cpu_core=cores_map))
    cmd = "sudo /usr/sbin/nvpmodel -f build/nvpmodel.temp.conf -m 0"
    logging.error(f"Setting current nvpmodel conf: {cmd}")
    # ignore error because disabling CPU cores makes nvpmodel command return 255, even though they
    # do get disabled
    subprocess.call(cmd, shell=True)

    return None


def reset_power_state_xavier(power_limits: List[float]) -> None:
    """Reset power limit using nvpmodel conf"""

    # Reset power limit to the specified value.
    cmd = "sudo /usr/sbin/nvpmodel -m 0 && sudo /usr/sbin/nvpmodel -d cool"
    logging.info(f"Resetting nvpmodel conf: {cmd}")
    run_command(cmd)


def reset_power_state_orin(power_limits: List[float]) -> None:
    """Reset power limit using nvpmodel conf"""

    # Reset power limit to the specified value.
    cmd = "sudo /usr/sbin/nvpmodel -f /etc/nvpmodel.conf -m 0 && sudo jetson_clocks"
    logging.info(f"Resetting nvpmodel conf: {cmd}")
    run_command(cmd)


class ScopedPowerLimit:
    """
        Create scope GPU power upper limit is overridden to the specified value.
        Setting power_limit to None to disable the scoped power limit.
    """

    def __init__(self, target_power_state: PowerState):
        self.target_power_state = target_power_state
        self.current_power_state = None
        #if SystemClassifications.is_xavier_agx():
        #    self.set_power_limits = set_power_state_xavier_agx
        #    self.reset_power_limits = reset_power_state_xavier
        #elif SystemClassifications.is_xavier_nx():
        #    self.set_power_limits = set_power_state_xavier_nx
        #    self.reset_power_limits = reset_power_state_xavier
        #elif SystemClassifications.is_orin():
        #    self.set_power_limits = set_power_state_orin
        #    self.reset_power_limits = reset_power_state_orin
        #else:
        #    self.set_power_limits = set_power_state_server
        #    self.reset_power_limits = reset_power_state_server

        self.set_power_limits = set_power_state_server
        self.reset_power_limits = reset_power_state_server

    def __enter__(self):
        if self.target_power_state is not None:
            self.current_power_state = self.set_power_limits(self.target_power_state)

    def __exit__(self, type, value, traceback):
        if self.target_power_state is not None:
            self.reset_power_limits(self.current_power_state)
