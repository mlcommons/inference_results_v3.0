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

import sys
import os
sys.path.insert(0, os.getcwd())

import itertools

from code.common.systems.system_list import KnownSystem
from code.common.constants import CPUArchitecture
from code.common import dict_get
from code.main import main, parse_main_args

if __name__ == "__main__":
    default_args = parse_main_args(custom=["--action", "generate_conf_files", "--config_ver", "all"])
    for system in KnownSystem:
        print("Generating measurements/ entries for", system.name)

        config_ver_keywords = ["maxq", "high_accuracy", "triton"]

        cross_products = itertools.product([False, True], repeat=len(config_ver_keywords))
        all_config_vers = []
        for config_product in cross_products:
            if not any(config_product):
                all_config_vers.append("default")
            else:
                all_config_vers.append("_".join(itertools.compress(config_ver_keywords, config_product)))

        num_migs = system.value.accelerator_conf.num_migs()
        if num_migs == 1:
            all_config_vers.extend(["hetero", "hetero_high_accuracy"])
        elif num_migs > 1:
            # 56-MIG and 32-MIG slice only uses Triton, not default config_ver
            all_config_vers.remove("default")

        # Triton CPU has special config_vers, and does not use 'default'
        if len(system.value.accelerator_conf.get_accelerators()) == 0 and \
                system.value.host_cpu_conf.get_architecture() == CPUArchitecture.x86_64:
            all_config_vers = ["openvino", "openvino_high_accuracy"]

        default_args["config_ver"] = ",".join(all_config_vers)
        main(default_args, system.value)
