# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


nvpmodel_template_orin = """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online
CORE_4 /sys/devices/system/cpu/cpu4/online
CORE_5 /sys/devices/system/cpu/cpu5/online
CORE_6 /sys/devices/system/cpu/cpu6/online
CORE_7 /sys/devices/system/cpu/cpu7/online
CORE_8 /sys/devices/system/cpu/cpu8/online
CORE_9 /sys/devices/system/cpu/cpu9/online
CORE_10 /sys/devices/system/cpu/cpu10/online
CORE_11 /sys/devices/system/cpu/cpu11/online

< PARAM TYPE=FILE NAME=TPC_POWER_GATING >
TPC_PG_MASK /sys/devices/gpu.0/tpc_pg_mask

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_A78_0 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
FREQ_TABLE_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ_KNEXT /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_A78_1 >
FREQ_TABLE /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq
FREQ_TABLE_KNEXT /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ_KNEXT /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ_KNEXT /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_A78_2 >
FREQ_TABLE /sys/devices/system/cpu/cpu8/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu8/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu8/cpufreq/scaling_min_freq
FREQ_TABLE_KNEXT /sys/devices/system/cpu/cpu8/cpufreq/scaling_available_frequencies
MAX_FREQ_KNEXT /sys/devices/system/cpu/cpu8/cpufreq/scaling_max_freq
MIN_FREQ_KNEXT /sys/devices/system/cpu/cpu8/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/available_frequencies
MAX_FREQ /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq
MIN_FREQ /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq
FREQ_TABLE_KNEXT /sys/devices/17000000.ga10b/devfreq_dev/available_frequencies
MAX_FREQ_KNEXT /sys/devices/17000000.ga10b/devfreq_dev/max_freq
MIN_FREQ_KNEXT /sys/devices/17000000.ga10b/devfreq_dev/min_freq

< PARAM TYPE=CLOCK NAME=DLA0_CORE >
MAX_FREQ /sys/devices/platform/13e40000.host1x/15880000.nvdla0/acm/clk_cap/dla0_core
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/15880000.nvdla0/acm/clk_cap/dla0_core

< PARAM TYPE=CLOCK NAME=DLA0_FALCON >
MAX_FREQ /sys/devices/platform/13e40000.host1x/15880000.nvdla0/acm/clk_cap/dla0_falcon
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/15880000.nvdla0/acm/clk_cap/dla0_falcon

< PARAM TYPE=CLOCK NAME=DLA1_CORE >
MAX_FREQ /sys/devices/platform/13e40000.host1x/158c0000.nvdla1/acm/clk_cap/dla1_core
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/158c0000.nvdla1/acm/clk_cap/dla1_core

< PARAM TYPE=CLOCK NAME=DLA1_FALCON >
MAX_FREQ /sys/devices/platform/13e40000.host1x/158c0000.nvdla1/acm/clk_cap/dla1_falcon
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/158c0000.nvdla1/acm/clk_cap/dla1_falcon

< PARAM TYPE=CLOCK NAME=PVA0_VPS >
MAX_FREQ /sys/devices/platform/13e40000.host1x/16000000.pva0/acm/clk_cap/pva0_vps
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/16000000.pva0/acm/clk_cap/pva0_vps

< PARAM TYPE=CLOCK NAME=PVA0_AXI >
MAX_FREQ /sys/devices/platform/13e40000.host1x/16000000.pva0/acm/clk_cap/pva0_cpu_axi
MAX_FREQ_KNEXT /sys/devices/platform/13e40000.host1x/16000000.pva0/acm/clk_cap/pva0_cpu_axi

< PARAM TYPE=FILE NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

< POWER_MODEL ID=0 NAME=MAX_Q >
CPU_ONLINE CORE_0 {cpu_core[0]}
CPU_ONLINE CORE_1 {cpu_core[1]}
CPU_ONLINE CORE_2 {cpu_core[2]}
CPU_ONLINE CORE_3 {cpu_core[3]}
CPU_ONLINE CORE_4 {cpu_core[4]}
CPU_ONLINE CORE_5 {cpu_core[5]}
CPU_ONLINE CORE_6 {cpu_core[6]}
CPU_ONLINE CORE_7 {cpu_core[7]}
CPU_ONLINE CORE_8 {cpu_core[8]}
CPU_ONLINE CORE_9 {cpu_core[9]}
CPU_ONLINE CORE_10 {cpu_core[10]}
CPU_ONLINE CORE_11 {cpu_core[11]}
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_A78_0 MIN_FREQ {cpu_clock}
CPU_A78_0 MAX_FREQ {cpu_clock}
CPU_A78_1 MIN_FREQ {cpu_clock}
CPU_A78_1 MAX_FREQ {cpu_clock}
CPU_A78_2 MIN_FREQ {cpu_clock}
CPU_A78_2 MAX_FREQ {cpu_clock}
GPU MIN_FREQ {gpu_clock}
GPU MAX_FREQ {gpu_clock}
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
DLA0_CORE MAX_FREQ {dla_clock}
DLA1_CORE MAX_FREQ {dla_clock}
DLA0_FALCON MAX_FREQ -1
DLA1_FALCON MAX_FREQ -1
PVA0_VPS MAX_FREQ -1
PVA0_AXI MAX_FREQ -1
EMC MAX_FREQ {emc_clock}
# mandatory section to configure the default power mode
< PM_CONFIG DEFAULT=0 >

"""
