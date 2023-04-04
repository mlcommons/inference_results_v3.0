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


nvpmodel_template_xavier_agx = """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online
CORE_4 /sys/devices/system/cpu/cpu4/online
CORE_5 /sys/devices/system/cpu/cpu5/online
CORE_6 /sys/devices/system/cpu/cpu6/online
CORE_7 /sys/devices/system/cpu/cpu7/online

< PARAM TYPE=FILE NAME=TPC_POWER_GATING >
TPC_PG_MASK /sys/devices/gpu.0/tpc_pg_mask

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_DENVER_0 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_1 >
FREQ_TABLE /sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_2 >
FREQ_TABLE /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_3 >
FREQ_TABLE /sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
MAX_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq
MIN_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq

< PARAM TYPE=CLOCK NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

< PARAM TYPE=CLOCK NAME=DLA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla

< PARAM TYPE=CLOCK NAME=DLA_FALCON >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon

< PARAM TYPE=CLOCK NAME=PVA_VPS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps

< PARAM TYPE=CLOCK NAME=PVA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_core
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_core

< PARAM TYPE=CLOCK NAME=CVNAS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_cvnas
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_cvnas

< POWER_MODEL ID=0 NAME=MAXN >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_ONLINE CORE_4 1
CPU_ONLINE CORE_5 1
CPU_ONLINE CORE_6 1
CPU_ONLINE CORE_7 1
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ 1200000
CPU_DENVER_0 MAX_FREQ -1
CPU_DENVER_1 MIN_FREQ 1200000
CPU_DENVER_1 MAX_FREQ -1
CPU_DENVER_2 MIN_FREQ 1200000
CPU_DENVER_2 MAX_FREQ -1
CPU_DENVER_3 MIN_FREQ 1200000
CPU_DENVER_3 MAX_FREQ -1
GPU MIN_FREQ 318750000
GPU MAX_FREQ -1
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 0
DLA_CORE MAX_FREQ -1
DLA_FALCON MAX_FREQ -1
PVA_VPS MAX_FREQ -1
PVA_CORE MAX_FREQ -1
CVNAS MAX_FREQ -1

< POWER_MODEL ID=8 NAME=MODE_MLPERF_V1_MAXQ >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
CPU_ONLINE CORE_6 0
CPU_ONLINE CORE_7 0
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ {cpu_clock}
CPU_DENVER_0 MAX_FREQ {cpu_clock}
GPU MIN_FREQ 318750000
GPU MAX_FREQ {gpu_clock}
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ {emc_clock}
DLA_CORE MAX_FREQ {dla_clock}
DLA_FALCON MAX_FREQ 630000000
PVA_VPS MAX_FREQ 760000000
PVA_CORE MAX_FREQ 532000000
CVNAS MAX_FREQ 1011200000

< PM_CONFIG DEFAULT=8 >
< FAN_CONFIG DEFAULT=cool >

"""

nvpmodel_template_xavier_nx = """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online
CORE_4 /sys/devices/system/cpu/cpu4/online
CORE_5 /sys/devices/system/cpu/cpu5/online

< PARAM TYPE=FILE NAME=TPC_POWER_GATING >
TPC_PG_MASK /sys/devices/gpu.0/tpc_pg_mask

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_DENVER_0 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_1 >
FREQ_TABLE /sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_2 >
FREQ_TABLE /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
MAX_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq
MIN_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq

< PARAM TYPE=CLOCK NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

< PARAM TYPE=CLOCK NAME=DLA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla

< PARAM TYPE=CLOCK NAME=DLA_FALCON >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon

< PARAM TYPE=CLOCK NAME=PVA_VPS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps

< PARAM TYPE=CLOCK NAME=PVA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_core
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_core

< PARAM TYPE=CLOCK NAME=CVNAS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_cvnas
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_cvnas

< POWER_MODEL ID=0 NAME=MODE_15W_2CORE >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
TPC_POWER_GATING TPC_PG_MASK 1
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ 1190400
CPU_DENVER_0 MAX_FREQ 1907200
GPU MIN_FREQ 0
GPU MAX_FREQ 1109250000
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 1600000000
DLA_CORE MAX_FREQ 1100800000
DLA_FALCON MAX_FREQ 640000000
PVA_VPS MAX_FREQ 819200000
PVA_CORE MAX_FREQ 601600000
CVNAS MAX_FREQ 576000000

< POWER_MODEL ID=8 NAME=MODE_MLPERF_V1_MAXQ >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
TPC_POWER_GATING TPC_PG_MASK 1
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ {cpu_clock}
CPU_DENVER_0 MAX_FREQ {cpu_clock}
GPU MIN_FREQ 0
GPU MAX_FREQ {gpu_clock}
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ {emc_clock}
DLA_CORE MAX_FREQ {dla_clock}
DLA_FALCON MAX_FREQ 640000000
PVA_VPS MAX_FREQ 819200000
PVA_CORE MAX_FREQ 601600000
CVNAS MAX_FREQ 576000000

< PM_CONFIG DEFAULT=8 >
< FAN_CONFIG DEFAULT=cool >

"""
