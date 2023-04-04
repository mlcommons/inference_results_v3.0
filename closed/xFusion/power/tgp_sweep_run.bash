#!/bin/bash
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

# This script run mlperf networks with different TGPs and log power

# suppose MLPERF_REPO is your mlperf-inference dir
# WKLD is your target network
# cd MLPERF_REPO/closed/NVIDIA
# make prebuild
# make generate_engines RUN_ARGS="--benchmarks=${WKLD} --scenarios=Offline"
# The above two lines will build mlperf container and execution plans
# then customize internal/scripts/tgp_sweep_run.bash to launch GPU TGP sweep
# this script uses nvidia-smi, ipmitool tool

#have to change the following two lines
MLPERF_ROOT=
NVIDIA_VISIBLE_DEVICES=

PROJECT_ROOT=$( pwd )
HOSTNAME=$( hostname | awk -F'.' '{print $1}' )
WKLD="resnet50,ssd-resnet34,3d-unet,dlrm,rnnt,bert"

set_powercap () {
  local PL=$1
  sudo nvidia-smi -pm 0 -i ${NVIDIA_VISIBLE_DEVICES}
  sudo nvidia-smi -pm 1 -i ${NVIDIA_VISIBLE_DEVICES}
  sudo nvidia-smi -rgc -i ${NVIDIA_VISIBLE_DEVICES}
  sudo nvidia-smi -pl ${PL} -i ${NVIDIA_VISIBLE_DEVICES}
  nvidia-smi
}

power_monitor () {
  local LOGFILE=$1
  local GPULOG=$2
  rm -f ${LOGFILE} ${GPULOG}
  while [ 1 -eq 1 ]; do
    date >> ${LOGFILE}
    sudo ipmitool sdr list >> ${LOGFILE}
    nvidia-smi --query-gpu=pstate,clocks.gr,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,display_mode,display_active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost,memory.used,utilization.gpu,utilization.memory,ecc.mode.current,enforced.power.limit,temperature.gpu --format=csv -i ${NVIDIA_VISIBLE_DEVICES} >> ${GPULOG}
  done
}

cd $PROJECT_ROOT
power_monitor ${HOSTNAME}_powersweep.ipmitrace ${HOSTNAME}_powersweep.nvidia_smi_trace &
cd $MLPERF_ROOT/closed/NVIDIA

for POWERCAP in 300 275 250 225 200 175 150; do
  set_powercap ${POWERCAP}
  sleep 60
  make launch_docker DOCKER_COMMAND='make run_harness RUN_ARGS="--benchmarks='${WKLD}' --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"'
done
sleep 60
cd $PROJECT_ROOT
