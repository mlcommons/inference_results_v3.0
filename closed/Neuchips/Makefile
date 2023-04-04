# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# Modified by Neuchips Corp. on 2023/01
#

SHELL := /bin/bash
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)
TIMESTAMP := $(shell date +'%Y.%m.%d-%H.%M.%S')
ifndef HOST_HOSTNAME
    HOST_HOSTNAME := $(HOSTNAME)
endif
HOST_HOSTNAME_SHORT := $(firstword $(subst ., , $(HOST_HOSTNAME)))
HOSTNAME_SHORT := $(firstword $(subst ., , $(HOSTNAME)))
#PYTHON3_CMD := $(shell if ! python3 --version | grep 3.8 &> /dev/null; then echo python3.8; else echo python3; fi)
PYTHON3_CMD := python3

USE_NIGHTLY ?= 0
USE_CPU ?= 1
#USE_INFERENTIA ?= 0
MINIMAL_REGRESSION_PRESUBMIT ?= 0

SUBMITTER ?= Neuchips

# Arch and board variables.
SYSTEM_NAME ?= $(shell $(PYTHON3_CMD) scripts/get_system_id.py 2> /dev/null)
TARGET_X86_64 := 1

ifeq ($(ARCH), x86_64)
    TARGET_X86_64 = 1
endif

SUPPORT_DOCKER := 0
PROJECT_ROOT := $(shell pwd)
BUILD_DIR    := $(PROJECT_ROOT)/build

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work
NO_DOCKER_PULL ?= 0
NO_BUILD ?= 0

# Set the include directory for Loadgen header files
INFERENCE_DIR = $(BUILD_DIR)/inference
INFERENCE_URL = https://github.com/mlcommons/inference.git
LOADGEN_INCLUDE_DIR ?= $(INFERENCE_DIR)/loadgen
LOADGEN_LIB_DIR ?= $(LOADGEN_INCLUDE_DIR)/build

INFERENCE_HASH = f5367250115ad4febf1334b34881ab74f2e55bfe

# Set the power-dev directory
POWER_DEV_DIR = $(BUILD_DIR)/power-dev
POWER_DEV_URL = https://github.com/mlcommons/power-dev.git
POWER_DEV_HASH = 678c23d9eb02f5d9719ed4c115831b4384809aab 	# v3.0
POWER_CLIENT_SCRIPT = $(BUILD_DIR)/power-dev/ptd_client_server/client.py
POWER_NTP_URL = time.stdtime.gov.tw
POWER_SERVER_IP = 192.168.50.46
POWER_SERVER_USERNAME = lab
POWER_SERVER_PASSWORD = labuser

POWER_SERVER_USER_IP = $(POWER_SERVER_USERNAME)@$(POWER_SERVER_IP)
POWER_SERVER_CONFIG = power/server-neuchips-linux.cfg
POWER_SERVER_SCRIPT_DIR = /home/$(POWER_SERVER_USERNAME)/power-dev/ptd_client_server
POWER_SSH = sshpass -p $(POWER_SERVER_PASSWORD) ssh -o "StrictHostKeyChecking no"
POWER_SCP = sshpass -p $(POWER_SERVER_PASSWORD) scp -o "StrictHostKeyChecking no"

# Set log directories for power logs
POWER_LOGS_TEMP_DIR = $(BUILD_DIR)/power_logs_temp

# Set Environment variables to extracted contents
export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):/usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR)
export LIBRARY_PATH := /usr/local/cuda/lib64:/usr/lib/$(ARCH)-linux-gnu:$(LOADGEN_LIB_DIR):$(LIBRARY_PATH)
export PATH := /usr/local/cuda/bin:$(PATH)
export CPATH := /usr/local/cuda/include:/usr/include/$(ARCH)-linux-gnu:/usr/include/$(ARCH)-linux-gnu/cub:$(CPATH)
export CUDA_PATH := /usr/local/cuda
export CCACHE_DISABLE=1
export NUMBA_CACHE_DIR=$(BUILD_DIR)/cache

# Set CUDA_DEVICE_MAX_CONNECTIONS to increase multi-stream performance.
#export CUDA_DEVICE_MAX_CONNECTIONS := 32

# Please run `export MLPERF_SCRATCH_PATH=<path>` to set your scratch space path.
# The below paths are for internal use only.
ifneq ($(wildcard /home/mlperf_inference_data),)
    MLPERF_SCRATCH_PATH ?= /home/mlperf_inference_data
endif
ifneq ($(wildcard /home/scratch.mlperf_inference),)
    DOCKER_MOUNTS += -v /home/scratch.mlperf_inference:/home/scratch.mlperf_inference
endif
ifneq ($(wildcard /home/scratch.svc_compute_arch),)
    DOCKER_MOUNTS += -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch
endif
ifneq ($(wildcard /home/scratch.computelab/sudo),)
    DOCKER_MOUNTS += -v /home/scratch.computelab/sudo:/home/scratch.computelab/sudo
endif
ifneq ($(wildcard /home/scratch.dlsim),)
    DOCKER_MOUNTS += -v /home/scratch.dlsim:/home/scratch.dlsim
endif
ifneq ($(wildcard $(PROJECT_ROOT)/../../regression),)
    DOCKER_MOUNTS += -v $(PROJECT_ROOT)/../../regression:/regression
endif
ifdef MLPERF_SCRATCH_PATH
    ifneq ($(wildcard $(MLPERF_SCRATCH_PATH)),)
        DOCKER_MOUNTS += -v $(MLPERF_SCRATCH_PATH):$(MLPERF_SCRATCH_PATH)
    else
        $(error Path set in MLPERF_SCRATCH_PATH does not exist!)
    endif
endif

# DATA_DIR is the actual location of data in the user-specified MLPERF_SCRATCH_PATH location.
# On the other hand, DATA_DIR_LINK is the location which our scripts assume the data to be located in. In the
# "link_dirs" target, we create a symbolic from DATA_DIR_LINK to DATA_DIR. The same applies to PREPROCESSED_DATA_DIR and
# MODEL_DIR as well.
DATA_DIR_LINK := $(BUILD_DIR)/data
PREPROCESSED_DATA_DIR_LINK := $(BUILD_DIR)/preprocessed_data
MODEL_DIR_LINK := $(BUILD_DIR)/models
DATA_DIR ?= $(MLPERF_SCRATCH_PATH)/data
PREPROCESSED_DATA_DIR ?= $(MLPERF_SCRATCH_PATH)/preprocessed_data
MODEL_DIR ?= $(MLPERF_SCRATCH_PATH)/models
export DATA_DIR
export PREPROCESSED_DATA_DIR
export MODEL_DIR

# Specify default dir for harness output logs.
ifndef LOG_DIR
    export LOG_DIR := $(BUILD_DIR)/logs/$(TIMESTAMP)
    export POWER_LOGS_DIR := $(BUILD_DIR)/power_logs/$(TIMESTAMP)
endif

# Specify debug options for build (default to Release build)
ifeq ($(DEBUG),1)
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

# Driver and cuda version check for x86 and aarch64 non-soc system
MIN_DRIVER_VER := 510
ifeq ($(IS_SOC)$(USE_CPU)$(USE_INFERENTIA)$(SKIP_DRIVER_CHECK), 0000)
    DRIVER_VER_MAJOR ?= $(shell nvidia-smi | /bin/grep -Eo 'Driver Version: [+-]?[0-9]+' | awk -F ' ' '{print $$NF}')
    # Check driver version and launch the appropriate container.
    ifeq ($(shell if [ $(DRIVER_VER_MAJOR) -ge $(MIN_DRIVER_VER) ]; then echo true; else echo false; fi), true)
	    CUDA_VER := 11.6
        DRIVER_VER_MAJOR := $(MIN_DRIVER_VER)
    else
        $(error MLPerf Inference v2.0 code requires NVIDIA Driver Version >= $(MIN_DRIVER_VER).xx)
    endif # Driver check
else
    CUDA_VER := 11.6
    DRIVER_VER_MAJOR := $(MIN_DRIVER_VER)
endif

ifneq ($(IS_SOC), 1)
    DOCKER_IMAGE_NAME := base-cuda$(CUDA_VER)-$(ARCH)-ubuntu20.04
    # Check if we are on intranet
    ifeq ($(shell bash $(PROJECT_ROOT)/scripts/check_intranet.sh),0)
        BASE_IMAGE ?= gitlab-master.nvidia.com/compute/mlperf-inference:$(DOCKER_IMAGE_NAME)
    else
        ifeq ($(CUDA_VER), 11.6)
            ifeq ($(TARGET_X86_64), 1)
                BASE_IMAGE ?= nvidia/cuda:11.6.0-devel-ubuntu20.04@sha256:f9a67849a6298e0cbeebfca5c9da8d3891ffa3e8d30bfd667326ffe232d8fc63
            else ifeq ($(TARGET_AARCH64), 1)
                BASE_IMAGE ?= nvidia/cuda:11.6.0-devel-ubuntu20.04@sha256:119c1041e2041f500cc943c1494c5aee587c325bbacede427071387d6aeaf94d
            else
                $(error MLPerf Inference only supports x86 and aarch64 system now.)
            endif
        else
            $(error MLPerf Inference v2.0 code requires cuda version 11.6)
        endif
    endif # check_intranet
endif # xavier check

TEST_FLAGS := --ignore=internal/test_data --ignore=internal/correlation --ignore=internal/__pycache__ -s --durations 0 -vv -rXfw
SM_GENCODE := $(shell ./scripts/get_gencode_$(ARCH) 2> /dev/null)

############################## PREBUILD ##############################
# Build the docker image and launch an interactive container.
# For CPU builds, first build the backend libraries and copy them into the working directory
.PHONY: prebuild
prebuild:
ifeq ($(USE_CPU), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_cpu_backends
endif
ifeq ($(USE_INFERENTIA), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) build_triton_inferentia_backends
endif
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker NO_BUILD?=1
ifneq ($(strip ${DOCKER_DETACH}), 1)
	@$(MAKE) -f $(MAKEFILE_NAME) configure_mig MIG_CONF=$(MIG_CONF)
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
	@$(MAKE) -f $(MAKEFILE_NAME) teardown_mig MIG_CONF=$(MIG_CONF)
endif

# Configure MIG
.PHONY: configure_mig
configure_mig:
	if [ $(MIG_CONF) != "OFF" ]; then MIG_CONF=$(MIG_CONF) ./scripts/mig_configure.sh; fi

# Tear down MIG
.PHONY: teardown_mig
teardown_mig:
	if [ $(MIG_CONF) != "OFF" ]; then ./scripts/mig_teardown.sh; fi

# Clone Triton.
.PHONY: clone_triton
clone_triton:
	@if [ ! -d $(TRITON_DIR) ]; then \
		echo "Cloning Triton Inference Server" \
			&& git clone $(TRITON_URL) $(TRITON_DIR); \
	fi
	@$(eval COMMIT_DISTANCE := $(shell cd $(TRITON_DIR) && git fetch && git rev-list --count origin/master...$(TRITON_HASH)))
	@if [ $(CHECK_TRITON_VERSION) == 1 ]; then \
		if [ $(COMMIT_DISTANCE) -ge 25 ] ; then \
			echo "Error: Triton hash is more than 25 commits behind main. Please update triton" && exit 1; \
		fi \
	fi
	@if [ $(CHECK_TRITON_VERSION) == 1 ]; then \
		if [ $(COMMIT_DISTANCE) -ge 15 ] ; then \
			if [ $(BYPASS_TRITON_WARNING) -lt 1 ] ; then \
				echo "Error: Triton hash is more than 15 commits behind main. Consider updating triton or run with BYPASS_TRITON_WARNING=1" && exit 1; \
			fi \
		fi \
	fi
	@cd $(TRITON_DIR) && git fetch && git checkout $(TRITON_HASH)

# Add symbolic links to scratch path if it exists.
.PHONY: link_dirs
link_dirs:
	@mkdir -p build
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(PREPROCESSED_DATA_DIR)
	@mkdir -p $(MODEL_DIR)
	@ln -sfn $(DATA_DIR) $(DATA_DIR_LINK)
	@ln -sfn $(PREPROCESSED_DATA_DIR) $(PREPROCESSED_DATA_DIR_LINK)
	@ln -sfn $(MODEL_DIR) $(MODEL_DIR_LINK)

############################## DOWNLOAD_MODEL ##############################

BENCHMARKS ?= dlrm

.PHONY: download_model
download_model: link_dirs
	$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_model.sh &&) \
		echo "Finished downloading all the models!"

############################## DOWNLOAD_DATA ##############################

.PHONY: download_data
download_data: link_dirs
	@$(foreach _benchmark,$(BENCHMARKS),bash code/$(_benchmark)/tensorrt/download_data.sh &&) \
		echo "Finished downloading all the datasets!"

############################## PREPROCESS_DATA ##############################

.PHONY: preprocess_data
preprocess_data: link_dirs
	@echo "$(BENCHMARKS),$(PYTHON3_CMD) code/$(_benchmark)/tensorrt/preprocess_data.py --data_dir=$(DATA_DIR) --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR)"
	@$(foreach _benchmark,$(BENCHMARKS),$(PYTHON3_CMD) code/$(_benchmark)/tensorrt/preprocess_data.py --data_dir=$(DATA_DIR) --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR) &&) \
		echo "Finished preprocessing all the datasets!"

############################### BUILD GPU ###############################

# Build all source codes.
.PHONY: build
build: clone_loadgen clone_power_dev link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness

# Clone LoadGen repo.
.PHONY: clone_loadgen
clone_loadgen:
	@if [ ! -d $(LOADGEN_INCLUDE_DIR) ]; then \
		echo "Cloning Official MLPerf Inference (For Loadgen Files)" \
			&& git clone $(INFERENCE_URL) $(INFERENCE_DIR); \
	fi
	@echo "Updating Loadgen" \
		&& cd $(INFERENCE_DIR) \
		&& git fetch \
		&& git checkout $(INFERENCE_HASH) \
		&& git submodule update --init tools/submission/power-dev \
		&& git submodule update --init third_party/pybind \
		&& git submodule update --init language/bert/DeepLearningExamples \
		&& git submodule update --init vision/medical_imaging/3d-unet-brats19/nnUnet

# Clone power-dev repo.
.PHONY: clone_power_dev
clone_power_dev:
	@if [ ! -d $(POWER_DEV_DIR) ]; then \
		echo "Cloning Official Power-Dev repo" \
			&& git clone $(POWER_DEV_URL) $(POWER_DEV_DIR); \
	fi
	@echo "Updating Power-Dev repo" \
		&& cd $(POWER_DEV_DIR) \
		&& git fetch \
		&& git checkout $(POWER_DEV_HASH)

# Build LoadGen.
.PHONY: build_loadgen
build_loadgen:
	@echo "Building loadgen..."
	@if [ ! -e $(LOADGEN_LIB_DIR) ]; then \
		mkdir $(LOADGEN_LIB_DIR); \
	fi
	@cd $(LOADGEN_LIB_DIR) \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) .. \
		&& make -j

# Build harness source codes.
.PHONY: build_harness
build_harness:
	@echo "Building harness..."
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DIS_SOC=$(IS_SOC) -DSOC_SM=$(SOC_SM) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building harness."

# Compile scripts
build_scripts:
	@nvcc scripts/get_gencode.cu -o scripts/get_gencode_$(ARCH)

.PHONY: build_dlrm_n3000
build_dlrm_n3000: clone_loadgen clone_power_dev link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) build_loadgen
	@$(MAKE) -f $(MAKEFILE_NAME) build_harness_n3000

NUMA_NUM=$(shell numactl --show | grep 'nodebind' | awk '{print $$NF}')
# Build N3000 harness source codes.
#.PHONY: build_harness_n3000
build_harness_n3000:
	@echo "Building N3000 harness..."
	@echo "NUMA number = ${NUMA_NUM}"
	@mkdir -p build/harness \
		&& cd build/harness \
		&& cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DLOADGEN_INCLUDE_DIR=$(LOADGEN_INCLUDE_DIR) -DLOADGEN_LIB_DIR=$(LOADGEN_LIB_DIR) -DUSE_CPU=$(USE_CPU) $(if $(strip $(NUMA_NUM),0),-DNO_NUMA=1) $(PROJECT_ROOT)/code/harness \
		&& make -j
	@echo "Finished building N3000 harness."

.PHONY: neu_card
neu_card:
	@cp configs/dlrm/Offline/init_$(NEU_CARDS)_card.py configs/dlrm/Offline/__init__.py
	@cp configs/dlrm/Server/init_$(NEU_CARDS)_card.py configs/dlrm/Server/__init__.py

###############################  RUN  ###############################
# Run the harness and check accuracy if in AccuracyOnly mode.
# Add "set -o pipefail" so that "<command> | tee output.txt" will fail when "<command>" fails because otherwise tee
# would return status 0 and clear the failure status.
.PHONY: run_harness
run_harness: link_dirs neu_card
	@mkdir -p $(LOG_DIR)
	@set -o pipefail && $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_harness" 2>&1 | tee $(LOG_DIR)/stdout.txt
	@set -o pipefail && $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS) 2>&1 | tee -a $(LOG_DIR)/stdout.txt

# Run the harness and measure power consumption using official MLPerf-Inference power workflow.
.PHONY: run_harness_power
run_harness_power: link_dirs neu_card
	@mkdir -p $(POWER_LOGS_DIR)
	@$(MAKE) -f $(MAKEFILE_NAME) power_prologue
	$(POWER_SSH) $(POWER_SERVER_USER_IP) echo "dummy connection to check if server is ready" || true
	set -o pipefail && $(PYTHON3_CMD) $(POWER_CLIENT_SCRIPT) -a $(POWER_SERVER_IP) -n "$(POWER_NTP_URL)" -S \
		-L $(POWER_LOGS_TEMP_DIR) -o $(POWER_LOGS_DIR) -f \
		-w 'LOG_DIR=$(POWER_LOGS_TEMP_DIR) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="run_harness" \
			2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt \
			&& if [ ! -d $(POWER_LOGS_DIR)/ranging_tmp ]; \
				then mkdir $(POWER_LOGS_DIR)/ranging_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/ranging_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; \
				else mkdir $(POWER_LOGS_DIR)/testing_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/testing_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; fi'
	@$(MAKE) -f $(MAKEFILE_NAME) power_epilogue


.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test05

AUDIT_HARNESS := run_audit_harness

AUDIT_VERIFICATION := run_audit_verification
ifeq ($(USE_CPU), 1)
	AUDIT_VERIFICATION := run_cpu_audit_verification
endif

.PHONY: run_audit_test01_once
run_audit_test01_once: link_dirs
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_HARNESS)"
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST01 --action="$(AUDIT_VERIFICATION)"

# TEST01 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test01
run_audit_test01: neu_card
	@for i in 1 2; do echo "TEST01 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test01_once && break; done

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04-A..." && sleep 20
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_HARNESS)"
	@echo "Sleep to reset thermal state before TEST04-B..." && sleep 20
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-B --action="$(AUDIT_HARNESS)"
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST04-A --action="$(AUDIT_VERIFICATION)"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test05_once
run_audit_test05_once: link_dirs
	@echo "Sleep to reset thermal state before TEST05..." && sleep 20
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_HARNESS)"
	@$(LOAD_PREBUILT_TRITON) $(PYTHON3_CMD) code/main.py $(RUN_ARGS) --audit_test=TEST05 --action="$(AUDIT_VERIFICATION)"

# TEST05 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test05
run_audit_test05: neu_card
	@for i in 1 2; do echo "TEST05 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test05_once && break; done

.PHONY: copy_profiles
copy_profiles:
	@echo "Copying yml and sqlite files"
	@cp *.yml /home/scratch.dlsim/data/mlperf-inference/
	@cp *.sqlite /home/scratch.dlsim/data/mlperf-inference/

.PHONY: run_harness_correlation
run_harness_correlation: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) lock_clocks
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness || ($(MAKE) -f $(MAKEFILE_NAME) free_clocks ; exit 1)
	@$(MAKE) -f $(MAKEFILE_NAME) free_clocks

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: link_dirs
	@$(PYTHON3_CMD) code/main.py $(RUN_ARGS) --action="calibrate"

# When running DLRM benchmark on multi-GPU systems, sometimes it is required to set this to reduce the likelihood of
# out-of-memory issues.
.PHONY: set_virtual_memory_overcommit
set_virtual_memory_overcommit:
	@sudo sysctl -w vm.max_map_count=16777216
	@sudo sysctl -w vm.overcommit_memory=2
	@sudo sysctl -w vm.overcommit_ratio=70

############################## POWER SUBMISSION ##############################
.PHONY: power_prologue
power_prologue:
	@mkdir -p $(POWER_LOGS_TEMP_DIR)
	@mkdir -p $(POWER_LOGS_DIR)
	@rm -rf $(POWER_LOGS_TEMP_DIR)
	@echo "+++ Check power-dev hash in Server"
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) "cat ~/power-dev/.git/HEAD"
	@echo "power-dev hash in SUT: $(POWER_DEV_HASH)"
	@echo "+++ Power Server: ps -ef | grep server.py"
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) "ps -ef | grep server.py"
	@echo "--- Power Server: ps -ef | grep server.py"
	@echo "Checking if someone else is using the Power Server machine..."
	@$(MAKE) -f $(MAKEFILE_NAME) power_check_server_not_running
	@echo "Copying Server config to the Power Server..."
	$(POWER_SCP) $(POWER_SERVER_CONFIG) $(POWER_SERVER_USER_IP):$(POWER_SERVER_SCRIPT_DIR)/server.cfg
	@echo "Running Server script on the Power Server..."
	$(POWER_SSH) $(POWER_SERVER_USER_IP) \
		nohup sudo python3 $(POWER_SERVER_SCRIPT_DIR)/server.py -c $(POWER_SERVER_SCRIPT_DIR)/server.cfg &
	@echo "Sleeping for 3 secs to make sure server is ready."
	@sleep 3
	@echo "Done sleeping. Power server should be listening to client now"

.PHONY: power_check_server_not_running
power_check_server_not_running:
	@echo "Killing python3 on Power Server if has..."
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) "sudo killall python3 || echo no python3"
	@echo "+++ Power Server: ps -ef | grep server.py"
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) "ps -ef | grep server.py"
	@echo "--- Power Server: ps -ef | grep server.py"

.PHONY: power_epilogue
power_epilogue:
	@echo "Clean up unnecessary files..."
	@mv $(POWER_LOGS_DIR)/20*/* $(POWER_LOGS_DIR)/
	@rm -rf $(POWER_LOGS_DIR)/20* $(POWER_LOGS_DIR)/*/mlperf_log*
	@mv $(POWER_LOGS_DIR)/ranging_tmp/* $(POWER_LOGS_DIR)/ranging/
	@mv $(POWER_LOGS_DIR)/testing_tmp/* $(POWER_LOGS_DIR)/run_1/
	@rm -rf $(POWER_LOGS_DIR)/ranging_tmp $(POWER_LOGS_DIR)/testing_tmp $(POWER_LOGS_TEMP_DIR)
	@set -o pipefail && LOG_DIR=$(POWER_LOGS_DIR)/run_1 $(PYTHON3_CMD) scripts/print_harness_result.py $(RUN_ARGS) 2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt
	@echo "Power logs are located in $(POWER_LOGS_DIR)."


############################## AUTOMATION AND SUBMISSION ##############################
# RUN_ID: ID for the run. For L1, this is the number of the run
# SYSTEM_NAME: Name of the current platform
# ARTIFACT_SRC_PATH: path/to/directory to compress
# ARTIFACT_NAME: name-of-artifact-to-push
# ARTIFACT_DST_PATH: path/in/artifactory/to/prepend
ARTIFACTORY_URL := https://urm.nvidia.com/artifactory/sw-mlpinf-generic

RUN_ID ?= manual-$(TIMESTAMP)
ARTIFACT_NAME ?= $(SYSTEM_NAME)_$(RUN_ID)
ARTIFACT_DST_PATH ?= artifacts

ARTIFACT_URL := $(ARTIFACTORY_URL)/$(ARTIFACT_DST_PATH)/$(ARTIFACT_NAME).gz

# Generate a raw results directory in build/full_results from LoadGen logs in build/logs
.PHONY: update_results
update_results:
	@$(PYTHON3_CMD) scripts/update_results.py --output_dir results --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: update_compliance
update_compliance:
	@$(PYTHON3_CMD) scripts/update_results.py --input_dir build/compliance_logs --output_dir compliance --assume_compliance --result_id $(ARTIFACT_NAME)
	@printf "If you would like to push results, run:\n\tmake truncate_results\n\tmake push_full_results ARTIFACT_NAME=$(ARTIFACT_NAME)\n"

.PHONY: truncate_results
truncate_results:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@rm -rf build/full_results
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/truncate_accuracy_log.py --input . --backup closed/$(SUBMITTER)/build/full_results --submitter $(SUBMITTER)
	@echo "Full accuracy logs stored in build/full_results/. Truncated results stored in results/."

.PHONY: summarize_results
summarize_results:
	@$(PYTHON3_CMD) scripts/internal/results_analysis/summarize_results.py $(RUN_ARGS)

.PHONY: check_submission
check_submission:
	@echo "WARNING: This script cannot be executed from within the docker container."
	@echo "It must have access to the project root at ../../"
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: check_submission_power
check_submission_power:
	@cd ../../ \
		&& $(PYTHON3_CMD) closed/$(SUBMITTER)/build/inference/tools/submission/submission-checker.py --more-power-check --input . --submitter $(SUBMITTER) 2>&1 \
		| tee closed/$(SUBMITTER)/results/submission_checker_log.txt

.PHONY: push_artifacts
push_artifacts:
	@mkdir -p build/artifacts
	@tar -cvzf build/artifacts/$(ARTIFACT_NAME).gz $(ARTIFACT_SRC_PATH)
	curl -u$(UNAME):$(ARTIFACTORY_API_KEY) -T build/artifacts/$(ARTIFACT_NAME).gz "$(ARTIFACT_URL)"

.PHONY: push_full_results
push_full_results:
	@$(MAKE) -f $(MAKEFILE_NAME) push_artifacts ARTIFACT_SRC_PATH=build/full_results ARTIFACT_DST_PATH=full_result_logs ARTIFACT_NAME=full-results_$(ARTIFACT_NAME)

############################## UTILITY ##############################

.PHONY: generate_conf_files
generate_conf_files:
	@$(PYTHON3_CMD) scripts/create_config_files.py

.PHONY: autotune
autotune:
	@$(PYTHON3_CMD) scripts/autotune/grid.py $(RUN_ARGS)

# Remove build directory.
.PHONY: clean
clean: clean_shallow
	rm -rf build

# Remove only the files necessary for a clean build.
.PHONY: clean_shallow
clean_shallow:
	rm -rf build/bin
	rm -rf build/harness
	rm -rf build/plugins
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(LOADGEN_LIB_DIR)
	rm -rf prebuild_triton_libs # Triton CPU libraries

.PHONY: clean_triton
clean_triton:
	rm -rf $(TRITON_DIR)

# Print out useful information.
.PHONY: info
info:
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(SYSTEM_NAME)"
	@echo "Architecture=$(ARCH)"
	@echo "SM_GENCODE=$(SM_GENCODE)"
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "PYTHON3_CMD=$(PYTHON3_CMD)"
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
	@echo "MIG_CONF=$(MIG_CONF)"

# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)

.PHONY: regression_presubmit
regression_presubmit:
	@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) --ignore=internal/e2e internal # All unit tests (non-End-to-End)
	@$(PYTHON3_CMD) -m pytest internal/e2e -rXfw -k "regression"
ifeq ($(MINIMAL_REGRESSION_PRESUBMIT), 0)
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "generate_engines"
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "run_harness"
		@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "accuracy"
endif
	@$(PYTHON3_CMD) -m pytest $(TEST_FLAGS) internal/e2e -k "functionality"
