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

# This file is the main entrypoint of NVIDIA's MLPerf Inference codebase, and includes all of the other Makefiles used
# in the project.
# This file contains the targets used to run the actual MLPerf Inference workloads, as well as some basic utility
# commands.

MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))  # Must be declared before includes
include $(CURDIR)/Makefile.const
include $(CURDIR)/Makefile.docker
include $(CURDIR)/Makefile.build
include $(CURDIR)/Makefile.data
include $(CURDIR)/Makefile.tests
include $(CURDIR)/Makefile.submission

# Generate TensorRT engines (plan files) and run the harness.
.PHONY: run
run:
	@$(MAKE) -f $(MAKEFILE_NAME) generate_engines
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness

# Generate TensorRT engines (plan files).
.PHONY: generate_engines
generate_engines: link_dirs
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="generate_engines"

# Run the harness and check accuracy if in AccuracyOnly mode.
# Add "set -o pipefail" so that "<command> | tee output.txt" will fail when "<command>" fails because otherwise tee
# would return status 0 and clear the failure status.
.PHONY: run_harness
run_harness: link_dirs
	@mkdir -p $(LOG_DIR)
	@set -o pipefail && LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness" 2>&1 | tee $(LOG_DIR)/stdout.txt
	@set -o pipefail && $(PYTHON3_CMD) -m scripts.print_harness_result $(RUN_ARGS) 2>&1 | tee -a $(LOG_DIR)/stdout.txt

# Run the harness and measure power consumption using official MLPerf-Inference power workflow.
# Use environment var USE_WIN_PTD=1 to control using windows power server or not
.PHONY: run_harness_power
run_harness_power: link_dirs
ifeq ($(IS_SOC), 0)
	@echo "WARNING: If you have not already, before you run MaxQ mode, `make power_set_maxq_state` must be run *OUTSIDE* the docker container."
endif
	@mkdir -p $(POWER_LOGS_DIR)
	@$(MAKE) -f Makefile.power power_prologue
	@$(POWER_SSH) $(POWER_SERVER_USER_IP) echo "dummy connection to check if server is ready" || true
	set -o pipefail && $(PYTHON3_CMD) $(POWER_CLIENT_SCRIPT) -a $(POWER_SERVER_IP) -n "$(POWER_NTP_URL)" -S \
		-L $(POWER_LOGS_TEMP_DIR) -o $(POWER_LOGS_DIR) -f \
		-w 'LOG_DIR=$(POWER_LOGS_TEMP_DIR) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="run_harness" \
			2>&1 | tee -a $(POWER_LOGS_DIR)/stdout.txt \
			&& if [ ! -d $(POWER_LOGS_DIR)/ranging_tmp ]; \
				then mkdir $(POWER_LOGS_DIR)/ranging_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/ranging_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/ranging_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; \
				else mkdir $(POWER_LOGS_DIR)/testing_tmp \
					&& mv $(POWER_LOGS_TEMP_DIR)/* $(POWER_LOGS_DIR)/testing_tmp/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_detail.txt $(POWER_LOGS_TEMP_DIR)/ \
					&& cp -v $(POWER_LOGS_DIR)/testing_tmp/*/*/*/mlperf_log_summary.txt $(POWER_LOGS_TEMP_DIR)/; fi' \
	|| $(MAKE) -f Makefile.power power_kill_server
	@$(MAKE) -f Makefile.power power_epilogue

.PHONY: run_audit_harness
run_audit_harness: link_dirs
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test01
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test04
	@$(MAKE) -f $(MAKEFILE_NAME) run_audit_test05

.PHONY: run_audit_test01_once
run_audit_test01_once: link_dirs
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST01 --server_target_qps_adj_factor=0.92 --action="run_audit_harness"

# TEST01 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test01
run_audit_test01:
	@for i in 1 2; do echo "TEST01 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test01_once && break; done

.PHONY: run_audit_test04_once
run_audit_test04_once: link_dirs
	@echo "Sleep to reset thermal state before TEST04..." && sleep 20
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST04 --server_target_qps_adj_factor=0.92 --action="run_audit_harness"

# TEST04 is so short that it is sometimes unstable. Try up to three times before failing.
.PHONY: run_audit_test04
run_audit_test04:
	@for i in 1 2 3; do echo "TEST04 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test04_once && break; done

.PHONY: run_audit_test05_once
run_audit_test05_once: link_dirs
	@echo "Sleep to reset thermal state before TEST05..." && sleep 20
	@LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH) $(PYTHON3_CMD) -m code.main $(RUN_ARGS) --audit_test=TEST05 --server_target_qps_adj_factor=0.96 --action="run_audit_harness"

# TEST05 is sometimes unstable. Try up to two times before failing.
.PHONY: run_audit_test05
run_audit_test05:
	@for i in 1 2; do echo "TEST05 trial $$i" && $(MAKE) -f $(MAKEFILE_NAME) run_audit_test05_once && break; done

.PHONY: run_harness_correlation
run_harness_correlation: link_dirs
	@$(MAKE) -f Makefile.power lock_clocks
	@$(MAKE) -f $(MAKEFILE_NAME) run_harness || ($(MAKE) -f Makefile.power free_clocks ; exit 1)
	@$(MAKE) -f Makefile.power free_clocks

# Re-generate TensorRT calibration cache.
.PHONY: calibrate
calibrate: link_dirs
	@$(PYTHON3_CMD) -m code.main $(RUN_ARGS) --action="calibrate"


############################## UTILITY ##############################

# When running DLRM benchmark on multi-GPU systems, sometimes it is required to set this to reduce the likelihood of
# out-of-memory issues.
.PHONY: set_virtual_memory_overcommit
set_virtual_memory_overcommit:
	@sudo sysctl -w vm.max_map_count=16777216
	@sudo sysctl -w vm.overcommit_memory=2
	@sudo sysctl -w vm.overcommit_ratio=70


.PHONY: copy_profiles
copy_profiles:
	@echo "Copying yml and sqlite files"
	@cp *.yml /home/scratch.dlsim/data/mlperf-inference/
	@cp *.sqlite /home/scratch.dlsim/data/mlperf-inference/


.PHONY: autotune
autotune:
	@$(PYTHON3_CMD) -m scripts.autotune.grid $(RUN_ARGS)


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
	rm -rf $(TRITON_PREBUILT_LIBS_DIR) # Triton CPU libraries
	rm -rf openvino*


.PHONY: clean_triton
clean_triton:
	rm -rf $(TRITON_DIR)
	rm -rf $(TRITON_OUT_DIR)
	rm -rf $(TRITON_PREBUILT_LIBS_DIR) # Triton CPU libraries


# Print out useful information.
.PHONY: info
info:
	@echo "==== System:"
	@echo "RUN_ID=$(RUN_ID)"
	@echo "SYSTEM_NAME=$(SYSTEM_NAME)"
	@echo "Architecture=$(ARCH)"
	@echo "MIG_CONF=$(MIG_CONF)"
	@echo "==== Docker: "
	@echo "User=$(UNAME)"
	@echo "UID=$(UID)"
	@echo "HOSTNAME=$(HOSTNAME)"
	@echo "Usergroup=$(GROUPNAME)"
	@echo "GroupID=$(GROUPID)"
	@echo "Docker info: {DETACH=$(DOCKER_DETACH), TAG=$(DOCKER_TAG)}"
	@echo "Docker file name: $(DOCKER_FILENAME)"
ifdef DOCKER_IMAGE_NAME
	@echo "Docker image used: $(DOCKER_IMAGE_NAME) -> [$(BASE_IMAGE)]"
endif
	@echo "==== Env Vars:"
	@echo "PYTHON3_CMD=$(PYTHON3_CMD)"
	@echo "PATH=$(PATH)"
	@echo "CPATH=$(CPATH)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "LIBRARY_PATH=$(LIBRARY_PATH)"
	@echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
	@echo "HARNESS_LD_LIBRARY_PATH=$(HARNESS_LD_LIBRARY_PATH)"
	@echo "==== Build flags:"
	@echo "HARNESS_BUILD_FLAGS=$(HARNESS_BUILD_FLAGS)"
	@echo "TRITON_BUILD_FLAGS=$(TRITON_BUILD_FLAGS)"
	@echo "BUILD_TRITON=$(BUILD_TRITON)"


# The shell target will start a shell that inherits all the environment
# variables set by this Makefile for convenience.
.PHONY: shell
shell:
	@$(SHELL)
