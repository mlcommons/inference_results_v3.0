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

from abc import ABC, abstractmethod, abstractclassmethod

import json
import os
import re
import shutil
import sys

from code.common import logging, run_command
from code.common.constants import G_MLCOMMONS_INF_REPO_PATH, G_RESULTS_STAGING_PATH, AuditTest


class AuditVerifier(ABC):
    """Verifies an MLPerf Inference compliance test py calling the verification script in the MLCommons repo.
    """

    def __init__(self, test_id: AuditTest, script_path: str):
        """Creates a new AuditVerifier for a specific test.
        """
        self.test_id = test_id
        self.script_path = script_path

    @abstractmethod
    def build_args(self) -> str:
        """Builds the argument string for the verification script
        """
        raise NotImplementedError

    def run(self) -> str:
        """Runs the audit verification script and returns the command output if a 0 exit code is returned.
        """
        arg_str = self.build_args()
        cmd = f"{sys.executable} {self.script_path} {arg_str}"
        return run_command(cmd, get_output=True)


class AuditTest01Verifier(AuditVerifier):
    def __init__(self, harness):
        super().__init__(AuditTest.TEST01, os.path.join(G_MLCOMMONS_INF_REPO_PATH, "compliance", "nvidia", "TEST01",
                                                        "run_verification.py"))
        self.harness = harness

    def build_args(self) -> str:
        results_path = os.path.join(G_RESULTS_STAGING_PATH,
                                    "closed/Inspur/results",
                                    self.harness.get_system_name(),
                                    self.harness._get_submission_benchmark_name(),
                                    self.harness.scenario.valstr())
        logging.info("Checking for truncated accuracy logs...")
        with open(os.path.join(results_path, "accuracy", "mlperf_log_accuracy.json")) as f:
            data = json.load(f)  # This will throw a json.decoder.JSONDecodeError if accuracy log is malformed
        log_dir_abs_path = self.harness.get_full_log_dir()
        return f"--results={results_path} --compliance={log_dir_abs_path} --output_dir={log_dir_abs_path}"

    def run(self) -> str:
        try:
            return super().run()
        except json.decoder.JSONDecodeError:
            logging.info("Malformed or truncated accuracy log in results directory. Restore full accuracy log or re-run accuracy test to regenerate result.")
            return "TEST01 TRUNCATED_ACC_LOG"
        except:
            # Handle test 01 failure
            logging.info('TEST01 verification failed. Proceeding to fallback approach')
            return 'TEST01 FALLBACK'  # Signal main.py to finish the process


class AuditTest04Verifier(AuditVerifier):
    def __init__(self, harness):
        super().__init__(AuditTest.TEST04, os.path.join(G_MLCOMMONS_INF_REPO_PATH, "compliance", "nvidia", "TEST04",
                                                        "run_verification.py"))
        self.harness = harness

    def build_args(self) -> str:
        results_path = os.path.join(G_RESULTS_STAGING_PATH,
                                    "closed/Inspur/results",
                                    self.harness.get_system_name(),
                                    self.harness._get_submission_benchmark_name(),
                                    self.harness.scenario.valstr())
        log_dir_abs_path = self.harness.get_full_log_dir()
        return f"--results_dir={results_path} --compliance_dir={log_dir_abs_path} --output_dir={log_dir_abs_path}"


class AuditTest05Verifier(AuditVerifier):
    def __init__(self, harness):
        super().__init__(AuditTest.TEST05, os.path.join(G_MLCOMMONS_INF_REPO_PATH, "compliance", "nvidia", "TEST05",
                                                        "run_verification.py"))
        self.harness = harness

    def build_args(self) -> str:
        results_path = os.path.join(G_RESULTS_STAGING_PATH,
                                    "closed/Inspur/results",
                                    self.harness.get_system_name(),
                                    self.harness._get_submission_benchmark_name(),
                                    self.harness.scenario.valstr())
        log_dir_abs_path = self.harness.get_full_log_dir()
        return f"--results_dir={results_path} --compliance_dir={log_dir_abs_path} --output_dir={log_dir_abs_path}"


def load(audit_test, benchmark):
    # Calculates path to audit.config
    src_config = os.path.join('build/inference/compliance/nvidia', audit_test, benchmark.valstr(), 'audit.config')
    logging.info('AUDIT HARNESS: Looking for audit.config in {}...'.format(src_config))
    if not os.path.isfile(src_config):
        # For tests that have one central audit.config instead of per-benchmark
        src_config = os.path.join('build/inference/compliance/nvidia', audit_test, 'audit.config')
        logging.info('AUDIT HARNESS: Search failed. Looking for audit.config in {}...'.format(src_config))
    # Destination is audit.config
    dest_config = 'audit.config'
    # Copy the file
    shutil.copyfile(src_config, dest_config)
    return dest_config


def cleanup():
    """Delete files for audit cleanup."""
    tmp_files = ["audit.config", "verify_accuracy.txt", "verify_performance.txt", "mlperf_log_accuracy_baseline.json", "accuracy.txt", "predictions.json"]
    for fname in tmp_files:
        if os.path.exists(fname):
            logging.info('Audit cleanup: Removing file {}'.format(fname))
            os.remove(fname)
