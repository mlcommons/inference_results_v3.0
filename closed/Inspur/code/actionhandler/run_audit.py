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

import glob
import importlib
import json
import os
import shutil
import sys
import traceback

import code.common.arguments as common_args
import code.common.auditing as auditing

from code import get_harness, G_HARNESS_CLASS_MAP
from code.actionhandler.base import ActionHandler
from code.actionhandler.run_harness import RunHarnessHandler
from code.common import logging, run_command
from code.common.accuracy_checker import check_accuracy
from code.common.constants import *
from code.common.fix_sys_path import fix_pythonpath_command
from code.common.protected_super import ProtectedSuper


def _move_file(src, dst):
    logging.info(f"=> Compliance harness: Moving file: {src} --> {dst}")
    shutil.move(src, dst)


def _copy_file(src, dst):
    logging.info(f"=> Compliance harness: Copying file: {src} --> {dst}")
    shutil.copy(src, dst)


def skip_if_exempt(default_retval=None):
    """Decorator to skip a RunAuditHandler function if the audit test is exempt
    """
    def _wrapper(f):
        def _f(self, *args, **kwargs):
            if self.is_exempt:
                return default_retval
            else:
                return f(self, *args, **kwargs)
        return _f
    return _wrapper


class RunAuditHandler(RunHarnessHandler):
    """Handles running audit tests. This changes the setup and teardown for RunHarnessHandler and calls it under the
    hood."""

    def __init__(self, test_name, benchmark_conf, power_controller, use_gpu=True, use_dla=True, profiler=None, verify=True):
        """Creates a new ActionHandler for RunAudit

        Args:
            test_name (str): The audit test name
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
            power_controller (PowerController): The PowerController to control to power settings of the system
            use_gpu (bool): Whether or not GPUs are used for this configuration.
            use_dla (bool): Whether or not DLAs are used for this configuration
            profiler (str): INTERNAL ONLY. Name of the profiler to use. (Default: None)
            verify (bool): If True, performs compliance verification after compliance harness is run. (Default: True)
        """
        self.test_name = test_name
        benchmark_conf["audit_test_name"] = test_name

        # Make sure the log_file override is valid
        benchmark_conf["log_dir"] = os.path.join("build/compliance_logs", self.test_name)
        os.makedirs(benchmark_conf["log_dir"], exist_ok=True)

        super().__init__(benchmark_conf, power_controller, use_gpu=use_gpu, use_dla=use_dla, profiler=profiler,
                         measure_power=False, skip_postprocess=True)  # Do not measure power during audit runs.
        self.action = Action.RunAuditHarness

        self.verify = verify
        if not self.verify:
            logging.warning("!! Compliance harness verification disabled!")

        if self.is_exempt:
            # Explicitly print that this audit test is exempted and the user can move on. In the past, some of our
            # submission partners were confused that exempted tests completed without logging, or that the logging was
            # unclear that the lack of running something was OK.
            benchmark_name = self.benchmark_conf["benchmark"].valstr()
            scenario_name = self.benchmark_conf["scenario"].valstr()
            logging.info(f"=> Compliance test {self.test_name} is exempted for {benchmark_name}-{scenario_name}. You can safely ignore this message.")

    @property
    def is_exempt(self):
        if self.test_name == AuditTest.TEST04:
            exclude_list = [Benchmark.BERT, Benchmark.DLRM, Benchmark.RNNT, Benchmark.UNET3D, Benchmark.Retinanet]
            if self.benchmark_conf["benchmark"] in exclude_list:
                return True
        return False

    @skip_if_exempt()
    def setup(self):
        """Called once before handle().
        """
        # Set the audit config before generating .conf files
        logging.info(f"=> Running compliance harness for test {self.test_name}")
        audit_config = auditing.load(self.test_name, self.benchmark_conf["benchmark"])

    @skip_if_exempt(default_retval=True)
    def handle(self) -> bool:
        """Run the action.

        Returns:
            bool: True if handle() succeeded, False otherwise.
        """
        # Run audit harness
        with ProtectedSuper(self) as duper:
            success = duper.run()  # Let base.ActionHandler do the heavy lifting

        if not self.verify:
            self.create_metadata_json({"audit_test": self.test_name,
                                       "audit_success": "unverified"}, append=True)
            return success

        # Run audit verification
        logging.info(f"=> Running compliance harness verification script...")
        verification = self._handle_verify()
        self.create_metadata_json({"audit_test": self.test_name,
                                   "audit_success": verification}, append=True)
        return verification

    def _handle_verify(self):
        if self.test_name == AuditTest.TEST01:
            return self._handle_verify_test01()
        elif self.test_name == AuditTest.TEST04:
            return self._handle_verify_test04()
        elif self.test_name == AuditTest.TEST05:
            return self._handle_verify_test05()
        else:
            raise ValueError(r"Invalid compliance test name: {self.test_name}")

    def _handle_verify_test01(self):
        result = auditing.AuditTest01Verifier(self.harness).run()
        if "Accuracy check pass: True" in result and "Performance check pass: True" in result:
            logging.info(f"=> Compliance test TEST01 passed without fallback")
            return True
        elif result == "TEST01 FALLBACK":
            # Signals a fallback for failed test
            logging.info(f"=> Running fallback for TEST01")

            # 1. Generate baseline_accuracy file
            SUBMITTER = os.environ.get("SUBMITTER", "Inspur")
            full_log_dir = self.harness.get_full_log_dir()
            results_path = os.path.join(G_RESULTS_STAGING_PATH,
                                        f"closed/{SUBMITTER}/results",
                                        self.harness.get_system_name(),
                                        self.harness._get_submission_benchmark_name(),
                                        self.harness.scenario.valstr())
            harness_accuracy_log = os.path.join(results_path, "accuracy", "mlperf_log_accuracy.json")
            compliance_accuracy_log = os.path.join(full_log_dir, "mlperf_log_accuracy.json")
            baseline_script_path = os.path.join(G_MLCOMMONS_INF_REPO_PATH,
                                                "compliance",
                                                "nvidia",
                                                "TEST01",
                                                "create_accuracy_baseline.sh")
            fallback_command = f"bash {baseline_script_path} {harness_accuracy_log} {compliance_accuracy_log}"
            run_command(fallback_command)

            # 2. Create accuracy and performance directories
            accuracy_dir = os.path.join(full_log_dir, "TEST01", "accuracy")
            os.makedirs(accuracy_dir, exist_ok=True)

            performance_dir = os.path.join(full_log_dir, "TEST01", "performance", "run_1")
            os.makedirs(performance_dir, exist_ok=True)

            # 3. Calculate the accuracy of baseline, using the benchmark's accuracy script
            fallback_result_baseline = check_accuracy("mlperf_log_accuracy_baseline.json",
                                                      self.benchmark_conf)["accuracy"]
            _move_file('accuracy.txt', os.path.join(accuracy_dir, 'baseline_accuracy.txt'))

            # 4. Calculate the accuracy of the compliance run, using the benchmark's accuracy script
            fallback_result_compliance = check_accuracy(
                os.path.join(full_log_dir, "mlperf_log_accuracy.json"),
                self.benchmark_conf)["accuracy"]

            # 5. Copy all the compliance files into log directory
            # Move it to the submission dir - check_accuracy stores accuracy.txt in the directory
            # name provided in its first argument. So this file will already be located inside get_full_log_dir()
            _move_file(os.path.join(full_log_dir, 'accuracy.txt'),
                       os.path.join(accuracy_dir, 'compliance_accuracy.txt'))
            # Required for run_verification.py
            # Move the required logs to their correct locations since run_verification.py has failed.
            _move_file('verify_accuracy.txt', os.path.join(full_log_dir, 'TEST01', 'verify_accuracy.txt'))
            _copy_file(os.path.join(full_log_dir, 'mlperf_log_accuracy.json'), os.path.join(accuracy_dir, 'mlperf_log_accuracy.json'))
            _copy_file(os.path.join(full_log_dir, 'mlperf_log_detail.txt'), os.path.join(performance_dir, 'mlperf_log_detail.txt'))
            _copy_file(os.path.join(full_log_dir, 'mlperf_log_summary.txt'), os.path.join(performance_dir, 'mlperf_log_summary.txt'))

            # 5. Check whether or not the accuracies are within a defined tolerance
            verify_performance_script = os.path.join(G_MLCOMMONS_INF_REPO_PATH,
                                                     "compliance",
                                                     "nvidia",
                                                     "TEST01",
                                                     "verify_performance.py")
            verify_performance_args = f"-r {results_path}/performance/run_1/mlperf_log_summary.txt -t {performance_dir}/mlperf_log_summary.txt"
            verify_performance_command = f"python3 {verify_performance_script} {verify_performance_args} | tee {full_log_dir}/TEST01/verify_performance.txt"
            verify_performance_command = fix_pythonpath_command(verify_performance_command)
            run_command(verify_performance_command)

            accuracy_level = self.benchmark_conf["workload_setting"].accuracy_target
            if accuracy_level == AccuracyTarget.k_99_9:
                logging.info('=> Compliance harness: High Accuracy benchmark detected. Tolerance set to 0.1%')
                rel_tol = 0.001
            elif accuracy_level == AccuracyTarget.k_99:
                logging.info('=> Compliance harness: Low Accuracy benchmark detected. Tolerance set to 1%')
                rel_tol = 0.01
            else:
                raise ValueError(f'Accuracy level not supported: {accuracy_level}')

            if not math.isclose(fallback_result_baseline, fallback_result_compliance, rel_tol=rel_tol):
                logging.warning(f'=> Compliance test TEST01 + Fallback failure: BASELINE ACCURACY: {fallback_result_baseline}, COMPLIANCE_ACCURACY: {fallback_result_compliance}')
                return False
            else:
                logging.info('AUDIT HARNESS: Success: TEST01 failure redeemed via fallback approach.')
                print('TEST PASS')  # TODO: Ask @garvitk if there is a better way of handling this. This stdout print actually notifies CI/CD of a successful run.
                return True
        else:
            logging.warning(f"=> Unexpected result from TEST01 verification script:\n\n{result}")
            return False

    def _handle_verify_test04(self):
        return "Performance check pass: True" in auditing.AuditTest04Verifier(self.harness).run()

    def _handle_verify_test05(self):
        return "Performance check pass: True" in auditing.AuditTest05Verifier(self.harness).run()

    @skip_if_exempt()
    def handle_failure(self):
        """Called after handle() if it errors.
        """
        raise RuntimeError("Run audit harness failed!")

    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.

        success (bool): Indicates whether or not self.handle() executed successfully.  This is useful when cleanup
                        behaves differently when handle fails, or the cleanup code depends on something that is only
                        done on successful runs.
        """
        logging.info("=> Audit harness: Cleaning up audit.config...")
        auditing.cleanup()

        if self.is_exempt:
            message = "PASS (by default, due to exemption)"
        elif success:
            message = "PASS"
        else:
            message = "FAIL"
        logging.info(f"=> Submission checker: Audit test {self.test_name} {message}")
