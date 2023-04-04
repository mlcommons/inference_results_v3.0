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

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Final, Dict, Tuple, Union

from code.common import run_command, logging
from code.common.constants import *
from code.common.fix_sys_path import ScopedRestrictedImport
from code.common.systems.system_list import DETECTED_SYSTEM, SystemClassifications

G_MLCOMMONS_INF_REPO_PATH: Final[str] = os.path.join(os.getcwd(), "build", "inference")
"""
Final[str]: Path to the MLCommons inference repo.
"""

# Since submission-checker uses a relative import, but we are running from main.py, we need to surface its directory
# into sys.path so it can successfully import it. Use a ScopedRestrictedImport for this.
_new_path = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, "tools", "submission")] + sys.path
with ScopedRestrictedImport(_new_path):
    submission_checker = import_module("submission_checker")
    G_ACC_PATTERNS = submission_checker.ACC_PATTERN
    # MLCommons doesn't add the current version until it is close to submission
    _submission_model_config = submission_checker.MODEL_CONFIG
    _version_str = VERSION if VERSION in _submission_model_config else "v2.0"
    G_ACC_TARGETS = _submission_model_config[_version_str]["accuracy-target"]
    """Dict[str, Tuple[str, float]]: A dictionary mapping the benchmark name to a tuple of (accuracy_metric, threshold)"""


G_MLCOMMONS_ACC_TARGETS: Final[Dict[AccuracyTarget, str]] = {AccuracyTarget.k_99: "99",
                                                             AccuracyTarget.k_99_9: "99.9"}
"""Dict[AccuracyTarget, str]: A dictionary mapping an AccuracyTarget to the string used in MLCommons keys"""


def get_pythonpath(pythonpath_extra: Optional[str] = None) -> str:
    """Gets the PYTHONPATH value used by ScopedRestrictedImport. In other words, gets the PYTHONPATH with all user
    package directories removed.

    Args:
        pythonpath_extra (str): If set, will be prepended to the PYTHONPATH. (Default: None)

    Returns:
        str: The PYTHONPATH value
    """
    with ScopedRestrictedImport() as sri:
        pythonpath = sri.path_as_string()
    if pythonpath_extra is not None:
        # TODO: Currently this method is only called once, but this is enforced by the programmer and is prone to bugs.
        # This should filter out the final PYTHONPATH for duplicate paths while preserving order.
        pythonpath = ":".join((pythonpath_extra, pythonpath))
    return pythonpath


@dataclass
class _AccuracyScriptCommand:
    """Contains metadata for the command to invoke an MLCommons Inference accuracy script"""

    executable: str
    """str: The executable name to run. Python accuracy scripts should NOT be invoked directly (i.e. ./path/to/script.py
            via the shebang). For Python-based accuracy scripts, this value should always be "python", "python3", or
            "python3.8".
    """

    argv: List[str]
    """List[str]: List of arguments to pass to the executable. For Python scripts, this should be sys.argv."""

    env: Dict[str, str]
    """Dict[str]: Dictionary of custom environment variables to pass to the executable."""

    def __str__(self) -> str:
        argv_str = " ".join(self.argv)
        s = f"{self.executable} {argv_str}"
        if len(self.env) > 0:
            env_str = " ".join(f"{k}={v}" for k, v in self.env.items())
            s = env_str + " " + s
        return s

    def fix_pythonpath(self):
        """
        For commands in the form "python(3|3.*) *", sets PYTHONPATH to use an import path that does not include ~/.local.
        For non-Python commands, this method is a no-op.

        By default, Python will automatically import [site.py](https://docs.python.org/3/library/site.html) when the
        Python session is initialized. This is what inserts ~/.local/lib/python*/site-packages into sys.path
        automatically.  However, this can by bypassed by using the -S flag. In doing so, we will need to add the system
        site-packages directories manually by generating a string and using PYTHONPATH.

        The reason this is not done within the Makefile is because it needs to be done at every invocation of Python
        within the Makefile, and also does not fix the problem with a subprocess is called (since the subprocess ALSO
        needs to pass in -S), and does not fix the issue when users run scripts by calling Python natively, without
        using Make.
        """
        if not self.executable.startswith("python"):
            return

        # Add the -S flag to disable site.py auto-import
        if " -S" not in self.executable:
            self.executable = f"{self.executable} -S"

        # Set PYTHONPATH in the environment vars
        self.env["PYTHONPATH"] = get_pythonpath(pythonpath_extra=self.env.get("PYTHONPATH", None))


class AccuracyChecker(ABC):
    """Utility class to run a particular accuracy script from the MLCommons Inference repo.
    """

    def __init__(self,
                 log_file: str,
                 benchmark_conf: Dict[str, Any],
                 full_benchmark_name: str,
                 mlcommons_module_path: str):
        """Creates an AccuracyChecker

        Args:
            log_file (str): Path to the accuracy log
            benchmark_conf (Dict[str, Any]): The benchmark configuration used to generate the accuracy result
            full_benchmark_name (str): The full submission name of the benchmark
            mlcommons_module_path (str): The relative filepath of the accuracy script in the MLCommons Inference repo
        """
        self.log_file = log_file
        self.benchmark_conf = benchmark_conf
        self.benchmark = self.benchmark_conf["benchmark"]
        self.full_benchmark_name = full_benchmark_name
        self.mlcommons_module_path = mlcommons_module_path

        self.acc_metric, self.threshold = G_ACC_TARGETS[self.full_benchmark_name]
        self.acc_pattern = G_ACC_PATTERNS[self.acc_metric]

    @abstractmethod
    def get_cmd(self) -> _AccuracyScriptCommand:
        """Constructs the command to run the accuracy script

        Returns:
            _AccuracyScriptCommand: The command to run
        """
        raise NotImplemented

    def run(self) -> List[str]:
        """Runs the accuracy checker script and returns the output if the script ran successfully.
        """
        cmd = self.get_cmd()
        cmd.fix_pythonpath()

        return run_command(str(cmd), get_output=True)

    def summarize(self, get_raw_result: bool = False, error_on_fail: bool = True) -> Union[str, float]:
        """Runs the accuracy script and summarizes the accuracy results.

        Args:
            get_raw_result (bool): If True, returns the accuracy metric as a float instead. (Default: False)
            error_on_fail (bool): If True, will raise a RuntimeError if the accuracy result is below the threshold.
                                  (Default: True)

        Returns:
            If get_raw_result is True, returns a float representing the accuracy metric. Otherwise, returns the accuracy
            string formatted for the MLPerf Inference summary.

        Raises:
            RuntimeError: If error_on_fail is true and the accuracy result is below the threshold
        """
        output = self.run()
        result_regex = re.compile(self.acc_pattern)

        # Copy the output to accuracy.txt
        accuracy = None
        with open(os.path.join(os.path.dirname(self.log_file), "accuracy.txt"), "w") as f:
            for line in output:
                print(line, file=f)

        # Extract the accuracy metric from the output
        for line in output:
            result_match = result_regex.match(line)
            if not result_match is None:
                accuracy = float(result_match.group(1))
                break

        accuracy_result = "PASSED" if accuracy is not None and accuracy >= self.threshold else "FAILED"
        acc_string = "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}.".format(accuracy,
                                                                                        self.threshold,
                                                                                        accuracy_result)
        if accuracy_result == "FAILED" and error_on_fail:
            raise RuntimeError(f"{acc_string}!")

        if get_raw_result:
            return accuracy
        else:
            return acc_string


class ResNet50AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "resnet",
                         "vision/classification_and_detection/tools/accuracy-imagenet.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                "--imagenet-val-file data_maps/imagenet/val_map.txt",
                "--dtype int32"]
        return _AccuracyScriptCommand("python3", argv, dict())


class RetinanetAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "retinanet",
                         "vision/classification_and_detection/tools/accuracy-openimages.py")
        self.openimages_dir = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "open-images-v6-mlperf")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--openimages-dir {self.openimages_dir}",
                "--output-file build/retinanet-results.json"]
        return _AccuracyScriptCommand("python3", argv, dict())


class SSDResNet34AccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "ssd-large",
                         "vision/classification_and_detection/tools/accuracy-coco.py")
        self.coco_dir = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--coco-dir {self.coco_dir}",
                "--output-file build/ssd-resnet34-results.json",
                "--use-inv-map"]
        return _AccuracyScriptCommand("python3", argv, dict())


class SSDMobileNetAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         "ssd-small",
                         "vision/classification_and_detection/tools/accuracy-coco.py")
        self.coco_dir = os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                f"--coco-dir {self.coco_dir}",
                "--output-file build/ssd-mobilenet-results.json"]
        return _AccuracyScriptCommand("python3", argv, dict())


class BERTAccuracyChecker(AccuracyChecker):
    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"}  # Use FP16 output for INT8 mode
    """Dict[str, str]: Remap MLPINF precision strings to a string that the BERT accuracy script understands"""

    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"bert-{_acc_target}",
                         "language/bert/accuracy-squad.py")
        self.squad_path = os.path.join(os.environ.get("DATA_DIR", "build/data"), "squad", "dev-v1.1.json")
        self.vocab_file_path = "build/data/squad/vocab.txt" if 'CPU' in self.benchmark_conf['config_name'] else "build/models/bert/vocab.txt"
        self.output_prediction_path = os.path.join(os.path.dirname(self.log_file), "predictions.json")

        _dtype = self.benchmark_conf["precision"].lower()
        self.dtype = BERTAccuracyChecker.dtype_expand_map.get(_dtype, _dtype)

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing tokenizers on SoC systems. Use custom BERT accuracy script.
        if SystemClassifications.is_soc():
            argv = ["code/bert/tensorrt/accuracy-bert.py",
                    f"--mlperf-accuracy-file {self.log_file}",
                    f"--squad-val-file {self.squad_path}"]
            env = dict()
        else:
            argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                    f"--log_file {self.log_file}",
                    f"--vocab_file {self.vocab_file_path}",
                    f"--val_data {self.squad_path}",
                    f"--out_file {self.output_prediction_path}",
                    f"--output_dtype {self.dtype}"]
            env = {"PYTHONPATH": "code/bert/tensorrt/helpers"}
        return _AccuracyScriptCommand("python3", argv, env)


class DLRMAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"dlrm-{_acc_target}",
                         "recommendation/dlrm/pytorch/tools/accuracy-dlrm.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                f"--mlperf-accuracy-file {self.log_file}",
                "--day-23-file build/data/criteo/day_23",
                "--aggregation-trace-file build/preprocessed_data/criteo/full_recalib/sample_partition_trace.txt"]
        return _AccuracyScriptCommand("python3", argv, dict())


class RNNTAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        super().__init__(log_file,
                         benchmark_conf,
                         f"rnnt",
                         "speech_recognition/rnnt/accuracy_eval.py")

    def get_cmd(self) -> _AccuracyScriptCommand:
        # Having issue installing librosa on SoC systems
        if SystemClassifications.is_soc():
            argv = ["code/rnnt/tensorrt/accuracy.py",
                    f"--loadgen_log {self.log_file}"]
        else:
            # RNNT output indices are in INT8
            argv = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, self.mlcommons_module_path),
                    f"--log_dir {os.path.dirname(self.log_file)}",
                    "--dataset_dir build/preprocessed_data/LibriSpeech/dev-clean-wav",
                    "--manifest build/preprocessed_data/LibriSpeech/dev-clean-wav.json",
                    "--output_dtype int8"]
        return _AccuracyScriptCommand("python3", argv, dict())


class UNET3DAccuracyChecker(AccuracyChecker):
    def __init__(self, log_file: str, benchmark_conf: Dict[str, Any]):
        _acc_target = G_MLCOMMONS_ACC_TARGETS[benchmark_conf["workload_setting"].accuracy_target]
        super().__init__(log_file,
                         benchmark_conf,
                         f"3d-unet-{_acc_target}",
                         None)

        postprocess_dir = "build/postprocessed_data"
        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)

    def get_cmd(self) -> _AccuracyScriptCommand:
        argv = ["code/3d-unet/tensorrt/accuracy_kits.py",
                f"--log_file {self.log_file}"]
        env = dict()
        # WAR for numpy linargerror_eigenvalues_nonconvergence happen in ARM64 (x86 don't see this)
        if DETECTED_SYSTEM.host_cpu_conf.get_architecture() == CPUArchitecture.aarch64:
            env["OPENBLAS_CORETYPE"] = "armv8"
        return _AccuracyScriptCommand("python3", argv, env)

    def run(self) -> List[str]:
        return super().run()


G_ACCURACY_CHECKER_MAP = {Benchmark.BERT: BERTAccuracyChecker,
                          Benchmark.DLRM: DLRMAccuracyChecker,
                          Benchmark.RNNT: RNNTAccuracyChecker,
                          Benchmark.ResNet50: ResNet50AccuracyChecker,
                          Benchmark.Retinanet: RetinanetAccuracyChecker,
                          Benchmark.SSDMobileNet: SSDMobileNetAccuracyChecker,
                          Benchmark.SSDResNet34: SSDResNet34AccuracyChecker,
                          Benchmark.UNET3D: UNET3DAccuracyChecker}
"""Dict[Benchmark, AccuracyChecker]: Maps a Benchmark to its AccuracyChecker"""


def check_accuracy(log_file, config, is_compliance=False):
    """Check accuracy of given benchmark."""
    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."

    # Check if log_file is empty by just reading first several bytes
    # The first 4B~6B is likely all we need to check: '', '[]', '[]\r', '[\n]\n', '[\r\n]\r\n', ...
    # but checking 8B for safety
    with open(log_file, 'r') as lf:
        first_8B = lf.read(8)
        if not first_8B or ('[' in first_8B and ']' in first_8B):
            return "No accuracy results in PerformanceOnly mode."

    benchmark = config["benchmark"]
    accuracy_checker = (G_ACCURACY_CHECKER_MAP.get(benchmark, None))(log_file, config)  # Create an instance
    if accuracy_checker is None:
        raise ValueError(f"Invalid benchmark {benchmark} does not have an AccuracyChecker.")

    return accuracy_checker.summarize(get_raw_result=is_compliance,
                                      error_on_fail=not is_compliance)
