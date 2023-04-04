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
from dataclasses import dataclass, asdict, field
from enum import Enum, unique
from typing import Any, Dict, Final, Optional, Union, Tuple

import os
import re
import math
import importlib
import importlib.util

# Conditional import. Sometimes, we may use the constants files outside of an environment that has TRT installed, for
# example certain scripts in CI/CD. Provide an environment variable 'OUTSIDE_MLPINF_ENV' to allow using constants.py
# outside of the docker.
if importlib.util.find_spec("tensorrt") is not None or os.environ.get("OUTSIDE_MLPINF_ENV", "0") == "0":
    import tensorrt as trt

    TRT_LOGGER: Final[trt.Logger] = trt.Logger(trt.Logger.INFO)
    """trt.Logger: TRT Logger Singleton. TRT Runtime requires its logger to be a singleton object, so if we create
    multiple runtimes with different trt.Logger objects, a segmentation fault will occur with 'The logger passed into
    createInferRuntime differs from one already assigned'"""


__doc__ = """Stores constants and Enums related to MLPerf Inference"""


VERSION: Final[str] = "v3.0"
"""str: Current version of MLPerf Inference"""

QUANTITY_UNIT_FORMAT = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)) ?(\S+)")
"""re.Pattern: Regex to parse strings that indicate numeric quantities with a unit. Derived from
               https://docs.python.org/3/library/re.html#simulating-scanf

               Disallows scientific notations (ie. 1e5).

               Will match:
                    - "12.5 KB"
                    - "15TB"
                    - "14 lbs"
                    - "0.2cm"
                    - "-19.7KB"     This will still match even though "negative memory" isn't valid
               Will not match:
                    - "1e6 GB"      Reason: Uses scientific notation
                    - "4  KB"       Reason: Too many spaces between quantity and unit
                    - "22"          Reason: No unit string

               match(1) - unitless, numeric quantity of memory
               match(2) - the byte suffix, denoting the unit
"""


@unique
class ByteSuffix(Enum):
    B = (1, 0)

    KiB = (1024, 1)
    MiB = (1024, 2)
    GiB = (1024, 3)
    TiB = (1024, 4)

    KB = (1000, 1)
    MB = (1000, 2)
    GB = (1000, 3)
    TB = (1000, 4)

    def to_bytes(self):
        base, exponent = self.value
        return base ** exponent


@dataclass(eq=True, frozen=True)
class Memory:
    """Represents an amount of computer memory."""

    quantity: float
    """float: Amount of memory units"""

    byte_suffix: ByteSuffix
    """ByteSuffix: The unit of memory to use"""

    _num_bytes: int = field(init=False)

    def __post_init__(self):
        num_bytes = self.quantity * self.byte_suffix.to_bytes()
        # This is a float, and potentially may look like xyz.0, xyz.99999998, xyz.0000518, or xyz.1abc.
        # The last case is not valid, but the second and third cases are floating point imprecision.
        # However, this is only in the case where self.byte_suffix is ByteSuffix.Byte or has a metric prefix, since
        # multiplying a float by 1024 is unlikely to produce an integer, ie. 1.5 KiB -> 1.5 * 1024 B is an integer, but
        # 3.14 KiB -> 3.14 * 1024 is not.
        dec = num_bytes - math.trunc(num_bytes)
        # This is the value after the decimal point. If there is floating point imprecision, we don't care because we
        # check with math.isclose. Also note we check math.isclose between 1+dec and 1, since rel_tol would fail in the
        # happy case where dec = 0.
        if self.byte_suffix.value[0] in (1, 1000) and not (
                math.isclose(1 + dec, 1, rel_tol=1e-3) or
                math.isclose(1, dec, rel_tol=1e-3)):
            raise ValueError(f"Memory({self.quantity}, {self.byte_suffix}) converts to {num_bytes} bytes, which is not an integer")
        # Note we cannot use '=' since this is a frozen dataclass. This is the official solution that dataclass uses:
        object.__setattr__(self, '_num_bytes', round(num_bytes))

    @classmethod
    def from_string(cls, s):
        """Creates a Memory object from a string, formatted like '[float][optional space][byte suffix]'
        ByteSuffix MUST be formatted in capitalized letters except for -ibibyte values. i.e. 'GB' and 'GiB' will be
        accepted, but 'gb' will not.
        """
        m = QUANTITY_UNIT_FORMAT.fullmatch(s)
        if m is None:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Invalid format.")
        suffix = m.group(2)
        if suffix not in ByteSuffix._member_map_:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Invalid suffix.")
        quant = float(m.group(1))
        if quant < 0:
            raise ValueError(f"Cannot convert string '{s}' to a Memory object. Negative quantity.")
        return Memory(quant, ByteSuffix[suffix])

    def convert(self, byte_suffix):
        """Converts a memory representation to an equivalent memory representation with a different ByteSuffix unit
        (i.e. from MB to GB), maintaining the same number of bytes."""
        return Memory._to_base(byte_suffix, self._num_bytes)

    def __eq__(self, o):
        if o.__class__ is not Memory:
            return NotImplemented
        return self._num_bytes == o._num_bytes

    @classmethod
    def _to_base(cls, base, n):
        max_exp = float("+inf")
        if type(base) is ByteSuffix:
            base, max_exp = base.value

        if n < base:
            return Memory(n, ByteSuffix.B)

        exp = 0
        while n >= base and exp < max_exp:
            exp += 1
            n /= base
        return Memory(n, ByteSuffix((base, exp)))

    @classmethod
    def to_1000_base(cls, n):
        return cls._to_base(1000, n)

    @classmethod
    def to_1024_base(cls, n):
        return cls._to_base(1024, n)

    def to_bytes(self):
        return self._num_bytes

    def pretty_string(self) -> str:
        """Formatted, human-readable string displaying the data in the Memory

        Returns:
            str: 'Pretty-print' string representation of the Memory
        """
        return f"{self.quantity:.2f} {self.byte_suffix.name}"


@dataclass(eq=False, frozen=True)
class AliasedName:
    """
    Represents a name that has given aliases that are considered equivalent to the original name.
    """

    name: Optional[str]
    """Optional[str]: The main name this AliasedName is referred to as. None is reserved as a special value."""

    aliases: Tuple[str, ...] = tuple()
    """Tuple[str, ...]: A tuple of aliases that are considered equivalent, and should map to the main name."""

    patterns: Tuple[re.Pattern, ...] = tuple()
    """Tuple[re.Pattern, ...]: A tuple of regex patterns this AliasedName can match with. These have lower precendence
    than aliases and will only be checked if aliases has been exhausted without a match. Patterns are only used if
    the object to match is an instance of str."""

    def __str__(self):
        return self.name

    def __hash__(self):
        # Needs to be implemented since we specify eq=False. See Python dataclass documentation.
        if self.name is None:
            return hash(None)
        return hash(self.name.lower())

    def __add__(self, other):
        if other.__class__ is AliasedName:
            raise Exception("Concatenating AliasedNames is ambiguous. One parameter should be a str.")
        elif other.__class__ is str:
            if len(self.patterns) > 0:
                raise Exception("Cannot concatenate AliasedName with regex patterns.")
            return AliasedName(self.name + other, [alias + other for alias in self.aliases])
        return NotImplemented

    def __eq__(self, other: Union[AliasedName, str, None]) -> bool:
        """
        Case insensitive equality check. Can be compared with another AliasedName or a str.

        If other is an AliasedName, returns True if self.name is case-insensitive equivalent to other.name.

        If other is a str, returns True if other is case-insensitive equivalent to self.name or any of the elements of
        self.aliases, or if it is a full match of any regex patterns in self.patterns.

        Args:
            other (Union[AliasedName, str, None]):
                The object to compare to

        Returns:
            bool: True if other is considered equal by the above rules. False otherwise, or if other is of an
            unrecognized type.
        """

        if other is None:
            return self.name is None

        if isinstance(other, AliasedName):
            if self.name is None or other.name is None:
                return self.name == other.name
            return self.name.lower() == other.name.lower()
        elif isinstance(other, str):
            if self.name is None:
                return False
            other_lower = other.lower()
            if self.name.lower() == other_lower or other_lower in (x.lower() for x in self.aliases):
                return True

            # No aliases matched, iterate through each regex.
            for pattern in self.patterns:
                if pattern.fullmatch(other):
                    return True
            return False
        else:
            return NotImplemented


class AliasedNameEnum(Enum):
    """
    Meant to be used as a parent class for any Enum that has AliasedName values..
    """

    @classmethod
    def as_aliased_names(cls) -> List[AliasedName]:
        return [elem.value for elem in cls]

    @classmethod
    def as_strings(cls) -> List[str]:
        return list(map(str, cls.as_aliased_names()))

    @classmethod
    def get_match(cls, name: Union[AliasedName, str]) -> Optional[AliasedNameEnum]:
        """
        Attempts to return the element of this enum that is equivalent to `name`.

        Args:
            name (Union[AliasedName, str]):
                The name of an element we want

        Returns:
            Optional[AliasedName]: The AliasedName if found, None otherwise
        """
        for elem in cls:
            if elem.value == name or elem == name:  # candidate must be on the left to use __eq__
                return elem
        return None

    def __eq__(self, other: Any) -> bool:
        """
        __eq__ override for members. Will compare directly if `other` is of the same __class__. Otherwise will attempt
        to use the __eq__ of the value.

        Addendum on how __eq__ interacts with AliasedNameEnum members:
            See https://stackoverflow.com/a/63310298 to learn how exactly __eq__ is evaluated in Python3 internally.

            As an example, if we add a 'print' to the beginning of this method like so:

            print("calling AliasedNameEnum.__eq__")

            The behavior is rather interesting in that this method is always the 'source of truth' when comparing
            against a Python string:

                >>> from code.common.constants import Benchmark
                >>> class MyStr(str):
                ...     def __eq__(self, other):
                ...             print("Calling MyStr.__eq__")
                ...             return super().__eq__(other)
                ...
                >>> MyStr("ResNet50") == Benchmark.ResNet50
                Calling MyStr.__eq__
                calling AliasedNameEnum.__eq__
                True
                >>> Benchmark.ResNet50 == MyStr("ResNet50")
                calling AliasedNameEnum.__eq__
                True
                >>> Benchmark.ResNet50 != MyStr("ResNet50")
                calling AliasedNameEnum.__eq__
                False
                >>> Benchmark.ResNet50 != MyStr("ResNet50")
                calling AliasedNameEnum.__eq__
                False
                >>> MyStr("ResNet50") == Benchmark.SSDMobileNet
                Calling MyStr.__eq__
                calling AliasedNameEnum.__eq__
                False

            The reasoning for this behavior in accordance with the rules is that the C implementation of a Python `str`
            (or as it called in Python source, a 'UnicodeObject') returns NotImplemented if 'other' is not another
            UnicodeObject:
            https://github.com/python/cpython/blob/7d25254cf0763b62f4c4a3019e56385cab597b9f/Objects/unicodeobject.c#L11283-L11284

            **However** it is STILL best practices to put the AliasedNameEnum on the LHS of an equality check to ensure
            that this implementation of __eq__ is used, i.e.:

                DO:
                Benchmarks.ResNet50 == "resnet50"

                DON'T:
                "resnet50" == Benchmarks.ResNet50

        Args:
            other (Any):
                The object to compare to

        Returns:
            bool: True if other is equivalent to self directly, or self.value. False otherwise.
        """
        if self.__class__ is other.__class__:
            return self is other
        else:
            return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def valstr(self) -> str:
        """
        Convenience method to get the string representation of this Enum member's value.

        Returns:
            str: self.value.__str__()
        """
        return str(self.value)


@unique
class Benchmark(AliasedNameEnum):
    """Names of supported Benchmarks in MLPerf Inference."""

    BERT: AliasedName = AliasedName("bert")
    DLRM: AliasedName = AliasedName("dlrm")
    RNNT: AliasedName = AliasedName("rnnt", ("rnn-t",))
    ResNet50: AliasedName = AliasedName("resnet50", ("resnet",))
    SSDMobileNet: AliasedName = AliasedName("ssd-mobilenet", ("ssdmobilenet", "ssd-small"))
    SSDResNet34: AliasedName = AliasedName("ssd-resnet34", ("ssdresnet34", "ssd-large"))
    UNET3D: AliasedName = AliasedName("3d-unet", ("3dunet", "unet", "3d-unet-kits", "3d-unet-kits19"))
    Retinanet: AliasedName = AliasedName("retinanet", ("ssd-retinanet", "resnext", "ssd-resnext"))


@unique
class Scenario(AliasedNameEnum):
    """Names of supported workload scenarios in MLPerf Inference."""

    Offline: AliasedName = AliasedName("Offline")
    Server: AliasedName = AliasedName("Server")
    SingleStream: AliasedName = AliasedName("SingleStream", ("single-stream", "single_stream"))
    MultiStream: AliasedName = AliasedName("MultiStream", ("multi-stream", "multi_stream"))


@unique
class Action(AliasedNameEnum):
    """Names of actions performed by our MLPerf Inference pipeline."""

    GenerateConfFiles: AliasedName = AliasedName("generate_conf_files")
    GenerateEngines: AliasedName = AliasedName("generate_engines")
    Calibrate: AliasedName = AliasedName("calibrate")
    RunHarness: AliasedName = AliasedName("run_harness")
    RunAuditHarness: AliasedName = AliasedName("run_audit_harness")


@unique
class AuditTest(AliasedNameEnum):
    """Audit test names"""

    TEST01: AliasedName = AliasedName("TEST01")
    TEST04: AliasedName = AliasedName("TEST04")
    TEST05: AliasedName = AliasedName("TEST05")


@unique
class Precision(AliasedNameEnum):
    """Different numeric precisions that can be used by benchmarks. Not all benchmarks can use all precisions."""

    INT8: AliasedName = AliasedName("int8")
    FP16: AliasedName = AliasedName("fp16")
    FP32: AliasedName = AliasedName("fp32")


@unique
class InputFormats(AliasedNameEnum):
    """Different input formats that can be used by benchmarks. Not all benchmarks can use all input formats."""
    Linear: AliasedName = AliasedName("linear")
    CHW4: AliasedName = AliasedName("chw4")
    DHWC8: AliasedName = AliasedName("dhwc8")
    CDHW32: AliasedName = AliasedName("cdhw32")


@unique
class CPUArchitecture(AliasedNameEnum):
    """Various CPU Architectures"""
    x86_64: AliasedName = AliasedName("x86_64")
    aarch64: AliasedName = AliasedName("aarch64")


@unique
class AcceleratorType(AliasedNameEnum):
    """Various accelerator types to describe how the CPU and GPU share memory"""
    Discrete: AliasedName = AliasedName("Discrete")  # Accelerator has independent memory
    Integrated: AliasedName = AliasedName("Integrated")  # Accelerator shares memory with CPU


@unique
class HarnessType(AliasedNameEnum):
    """Possible harnesses a benchmark can use."""
    LWIS: AliasedName = AliasedName("lwis")
    Custom: AliasedName = AliasedName("custom")
    Triton: AliasedName = AliasedName("triton")
    HeteroMIG: AliasedName = AliasedName("hetero")
    # TODO: once triton harness unification is completed, TritonUnified will replace Triton
    TritonUnified: AliasedName = AliasedName("triton_unified")
    # FIXME: Better way than add LON/SUT node here?
    LON_Node: AliasedName = AliasedName("lon_node")
    SUT_Node: AliasedName = AliasedName("sut_node")


@unique
class AccuracyTarget(Enum):
    """Possible accuracy targets a benchmark must meet. Determined by MLPerf Inference committee."""
    k_99: float = .99
    k_99_9: float = .999


@unique
class PowerSetting(AliasedNameEnum):
    """Possible power settings the system can be set in when running a benchmark."""
    MaxP: AliasedName = AliasedName("MaxP")
    MaxQ: AliasedName = AliasedName("MaxQ")


@unique
class CPUArch(AliasedNameEnum):
    """Possible power settings the system can be set in when running a benchmark."""
    x86_64: AliasedName = AliasedName("x86_64")
    aarch64: AliasedName = AliasedName("aarch64")


@dataclass(frozen=True)
class WorkloadSetting:
    """
    Describes the various settings used when running a benchmark workload. These are usually for different use cases that
    MLPerf Inference allows (i.e. power submission), or running the same workload with different software (i.e. Triton).
    """
    harness_type: HarnessType = HarnessType.Custom
    """HarnessType: Harness to use for this workload. Default: HarnessType.Custom"""

    accuracy_target: AccuracyTarget = AccuracyTarget.k_99
    """AccuracyTarget: Accuracy target for the benchmark. Default: AccuracyTarget.k_99"""

    power_setting: PowerSetting = PowerSetting.MaxP
    """PowerSetting: Power setting for the system during this workload. Default: PowerSetting.MaxP"""

    def __str__(self) -> str:
        return f"WorkloadSetting({self.harness_type}, {self.accuracy_target}, {self.power_setting})"

    def shortname(self) -> str:
        return f"{self.harness_type.value.name}_{self.accuracy_target.name}_{self.power_setting.value.name}"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convenience wrapper around dataclasses.asdict to convert this WorkloadSetting to a dict().

        Returns:
            Dict[str, Any]: This WorkloadSetting as a dict
        """
        return asdict(self)


G_HIGH_ACC_ENABLED_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.DLRM,
    Benchmark.UNET3D,
)
"""Tuple[Benchmark, ...]: Benchmarks that have 99.9% accuracy targets"""


G_DATACENTER_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.DLRM,
    Benchmark.RNNT,
    Benchmark.ResNet50,
    Benchmark.Retinanet,
    Benchmark.SSDResNet34,
    Benchmark.UNET3D,
)
"""Tuple[Benchmark, ...]: Benchmarks for the Datacenter submission category"""


G_EDGE_BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark.BERT,
    Benchmark.RNNT,
    Benchmark.ResNet50,
    Benchmark.Retinanet,
    Benchmark.SSDMobileNet,
    Benchmark.SSDResNet34,
    Benchmark.UNET3D,
)
"""Tuple[Benchmark, ...]: Benchmarks for the Edge submission category"""


G_DEFAULT_HARNESS_TYPES: Dict[Benchmark, HarnessType] = {
    Benchmark.BERT: HarnessType.Custom,
    Benchmark.DLRM: HarnessType.Custom,
    Benchmark.RNNT: HarnessType.Custom,
    Benchmark.ResNet50: HarnessType.LWIS,
    Benchmark.Retinanet: HarnessType.LWIS,
    Benchmark.SSDMobileNet: HarnessType.LWIS,
    Benchmark.SSDResNet34: HarnessType.LWIS,
    Benchmark.UNET3D: HarnessType.Custom,
}
"""Dict[Benchmark, HarnessType]: Defines the default harnesses (non-Triton) used for each benchmark."""


def config_ver_to_workload_setting(benchmark: Benchmark, config_ver: str) -> WorkloadSetting:
    """This method is a temporary workaround to retain legacy behavior as the codebase is incrementally refactored to
    use the new Python-style BenchmarkConfiguration instead of the old config.json files.

    Converts a legacy 'config_ver' ID to a new-style WorkloadSetting.

    Args:
        benchmark (Benchmark):
            The benchmark that is being processed. Used to decide the HarnessType.
        config_ver (str):
            The old-style 'config_ver' ID

    Returns:
        WorkloadSetting: The equivalent WorkloadSetting for the benchmark/config_ver.
    """
    harness_type = G_DEFAULT_HARNESS_TYPES[benchmark]
    if "triton_unified" in config_ver:
        harness_type = HarnessType.TritonUnified
    elif "openvino" in config_ver or "triton" in config_ver:
        harness_type = HarnessType.Triton
    elif "hetero" in config_ver:
        harness_type = HarnessType.HeteroMIG
    # FIXME: adding LON/SUT node here as separate harness type
    elif "lon_node" in config_ver:
        harness_type = HarnessType.LON_Node
    elif "sut_node" in config_ver:
        harness_type = HarnessType.SUT_Node

    accuracy_target = AccuracyTarget.k_99
    if "high_accuracy" in config_ver:
        accuracy_target = AccuracyTarget.k_99_9

    power_setting = PowerSetting.MaxP
    if "maxq" in config_ver:
        power_setting = PowerSetting.MaxQ

    return WorkloadSetting(harness_type=harness_type, accuracy_target=accuracy_target, power_setting=power_setting)
