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

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging
from code.common.systems.systems import SystemConfiguration
from code.common.systems.system_list import KnownSystem, match_known_system
from code.common.constants import (
    Benchmark,
    Scenario,
    HarnessType,
    AccuracyTarget,
    PowerSetting,
    CPUArch,
    WorkloadSetting,
    G_DEFAULT_HARNESS_TYPES
)
from code.common.utils import Tree
from code.common.fields import get_applicable_fields
from configs.error import *

import ast
import inspect
import textwrap
import importlib

from collections import namedtuple
from enum import Enum, unique
from typing import List, Tuple


def _parse_sourcelines(lines):
    """
    Returns the Abstract Syntax Tree of the source code represented in `lines`. Since source may be local to a scope and
    have indents, it is possible that the parsing of `lines` may fail due to improper indentation.

    This method will normalize the indentation of the sourcelines so that the first line in `lines` will not have any
    indentation.
    """
    if len(lines) > 0:
        indent_str_len = len(lines[0]) - len(textwrap.dedent(lines[0]))
    normalized = [s[indent_str_len:] for s in lines]
    return ast.parse("".join(normalized))


InvalidConfig = namedtuple("InvalidConfig", ("config_cls", "workload_setting", "error"))


class ConfigRegistry(object):
    """
    Singleton, global data structure to store and index benchmark-system configurations and provide metadata about them.

    A config cannot be registered unless it passes certain validation checks, and will not be able to be used at runtime
    unless it is registered. This serves as a 'security check' at runtime to ensure best practices when writing configs.

    Internal structure for storing configs. Should not be accessed outside the class. This Tree will have the structure:
        {
            <benchmark>: {
                <scenario>: {
                    <system id>: {
                        <setting>: <config>, # 'setting' is a string that describes the workload setting
                        ...
                    },
                    ...
                },
                ...
            },
            ...
        }
    """

    _registry: Tree = Tree()
    """Tree: See class docstring for structure"""

    _permissive_register: bool = False
    """bool: If True, will not raise an exception if an invalid BenchmarkConfiguration is attempted to be registered."""

    _failed_register: List[InvalidConfig] = []
    """List[InvalidConfig]: Maintains an internal list of BenchmarkConfigurations that failed to register"""

    @staticmethod
    def load_configs(benchmark: Benchmark, scenario: Scenario, ignore_custom: bool = False) -> bool:
        """
        Bulk registers all the configs for a given benchmark-scenario pair by importing its file from configs/. If a
        module for benchmark.scenario does not exist, will return False.

        Args:
            benchmark (Benchmark):
                The benchmark to load configs for
            scenario (Scenario):
                The scenario to load configs for
            ignore_custom (bool):
                If True, ignores loading any detected custom configs. (Default: False)

        Returns:
            bool: Whether or not the module was loaded successfully
        """
        try:
            base_module = f"configs.{benchmark.valstr()}.{scenario.valstr()}"
            if ignore_custom:
                logging.debug("Skipping custom configs")
                importlib.import_module(base_module)
            else:
                _custom_definition_path = os.environ.get("MLPINF_CUSTOM_DEFINITION_PATH", None)
                if _custom_definition_path is None:
                    logging.debug(f"MLPINF_CUSTOM_DEFINITION_PATH not set. Searching for custom configs in {base_module}.custom")
                    if importlib.util.find_spec(f"{base_module}.custom") is not None:
                        logging.debug(f"Found custom configs in {base_module}.custom")
                        importlib.import_module(f"{base_module}.custom")
                elif not os.path.exists(_custom_definition_path):
                    raise FileNotFoundError(f"MLPINF_CUSTOM_DEFINITION_PATH {_custom_definition_path} does not exist.")
                else:
                    custom_module = f"custom_configs.{benchmark.valstr()}.{scenario.valstr()}.custom"
                    logging.debug(f"Searching for custom configs in {custom_module}")
                    from code.common.fix_sys_path import ScopedRestrictedImport
                    with ScopedRestrictedImport(restricted_path=[_custom_definition_path] + sys.path) as sri:
                        if importlib.util.find_spec(custom_module) is not None:
                            logging.debug(f"Found custom configs in {custom_module}")
                            importlib.import_module(custom_module)
                # Since the non-custom configs are in __init__.py, they are auto-imported if you import custom.
            return True
        except ModuleNotFoundError:
            return False

    @classmethod
    def load_module_dryrun(cls, module_path: str) -> Tuple[Tree, Tuple[InvalidConfig, ...]]:
        """
        Checks the configs of the given module, while preserving the state of the ConfigRegistry (i.e. the state of the
        ConfigRegistry before and after the execution of this method should be the same).

        Args:
            module_path (str):
                The path of the module to check

        Returns:
            Tree: a Tree representation of the ConfigRegistry of the BenchmarkConfigurations in the given module which
                  were loaded successfully.
            Tuple[InvalidConfig, ...]: A tuple of InvalidConfigs that represent BenchmarkConfigurations that were not loaded
                                       successfully from the module.

        Raises:
            ModuleNotFoundError: If the module_path is not well-formed or does not exist.
        """
        backup_registry = cls._registry
        cls._reset()
        cls._permissive_register = True

        tmp_registry = None
        invalid = tuple()
        try:
            importlib.import_module(module_path)
            tmp_registry = cls._registry
            invalid = tuple(cls._failed_register)
        finally:
            cls._reset()
            cls._registry = backup_registry
        return tmp_registry, invalid

    @classmethod
    def _reset(cls):
        """
        Clears the registry.
        """
        cls._registry = Tree()
        cls._failed_register = []
        cls._permissive_register = False

    @classmethod
    def register(cls, harness_type, accuracy_target, power_setting):
        """
        Used as a decorator to register a config for the given workload setting.

        Returns:
            A func that will attempt to register the config, and will return the config.

            This func will attempts to register the config if the config passes validation checks. If the config is
            invalid or is already exists in the registry, this returned func will error.
        """
        workload_setting = WorkloadSetting(harness_type, accuracy_target, power_setting)

        def _do_register(config):
            try:
                cls.validate(config, workload_setting)  # Will raise error if validation fails
            except ConfigError as e:
                if cls._permissive_register:
                    cls._failed_register.append(InvalidConfig(config, workload_setting, e))
                    return config
                else:
                    raise e

            keyspace = [config.benchmark, config.scenario, config.system, workload_setting]
            if cls._registry.get(keyspace) != None:
                raise KeyError("Config for {} is already registered.".format("/".join(map(str, keyspace))))
            cls._registry.insert(keyspace, config)
            return config
        return _do_register

    @classmethod
    def get(cls, benchmark, scenario, system, harness_type=HarnessType.Custom, accuracy_target=AccuracyTarget.k_99,
            power_setting=PowerSetting.MaxP):
        """
        Returns the config specified, None if it doesn't exist.
        """
        if type(system) is SystemConfiguration:
            system = match_known_system(system)
        workload_setting = WorkloadSetting(harness_type, accuracy_target, power_setting)
        return cls._registry.get([benchmark, scenario, system, workload_setting])

    @classmethod
    def available_workload_settings(cls, benchmark, scenario):
        """
        Returns the registered WorkloadSettings for a given benchmark-scenario pair. None if the benchmark-scenario
        pair does not exist.
        """
        workloads = cls._registry.get([benchmark, scenario])
        if workloads is None:
            return None
        return list(workloads.keys())

    @classmethod
    def validate(cls, config, workload_setting):
        """
        Validates the config's settings based on certain rules. The config must satisfy the following:
            1. Must have 'BenchmarkConfiguration' in its inheritance parent classes
            2. Can only inherit from configs that use the same:
                - Benchmark
                - Scenario
                - Accelerator
            3. Does not include any config fields that are not applicable to the workload
            4. Contains all required keys necessary to run the workload
            5. Does not define fields multiple times within the class
            6. Does not extend multiple classes at any point within its inheritance chain
            7. Does not use an unsupported harness type

        If the config fails any of these criteria, it will raise the corresponding configs.error.ConfigFieldError.
        This method will finish successfully without errors raised if the config is valid.
        """
        # Criteria 1,2,6
        cls._check_inheritance_constraints(config)

        # Criteria 7
        cls._check_harness_support(config, workload_setting)

        # Criteria 3,4
        cls._check_field_descriptions(config, workload_setting)

        # Criteria 5: Cannot define the same field multiple times within the class
        cls._check_field_reassignment(config)

    @classmethod
    def _check_harness_support(cls, config, workload_setting):
        """
        Checks that the harness types are supported by the benchmark.
        """
        benchmark = config.benchmark
        harness_type = workload_setting.harness_type
        if harness_type in [HarnessType.Custom, HarnessType.LWIS]:
            # LWIS is technically a 'Custom' harness, but is widely across multiple benchmarks
            if harness_type != G_DEFAULT_HARNESS_TYPES[benchmark]:
                raise ConfigInvalidHarnessTypeError(config.__name__, harness_type)
        if harness_type == HarnessType.HeteroMIG:
            if config.system.value.accelerator_conf.num_migs() == 0:
                raise ConfigInvalidHarnessTypeError(config.__name__, harness_type)

    @classmethod
    def _check_field_descriptions(cls, config, workload_setting, enforce_full=True):
        """
        Checks the names and types of all the fields in config.

        This method will:
            1. Checks if the config has 'benchmark', 'scenario', and 'system'
            2. Builds a list of mandatory and optional Fields given the 'benchmark', 'scenario', and 'system'
            3. Check if any fields exist in config that are not in these mandatory and optional sets.

        This method will run successfully if and only if:
            - config.fields is a subset of (mandatory UNION optional)
            - If enforce_full is True, mandatory is a subset of config.fields

        Otherwise, this method will throw a ConfigFieldMissingError or ConfigFieldInvalidTypeError.
        """
        # Check for benchmark, scenario, and system:
        identifiers = {
            "action": None,
            "benchmark": None,
            "scenario": None,
            "system": None,
            "workload_setting": workload_setting,
        }

        ConfigFieldDescription = namedtuple("ConfigFieldDescription", ("name", "type"))
        for f in (
            ConfigFieldDescription(name="benchmark", type=Benchmark),
            ConfigFieldDescription(name="scenario", type=Scenario),
            ConfigFieldDescription(name="system", type=KnownSystem)
        ):
            if not hasattr(config, f.name):
                if enforce_full:
                    raise ConfigFieldMissingError(config.__name__, f.name)
            elif type(getattr(config, f.name)) != f.type:
                raise ConfigFieldInvalidTypeError(config.__name__, f.name, f.type)
            else:
                identifiers[f.name] = getattr(config, f.name)

        # Grab a set of mandatory and optional Fields from fields.py.
        mandatory, optional = get_applicable_fields(**identifiers)
        mandatory = set([f.name for f in mandatory])
        optional = set([f.name for f in optional])
        possible_fields = mandatory.union(optional)
        # Remove 'action' and 'harness_type', as these are metadata about the config, not fields.
        declared_fields = set(config.all_fields()) - {"action", "harness_type"}

        disallowed_fields = declared_fields - possible_fields
        if len(disallowed_fields) > 0:
            raise ConfigFieldInvalidError(config.__name__, list(disallowed_fields)[0])

        missing_fields = mandatory - declared_fields
        if enforce_full and len(missing_fields) > 0:
            raise ConfigFieldMissingError(config.__name__, list(missing_fields)[0])

        if HarnessType.Triton == workload_setting.harness_type:
            if config.use_triton != True:
                raise ConfigFieldError(f"<config '{config.__name__}'> uses Triton, but use_triton is not set to True")

    @classmethod
    def _check_inheritance_constraints(cls, config):
        """
        Checks that the config satisfies the following rules:
            1. Must have 'BenchmarkConfiguration' in its inheritance parent classes
            2. Can only inherit from configs that use the same:
                - Benchmark
                - Scenario
                - Accelerator
            3. Does not extend multiple classes at any point within its inheritance chain
        """
        # Check inheritance rules via MRO (method resolution order)
        parents = inspect.getmro(config)

        # Criteria 1:
        if BenchmarkConfiguration not in parents:
            raise ConfigInvalidTypeError(config.__name__)

        # Criteria 3:
        for curr in parents[:-2]:
            # Search for the field in the current class:
            source, start_lineno = inspect.getsourcelines(curr)
            syntax_tree = _parse_sourcelines(source)
            classdefs = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.ClassDef)]
            assert(len(classdefs) > 0)

            # We only care about the 0th class definition, as any class definition after that would be an internal class
            # defined within the Configuration
            if len(classdefs[0].bases) != 1:
                raise ConfigMultipleExtendsError(curr.__name__)

        # Criteria 2: Check every parent for validity
        for parent in parents[:-2]:
            # Must have the same benchmark if a parent defined a benchmark.
            if hasattr(parent, 'benchmark'):
                if config.benchmark != parent.benchmark:
                    raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "benchmark", config.benchmark,
                                                      parent.benchmark)

            # Must have the same scenario if a parent defined a scenario.
            if hasattr(parent, 'scenario'):
                if config.scenario != parent.scenario:
                    raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "scenario", config.scenario,
                                                      parent.scenario)

            # Must have the same accelerator if a parent defined an accelerator.
            if hasattr(config, 'system'):
                if type(config.system) != KnownSystem:
                    raise ConfigFieldInvalidTypeError(config.__name__, "system", KnownSystem.__name__)

                if hasattr(parent, 'system'):
                    if type(parent.system) != KnownSystem:
                        raise ConfigFieldInvalidTypeError(parent.__name__, "system", KnownSystem.__name__)

                    parent_accelerators = set(parent.system.value.accelerator_conf.get_accelerators())
                    child_accelerators = set(config.system.value.accelerator_conf.get_accelerators())
                    diff = parent_accelerators - child_accelerators
                    if len(diff) > 0:
                        raise ConfigFieldInheritanceError(config.__name__, parent.__name__, "system.gpu",
                                                          config.system.value.accelerator_conf,
                                                          parent.system.value.accelerator_conf)

    @classmethod
    def _check_field_reassignment(cls, config):
        """
        Checks that the config does not define fields multiple times within the class
        """
        trace = config.get_field_trace()
        for field, assignments in trace.trace.items():
            # Note that since assignments is an ordered sequence, we can just iterate once to check for duplicates
            for i in range(len(assignments) - 1):
                if assignments[i].klass == assignments[i + 1].klass:
                    raise ConfigFieldReassignmentError(assignments[i].klass.__name__,
                                                       field,
                                                       (assignments[i].lineno, assignments[i + 1].lineno))


class BenchmarkConfiguration(object):
    """
    Describes a configuration for a BenchmarkScenario pair for a given system. If the config is meant to be used as a
    full config, the derived class must be registered with the @ConfigRegistry.register decorator. For example, there
    might be an INT8Configuration with generic, default settings for an INT8 benchmark that is not registered, which is
    extended by ResNet50_INT8_Configuration that will be registered.

    Fields for the configuration are all non-callable class variables that are not prefixed with '_'. For instance, if
    you wish for a variable to be hidden from code that uses the config, prefix it with '_'.

    A configuration is defined like follows:

        class MyConfiguration(<must have BenchmarkConfiguration somewhere in its inheritance chain>):
            system = System(...)
            scenario = Scenario.<...>
            _N = 12 # This field will not be visible when a list of fields is requested
            batch_size = 64 * _N # This field uses __N to calculate it, and will be visible
            depth = _N + 1
            some_field = "value"
            ...

    For best practices, Configurations should **NOT** have multiple inherits at any level of their inheritance chain.
    This makes it easier to detect where fields are introduced along the inheritance chain.

    Example:
        class Foo(BenchmarkConfiguration):
            ...

        class Bar(Foo): # This is fine
            ...

        class Baz(Bar, SomeOtherClass): # This is not best practices
            ...

        class Boo(Bar): # This is still fine
            ...

        class Faz(Baz): # Even though Faz only inherits Baz, Baz has multiple inherits which makes this not advised.
            ...
    """

    @classmethod
    def all_fields(cls):
        """
        Returns all visible fields in this config. Visible fields are all non-callable class variables that are not
        prepended with '_'.
        """
        return tuple([
            attr
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("_") and getattr(cls, attr) is not None
        ])

    @classmethod
    def as_dict(cls):
        """
        Returns all fields of the class as a shallow dict of {key:value}.
        """
        return {
            attr: getattr(cls, attr)
            for attr in cls.all_fields()
        }

    @classmethod
    def get_field_trace(cls):
        """
        Returns a FieldTrace of this class. See FieldTrace documentation for more details. This is useful to track down
        where fields are overridden or inherited from.
        """
        return FieldTrace(cls)


class FieldTrace(object):
    """
    Represents the trace of all the fields of a BenchmarkConfiguration as they are declared in Method Resolution Order.

    {
        <field>: [ list of namedtuple(klass=<class>, lineno=int, value=<ast.Value object>) ]
        ...
    }
    The list is ordered in method resolution order, where the first element is the true resolution of the field at
    runtime, and subsequent elements are resolutions of the field further down the resolution chain.

    i.e. Given
        class A(object):
            foo = 1

        class B(A):
            foo = 2

        class C(B):
            foo = 3

    The FieldTrace of C will be:
        {
            "foo": [(klass=C, lineno=..., value=ast.Num(3)),
                    (klass=B, lineno=..., value=ast.Num(2)),
                    (klass=A, lineno=..., value=ast.Num(1))]
        }
    """

    def __init__(self, config_cls):
        """
        Initializes a FieldTrace for the given class.
        """
        self.cls_name = config_cls.__name__

        Assignment = namedtuple("Assignment", ["klass", "lineno", "value"])
        parents = inspect.getmro(config_cls)
        trace = dict()

        def _add_trace(k, v):
            if k not in trace:
                trace[k] = []
            trace[k].append(v)
        fields = config_cls.all_fields()

        # The last 2 fields in MRO will be <BenchmarkConfiguration> and <object>, which we don't want to check.
        for curr in parents[:-2]:
            # Search for the field in the current class:
            source, start_lineno = inspect.getsourcelines(curr)
            syntax_tree = _parse_sourcelines(source)
            assignments = [node for node in ast.walk(syntax_tree) if isinstance(node, ast.Assign)]

            # Later assignments will take precendence in resolution, so reverse the order of assignments
            for assignment in assignments[::-1]:
                for target in assignment.targets:
                    if target.id in fields:
                        _add_trace(
                            target.id,
                            Assignment(
                                klass=curr,
                                lineno=(start_lineno + assignment.lineno - 1),
                                value=assignment.value
                            ))
        self.trace = trace

    def __str__(self):
        return self.dump()

    def dump(self, sort_fields=True, indent=4):
        """
        Returns a string representing this FieldTrace in a human-readable format.
        """
        keys = self.trace.keys()
        if sort_fields:
            keys = list(sorted(keys))

        indent_str = " " * indent
        lines = [f"FieldTrace({self.cls_name})" + "{"]
        for key in keys:
            lines.append(f"{indent_str}'{key}': [")
            for assignment in self.trace[key]:
                lines.append(f"{indent_str * 2}Assignment(")
                lines.append(f"{indent_str * 3}klass={assignment.klass}")
                lines.append(f"{indent_str * 3}lineno={assignment.lineno}")
                lines.append(f"{indent_str * 3}value={ast.dump(assignment.value)}")
                lines.append(f"{indent_str * 2}),")
            lines.append(f"{indent_str}]")
        lines.append("}")
        return "\n".join(lines)
