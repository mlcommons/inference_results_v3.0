# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


__doc__ = """Defines custom Error types that are thrown by the ConfigRegistry when attempting to validate / register bad
Configs.
"""


class ConfigError(Exception):
    """
    Base class for Errors involving Config classes themselves
    """

    def __init__(self, msg):
        super().__init__(msg)


class ConfigInvalidTypeError(ConfigError):
    """
    Exception raised when a Config does not extend configs.configuration.BenchmarkConfiguration.
    """

    def __init__(self, config_name):
        super().__init__(f"<config '{config_name}'> does not inherit from BenchmarkConfiguration")


class ConfigMultipleExtendsError(ConfigError):
    """
    Exception raised when a class that extends BenchmarkConfiguration has multiple base classes, which is disallowed.
    """

    def __init__(self, config_name):
        super().__init__(f"<config '{config_name}'> extends multiple base classes, which is disallowed.")


class ConfigFieldError(Exception):
    """
    Base class for Errors involving Config Fields.
    """

    def __init__(self, msg):
        super().__init__(msg)


class ConfigFieldMissingError(ConfigFieldError):
    """
    Exception raised for errors when a Config is missing a mandatory Field
    """

    def __init__(self, config_name, field_name):
        super().__init__(f"<config '{config_name}'> is missing mandatory Field '{field_name}'")


class ConfigFieldInvalidError(ConfigFieldError):
    """
    Exception raised for errors when a Field should not exist in a Config
    """

    def __init__(self, config_name, field_name):
        super().__init__(f"<config '{config_name}'> contains unsupported Field '{field_name}'")


class ConfigFieldInvalidTypeError(ConfigFieldError):
    """
    Exception raised for errors when a Config Field has a value that is not the correct Type
    """

    def __init__(self, config_name, field_name, expected_type):
        super().__init__(f"<config '{config_name}'>.{field_name} is not of type '{expected_type}'")


class ConfigFieldInheritanceError(ConfigFieldError):
    """
    Exception raised for errors when a Config Field extends a parent Config, but overrides a field with an invalid
    Value. This will occur for the following fields:
        - benchmark
        - scenario
        - system.gpu
    This error exists for the inheritance constraint that configurations may only extend configurations that are of the
    same workload and system.
    """

    def __init__(self, config_name, parent_name, field_name, new_value, old_value):
        super().__init__(f"<config '{config_name}'>.{field_name}={new_value}, but inherits <config '{parent_name}'>.{field_name}={old_value}")


class ConfigFieldReassignmentError(ConfigFieldError):
    """
    Exception raised for errors when a Config Field is (re)assigned multiple times within the same class.
    """

    def __init__(self, config_name, field_name, line_nos):
        line_str = ",".join([str(n) for n in line_nos])
        super().__init__(f"<config '{config_name}'>.{field_name} defined multiple times on lines {line_str}")


class ConfigInvalidHarnessTypeError(ConfigError):
    """
    Exception raised for errors when a Config's workload setting uses an unsupported harness type.
    """

    def __init__(self, config_name, harness_type):
        super().__init__(f"<config '{config_name}'> attempted to register unsupported harness type '{harness_type}'.")
