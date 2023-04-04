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
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum
from numbers import Number
from typing import Callable, Final, Optional
from multiprocessing import Process

import dataclasses
import math

from code.common import logging
from code.common.constants import *
from code.common.systems.systems import SystemConfiguration


class ActionHandler(ABC):
    """Handles one of the 'Actions' that represents one of the phases of the MLPerf Inference pipeline. Each action is
    to correspond to a single Make target.

    See code.common.constants.Action for a list of supported Actions.
    """

    def __init__(self, action: Action):
        """Constructs an ActionHandler
        """
        self.action = action

    @abstractmethod
    def setup(self):
        """Called once before handle().
        """
        pass

    @abstractmethod
    def handle(self) -> bool:
        """Run the action.

        Returns:
            bool: True if handle() succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def handle_failure(self):
        """Called after handle() if it errors.
        """
        pass

    @abstractmethod
    def cleanup(self, success: bool):
        """Called after handle(), regardless if it errors.

        success (bool): Indicates whether or not self.handle() executed successfully.  This is useful when cleanup
                        behaves differently when handle fails, or the cleanup code depends on something that is only
                        done on successful runs.
        """
        pass

    def run(self) -> bool:
        """Runs through all the steps of the ActionHandler"""
        self.setup()
        success = False
        try:
            success = self.handle()
        finally:  # Always run cleanup
            self.cleanup(success)
            if not success:
                self.handle_failure()
        return success


class SubprocessActionHandler(ActionHandler):
    """Handles one of the 'Actions' in a subprocess. Child classes must implement:

        - setup()
        - cleanup()
        - handle_failure()
        - subprocess_target()

    SubprocessActionHandler changes the behavior of the following:
        - setup() is only called before the FIRST try
        - cleanup() is called after the FINAL try (before handle_failure())
        - handle_failure() is called if ALL retries of handle() fail
    """

    def __init__(self,
                 action: Action,
                 num_retries: int = 0,
                 timeout: int = 7200):
        """Creates an ActionHandler that runs handle() in a subprocess. This subprocess is controlled by the Python
        multiprocessing library, and therefore is affected by the multiprocessing start method.

        See https://docs.python.org/3/library/multiprocessing.html for more details.

        By default the MLPerf Inference codebase uses the 'spawn' start method, which will call the subprocess target in
        a *fresh* Python interpreter, only inheriting the resources required for running the target itself. In
        particular, the only resources that will persist between the parent and child process are those in the *args and
        **kwargs for the subprocess target. Because of this, in self.handle, do not update any variables or values that
        are supposed to persist in the parent process.

        There are a few reasons to use "spawn" over "fork" for MLPerf Inference's use-case:
            - One reason is "fork" would still not fix the issue with modifying global variables in self.handle. While
              the globals would be copied over from parent to child process, the other way around would not work, so any
              updates the child processes do would not be reflected in the parent. "spawn" is more light-weight than
              "fork" so it is better here.
            - The other, larger reason is that every time an engine is built on a retry, a new child process will be
              spawned. In this case, we want a clean CUDA runtime state before re-attempting to build the engine. Using
              "spawn" ensures that the CUDA context in the child process is isolated and clean.

        Args:
            action (Action): The Action this handler handles.
            num_retries (int): Number of times to retry the subprocess target if it fails. Setting this to 0 will only
                               call the subprocess target once, and not attempt to re-run if it fails. (Default: 0)
            timeout (int): Number of seconds to wait for the subprocess to finish running. Must be a positive number.
                           (Default: 7200)
        """
        super().__init__(action)
        self.n_retries = num_retries
        self.timeout = timeout

    @abstractmethod
    def subprocess_target(self):
        """Function to run in a subprocess to perform the action
        """
        pass

    def handle(self):
        success = False
        for i in range(self.n_retries + 1):
            logging.debug(f"Launching subprocess for action {self.action.valstr()}. Attempt #{i + 1}")
            p = Process(target=self.subprocess_target)
            p.start()
            try:
                p.join(self.timeout)
            except KeyboardInterrupt:
                p.terminate()
                p.join(self.timeout)
                raise KeyboardInterrupt
            if p.exitcode == 0:
                success = True
                break
        return success


class SubprocessActionHandlerWrapper(SubprocessActionHandler):
    """Wrapper class that takes in an ActionHandler and converts it to a SubprocessActionHandler"""

    def __init__(self,
                 action_handler: ActionHandler,
                 num_retries: int = 1,
                 timeout: int = 7200):
        """Creates a SubprocessActionHandler out of a normal ActionHandler.

        Args:
            action_handler (ActionHandler): The ActionHandler to convert
            num_retries (int): Number of retries for the SubprocessActionHandler. See the docstring for
                               SubprocessActionHandler for more information.
            timeout (int): Timeout in seconds for the SubprocessActionHandler. See the docstring for
                           SubprocessActionHandler for more information.
        """
        super().__init__(action_handler.action, num_retries=num_retries, timeout=timeout)

        self.action_handler = action_handler

    def setup(self):
        self.action_handler.setup()

    def cleanup(self, success: bool):
        self.action_handler.cleanup(success)

    def handle_failure(self):
        self.action_handler.handle_failure()

    def subprocess_target(self):
        return self.action_handler.handle()
